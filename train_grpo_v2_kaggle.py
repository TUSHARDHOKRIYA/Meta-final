"""
EduPath GRPO v2 — Anti-Mode-Collapse Training Script
Run on Kaggle with T4 GPU. Copy cells into notebook.
"""
# ===== CELL 1: Install =====
# !pip install -q trl>=0.15.0 transformers accelerate peft datasets bitsandbytes scipy
# !pip install -q --no-deps unsloth 2>/dev/null || echo 'Unsloth N/A'

# ===== CELL 2: Setup Environment =====
import subprocess,os,sys,json,re,random,logging,inspect
from collections import defaultdict
import numpy as np, torch

WORK='/kaggle/working'
REPO=f'{WORK}/edupath'
SAVE_DIR=f'{WORK}/edupath_grpo_v2'
FINAL_DIR=f'{WORK}/edupath_grpo_v2_final'
os.makedirs(SAVE_DIR,exist_ok=True)
os.makedirs(FINAL_DIR,exist_ok=True)

if not os.path.exists(REPO):
    subprocess.run(['git','clone',
        'https://huggingface.co/spaces/degree-checker-01/meta-new-space',REPO],check=True)
os.chdir(REPO); sys.path.insert(0,f'{REPO}/backend')

from unittest.mock import MagicMock
for m in ['llm_blender','llm_blender.agents','llm_blender.pair_ranker']:
    if m not in sys.modules: sys.modules[m]=MagicMock()

from environment.env import EduPathEnv
from environment.models import Action, ActionType
from environment.student import student_manager
from environment.curriculum import TOPIC_GRAPH
print(f'Topics: {list(TOPIC_GRAPH.keys())[:10]}...')
print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# Get ALL valid topic IDs
ALL_TOPIC_IDS = set(TOPIC_GRAPH.keys())
print(f'Total topics: {len(ALL_TOPIC_IDS)}')
print('Setup ✅')

# ===== CELL 3: State-Aware Reward Function =====
logger = logging.getLogger('edupath')
_parse_fails = 0
_action_counts = defaultdict(int)
_total_actions = 0

def parse_state_from_prompt(prompt):
    """Extract student state from the prompt text."""
    state = {'completed': [], 'available': [], 'job_readiness': 0.0,
             'current_topic': 'None', 'n_completed': 0}
    try:
        m = re.search(r'Completed:\s*\[([^\]]*)\]', prompt)
        if m and m.group(1).strip():
            state['completed'] = [x.strip().strip("'\"") for x in m.group(1).split(',') if x.strip()]
        state['n_completed'] = len(state['completed'])

        m = re.search(r'Available:\s*\[([^\]]*)\]', prompt)
        if m and m.group(1).strip():
            state['available'] = [x.strip().strip("'\"") for x in m.group(1).split(',') if x.strip()]

        m = re.search(r'Job readiness:\s*([\d.]+)', prompt)
        if m: state['job_readiness'] = float(m.group(1))

        m = re.search(r'Current topic:\s*(\S+)', prompt)
        if m: state['current_topic'] = m.group(1)
    except: pass
    return state


def edupath_reward_v2(completions, prompts=None, **kwargs):
    """State-aware reward that prevents mode collapse."""
    global _parse_fails, _total_actions
    rewards = []

    for i, comp in enumerate(completions):
        # Extract text
        if isinstance(comp, list):
            text = ''
            for msg in comp:
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    text = msg.get('content', ''); break
            if not text and comp:
                last = comp[-1]
                text = last.get('content', '') if isinstance(last, dict) else str(last)
        elif isinstance(comp, dict): text = comp.get('content', str(comp))
        else: text = str(comp)

        # Parse JSON
        match = re.search(r'\{[^}]+\}', text)
        if not match:
            _parse_fails += 1; rewards.append(-0.5); continue
        try:
            d = json.loads(match.group())
        except:
            _parse_fails += 1; rewards.append(-0.5); continue

        atype = d.get('type', '')
        topic_id = d.get('topic_id')
        if isinstance(topic_id, (int, float)): topic_id = None  # Reject numeric IDs

        # Validate action type
        valid_types = {a.value for a in ActionType}
        if atype not in valid_types:
            _parse_fails += 1; rewards.append(-0.3); continue

        # ── Format bonus ──
        reward = 0.1  # Valid JSON with valid type

        # ── Parse student state from prompt ──
        prompt_text = ''
        if prompts is not None:
            idx = i // 4 if len(prompts) < len(completions) else i  # GRPO generates num_generations per prompt
            if idx < len(prompts):
                p = prompts[idx]
                prompt_text = p if isinstance(p, str) else str(p)
        state = parse_state_from_prompt(prompt_text)
        n_comp = state['n_completed']
        available = state['available']
        jr = state['job_readiness']
        cur = state['current_topic']

        # ── State-aware scoring ──
        if atype == 'recommend_topic':
            if topic_id and topic_id in available:
                reward += 0.7  # Perfect: real available topic
            elif topic_id and topic_id in ALL_TOPIC_IDS:
                reward += 0.2  # Real topic but not available
            elif topic_id:
                reward += -0.1  # Fake topic ID
            else:
                reward += -0.2  # No topic_id

        elif atype == 'assign_quiz':
            if cur and cur != 'None':
                reward += 0.5  # Good: quiz on current topic
            elif n_comp > 0:
                reward += 0.2  # OK: quiz on something learned
            else:
                reward += -0.2  # Bad: nothing to quiz

        elif atype == 'assign_mini_project':
            if n_comp >= 3:
                reward += 0.5  # Good: enough knowledge
            elif n_comp >= 1:
                reward += 0.1  # Early but OK
            else:
                reward += -0.3  # Bad: no knowledge at all

        elif atype == 'assign_capstone':
            if n_comp >= 5 and jr >= 0.4:
                reward += 0.7  # Good: advanced student
            elif n_comp >= 3:
                reward += 0.1  # Premature
            else:
                reward += -0.4  # Way too early

        elif atype == 'recommend_resource':
            if cur and cur != 'None':
                reward += 0.3  # Good: supplement current study
            else:
                reward += 0.05  # Meh

        elif atype == 'mark_job_ready':
            if jr >= 0.8:
                reward += 0.9  # Perfect!
            elif jr >= 0.5:
                reward += -0.1  # Too early
            else:
                reward += -0.5  # Way too early

        # ── Diversity penalty ──
        _action_counts[atype] += 1
        _total_actions += 1
        if _total_actions > 50:
            freq = _action_counts[atype] / _total_actions
            if freq > 0.5:
                reward -= 0.15 * (freq - 0.5)  # Penalize dominant action

        # ── Topic ID validation bonus ──
        if topic_id and topic_id in ALL_TOPIC_IDS:
            reward += 0.1

        rewards.append(max(min(reward, 1.0), -1.0))  # Clamp

    return rewards


# Sanity check
tests = [
    '{"type":"recommend_topic","topic_id":"python_basics"}',
    '{"type":"assign_mini_project","topic_id":"123456"}',
    '{"type":"mark_job_ready"}',
    'garbage',
]
test_prompts = ['Completed: []\nAvailable: [python_basics, statistics]\nJob readiness: 0.00\nCurrent topic: None'] * 4
for t, r in zip(tests, edupath_reward_v2(tests, prompts=test_prompts)):
    print(f'  {t[:55]:55s} r={r:+.2f}')
# Reset counters after test
_action_counts.clear(); _total_actions = 0
print('Reward v2 ✅')


# ===== CELL 4: Diverse Dataset =====
from datasets import Dataset

PROFILES = [
  {'field':'tech','goal':'ML Engineer','hours':10,'skills':[]},
  {'field':'tech','goal':'Data Analyst','hours':15,'skills':[{'skill':'Python','level':'beginner'}]},
  {'field':'tech','goal':'Web Developer','hours':8,'skills':[{'skill':'JavaScript','level':'beginner'}]},
  {'field':'healthcare','goal':'AI in Medicine','hours':10,'skills':[{'skill':'Biology','level':'intermediate'}]},
  {'field':'business','goal':'Business Analytics','hours':12,'skills':[{'skill':'Excel','level':'intermediate'}]},
]

def make_prompt(o):
    available = o.get('available_topics', [])[:6]
    return f"""You are an AI tutoring agent. Given the student state, choose the SINGLE BEST next action.

RULES:
- recommend_topic: Use a topic_id from Available list. Best when student needs new material.
- assign_quiz: Best when student has a current_topic to test on.
- assign_mini_project: Best when student has completed 2+ topics.
- assign_capstone: Best when job_readiness >= 0.4 and 5+ topics completed.
- recommend_resource: Supplements current topic study.
- mark_job_ready: ONLY when job_readiness >= 0.8.

STATE:
- Completed: {o.get('completed_topics',[])}
- Available: {available}
- Quiz scores: {o.get('quiz_history_summary',{})}
- Job readiness: {o.get('job_readiness_score',0):.2f}
- Current topic: {o.get('current_topic','None')}

VALID TOPIC IDS: {available}
Respond ONLY with JSON: {{"type":"<action>","topic_id":"<id_from_available_or_null>"}}"""

train_prompts, eval_prompts = [], []
random.seed(42)

for seed in range(600):
    p = PROFILES[seed % len(PROFILES)]
    e = EduPathEnv(seed=seed)
    s = student_manager.create(name=f'v2_{seed}')
    onboard = {'target_field': p['field'], 'learning_goal': p['goal'], 'weekly_hours': p['hours']}
    if p['skills']: onboard['skills'] = p['skills']
    student_manager.update_from_onboarding(s.id, onboard)
    o = e.reset(student_id=s.id, seed=seed)

    # Advance environment randomly to create diverse states
    n_steps = random.randint(0, 6)
    for _ in range(n_steps):
        avail = o.available_topics
        if avail and random.random() < 0.6:
            action = Action(type=ActionType.RECOMMEND_TOPIC, topic_id=random.choice(avail))
        elif random.random() < 0.3:
            action = Action(type=ActionType.ASSIGN_QUIZ, topic_id=random.choice(avail) if avail else None)
        else:
            action = Action(type=ActionType.RECOMMEND_RESOURCE, topic_id=random.choice(avail) if avail else None)
        try:
            result = e.step(action)
            if result.done: break
            o = result.observation
        except: break

    prompt = make_prompt(o.model_dump())
    if seed < 500: train_prompts.append(prompt)
    else: eval_prompts.append(prompt)

train_dataset = Dataset.from_dict({'prompt': train_prompts})
print(f'Train: {len(train_prompts)} | Eval: {len(eval_prompts)}')

# Show diversity of states
n_comp_counts = []
for p in train_prompts[:50]:
    st = parse_state_from_prompt(p)
    n_comp_counts.append(st['n_completed'])
print(f'Completed topics distribution (first 50): {dict(zip(*np.unique(n_comp_counts, return_counts=True)))}')
print('Dataset v2 ✅')


# ===== CELL 5: Load Model =====
from trl import GRPOTrainer, GRPOConfig
MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
model = tokenizer = None
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        'unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit', max_seq_length=1024, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=16,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_alpha=32, lora_dropout=0.05, bias='none', use_gradient_checkpointing='unsloth')
    print('Unsloth 4-bit ✅')
except Exception as e:
    print(f'Unsloth failed: {e}')
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    from peft import get_peft_model, LoraConfig, TaskType
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','k_proj','v_proj','o_proj'],
        lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    print('HuggingFace 4-bit ✅')

total = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {total/1e6:.0f}M total, {train_p/1e6:.1f}M trainable')


# ===== CELL 6: Evaluate Function =====
def evaluate_model(model, tokenizer, prompts, label='Model', n=50):
    model.eval()
    rewards_all, actions_count, valid_json = [], {}, 0
    device = next(model.parameters()).device
    try:
        for p in prompts[:n]:
            inp = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=80, temperature=0.7, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            r = edupath_reward_v2([text], prompts=[p])[0]
            rewards_all.append(r)
            m = re.search(r'\{[^}]+\}', text)
            if m:
                valid_json += 1
                try:
                    d = json.loads(m.group()); at = d.get('type', 'unknown')
                    vt = {a.value for a in ActionType}
                    if at not in vt: at = f'invalid:{at}'
                    actions_count[at] = actions_count.get(at, 0) + 1
                except: actions_count['parse_fail'] = actions_count.get('parse_fail', 0) + 1
            else: actions_count['no_json'] = actions_count.get('no_json', 0) + 1
    finally: model.train()
    return {'label': label, 'mean': float(np.mean(rewards_all)), 'std': float(np.std(rewards_all)),
        'pos_rate': float(np.mean([r > 0 for r in rewards_all])), 'json_rate': valid_json / max(n, 1),
        'actions': actions_count, 'rewards': rewards_all}

# Reset diversity counters before baseline
_action_counts.clear(); _total_actions = 0
print('Baseline eval (50 held-out)...')
baseline = evaluate_model(model, tokenizer, eval_prompts, 'Baseline', 50)
print(f"  Mean reward: {baseline['mean']:+.3f} +/- {baseline['std']:.3f}")
print(f"  Positive %:  {baseline['pos_rate']:.0%}")
print(f"  Valid JSON:  {baseline['json_rate']:.0%}")
print(f"  Actions:     {baseline['actions']}")
print('Baseline ✅')


# ===== CELL 7: GRPO Training =====
import trl
from datetime import datetime

# Reset diversity counters for training
_action_counts.clear(); _total_actions = 0

_sig = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
args = dict(output_dir=SAVE_DIR, max_steps=600, per_device_train_batch_size=2,
    gradient_accumulation_steps=4, learning_rate=5e-5, lr_scheduler_type='cosine',
    warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=0.3,
    logging_steps=10, save_steps=100, save_total_limit=3,
    report_to='none', seed=42, bf16=_bf16, fp16=not _bf16)
for k, v in {'num_generations': 4, 'max_prompt_length': 512, 'beta': 0.1}.items():
    if k in _sig: args[k] = v
for c in ['max_completion_length', 'max_new_tokens', 'generation_max_new_tokens']:
    if c in _sig: args[c] = 100; break

config = GRPOConfig(**args)
trainer_kw = dict(model=model, args=config, train_dataset=train_dataset, reward_funcs=[edupath_reward_v2])
if trl.__version__ >= '0.15.0': trainer_kw['processing_class'] = tokenizer
else: trainer_kw['tokenizer'] = tokenizer
trainer = GRPOTrainer(**trainer_kw)

print(f'Training {config.max_steps} steps (beta={0.1}, saves to {SAVE_DIR})...')
t0 = datetime.now()
result = trainer.train()
mins = (datetime.now() - t0).total_seconds() / 60
print(f'\nDone! {mins:.1f}min | parse_fails={_parse_fails} ✅')


# ===== CELL 8: Post-Training Eval =====
_action_counts.clear(); _total_actions = 0  # Reset for clean eval
print('Post-training eval...')
trained = evaluate_model(model, tokenizer, eval_prompts, 'Trained', 50)
print(f"  Mean reward: {trained['mean']:+.3f} +/- {trained['std']:.3f}")
print(f"  Positive %:  {trained['pos_rate']:.0%}")
print(f"  Valid JSON:  {trained['json_rate']:.0%}")
print(f"  Actions:     {trained['actions']}")
imp = trained['mean'] - baseline['mean']
sym = 'BETTER' if imp > 0 else 'WORSE'
print(f'\n  Improvement: {imp:+.3f} ({sym})')


# ===== CELL 9: Plots =====
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

log = trainer.state.log_history
st = [e['step'] for e in log if 'loss' in e]; lo = [e['loss'] for e in log if 'loss' in e]
rs = [e['step'] for e in log if 'reward' in e]; rw = [e['reward'] for e in log if 'reward' in e]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes[0,0].plot(st, lo, 'royalblue', lw=1, alpha=0.4)
if len(lo) > 5: axes[0,0].plot(st, uniform_filter1d(lo, 5), 'royalblue', lw=2.5, label='Smoothed')
axes[0,0].set(xlabel='Step', ylabel='Loss', title='GRPO Training Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

if rs:
    axes[0,1].plot(rs, rw, 'forestgreen', lw=1, alpha=0.4)
    if len(rw) > 5: axes[0,1].plot(rs, uniform_filter1d(rw, 5), 'forestgreen', lw=2.5, label='Smoothed')
    axes[0,1].axhline(baseline['mean'], color='red', ls='--', lw=1.5, label=f"Baseline ({baseline['mean']:+.2f})")
    axes[0,1].legend()
axes[0,1].set(xlabel='Step', ylabel='Reward', title='Environment Reward'); axes[0,1].grid(True, alpha=0.3)

lbl = ['Mean\nReward', 'Positive\nRate', 'Valid\nJSON']
bv = [baseline['mean'], baseline['pos_rate'], baseline['json_rate']]
av = [trained['mean'], trained['pos_rate'], trained['json_rate']]
x = np.arange(3); w = 0.35
b1 = axes[1,0].bar(x-w/2, bv, w, label='Before', color='#e74c3c', alpha=0.85)
b2 = axes[1,0].bar(x+w/2, av, w, label='After', color='#2ecc71', alpha=0.85)
for bar in list(b1) + list(b2):
    h = bar.get_height(); off = 0.01 if h >= 0 else -0.04
    axes[1,0].text(bar.get_x()+bar.get_width()/2, h+off, f'{h:.2f}', ha='center', fontsize=9)
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(lbl)
axes[1,0].set(ylabel='Score', title='Before vs After GRPO'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3, axis='y')

all_a = sorted(set(list(baseline['actions'].keys()) + list(trained['actions'].keys())))
bv2 = [baseline['actions'].get(a, 0) for a in all_a]; av2 = [trained['actions'].get(a, 0) for a in all_a]
x2 = np.arange(len(all_a)); w2 = 0.35
axes[1,1].bar(x2-w2/2, bv2, w2, label='Before', color='#e74c3c', alpha=0.85)
axes[1,1].bar(x2+w2/2, av2, w2, label='After', color='#2ecc71', alpha=0.85)
axes[1,1].set_xticks(x2); axes[1,1].set_xticklabels(all_a, rotation=30, ha='right', fontsize=8)
axes[1,1].set(ylabel='Count', title='Action Type Distribution'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3, axis='y')

plt.suptitle(f'EduPath AI — GRPO v2 Results ({MODEL_ID})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{WORK}/grpo_training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {WORK}/grpo_training_results.png ✅')


# ===== CELL 10: Save & Upload =====
model.eval()
trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

results = {'model': MODEL_ID, 'steps': config.max_steps, 'time_min': round(mins, 1),
    'baseline': {'mean': baseline['mean'], 'pos_rate': baseline['pos_rate'],
                 'json_rate': baseline['json_rate'], 'actions': baseline['actions']},
    'trained': {'mean': trained['mean'], 'pos_rate': trained['pos_rate'],
                'json_rate': trained['json_rate'], 'actions': trained['actions']},
    'improvement': round(imp, 4), 'parse_fails': _parse_fails}
with open(f'{WORK}/grpo_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Sample outputs
print('Sample outputs:\n')
inp = tokenizer(eval_prompts[0], return_tensors='pt', truncation=True, max_length=512).to(next(model.parameters()).device)
for i in range(5):
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=80, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    r = edupath_reward_v2([text], prompts=[eval_prompts[0]])[0]
    print(f'  {i+1}. {text[:80]:80s} r={r:+.2f}')

print(f'\n{"="*55}')
print(f'  Model:  {MODEL_ID}')
print(f'  Steps:  {config.max_steps} | Time: {mins:.1f}min')
print(f'  Before: reward={baseline["mean"]:+.3f} pos={baseline["pos_rate"]:.0%} json={baseline["json_rate"]:.0%}')
print(f'  After:  reward={trained["mean"]:+.3f} pos={trained["pos_rate"]:.0%} json={trained["json_rate"]:.0%}')
print(f'  Delta:  {imp:+.3f}')
print(f'  Actions before: {baseline["actions"]}')
print(f'  Actions after:  {trained["actions"]}')
print(f'{"="*55}')


# ===== CELL 11: Merge & Upload to HF =====
# from peft import AutoPeftModelForCausalLM
# from huggingface_hub import HfApi, login
# login(token="YOUR_TOKEN")
# merged = AutoPeftModelForCausalLM.from_pretrained(FINAL_DIR, torch_dtype=torch.float16, device_map='auto')
# merged = merged.merge_and_unload()
# merged.save_pretrained(f'{WORK}/merged_v2')
# tokenizer.save_pretrained(f'{WORK}/merged_v2')
# api = HfApi()
# api.create_repo("degree-checker-01/edupath-grpo-tutor", repo_type="model", exist_ok=True)
# api.upload_folder(folder_path=f'{WORK}/merged_v2', repo_id='degree-checker-01/edupath-grpo-tutor', repo_type='model')
# print("Model uploaded ✅")
