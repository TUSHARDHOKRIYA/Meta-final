# ========== CELL 4: Multi-Task Dataset ==========
from datasets import Dataset

PROFILES = [
  {'field':'tech','goal':'ML Engineer','hours':10,'skills':[]},
  {'field':'tech','goal':'Data Analyst','hours':15,'skills':[{'skill':'Python','level':'beginner'}]},
  {'field':'tech','goal':'Web Developer','hours':8,'skills':[]},
  {'field':'healthcare','goal':'AI in Medicine','hours':10,'skills':[]},
  {'field':'business','goal':'Business Analytics','hours':12,'skills':[]},
  {'field':'law','goal':'Legal AI','hours':8,'skills':[]},
  {'field':'design','goal':'UX Designer','hours':10,'skills':[]},
]

def make_action_prompt(obs):
    avail = obs.get('available_topics', [])[:6]
    return f"""[ACTION] You are an AI tutoring agent. Choose the BEST next action for this student.
RULES:
- recommend_topic: pick from Available. Best when student needs new material.
- assign_quiz: when student has a current_topic to test.
- assign_mini_project: when 2+ topics completed.
- assign_capstone: when job_readiness >= 0.4 and 5+ completed.
- recommend_resource: supplements current study.
- mark_job_ready: ONLY when job_readiness >= 0.8.
STATE:
- Completed: {obs.get('completed_topics',[])}
- Available: {avail}
- Job readiness: {obs.get('job_readiness_score',0):.2f}
- Current topic: {obs.get('current_topic','None')}
VALID TOPIC IDS: {avail}
Respond ONLY with JSON: {{"type":"<action>","topic_id":"<from_available>"}}"""

def make_roadmap_prompt(field, goal):
    topics = FIELD_TOPICS.get(field, FIELD_TOPICS['tech'])[:8]
    return f"""[ROADMAP] Create a personalized learning roadmap for a student.
Student field: {field}
Student goal: {goal}
Available topics in this field: {topics}
Respond with JSON: {{"roadmap": [{{"topic_id": "<real_id>", "name": "<topic_name>", "reason": "<why_this_topic>", "estimated_hours": <hours>}}, ...]}}
Order topics by prerequisites. Use ONLY real topic IDs from the list above."""

def make_quiz_prompt(topic_id):
    t = TOPIC_GRAPH.get(topic_id)
    name = t.name if t else topic_id
    return f"""[QUIZ] Generate a quiz for the topic: {name} (id: {topic_id})
Create 4 multiple-choice questions testing understanding of {name}.
Respond with JSON: {{"questions": [{{"question": "<question_text>", "options": ["<A>", "<B>", "<C>", "<D>"], "correct": <0-3>, "explanation": "<why>"}}]}}"""

def make_resource_prompt(topic_id, level='beginner'):
    t = TOPIC_GRAPH.get(topic_id)
    name = t.name if t else topic_id
    real_res = RESOURCE_DB.get(topic_id, [])
    hint = f"Known resources: {[r.title for r in real_res[:3]]}" if real_res else ""
    return f"""[RESOURCE] Recommend learning resources for: {name} (id: {topic_id})
Student level: {level}
{hint}
Respond with JSON: {{"resources": [{{"title": "<name>", "url": "<url>", "type": "<course|article|video>", "difficulty": "<beginner|intermediate|advanced>", "reason": "<why_this_resource>"}}]}}
Recommend 3-5 high-quality, real resources."""

random.seed(42)
all_prompts = []

# Action prompts (40%) — diverse student states
for seed in range(240):
    p = PROFILES[seed % len(PROFILES)]
    e = EduPathEnv(seed=seed)
    s = student_manager.create(name=f'mt_{seed}')
    student_manager.update_from_onboarding(s.id, {'target_field':p['field'],'learning_goal':p['goal'],'weekly_hours':p['hours']})
    o = e.reset(student_id=s.id, seed=seed)
    for _ in range(random.randint(0, 6)):
        avail = o.available_topics
        if not avail: break
        at = random.choice([ActionType.RECOMMEND_TOPIC, ActionType.ASSIGN_QUIZ, ActionType.RECOMMEND_RESOURCE])
        try:
            result = e.step(Action(type=at, topic_id=random.choice(avail) if avail else None))
            if result.done: break
            o = result.observation
        except: break
    all_prompts.append(make_action_prompt(o.model_dump()))

# Roadmap prompts (20%)
for i in range(120):
    p = PROFILES[i % len(PROFILES)]
    all_prompts.append(make_roadmap_prompt(p['field'], p['goal']))

# Quiz prompts (20%)
topic_list = list(ALL_TOPICS)
for i in range(120):
    all_prompts.append(make_quiz_prompt(topic_list[i % len(topic_list)]))

# Resource prompts (20%)
levels = ['beginner', 'intermediate', 'advanced']
for i in range(120):
    all_prompts.append(make_resource_prompt(topic_list[i % len(topic_list)], levels[i % 3]))

random.shuffle(all_prompts)
train_prompts = all_prompts[:500]
eval_prompts = all_prompts[500:]

train_dataset = Dataset.from_dict({'prompt': train_prompts})
task_dist = defaultdict(int)
for p in train_prompts: task_dist[detect_task(p)] += 1
print(f'Train: {len(train_prompts)} | Eval: {len(eval_prompts)}')
print(f'Task distribution: {dict(task_dist)}')
print('Dataset ✅')


# ========== CELL 5: Load 7B Model ==========
MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = tokenizer = None

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        'unsloth/Qwen2.5-7B-Instruct-bnb-4bit', max_seq_length=768, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=16,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_alpha=32, lora_dropout=0.05, bias='none', use_gradient_checkpointing='unsloth')
    print('Unsloth 7B 4-bit ✅')
except Exception as e:
    print(f'Unsloth N/A ({e}), using HF')
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
        device_map='auto', attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    from peft import get_peft_model, LoraConfig, TaskType
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','k_proj','v_proj','o_proj'],
        lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    print('HF 7B 4-bit ✅')

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {total/1e6:.0f}M total, {trainable/1e6:.1f}M trainable')
print(f'VRAM used: {torch.cuda.memory_allocated()/1e9:.1f}GB')


# ========== CELL 6: Baseline Eval ==========
def evaluate_model(model, tokenizer, prompts, label='Model', n=40):
    model.eval(); rewards_all = []; task_rewards = defaultdict(list)
    actions_count = {}; valid_json = 0; device = next(model.parameters()).device
    try:
        for p in prompts[:n]:
            inp = tokenizer(p, return_tensors='pt', truncation=True, max_length=384).to(device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=150, temperature=0.7, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            r = multitask_reward([text], prompts=[p])[0]
            rewards_all.append(r); task = detect_task(p); task_rewards[task].append(r)
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                valid_json += 1
                try:
                    d = json.loads(m.group())
                    if 'type' in d: actions_count[d['type']] = actions_count.get(d['type'],0)+1
                except: pass
    finally: model.train()
    per_task = {k: float(np.mean(v)) for k,v in task_rewards.items()}
    return {'label':label,'mean':float(np.mean(rewards_all)),'std':float(np.std(rewards_all)),
        'pos_rate':float(np.mean([r>0 for r in rewards_all])),'json_rate':valid_json/max(n,1),
        'actions':actions_count,'per_task':per_task,'rewards':rewards_all}

_stats['actions'].clear(); _stats['total']=0
print('Baseline eval...')
baseline = evaluate_model(model, tokenizer, eval_prompts, 'Baseline', 40)
print(f"  Mean: {baseline['mean']:+.3f} | Pos: {baseline['pos_rate']:.0%} | JSON: {baseline['json_rate']:.0%}")
print(f"  Per-task: {baseline['per_task']}")
print('Baseline ✅')


# ========== CELL 7: GRPO Training ==========
import trl; from trl import GRPOTrainer, GRPOConfig; from datetime import datetime
_stats['actions'].clear(); _stats['total']=0; _stats['parse_fail']=0
_sig = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
_bf16 = torch.cuda.is_bf16_supported()

# Conservative config for 7B on T4
args = dict(output_dir=SAVE_DIR, max_steps=600, per_device_train_batch_size=1,
    gradient_accumulation_steps=8, learning_rate=3e-5, lr_scheduler_type='cosine',
    warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=0.3,
    logging_steps=10, save_steps=150, save_total_limit=2,
    gradient_checkpointing=True, report_to='none', seed=42, bf16=_bf16, fp16=not _bf16)
# Explicitly disable vLLM — not installed in this environment
if 'use_vllm' in _sig: args['use_vllm'] = False
for k, v in {'num_generations':2, 'max_prompt_length':384, 'beta':0.08}.items():
    if k in _sig: args[k] = v
for c in ['max_completion_length','max_new_tokens','generation_max_new_tokens']:
    if c in _sig: args[c] = 160; break

config = GRPOConfig(**args)
if not hasattr(model, 'warnings_issued'):
    model.warnings_issued = {}
kw = dict(model=model, args=config, train_dataset=train_dataset, reward_funcs=[multitask_reward])
if trl.__version__ >= '0.15.0': kw['processing_class'] = tokenizer
else: kw['tokenizer'] = tokenizer
trainer = GRPOTrainer(**kw)

print(f'Training 7B | {config.max_steps} steps | batch=1 | grad_accum=8 | beta=0.08')
print(f'VRAM before train: {torch.cuda.memory_allocated()/1e9:.1f}GB')
t0 = datetime.now()
result = trainer.train()
mins = (datetime.now() - t0).total_seconds() / 60
print(f'\nDone! {mins:.1f}min ✅')


# ========== CELL 8: Post-Training Eval ==========
_stats['actions'].clear(); _stats['total']=0
trained = evaluate_model(model, tokenizer, eval_prompts, 'Trained', 40)
imp = trained['mean'] - baseline['mean']
print(f"  Mean: {trained['mean']:+.3f} | Pos: {trained['pos_rate']:.0%} | JSON: {trained['json_rate']:.0%}")
print(f"  Per-task: {trained['per_task']}")
print(f"  Improvement: {imp:+.3f} ({'BETTER' if imp>0 else 'WORSE'})")


# ========== CELL 9: Plots ==========
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

log = trainer.state.log_history
st=[e['step'] for e in log if 'loss' in e]; lo=[e['loss'] for e in log if 'loss' in e]
rs=[e['step'] for e in log if 'reward' in e]; rw=[e['reward'] for e in log if 'reward' in e]

fig,axes = plt.subplots(2,2,figsize=(16,10))
axes[0,0].plot(st,lo,'royalblue',lw=1,alpha=0.4)
if len(lo)>5: axes[0,0].plot(st,uniform_filter1d(lo,5),'royalblue',lw=2.5,label='Smooth')
axes[0,0].set(xlabel='Step',ylabel='Loss',title='Training Loss'); axes[0,0].legend(); axes[0,0].grid(True,alpha=0.3)

if rs:
    axes[0,1].plot(rs,rw,'forestgreen',lw=1,alpha=0.4)
    if len(rw)>5: axes[0,1].plot(rs,uniform_filter1d(rw,5),'forestgreen',lw=2.5,label='Smooth')
    axes[0,1].axhline(baseline['mean'],color='red',ls='--',lw=1.5,label=f"Baseline ({baseline['mean']:+.2f})")
    axes[0,1].legend()
axes[0,1].set(xlabel='Step',ylabel='Reward',title='Multi-Task Reward'); axes[0,1].grid(True,alpha=0.3)

lbl=['Mean\nReward','Positive\nRate','Valid\nJSON']
bv=[baseline['mean'],baseline['pos_rate'],baseline['json_rate']]
av=[trained['mean'],trained['pos_rate'],trained['json_rate']]
x=np.arange(3); w=0.35
b1=axes[1,0].bar(x-w/2,bv,w,label='Before',color='#e74c3c',alpha=0.85)
b2=axes[1,0].bar(x+w/2,av,w,label='After',color='#2ecc71',alpha=0.85)
for bar in list(b1)+list(b2):
    h=bar.get_height(); axes[1,0].text(bar.get_x()+bar.get_width()/2,h+0.01,f'{h:.2f}',ha='center',fontsize=9)
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(lbl)
axes[1,0].set(ylabel='Score',title='Before vs After GRPO'); axes[1,0].legend(); axes[1,0].grid(True,alpha=0.3,axis='y')

tasks = list(set(list(baseline.get('per_task',{}).keys())+list(trained.get('per_task',{}).keys())))
if tasks:
    bpt=[baseline.get('per_task',{}).get(t,0) for t in tasks]
    apt=[trained.get('per_task',{}).get(t,0) for t in tasks]
    x2=np.arange(len(tasks)); w2=0.35
    axes[1,1].bar(x2-w2/2,bpt,w2,label='Before',color='#e74c3c',alpha=0.85)
    axes[1,1].bar(x2+w2/2,apt,w2,label='After',color='#2ecc71',alpha=0.85)
    axes[1,1].set_xticks(x2); axes[1,1].set_xticklabels(tasks,fontsize=9)
    axes[1,1].set(ylabel='Reward',title='Per-Task Improvement'); axes[1,1].legend()
axes[1,1].grid(True,alpha=0.3,axis='y')

plt.suptitle('EduPath AI — Multi-Task GRPO (7B)',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig(f'{WORK}/grpo_training_results.png',dpi=150,bbox_inches='tight')
plt.show()

# Save JSON
results = {'model':MODEL_ID,'steps':600,'time_min':round(mins,1),
    'baseline':{'mean':baseline['mean'],'pos_rate':baseline['pos_rate'],'json_rate':baseline['json_rate'],'per_task':baseline.get('per_task',{})},
    'trained':{'mean':trained['mean'],'pos_rate':trained['pos_rate'],'json_rate':trained['json_rate'],'per_task':trained.get('per_task',{})},
    'improvement':round(imp,4)}
with open(f'{WORK}/grpo_results.json','w') as f: json.dump(results,f,indent=2)
print(f'Saved plot + JSON ✅')


# ========== CELL 10: Save Model ==========
trainer.save_model(FINAL); tokenizer.save_pretrained(FINAL)
print(f'Model saved to {FINAL} ✅')

# Print summary + samples
print(f'\n{"="*60}')
print(f'  Model:  {MODEL_ID} (7B)')
print(f'  Steps:  600 | Time: {mins:.1f}min')
print(f'  Before: reward={baseline["mean"]:+.3f} pos={baseline["pos_rate"]:.0%} json={baseline["json_rate"]:.0%}')
print(f'  After:  reward={trained["mean"]:+.3f} pos={trained["pos_rate"]:.0%} json={trained["json_rate"]:.0%}')
print(f'  Delta:  {imp:+.3f}')
for t in ['action','roadmap','quiz','resource']:
    b = baseline.get('per_task',{}).get(t,0); a = trained.get('per_task',{}).get(t,0)
    print(f'    {t:10s}: {b:+.3f} → {a:+.3f} ({"+" if a>b else ""}{a-b:.3f})')
print(f'{"="*60}')


# ========== CELL 11: Merge & Upload ==========
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if HF_TOKEN:
    from peft import AutoPeftModelForCausalLM
    from huggingface_hub import HfApi, login
    login(token=HF_TOKEN)
    print("Merging LoRA...")
    merged = AutoPeftModelForCausalLM.from_pretrained(FINAL, dtype=torch.float16, device_map='auto')
    merged = merged.merge_and_unload()
    merged.save_pretrained(f'{WORK}/merged_v3')
    tokenizer.save_pretrained(f'{WORK}/merged_v3')
    api = HfApi()
    api.create_repo("degree-checker-01/edupath-grpo-tutor", repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=f'{WORK}/merged_v3', repo_id='degree-checker-01/edupath-grpo-tutor', repo_type='model')
    api.upload_file(path_or_fileobj=f'{WORK}/grpo_training_results.png', path_in_repo='grpo_training_results.png', repo_id='degree-checker-01/edupath-grpo-tutor', repo_type='model')
    api.upload_file(path_or_fileobj=f'{WORK}/grpo_results.json', path_in_repo='grpo_results.json', repo_id='degree-checker-01/edupath-grpo-tutor', repo_type='model')
    print("✅ Model + evidence uploaded to HuggingFace")
else:
    print("⚠️ HF_TOKEN not set — skipping upload. Set it as a Space secret.")

