# ========== CELL 4: Multi-Task Dataset (4K Diverse Prompts) ==========
from datasets import Dataset

PROFILES = [
  # Tech profiles — varied goals and skill levels
  {'field':'tech','goal':'ML Engineer','hours':10,'skills':[]},
  {'field':'tech','goal':'Data Analyst','hours':15,'skills':[{'skill':'Python','level':'beginner'}]},
  {'field':'tech','goal':'Web Developer','hours':8,'skills':[]},
  {'field':'tech','goal':'Backend Engineer','hours':12,'skills':[{'skill':'Python','level':'intermediate'}]},
  {'field':'tech','goal':'DevOps Engineer','hours':6,'skills':[]},
  {'field':'tech','goal':'Full Stack Developer','hours':20,'skills':[{'skill':'JavaScript','level':'beginner'}]},
  # Healthcare
  {'field':'healthcare','goal':'AI in Medicine','hours':10,'skills':[]},
  {'field':'healthcare','goal':'Health Data Scientist','hours':14,'skills':[{'skill':'Statistics','level':'beginner'}]},
  {'field':'healthcare','goal':'Clinical Informaticist','hours':8,'skills':[]},
  # Business
  {'field':'business','goal':'Business Analytics','hours':12,'skills':[]},
  {'field':'business','goal':'Product Manager','hours':10,'skills':[{'skill':'Data Analysis','level':'beginner'}]},
  {'field':'business','goal':'Management Consultant','hours':16,'skills':[]},
  # Law
  {'field':'law','goal':'Legal AI','hours':8,'skills':[]},
  {'field':'law','goal':'Compliance Analyst','hours':10,'skills':[]},
  # Design
  {'field':'design','goal':'UX Designer','hours':10,'skills':[]},
  {'field':'design','goal':'Product Designer','hours':12,'skills':[{'skill':'Design Thinking','level':'beginner'}]},
  {'field':'design','goal':'UI Engineer','hours':8,'skills':[]},
  # Cross-disciplinary
  {'field':'tech','goal':'AI Researcher','hours':20,'skills':[{'skill':'Math','level':'intermediate'}]},
  {'field':'business','goal':'Data-Driven Strategist','hours':10,'skills':[]},
  {'field':'healthcare','goal':'Biostatistician','hours':12,'skills':[{'skill':'R','level':'beginner'}]},
]

# --- Action prompt templates (varied phrasings) ---
ACTION_TEMPLATES = [
    lambda obs: f"""[ACTION] You are an AI tutoring agent. Choose the BEST next action for this student.
RULES:
- recommend_topic: pick from Available. Best when student needs new material.
- assign_quiz: when student has a current_topic to test.
- assign_mini_project: when 2+ topics completed.
- assign_capstone: when job_readiness >= 0.4 and 5+ completed.
- recommend_resource: supplements current study.
- mark_job_ready: ONLY when job_readiness >= 0.8.
STATE:
- Completed: {obs.get('completed_topics',[])}
- Available: {obs.get('available_topics',[])[:6]}
- Job readiness: {obs.get('job_readiness_score',0):.2f}
- Current topic: {obs.get('current_topic','None')}
VALID TOPIC IDS: {obs.get('available_topics',[])[:6]}
Respond ONLY with JSON: {{"type":"<action>","topic_id":"<from_available>"}}""",

    lambda obs: f"""[ACTION] As a personalized tutoring agent, decide the optimal next step.
Student progress: {len(obs.get('completed_topics',[]))} topics completed out of available {len(obs.get('available_topics',[]))}
Completed so far: {obs.get('completed_topics',[])}
Available next: {obs.get('available_topics',[])[:6]}
Current study: {obs.get('current_topic','None')}
Job readiness score: {obs.get('job_readiness_score',0):.2f}
Choose from: recommend_topic, assign_quiz, assign_mini_project, assign_capstone, recommend_resource, mark_job_ready
Output JSON: {{"type":"<action_type>","topic_id":"<topic_from_available>"}}""",

    lambda obs: f"""[ACTION] Analyze this student's learning state and select the most impactful action.
Learning state summary:
  - Topics mastered: {obs.get('completed_topics',[])}
  - Topics available: {obs.get('available_topics',[])[:6]}
  - Currently studying: {obs.get('current_topic','None')}
  - Readiness metric: {obs.get('job_readiness_score',0):.2f}
Guidelines: recommend_topic for new learners, assign_quiz to test current knowledge, assign_mini_project after 2+ completions, assign_capstone when ready (>=0.4 readiness, 5+ done), mark_job_ready only at >=0.8.
Return JSON: {{"type":"<action>","topic_id":"<valid_topic_id>"}}""",
]

# --- Roadmap prompt templates ---
ROADMAP_TEMPLATES = [
    lambda field, goal, topics: f"""[ROADMAP] Create a personalized learning roadmap for a student.
Student field: {field}
Student goal: {goal}
Available topics in this field: {topics}
Respond with JSON: {{"roadmap": [{{"topic_id": "<real_id>", "name": "<topic_name>", "reason": "<why_this_topic>", "estimated_hours": <hours>}}, ...]}}
Order topics by prerequisites. Use ONLY real topic IDs from the list above.""",

    lambda field, goal, topics: f"""[ROADMAP] Design a structured learning path for a {field} student aiming to become a {goal}.
Available curriculum topics: {topics}
Create an ordered sequence respecting prerequisites.
Output JSON: {{"roadmap": [{{"topic_id": "<id>", "name": "<name>", "reason": "<rationale>", "estimated_hours": <hrs>}}, ...]}}""",

    lambda field, goal, topics: f"""[ROADMAP] A student in {field} wants to achieve: {goal}
Build a step-by-step roadmap using these topics: {topics}
Ensure prerequisites come first. Include 4-8 topics.
JSON format: {{"roadmap": [{{"topic_id": "<id>", "name": "<name>", "reason": "<why>", "estimated_hours": <h>}}, ...]}}""",
]

# --- Quiz prompt templates ---
QUIZ_TEMPLATES = [
    lambda name, tid: f"""[QUIZ] Generate a quiz for the topic: {name} (id: {tid})
Create 4 multiple-choice questions testing understanding of {name}.
Respond with JSON: {{"questions": [{{"question": "<question_text>", "options": ["<A>", "<B>", "<C>", "<D>"], "correct": <0-3>, "explanation": "<why>"}}]}}""",

    lambda name, tid: f"""[QUIZ] Create an assessment for {name} (topic_id: {tid}).
Write 4 MCQ questions at varying difficulty levels (easy to hard).
Each question must have exactly 4 options with one correct answer.
Output: {{"questions": [{{"question": "<text>", "options": ["<a>","<b>","<c>","<d>"], "correct": <index>, "explanation": "<reasoning>"}}]}}""",

    lambda name, tid: f"""[QUIZ] Design a knowledge check for students studying {name} ({tid}).
Include 3-5 questions covering key concepts, common misconceptions, and practical application.
Format: {{"questions": [{{"question": "<q>", "options": ["<o1>","<o2>","<o3>","<o4>"], "correct": <0-3>, "explanation": "<exp>"}}]}}""",
]

# --- Resource prompt templates ---
RESOURCE_TEMPLATES = [
    lambda name, tid, level, hint: f"""[RESOURCE] Recommend learning resources for: {name} (id: {tid})
Student level: {level}
{hint}
Respond with JSON: {{"resources": [{{"title": "<name>", "url": "<url>", "type": "<course|article|video>", "difficulty": "<beginner|intermediate|advanced>", "reason": "<why_this_resource>"}}]}}
Recommend 3-5 high-quality, real resources.""",

    lambda name, tid, level, hint: f"""[RESOURCE] Find the best study materials for {name} ({tid}) suitable for a {level} learner.
{hint}
Include a mix of formats (videos, articles, courses).
JSON: {{"resources": [{{"title": "<t>", "url": "<u>", "type": "<course|article|video|tutorial>", "difficulty": "{level}", "reason": "<why>"}}]}}""",

    lambda name, tid, level, hint: f"""[RESOURCE] A {level}-level student needs resources to learn {name} (id: {tid}).
{hint}
Suggest 3-5 diverse, high-quality resources with explanations.
Output: {{"resources": [{{"title": "<title>", "url": "<link>", "type": "<type>", "difficulty": "<level>", "reason": "<rationale>"}}]}}""",
]

random.seed(42)
all_prompts = []

# Action prompts (40% = ~1600) — diverse student states with varied templates
for seed in range(1600):
    p = PROFILES[seed % len(PROFILES)]
    e = EduPathEnv(seed=seed)
    s = student_manager.create(name=f'mt_{seed}')
    student_manager.update_from_onboarding(s.id, {'target_field':p['field'],'learning_goal':p['goal'],'weekly_hours':p['hours']})
    o = e.reset(student_id=s.id, seed=seed)
    # Vary depth of exploration per student
    depth = random.randint(0, 8)
    for _ in range(depth):
        avail = o.available_topics
        if not avail: break
        at = random.choice([ActionType.RECOMMEND_TOPIC, ActionType.ASSIGN_QUIZ, ActionType.RECOMMEND_RESOURCE])
        try:
            result = e.step(Action(type=at, topic_id=random.choice(avail) if avail else None))
            if result.done: break
            o = result.observation
        except: break
    # Pick a random template for variety
    template = ACTION_TEMPLATES[seed % len(ACTION_TEMPLATES)]
    all_prompts.append(template(o.model_dump()))

# Roadmap prompts (20% = ~800) — varied fields, goals, templates
goals_extra = {
    'tech': ['AI Startup Founder', 'Robotics Engineer', 'Cloud Architect', 'Mobile Developer', 'Game Developer'],
    'healthcare': ['Digital Health PM', 'Medical AI Researcher', 'Health Tech Analyst'],
    'business': ['Startup Founder', 'Financial Analyst', 'Operations Manager'],
    'law': ['Legal Tech Developer', 'IP Analyst', 'RegTech Specialist'],
    'design': ['Design Lead', 'Creative Director', 'Interaction Designer'],
}
roadmap_idx = 0
for field in FIELD_TOPICS.keys():
    topics = FIELD_TOPICS.get(field, FIELD_TOPICS['tech'])[:8]
    base_goals = [p['goal'] for p in PROFILES if p['field'] == field]
    extra_goals = goals_extra.get(field, [])
    all_goals = list(set(base_goals + extra_goals))
    for goal in all_goals:
        for tmpl in ROADMAP_TEMPLATES:
            all_prompts.append(tmpl(field, goal, topics))
            roadmap_idx += 1
# Pad to ~800
while roadmap_idx < 800:
    p = PROFILES[roadmap_idx % len(PROFILES)]
    topics = FIELD_TOPICS.get(p['field'], FIELD_TOPICS['tech'])[:8]
    tmpl = ROADMAP_TEMPLATES[roadmap_idx % len(ROADMAP_TEMPLATES)]
    all_prompts.append(tmpl(p['field'], p['goal'], topics))
    roadmap_idx += 1

# Quiz prompts (20% = ~800) — all topics × all templates × varied
topic_list = list(ALL_TOPICS)
quiz_idx = 0
for i, tid in enumerate(topic_list):
    t = TOPIC_GRAPH.get(tid)
    name = t.name if t else tid
    for tmpl in QUIZ_TEMPLATES:
        all_prompts.append(tmpl(name, tid))
        quiz_idx += 1
# Pad with random combos to reach ~800
while quiz_idx < 800:
    tid = topic_list[quiz_idx % len(topic_list)]
    t = TOPIC_GRAPH.get(tid)
    name = t.name if t else tid
    tmpl = QUIZ_TEMPLATES[quiz_idx % len(QUIZ_TEMPLATES)]
    all_prompts.append(tmpl(name, tid))
    quiz_idx += 1

# Resource prompts (20% = ~800) — all topics × levels × templates
levels = ['beginner', 'intermediate', 'advanced']
res_idx = 0
for tid in topic_list:
    t = TOPIC_GRAPH.get(tid)
    name = t.name if t else tid
    real_res = RESOURCE_DB.get(tid, [])
    hint = f"Known resources: {[r.title for r in real_res[:3]]}" if real_res else ""
    for level in levels:
        tmpl = RESOURCE_TEMPLATES[res_idx % len(RESOURCE_TEMPLATES)]
        all_prompts.append(tmpl(name, tid, level, hint))
        res_idx += 1
# Pad to ~800
while res_idx < 800:
    tid = topic_list[res_idx % len(topic_list)]
    t = TOPIC_GRAPH.get(tid)
    name = t.name if t else tid
    real_res = RESOURCE_DB.get(tid, [])
    hint = f"Known resources: {[r.title for r in real_res[:3]]}" if real_res else ""
    level = levels[res_idx % 3]
    tmpl = RESOURCE_TEMPLATES[res_idx % len(RESOURCE_TEMPLATES)]
    all_prompts.append(tmpl(name, tid, level, hint))
    res_idx += 1

random.shuffle(all_prompts)
train_prompts = all_prompts[:3500]
eval_prompts = all_prompts[3500:]

train_dataset = Dataset.from_dict({'prompt': train_prompts})
task_dist = defaultdict(int)
for p in train_prompts: task_dist[detect_task(p)] += 1
print(f'Train: {len(train_prompts)} | Eval: {len(eval_prompts)}')
print(f'Task distribution: {dict(task_dist)}')
print(f'Total prompts generated: {len(all_prompts)}')
print(f'Unique profiles: {len(PROFILES)} | Unique topics: {len(topic_list)}')
print('Dataset ✅')


# ========== CELL 5: Load 3B Model ==========
MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = tokenizer = None
print(f'[MODEL] Downloading {MODEL_ID}...')

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        'unsloth/Qwen2.5-3B-Instruct-bnb-4bit', max_seq_length=768, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=16,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_alpha=32, lora_dropout=0.05, bias='none', use_gradient_checkpointing='unsloth')
    print('Unsloth 3B 4-bit ✅')
except Exception as e:
    print(f'Unsloth N/A ({e}), using HF 3B')
    print('[MODEL] Loading tokenizer...')
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
        device_map='auto', attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    print('[MODEL] Applying LoRA adapters...')
    from peft import get_peft_model, LoraConfig, TaskType
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    print('HF 3B 4-bit ✅')
    print('[MODEL] Model ready.')

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {total/1e6:.0f}M total, {trainable/1e6:.1f}M trainable')
print(f'VRAM used: {torch.cuda.memory_allocated()/1e9:.1f}GB')


# ========== CELL 6: Baseline Eval ==========
def evaluate_model(model, tokenizer, prompts, label='Model', n=40):
    model.eval(); rewards_all = []; task_rewards = defaultdict(list)
    actions_count = {}; valid_json = 0; device = next(model.parameters()).device
    try:
        for idx, p in enumerate(prompts[:n]):
            print(f'  [EVAL] {label} sample {idx+1}/{min(n,len(prompts))}...', end=' ', flush=True)
            inp = tokenizer(p, return_tensors='pt', truncation=True, max_length=384).to(device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=150, temperature=0.7, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            r = multitask_reward([text], prompts=[p])[0]
            print(f'r={r:+.2f}', flush=True)
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
baseline = evaluate_model(model, tokenizer, eval_prompts, 'Baseline', 20)
print(f"  Mean: {baseline['mean']:+.3f} | Pos: {baseline['pos_rate']:.0%} | JSON: {baseline['json_rate']:.0%}")
print(f"  Per-task: {baseline['per_task']}")
print('Baseline ✅')


# ========== CELL 7: GRPO Training ==========
print('[TRAIN] Importing TRL...')
import trl; from trl import GRPOTrainer, GRPOConfig; from datetime import datetime
print(f'[TRAIN] TRL v{trl.__version__} loaded ✅')
_stats['actions'].clear(); _stats['total']=0; _stats['parse_fail']=0
_sig = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
_bf16 = torch.cuda.is_bf16_supported()

# Optimized config for 3B on L40S (48GB VRAM) — max out batch & generations
args = dict(output_dir=SAVE_DIR, max_steps=1500, per_device_train_batch_size=4,
    gradient_accumulation_steps=4, learning_rate=3e-5, lr_scheduler_type='cosine',
    warmup_steps=120, weight_decay=0.01, max_grad_norm=0.3,
    logging_steps=1, save_steps=300, save_total_limit=3,
    gradient_checkpointing=True, report_to='none', seed=42, bf16=_bf16, fp16=not _bf16)
# Explicitly disable vLLM — not installed in this environment
if 'use_vllm' in _sig: args['use_vllm'] = False
for k, v in {'num_generations':8, 'max_prompt_length':768, 'beta':0.04}.items():
    if k in _sig: args[k] = v
for c in ['max_completion_length','max_new_tokens','generation_max_new_tokens']:
    if c in _sig: args[c] = 256; break

config = GRPOConfig(**args)
print('[TRAIN] Initializing GRPOTrainer...')
if not hasattr(model, 'warnings_issued'):
    model.warnings_issued = {}
kw = dict(model=model, args=config, train_dataset=train_dataset, reward_funcs=[multitask_reward])
_trainer_sig = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
if 'processing_class' in _trainer_sig:
    kw['processing_class'] = tokenizer
else:
    kw['tokenizer'] = tokenizer
trainer = GRPOTrainer(**kw)

print(f'Training 3B | {config.max_steps} steps | batch=4 | grad_accum=4 | gen=8 | beta=0.04 | L40S 48GB')
print(f'VRAM before train: {torch.cuda.memory_allocated()/1e9:.1f}GB')
t0 = datetime.now()
result = trainer.train()
mins = (datetime.now() - t0).total_seconds() / 60
print(f'\nDone! {mins:.1f}min ✅')


# ========== CELL 8: Post-Training Eval ==========
_stats['actions'].clear(); _stats['total']=0
trained = evaluate_model(model, tokenizer, eval_prompts, 'Trained', 20)
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

plt.suptitle('EduPath AI — Multi-Task GRPO (3B) — L40S 48GB',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig(f'{WORK}/grpo_training_results.png',dpi=150,bbox_inches='tight')
plt.show()

# Save JSON
results = {'model':MODEL_ID,'steps':1500,'hardware':'L40S-48GB','time_min':round(mins,1),
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
print(f'  Model:  {MODEL_ID} (3B)')
print(f'  Steps:  1500 | Time: {mins:.1f}min | GPU: L40S 48GB')
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
    api.create_repo("degree-checker-01/edupath-grpo-3b-l40s", repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=f'{WORK}/merged_v3', repo_id='degree-checker-01/edupath-grpo-3b-l40s', repo_type='model')
    api.upload_file(path_or_fileobj=f'{WORK}/grpo_training_results.png', path_in_repo='grpo_training_results.png', repo_id='degree-checker-01/edupath-grpo-3b-l40s', repo_type='model')
    api.upload_file(path_or_fileobj=f'{WORK}/grpo_results.json', path_in_repo='grpo_results.json', repo_id='degree-checker-01/edupath-grpo-3b-l40s', repo_type='model')
    print("✅ Model + evidence uploaded to HuggingFace")
else:
    print("⚠️ HF_TOKEN not set — skipping upload. Set it as a Space secret.")

