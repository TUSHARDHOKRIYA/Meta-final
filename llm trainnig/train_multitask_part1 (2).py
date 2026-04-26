# ========== CELL 1: Setup (deps installed via Dockerfile) ==========

# ========== CELL 2: Setup ==========
import subprocess,os,sys,json,re,random,logging,inspect,time,types
from collections import defaultdict
import numpy as np, torch

WORK=os.environ.get('WORK','/app/output'); REPO=f'{WORK}/edupath'
SAVE_DIR=f'{WORK}/grpo_multitask'; FINAL=f'{WORK}/grpo_final_v3'
for d in [SAVE_DIR,FINAL]: os.makedirs(d,exist_ok=True)

if not os.path.exists(REPO):
    for _attempt in range(3):
        result = subprocess.run(['git','clone','https://huggingface.co/spaces/degree-checker-01/meta-new-space',REPO],
            capture_output=True, text=True)
        if result.returncode == 0:
            break
        print(f'  git clone attempt {_attempt+1}/3 failed: {result.stderr.strip()}')
        if _attempt < 2: time.sleep(5)
    else:
        raise RuntimeError(f'git clone failed after 3 attempts: {result.stderr}')
os.chdir(REPO); sys.path.insert(0,f'{REPO}/backend')
# Mock llm_blender with proper ModuleSpec — TRL 0.15.0 both:
# 1) calls find_spec() which rejects __spec__=None
# 2) does `import llm_blender` which needs the mock present in sys.modules
from importlib.machinery import ModuleSpec
for m in ['llm_blender','llm_blender.agents','llm_blender.pair_ranker']:
    if m not in sys.modules:
        _mod = types.ModuleType(m)
        _mod.__spec__ = ModuleSpec(m, None)  # real spec so find_spec() passes
        _mod.__path__ = []
        _mod.__package__ = m.split('.')[0]
        _mod.__file__ = None
        sys.modules[m] = _mod

# Mock vllm — TRL 0.15.0 imports it unconditionally at module load time
# even when use_vllm=False. The mock just needs LLM and SamplingParams present.
for m in ['vllm', 'vllm.sampling_params']:
    if m not in sys.modules:
        _mod = types.ModuleType(m)
        _mod.__spec__ = ModuleSpec(m, None)
        _mod.__path__ = []
        _mod.__package__ = m.split('.')[0]
        _mod.__file__ = None
        sys.modules[m] = _mod

# Stub out the two names TRL actually imports from vllm
class _LLM:
    def __init__(self, *a, **kw): raise RuntimeError("vllm not installed — use_vllm must be False")

class _SamplingParams:
    def __init__(self, *a, **kw): pass

sys.modules['vllm'].LLM = _LLM
sys.modules['vllm'].SamplingParams = _SamplingParams

from environment.env import EduPathEnv
from environment.models import Action, ActionType
from environment.student import student_manager
from environment.curriculum import TOPIC_GRAPH, RESOURCE_DB

ALL_TOPICS = set(TOPIC_GRAPH.keys())
FIELD_TOPICS = defaultdict(list)
TOPIC_PREREQS = {}
for tid, t in TOPIC_GRAPH.items():
    FIELD_TOPICS[t.field].append(tid)
    TOPIC_PREREQS[tid] = t.prerequisites

print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
print(f'Topics: {len(ALL_TOPICS)} | Fields: {list(FIELD_TOPICS.keys())}')

# ========== CELL 3: Multi-Task Reward ==========
_stats = {'parse_fail':0, 'actions': defaultdict(int), 'total': 0}

TASK_TAG = {'[ACTION]':'action','[ROADMAP]':'roadmap','[QUIZ]':'quiz','[RESOURCE]':'resource'}

def detect_task(prompt):
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content', '')
                for tag, task in TASK_TAG.items():
                    if tag in content[:200]:
                        return task
        return 'action'
    for tag, task in TASK_TAG.items():
        if tag in str(prompt)[:200]: return task
    return 'action'

def check_prereq_order(topic_list):
    seen = set()
    violations = 0
    for tid in topic_list:
        prereqs = TOPIC_PREREQS.get(tid, [])
        for p in prereqs:
            if p not in seen and p in set(topic_list): violations += 1
        seen.add(tid)
    return violations

def reward_action(d, prompt):
    atype = d.get('type',''); tid = d.get('topic_id')
    if isinstance(tid,(int,float)): tid = None
    valid = {a.value for a in ActionType}
    if atype not in valid: return -0.3
    r = 0.1
    st = {}
    m = re.search(r'Completed:\s*\[([^\]]*)\]', str(prompt))
    nc = len([x for x in m.group(1).split(',') if x.strip()]) if m and m.group(1).strip() else 0
    m2 = re.search(r'Available:\s*\[([^\]]*)\]', str(prompt))
    avail = [x.strip().strip("'\"") for x in m2.group(1).split(',') if x.strip()] if m2 and m2.group(1).strip() else []
    m3 = re.search(r'Job readiness:\s*([\d.]+)', str(prompt))
    jr = float(m3.group(1)) if m3 else 0.0
    m4 = re.search(r'Current topic:\s*(\S+)', str(prompt))
    cur = m4.group(1) if m4 else 'None'

    if atype == 'recommend_topic':
        if tid and tid in avail: r += 0.7
        elif tid and tid in ALL_TOPICS: r += 0.15
        else: r -= 0.1
    elif atype == 'assign_quiz':
        if cur != 'None': r += 0.5
        elif nc > 0: r += 0.2
        else: r -= 0.2
    elif atype == 'assign_mini_project':
        if nc >= 3: r += 0.5
        elif nc >= 1: r += 0.1
        else: r -= 0.3
    elif atype == 'assign_capstone':
        if nc >= 5 and jr >= 0.4: r += 0.7
        elif nc >= 3: r += 0.1
        else: r -= 0.4
    elif atype == 'recommend_resource':
        r += 0.3 if cur != 'None' else 0.05
    elif atype == 'mark_job_ready':
        if jr >= 0.8: r += 0.9
        elif jr >= 0.5: r -= 0.1
        else: r -= 0.5
    if tid and tid in ALL_TOPICS: r += 0.1
    _stats['actions'][atype] += 1; _stats['total'] += 1
    if _stats['total'] > 50:
        freq = _stats['actions'][atype] / _stats['total']
        if freq > 0.5: r -= 0.15 * (freq - 0.5)
    return r

def reward_roadmap(d, prompt):
    r = 0.0
    roadmap = d.get('roadmap', d.get('topics', []))
    if not isinstance(roadmap, list) or len(roadmap) == 0: return -0.3
    if 3 <= len(roadmap) <= 12: r += 0.2
    else: r += 0.05
    valid_ids = 0
    for item in roadmap:
        if isinstance(item, dict):
            tid = item.get('topic_id','')
            if tid in ALL_TOPICS: valid_ids += 1
            if item.get('reason') or item.get('description'): r += 0.02
    r += min(0.3, valid_ids * 0.05)
    tids = [i.get('topic_id','') for i in roadmap if isinstance(i,dict)]
    v = check_prereq_order(tids)
    r += 0.2 if v == 0 else max(0, 0.2 - v * 0.05)
    m = re.search(r'field["\s:]+(\w+)', str(prompt), re.I)
    if m:
        field = m.group(1).lower()
        field_match = sum(1 for t in tids if t in FIELD_TOPICS.get(field,[]))
        r += min(0.15, field_match * 0.03)
    return r

def reward_quiz(d, prompt):
    r = 0.0
    qs = d.get('questions', [])
    if not isinstance(qs, list) or len(qs) == 0: return -0.3
    if 3 <= len(qs) <= 5: r += 0.2
    elif 1 <= len(qs) <= 7: r += 0.1
    for q in qs:
        if not isinstance(q, dict): continue
        if q.get('question') and len(str(q['question'])) > 10: r += 0.05
        opts = q.get('options', [])
        if isinstance(opts, list) and len(opts) == 4: r += 0.05
        ca = q.get('correct', q.get('correct_answer', q.get('answer')))
        if ca is not None: r += 0.05
        if q.get('explanation'): r += 0.03
    return r

def reward_resource(d, prompt):
    r = 0.0
    res = d.get('resources', [])
    if not isinstance(res, list) or len(res) == 0: return -0.3
    if 2 <= len(res) <= 5: r += 0.15
    for item in res:
        if not isinstance(item, dict): continue
        if item.get('title'): r += 0.05
        if item.get('url') and 'http' in str(item.get('url','')): r += 0.05
        if item.get('type') in ['course','article','video','tutorial','documentation','platform']: r += 0.03
        if item.get('reason') or item.get('description'): r += 0.03
    m = re.search(r'topic["\s:]+(\w+)', str(prompt), re.I)
    if m:
        topic = m.group(1)
        topic_node = TOPIC_GRAPH.get(topic)
        GOOD_DOMAINS = ['coursera','udemy','docs.python','github','youtube',
                        'khanacademy','mit.edu','arxiv','tensorflow','pytorch',
                        'scikit-learn','kaggle','fast.ai','deeplearning.ai']
        for item in res:
            url = str(item.get('url',''))
            title = str(item.get('title',''))
            # Partial credit for reputable domains
            if any(d in url for d in GOOD_DOMAINS): r += 0.05
            # Credit for topic name appearing in title
            if topic_node and topic_node.name.lower() in title.lower(): r += 0.03
            # Exact RESOURCE_DB match still gives bonus if it happens
            if topic in RESOURCE_DB:
                real_urls = {res_item.url for res_item in RESOURCE_DB[topic]}
                if item.get('url') in real_urls: r += 0.08
    return r

def multitask_reward(completions, prompts=None, **kwargs):
    import random
    rewards = []
    for i, comp in enumerate(completions):
        if isinstance(comp, list):
            text = ''
            for msg in comp:
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    text = msg.get('content',''); break
            if not text and comp:
                text = comp[-1].get('content','') if isinstance(comp[-1],dict) else str(comp[-1])
        elif isinstance(comp, dict): text = comp.get('content', str(comp))
        else: text = str(comp)

        # Strip Qwen wrapper tokens from raw completions
        text = text.replace('<|im_start|>assistant', '')
        text = text.replace('<|im_end|>', '')
        text = text.replace('<|endoftext|>', '')
        text = text.strip()

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match: 
            _stats['parse_fail'] += 1
            rewards.append(-0.5 + random.uniform(-0.05, 0.05))
            continue
        try: d = json.loads(match.group())
        except:
            try:
                fixed = re.sub(r',\s*}', '}', re.sub(r',\s*]', ']', match.group()))
                d = json.loads(fixed)
            except: 
                _stats['parse_fail'] += 1
                rewards.append(-0.4 + random.uniform(-0.05, 0.05))
                continue

        prompt_text = ''
        if prompts is not None:
            idx = min(i, len(prompts)-1)
            p = prompts[idx]
            if isinstance(p, list):
                for msg in p:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        prompt_text = msg.get('content', '')
                        break
            if not prompt_text:
                prompt_text = p if isinstance(p, str) else str(p)

        task = detect_task(prompt_text)
        if task == 'action': r = reward_action(d, prompt_text)
        elif task == 'roadmap': r = reward_roadmap(d, prompt_text)
        elif task == 'quiz': r = reward_quiz(d, prompt_text)
        elif task == 'resource': r = reward_resource(d, prompt_text)
        else: r = 0.0
        # Soft clamping to prevent hard boundaries
        rewards.append(max(min(r, 1.0), -1.0))
        
    # Prevent zero-variance collapse robustly
    if len(rewards) > 1:
        std = float(np.std(rewards))
        if std < 0.01:
            rewards = [r + random.uniform(-0.05, 0.05) for r in rewards]
        
    return rewards

# Test
print('Reward tests:')
tests = [
    ('[ACTION] Completed: []\nAvailable: [python_basics]', '{"type":"recommend_topic","topic_id":"python_basics"}'),
    ('[ROADMAP] field: tech', '{"roadmap":[{"topic_id":"python_basics","reason":"foundation"}]}'),
    ('[QUIZ] topic: python_basics', '{"questions":[{"question":"What is Python?","options":["A","B","C","D"],"correct":0,"explanation":"..."}]}'),
    ('[RESOURCE] topic: python_basics', '{"resources":[{"title":"Python Tutorial","url":"https://docs.python.org","type":"course","reason":"official"}]}'),
]
for p, t in tests:
    r = multitask_reward([t], prompts=[p])[0]
    print(f'  {p[:30]:30s} → r={r:+.3f}')
_stats['actions'].clear(); _stats['total'] = 0; _stats['parse_fail'] = 0
print('Reward ✅')
