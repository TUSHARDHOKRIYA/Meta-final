"""
EduPath AI - GRPO LLM Fine-Tuning Script
Team KRIYA | Meta Hackathon 2026

Trains an LLM tutoring agent using Group Relative Policy Optimization.
The EduPath environment acts as the reward verifier.

Usage (Colab/GPU):
    python llm_training.py
    python llm_training.py --steps 500 --model Qwen/Qwen2.5-1.5B-Instruct
"""
import os, sys, json, re, random, warnings, logging, inspect, argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

# ── Setup paths ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# ── Mock optional deps that aren't needed for training ──
from unittest.mock import MagicMock
for _m in ["llm_blender", "llm_blender.agents", "llm_blender.pair_ranker"]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS — pulled from the real EduPath curriculum
# ═══════════════════════════════════════════════════════════════
from environment.curriculum import TOPIC_GRAPH, get_available_topics

ALL_TOPIC_IDS = list(TOPIC_GRAPH.keys())
TECH_TOPICS = [t for t, v in TOPIC_GRAPH.items() if v.field == "tech"]
VALID_ACTIONS = {
    "recommend_topic", "assign_quiz", "assign_mini_project",
    "assign_capstone", "recommend_resource", "mark_job_ready",
}

# ═══════════════════════════════════════════════════════════════
#  REWARD FUNCTIONS  (environment-aligned)
# ═══════════════════════════════════════════════════════════════
def parse_action(text: str):
    text = re.sub(r"```json\s*", "", text.strip())
    text = re.sub(r"```\s*", "", text).strip()
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None, None
    try:
        d = json.loads(match.group(0))
        atype = d.get("type", "").strip()
        tid = d.get("topic_id")
        if tid in ("null", "none", "", "None", None):
            tid = None
        return atype, tid
    except json.JSONDecodeError:
        return None, None


def get_optimal_action(state: dict) -> str:
    completed = state.get("completed_topics", [])
    job_score = state.get("job_readiness", 0.0)
    quiz_scores = state.get("quiz_scores", {})
    unquizzed = [t for t in completed if t not in quiz_scores]
    available = [t for t in TECH_TOPICS if t not in completed]
    n = len(completed)
    if job_score >= 0.8 and n >= 4:
        return "mark_job_ready"
    if unquizzed:
        return "assign_quiz"
    if n >= 3:
        return "assign_capstone"
    if available:
        return "recommend_topic"
    return "assign_mini_project"


def compute_reward(text: str, state: dict) -> float:
    completed = state.get("completed_topics", [])
    job_score = state.get("job_readiness", 0.0)
    quiz_scores = state.get("quiz_scores", {})
    n = len(completed)
    unquizzed = [t for t in completed if t not in quiz_scores]
    available = [t for t in TECH_TOPICS if t not in completed]
    atype, tid = parse_action(text)
    if atype is None:
        return -0.40
    if atype not in VALID_ACTIONS:
        return -0.30
    r = 0.10
    topic_set = set(ALL_TOPIC_IDS)
    if atype == "recommend_topic":
        if tid is None:
            r -= 0.15
        elif tid not in topic_set:
            r -= 0.20
        elif tid in completed:
            r -= 0.20
        else:
            r += 0.45
    elif atype == "assign_quiz":
        if tid and tid in completed and tid not in quiz_scores:
            r += 0.50
        elif tid and tid in quiz_scores:
            r += 0.10
        else:
            r -= 0.10
    elif atype == "assign_mini_project":
        r += 0.25 if n >= 1 else -0.10
    elif atype == "assign_capstone":
        r += 0.45 if n >= 3 else -0.25
    elif atype == "recommend_resource":
        r += 0.15
    elif atype == "mark_job_ready":
        if job_score >= 0.8 and n >= 4:
            r += 1.00
        else:
            r -= 0.35
    if atype == get_optimal_action(state):
        r += 0.20
    return float(np.clip(r, -0.5, 1.0))


def edupath_reward_fn(completions, **kwargs) -> list:
    """GRPO-compatible reward function. Accepts **kwargs for forward compat."""
    rewards = []
    states = kwargs.get("state", None)
    for i, completion in enumerate(completions):
        # Extract text from various completion formats
        if isinstance(completion, list):
            txt = ""
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    txt = msg.get("content", "")
                    break
            if not txt and completion:
                last = completion[-1]
                txt = last.get("content", "") if isinstance(last, dict) else str(last)
        elif isinstance(completion, dict):
            txt = completion.get("content", str(completion))
        else:
            txt = str(completion)
        state = {}
        if states is not None:
            try:
                raw = states[i] if i < len(states) else "{}"
                state = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                state = {}
        rewards.append(compute_reward(txt, state))
    return rewards


# ═══════════════════════════════════════════════════════════════
#  DATASET BUILDER
# ═══════════════════════════════════════════════════════════════
def build_prompt(state: dict) -> str:
    completed = state.get("completed_topics", [])
    quiz_scores = state.get("quiz_scores", {})
    job_score = state.get("job_readiness", 0.0)
    available = [t for t in TECH_TOPICS if t not in completed]
    unquizzed = [t for t in completed if t not in quiz_scores]
    qs = ", ".join(f"{k}: {v:.0%}" for k, v in quiz_scores.items()) if quiz_scores else "None"
    return (
        "You are an adaptive tutoring agent for EduPath AI.\n"
        "Analyze the student state and choose the single best action.\n\n"
        "STUDENT STATE:\n"
        f"  completed_topics : {completed}\n"
        f"  remaining_topics : {available}\n"
        f"  unquizzed_topics : {unquizzed}\n"
        f"  quiz_scores      : {qs}\n"
        f"  job_readiness    : {job_score:.2f}\n\n"
        "ACTIONS:\n"
        "  recommend_topic      - assign from remaining_topics\n"
        "  assign_quiz          - quiz a completed topic\n"
        "  assign_mini_project  - small project (need 1+ completed)\n"
        "  assign_capstone      - major project (need 3+ completed)\n"
        "  recommend_resource   - share a resource\n"
        "  mark_job_ready       - ONLY if job_readiness >= 0.80 AND 4+ completed\n\n"
        'Respond ONLY with JSON: {"type": "<action>", "topic_id": "<id_or_null>"}'
    )


def build_messages(state: dict) -> list:
    return [
        {"role": "system", "content": "You are an intelligent adaptive tutoring agent. Output only valid JSON."},
        {"role": "user", "content": build_prompt(state)},
    ]


def make_state(difficulty: str) -> dict:
    if difficulty == "easy":
        templates = [
            {"completed_topics": [], "quiz_scores": {}, "job_readiness": 0.0},
            {"completed_topics": ["python_basics"], "quiz_scores": {}, "job_readiness": 0.10},
            {"completed_topics": TECH_TOPICS[:4],
             "quiz_scores": {t: round(random.uniform(0.85, 1.0), 2) for t in TECH_TOPICS[:4]},
             "job_readiness": round(random.uniform(0.85, 0.95), 2)},
        ]
    elif difficulty == "medium":
        templates = [
            {"completed_topics": TECH_TOPICS[:2],
             "quiz_scores": {t: round(random.uniform(0.7, 0.9), 2) for t in TECH_TOPICS[:2]},
             "job_readiness": round(random.uniform(0.25, 0.45), 2)},
            {"completed_topics": TECH_TOPICS[:3],
             "quiz_scores": {t: round(random.uniform(0.75, 0.92), 2) for t in TECH_TOPICS[:3]},
             "job_readiness": round(random.uniform(0.50, 0.70), 2)},
        ]
    else:
        templates = [
            {"completed_topics": TECH_TOPICS[:4],
             "quiz_scores": {t: 0.82 for t in TECH_TOPICS[:4]},
             "job_readiness": 0.79},
            {"completed_topics": TECH_TOPICS[:5],
             "quiz_scores": {t: round(random.uniform(0.88, 1.0), 2) for t in TECH_TOPICS[:5]},
             "job_readiness": round(random.uniform(0.88, 0.98), 2)},
        ]
    s = random.choice(templates).copy()
    s["job_readiness"] = float(np.clip(s["job_readiness"] + random.uniform(-0.02, 0.02), 0, 1))
    return s


def build_dataset(tokenizer, n_easy=120, n_med=160, n_hard=120):
    from datasets import Dataset
    data, counts = [], defaultdict(int)
    for diff, n in [("easy", n_easy), ("medium", n_med), ("hard", n_hard)]:
        for _ in range(n):
            state = make_state(diff)
            msgs = build_messages(state)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            data.append({
                "prompt": prompt,
                "messages": msgs,
                "state": json.dumps(state),
                "difficulty": diff,
                "optimal": get_optimal_action(state),
            })
            counts[diff] += 1
    total = sum(counts.values())
    print(f"  Dataset: {total} samples (easy={counts['easy']}, med={counts['medium']}, hard={counts['hard']})")
    return Dataset.from_list(data)


# ═══════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════
def run_eval(model, tokenizer, n=60, label="Model"):
    import torch
    model.eval()
    results = defaultdict(list)
    matches = []
    diffs = ["easy"] * 20 + ["medium"] * 20 + ["hard"] * 20
    states = [make_state(d) for d in diffs]
    for i, state in enumerate(states):
        msgs = build_messages(state)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=80, temperature=0.3, do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        r = compute_reward(resp, state)
        atype, _ = parse_action(resp)
        results[diffs[i]].append(r)
        matches.append(atype == get_optimal_action(state))
    model.train()
    all_r = results["easy"] + results["medium"] + results["hard"]
    return {
        "label": label,
        "mean_reward": float(np.mean(all_r)),
        "positive_rate": float(np.mean([x > 0 for x in all_r])),
        "optimal_rate": float(np.mean(matches)),
        "easy": float(np.mean(results["easy"])),
        "medium": float(np.mean(results["medium"])),
        "hard": float(np.mean(results["hard"])),
        "all_rewards": all_r,
    }


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════
def load_model(model_id=None):
    import torch
    print("\n── Detecting hardware ──")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram:.1f} GB")
    else:
        vram = 0
        print("  No GPU — will use CPU (slow)")

    # Pick model by VRAM
    if model_id is None:
        if vram >= 20:
            model_id = "Qwen/Qwen2.5-7B-Instruct"
        elif vram >= 12:
            model_id = "Qwen/Qwen2.5-3B-Instruct"
        else:
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_r = 32 if vram >= 12 else 16

    # Try Unsloth first
    backend = None
    model = tokenizer = None
    try:
        from unsloth import FastLanguageModel
        unsloth_id = model_id
        # Try unsloth quantized variant
        if "unsloth/" not in model_id:
            unsloth_id = f"unsloth/{model_id.split('/')[-1]}-bnb-4bit"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=unsloth_id, max_seq_length=1024, load_in_4bit=True, dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
            use_gradient_checkpointing="unsloth", random_state=42,
        )
        backend = "unsloth"
        print(f"  Loaded {unsloth_id} via Unsloth")
    except Exception as e:
        print(f"  Unsloth unavailable ({e}), using HuggingFace...")

    if backend is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_r * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)
            backend = "hf-4bit"
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
            backend = "hf-fp"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backend   : {backend}")
    print(f"  Params    : {total/1e9:.2f}B total, {trainable/1e6:.1f}M trainable")
    return model, tokenizer, backend, model_id, lora_r


# ═══════════════════════════════════════════════════════════════
#  GRPO TRAINING
# ═══════════════════════════════════════════════════════════════
def train(model, tokenizer, dataset, steps=300, batch_size=2, grad_accum=4, lr=2e-6, lora_r=16):
    import torch
    from trl import GRPOConfig, GRPOTrainer

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "grpo_edupath")
    os.makedirs(output_dir, exist_ok=True)

    # Build config with version-safe params
    _sig = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    args = dict(
        output_dir=output_dir,
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        weight_decay=0.01,
        max_grad_norm=0.3,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=max(steps // 3, 50),
        save_total_limit=2,
        seed=42,
        report_to="none",
    )
    optional = {"num_generations": 4, "max_prompt_length": 900, "beta": 0.01, "temperature": 0.9}
    for k, v in optional.items():
        if k in _sig:
            args[k] = v
    for cand in ["max_completion_length", "max_new_tokens", "generation_max_new_tokens"]:
        if cand in _sig:
            args[cand] = 120
            break

    config = GRPOConfig(**args)
    trainer = GRPOTrainer(
        model=model, args=config, train_dataset=dataset,
        reward_funcs=[edupath_reward_fn], processing_class=tokenizer,
    )

    print(f"\n{'='*55}")
    print(f"  GRPO Training | steps={steps} | bs={batch_size}x{grad_accum}")
    print(f"{'='*55}\n")
    t0 = datetime.now()
    result = trainer.train()
    mins = (datetime.now() - t0).total_seconds() / 60
    print(f"\n  Training done in {mins:.1f} min | loss={result.training_loss:.4f}")

    log = trainer.state.log_history
    return trainer, result, log, mins


# ═══════════════════════════════════════════════════════════════
#  SAVE & REPORT
# ═══════════════════════════════════════════════════════════════
def save_results(model, tokenizer, baseline, trained, mins, model_id, lora_r, steps):
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "grpo_edupath")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    res = {
        "model": model_id, "method": f"GRPO+LoRA r={lora_r}",
        "steps": steps, "time_mins": round(mins, 1),
        "baseline": {k: round(v, 4) for k, v in baseline.items() if k != "all_rewards"},
        "trained": {k: round(v, 4) for k, v in trained.items() if k != "all_rewards"},
        "improvement": round(trained["mean_reward"] - baseline["mean_reward"], 4),
        "timestamp": datetime.now().isoformat(),
    }
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "grpo_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  Model  -> {save_dir}")
    print(f"  Results-> {results_dir}/grpo_results.json")
    return res


def print_comparison(baseline, trained):
    imp = trained["mean_reward"] - baseline["mean_reward"]
    print(f"\n{'='*60}")
    print(f"  {'Metric':<20} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    for label, key in [("Mean Reward","mean_reward"),("Positive%","positive_rate"),
                       ("Optimal%","optimal_rate"),("Easy","easy"),("Medium","medium"),("Hard","hard")]:
        b, a = baseline[key], trained[key]
        arrow = "▲" if a > b + 0.005 else "▼" if a < b - 0.005 else "="
        print(f"  {label:<20} {b:>+10.4f} {a:>+10.4f} {arrow}{abs(a-b):>+9.4f}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="EduPath GRPO LLM Training")
    parser.add_argument("--model", type=str, default=None, help="HF model ID (auto-detected by VRAM)")
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--eval-only", action="store_true", help="Run eval without training")
    args = parser.parse_args()

    # 1. Load model
    model, tokenizer, backend, model_id, lora_r = load_model(args.model)

    # 2. Build dataset
    print("\n── Building curriculum dataset ──")
    dataset = build_dataset(tokenizer)

    # 3. Baseline eval
    print("\n── Baseline evaluation ──")
    baseline = run_eval(model, tokenizer, label="Baseline")
    print(f"  Mean reward: {baseline['mean_reward']:+.4f} | Optimal: {baseline['optimal_rate']:.0%}")

    if args.eval_only:
        return

    # 4. Train
    trainer, result, log, mins = train(
        model, tokenizer, dataset,
        steps=args.steps, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr, lora_r=lora_r,
    )

    # 5. Post-training eval
    print("\n── Post-training evaluation ──")
    trained = run_eval(model, tokenizer, label="Trained")
    print_comparison(baseline, trained)

    # 6. Save
    save_results(model, tokenizer, baseline, trained, mins, model_id, lora_r, args.steps)

    # 7. Quick demo
    print(f"\n── Demo ──")
    import torch
    model.eval()
    demos = [
        ("New student", {"completed_topics": [], "quiz_scores": {}, "job_readiness": 0.0}),
        ("Needs quiz", {"completed_topics": ["python_basics"], "quiz_scores": {}, "job_readiness": 0.1}),
        ("Job ready", {"completed_topics": TECH_TOPICS[:5],
                       "quiz_scores": {t: 0.9 for t in TECH_TOPICS[:5]}, "job_readiness": 0.9}),
    ]
    for label, state in demos:
        msgs = build_messages(state)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, temperature=0.1, do_sample=True,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"```json\s*", "", resp)
        resp = re.sub(r"```\s*", "", resp).strip()
        r = compute_reward(resp, state)
        atype, _ = parse_action(resp)
        print(f"  {label:15s} -> {atype or 'INVALID':20s} reward={r:+.3f}  {resp[:50]}")

    print(f"\n{'='*55}")
    print(f"  DONE | {model_id} | improvement={trained['mean_reward']-baseline['mean_reward']:+.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
