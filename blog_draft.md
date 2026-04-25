# EduPath AI: Training LLMs to Become Adaptive Tutors with Multi-Task GRPO

**Team KRIYA** | OpenEnv Hackathon India 2026

---

## The Problem

LLMs can answer questions, but they can't *teach*. A real tutor doesn't just respond — they decide what to teach next, assess understanding through quizzes, adapt when students struggle, and plan learning paths that span weeks. We built **EduPath AI**, an RL environment that trains LLMs to do exactly this.

## What We Built

EduPath AI is an OpenEnv-compliant environment with:

- **36 topics** across 5 fields (tech, healthcare, business, law, design) connected by a prerequisite DAG
- **A Bayesian Knowledge Tracing (BKT) model** that simulates realistic student learning — quiz scores aren't random, they reflect actual mastery
- **15 specialized agents** including a 6-agent council that debates roadmap decisions, a 3-agent learning loop (scout → quiz → critic), and an adaptation agent for course correction
- **7 actions** the agent can take: recommend topics, assign quizzes, create projects, suggest resources, and mark students job-ready

## How We Trained

We fine-tuned **Qwen 2.5 (3B and 7B)** using **Group Relative Policy Optimization (GRPO)** on 4 tasks simultaneously:

1. **Action Selection** (40%) — choosing the right pedagogical action based on student state
2. **Roadmap Generation** (20%) — creating prerequisite-ordered learning paths
3. **Quiz Generation** (20%) — producing topic-specific MCQ quizzes
4. **Resource Recommendation** (20%) — suggesting real, curated learning materials

Our reward function is state-aware and multi-dimensional — it checks prerequisite ordering, validates topic IDs against the real curriculum, penalizes mode collapse, and provides graduated scoring instead of binary pass/fail.

## Results

<!-- TODO: Fill in after training completes -->

### Training Metrics

| Metric | Before GRPO | After GRPO |
|---|---|---|
| Mean Reward | <!-- TODO --> | <!-- TODO --> |
| Positive Rate | <!-- TODO --> | <!-- TODO --> |
| Valid JSON | 100% | <!-- TODO --> |

### Training Plot

<!-- TODO: Embed grpo_training_results.png -->
> ⏳ Training in progress — results coming soon

### Key Observations

<!-- TODO: Fill in observations after training -->
- The baseline model produces valid JSON but fails to use correct topic IDs or contextually appropriate actions
- After GRPO training, the model learns to...
- Per-task improvement was most significant in...

## Why It Matters

Most LLM training environments test narrow skills (math, coding, games). EduPath tests something harder: **multi-step decision making in a personalized, partially observable environment**. The agent must maintain mental models of student knowledge, plan across a prerequisite graph, and balance multiple pedagogical objectives — exactly the kind of capability gap where RL fine-tuning makes a difference.

## Try It

- **🌐 Environment**: [HuggingFace Space](https://huggingface.co/spaces/degree-checker-01/meta-new-space)
- **🤖 Model**: [edupath-grpo-tutor](https://huggingface.co/degree-checker-01/edupath-grpo-tutor)
- **📊 Live Demo**: [Baseline vs GRPO Comparison](https://degree-checker-01-meta-new-space.hf.space/comparison)
- **💻 Code**: [GitHub / HF Space Files](https://huggingface.co/spaces/degree-checker-01/meta-new-space/tree/main)

---

*Built with ❤️ by Team KRIYA for the Meta Hackathon 2026*
