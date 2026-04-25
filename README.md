# 🎓 EduPath AI — Personalized Learning Tutor Environment

> **Team KRIYA** | OpenEnv Hackathon India 2026  
> An OpenEnv-compliant RL environment for training LLMs to become adaptive, multi-agent AI tutors

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/degree-checker-01/meta-new-space)
[![GRPO Model](https://img.shields.io/badge/🤗%20Model-edupath--grpo--tutor-green)](https://huggingface.co/degree-checker-01/edupath-grpo-tutor)
[![Live Demo](https://img.shields.io/badge/🔗%20Live-Comparison%20Demo-orange)](https://degree-checker-01-meta-new-space.hf.space/comparison)

---

## 📌 Links

| Resource | URL |
|---|---|
| **🌐 HuggingFace Space (Environment)** | [degree-checker-01/meta-new-space](https://huggingface.co/spaces/degree-checker-01/meta-new-space) |
| **🤖 GRPO Fine-Tuned Model** | [degree-checker-01/edupath-grpo-tutor](https://huggingface.co/degree-checker-01/edupath-grpo-tutor) |
| **📊 Live Comparison Demo** | [Baseline vs GRPO](https://degree-checker-01-meta-new-space.hf.space/comparison) |
| **📝 Blog Post** | <!-- TODO: Add HuggingFace blog link --> |
| **🎥 Video (< 2 min)** | <!-- TODO: Add YouTube link --> |
| **📓 Training Notebook** | <!-- TODO: Add Kaggle notebook link --> |

---

## 🧠 Problem: Why Can't LLMs Teach?

Current LLMs can answer questions, but they **can't tutor**. Real tutoring requires:

- **Knowing what to teach next** — not just answering, but proactively choosing the right topic based on what a student already knows
- **Remembering student state** across a multi-week learning journey (long-horizon planning)
- **Adapting in real-time** — if a student fails a quiz, the tutor must adjust the roadmap, not blindly continue
- **Multi-agent coordination** — different cognitive tasks (profiling, curriculum design, assessment, adaptation) require different reasoning strategies

**EduPath AI** is an OpenEnv-compliant RL environment that trains LLMs to handle all four of these challenges simultaneously through a **multi-task GRPO fine-tuning pipeline**.

---

## 🏗️ Environment Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    EduPath AI Environment                        │
│                                                                  │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │  Student     │    │  Curriculum DAG   │    │  BKT Student   │  │
│  │  Manager     │◄──►│  36 Topics × 5    │◄──►│  Model         │  │
│  │  (JSON/DB)   │    │  Fields           │    │  (Bayesian)    │  │
│  └──────┬───────┘    └────────┬─────────┘    └───────┬────────┘  │
│         │                     │                       │          │
│         ▼                     ▼                       ▼          │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              OpenEnv API: /reset  /step  /state          │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  15-Agent System                          │    │
│  │                                                           │    │
│  │  Stage 1: Profiling Agent (conversational onboarding)     │    │
│  │  Stage 2: Roadmap Council (6 agents debate curriculum)    │    │
│  │  Stage 3: Learning Loop (scout → quiz → critic)           │    │
│  │  Stage 4: Adaptation Agent (dynamic course correction)    │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### What the Agent Sees (Observation)

```json
{
  "completed_topics": ["python_basics", "python_control_flow"],
  "available_topics": ["python_oop", "data_structures", "version_control"],
  "current_topic": "python_control_flow",
  "job_readiness_score": 0.15,
  "quiz_history": [{"topic": "python_basics", "score": 85, "passed": true}],
  "badges_earned": ["first_step", "first_quiz"],
  "step_count": 12
}
```

### What the Agent Does (7 Actions)

| Action | When to Use |
|---|---|
| `recommend_topic` | Student needs new material → pick from available topics |
| `assign_quiz` | Student is studying a topic → test understanding |
| `assign_mini_project` | 2+ topics completed → hands-on practice |
| `assign_capstone` | 5+ topics + readiness ≥ 0.4 → final project |
| `recommend_resource` | Supplement current study with courses/articles |
| `suggest_event` | Relevant hackathons or events |
| `mark_job_ready` | Readiness ≥ 0.8 → student is career-ready |

### How the Agent Gets Rewarded

The reward function is **multi-dimensional and state-aware** — not a simple 0/1 signal:

- **Prerequisite compliance**: Topics must follow the DAG order (+0.2 for correct order)
- **Context sensitivity**: `assign_quiz` rewarded when student has a current topic, penalized otherwise
- **Anti-mode-collapse**: Frequency penalty if agent repeats the same action >50% of the time
- **Real resource matching**: Bonus for recommending URLs that exist in our curated resource database
- **JSON structural scoring**: Graduated rewards for well-formed outputs (not binary parse/fail)

---

## 🏋️ Training: Multi-Task GRPO

We train a single model on **4 tasks simultaneously** using Group Relative Policy Optimization (GRPO):

| Task | Weight | What it Trains |
|---|---|---|
| **Action Selection** | 40% | Choose the right pedagogical action given student state |
| **Roadmap Generation** | 20% | Create prerequisite-ordered learning paths |
| **Quiz Generation** | 20% | Generate topic-specific MCQ quizzes with explanations |
| **Resource Recommendation** | 20% | Recommend real, curated learning resources |

### Training Configuration

| Parameter | 7B Model | 3B Model |
|---|---|---|
| Base Model | Qwen 2.5-7B-Instruct | Qwen 2.5-3B-Instruct |
| Quantization | 4-bit (NF4) | 4-bit (NF4) |
| LoRA Rank | 16 | 16 |
| Training Steps | 600 | 600 |
| Batch Size | 1 | 2 |
| Gradient Accumulation | 8 | 4 |
| Learning Rate | 3e-5 | 5e-5 |
| Beta (KL penalty) | 0.08 | 0.1 |
| Hardware | Kaggle T4 (16GB) | Kaggle T4 (16GB) |
| Dataset | 500 train / 100 eval | 500 train / 100 eval |

### Training Script

```bash
# Part 1: Setup + reward functions (shared)
# Part 2: Dataset generation + model loading + GRPO training

# Run on Kaggle with T4 GPU:
# Cell 1-3: train_multitask_part1.py
# Cell 4-11: train_multitask_part2.py (7B) or train_multitask_part2_3B.py (3B)
```

---

## 📊 Results

### Training Curves

<!-- TODO: Embed training plot after training completes -->
<!-- ![Training Results](grpo_training_results.png) -->
> **⏳ Training in progress on Kaggle — plots will be added upon completion**

### Before vs After GRPO

| Metric | Baseline (Untrained) | After GRPO | Improvement |
|---|---|---|---|
| Mean Reward | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Positive Rate | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Valid JSON Rate | 100% | <!-- TODO --> | — |

### Per-Task Breakdown

| Task | Before | After | Delta |
|---|---|---|---|
| Action Selection | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Roadmap Generation | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Quiz Generation | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Resource Recommendation | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

### Qualitative Examples

<details>
<summary>🎯 Action Selection — Before vs After</summary>

**Prompt**: Student has completed `python_basics` and `python_control_flow`. Current topic: `python_control_flow`. Available: `[python_oop, data_structures, version_control]`.

**Baseline Output:**
```json
// TODO: Add real baseline output
```

**GRPO Output:**
```json
// TODO: Add real GRPO output
```
</details>

<details>
<summary>🗺️ Roadmap Generation — Before vs After</summary>

**Prompt**: Tech student, goal: ML Engineer

**Baseline Output:**
```json
// TODO: Add real baseline output
```

**GRPO Output:**
```json
// TODO: Add real GRPO output
```
</details>

---

## 🔧 How to Run

### 1. Run the Environment (HuggingFace Space)

The environment is live at: **[degree-checker-01/meta-new-space](https://huggingface.co/spaces/degree-checker-01/meta-new-space)**

```bash
# Or run locally:
git clone https://huggingface.co/spaces/degree-checker-01/meta-new-space
cd meta-new-space
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

### 2. Interact with the Environment

```python
import requests

# Reset environment
resp = requests.post("http://localhost:7860/reset", json={
    "student_profile": {
        "name": "Alice",
        "target_field": "tech",
        "learning_goal": "ML Engineer",
        "weekly_hours": 10
    }
})
obs = resp.json()["observation"]

# Take an action
resp = requests.post("http://localhost:7860/step", json={
    "type": "recommend_topic",
    "topic_id": "python_basics"
})
result = resp.json()
print(f"Reward: {result['reward']}, Done: {result['done']}")
```

### 3. Train with GRPO

```bash
# On Kaggle (T4 GPU required):
# 1. Create a new notebook
# 2. Paste cells from train_multitask_part1.py (setup + rewards)
# 3. Paste cells from train_multitask_part2.py (7B) or train_multitask_part2_3B.py (3B)
# 4. Run all cells — training takes ~2-3 hours on T4
```

---

## 🎯 Hackathon Theme Alignment

| Theme | How EduPath Addresses It |
|---|---|
| **Multi-Agent Interactions** | 15 agents: 6-agent council debate for roadmaps, 3-agent learning loop, profiling + adaptation agents |
| **Long-Horizon Planning** | Learning journeys span weeks/months across 36 topics with prerequisite constraints |
| **Personalized Tasks** | BKT-driven student model adapts difficulty, resources, and pacing to individual learners |
| **Self-Improvement** | GRPO training loop: model generates → gets reward → improves. Trained on 4 tasks simultaneously |

---

## 📁 Repository Structure

```
meta-hacka/
├── backend/
│   ├── ai/                     # 15 agent implementations
│   │   ├── llm_client.py       # Unified LLM client (HF/Groq/OpenAI)
│   │   ├── profiling_agent.py  # Conversational student profiling
│   │   ├── roadmap_generator.py
│   │   ├── quiz_generator.py
│   │   ├── council/            # 6-agent roadmap council
│   │   ├── learning_loop/      # Scout → Quiz → Critic loop
│   │   └── adaptation/         # Adaptation + recap agents
│   ├── environment/            # OpenEnv RL environment
│   │   ├── env.py              # EduPathEnv (step/reset/state)
│   │   ├── models.py           # Pydantic data models
│   │   ├── curriculum.py       # Topic DAG + resource DB
│   │   ├── bkt_model.py        # Bayesian Knowledge Tracing
│   │   └── student.py          # Student state management
│   └── main.py                 # FastAPI server
├── dashboard/
│   ├── index.html              # Training dashboard
│   └── comparison.html         # Baseline vs GRPO demo
├── train_multitask_part1.py    # GRPO setup + reward functions
├── train_multitask_part2.py    # 7B training pipeline
├── train_multitask_part2_3B.py # 3B training pipeline
├── openenv.yaml                # OpenEnv manifest
├── Dockerfile                  # HuggingFace Space deployment
└── README.md                   # This file
```

---

## 👥 Team KRIYA

<!-- TODO: Add team member names and roles -->

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
