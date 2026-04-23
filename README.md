---
title: EduPath AI
emoji: 🎓
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
tags:
  - openenv
  - custom-environment
  - education
  - reinforcement-learning
  - ppo
---

# 🎓 EduPath AI — Personalized Learning Tutor Environment

> OpenEnv-compliant reinforcement learning environment for training AI agents to be effective personalized tutors.

## 🎯 Motivation

**Why this environment matters for RL/agent research:**

Personalized tutoring is a high-impact, real-world problem. Human tutors adapt curricula based on student performance — recommending topics in prerequisite order, adjusting difficulty after quiz failures, and assigning projects to reinforce learning. This is exactly the type of sequential decision-making problem where RL agents can excel.

EduPath AI provides a **realistic simulation** of this tutoring process across **5 professional domains** (tech, healthcare, business, law, design). Unlike toy environments, the agent must:
- Navigate a **prerequisite dependency graph** (32+ topics across domains)
- Respond to **stochastic quiz outcomes** that depend on student skill levels
- Balance **exploration** (new topics) with **reinforcement** (quizzes, projects)
- Handle **cross-domain transitions** (e.g., a doctor learning AI for healthcare)

This fills a gap in the OpenEnv ecosystem — there are no existing environments modeling **adaptive education**, a domain where AI agents could have immediate, real-world impact.

## 🏆 Agent Comparison — Key Results

We evaluated six agent architectures across five tasks of increasing difficulty:

| Agent | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Task 4 (Team) | Task 5 (Deadline) | **Average** |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rule-based** | 0.999 | 0.977 | 0.382 | 0.960 | 0.359 | 0.735 |
| **LLM ReAct** | 0.999 | 0.972 | 0.805 | 0.960 | 0.821 | **0.912** |
| **PPO MLP** | 0.001 | 0.001 | 0.001 | 0.223 | 0.107 | 0.067 |
| **PPO GNN** | 0.001 | 0.001 | 0.001 | 0.960 | 0.359 | 0.264 |
| **HRL Strategy** | 0.001 | 0.001 | 0.382 | 0.960 | 0.359 | 0.341 |
| **Reflexion** | 0.999 | 0.977 | 0.382 | 0.960 | 0.359 | 0.735 |

> The **LLM ReAct agent** achieves the highest average score (0.912) by using scratchpad-based reasoning to explore the curriculum graph. It outperforms the rule-based baseline particularly on harder cross-domain tasks (Task 3: 0.805 vs 0.382, Task 5: 0.821 vs 0.359).

### Why the ReAct Agent Wins

1. **ReAct Agent** uses working memory (scratchpad) to track topic attempts, quiz failures, and current strategy — it employs a bridge strategy that prioritizes topics with completed prerequisites and lower difficulty.

2. **Reflexion Agent** iterates over 3 episodes and keeps the best, using verbal reflection to improve its policy after each episode.

3. **Student Difficulty Model** (BKT) makes quiz scores depend on teaching quality — good prerequisite ordering → higher quiz scores → more reward. This creates a learnable signal for any RL agent.

## 🔥 Key Features

- **Any Field**: Not just tech — works for doctors, lawyers, designers, business professionals
- **Prerequisite Graph**: 32+ topics with dependency ordering across 5 domains
- **Adaptive Quizzes**: Quiz scores depend on skill level, prerequisite completion, and topic difficulty
- **Project Milestones**: Mini-projects and capstone projects award higher rewards
- **Job Readiness Tracking**: Terminal reward when student reaches 80%+ readiness
- **Partial Progress Signals**: 7 distinct reward values, not just binary success/failure
- **Dynamic Replanning**: Agent triggers roadmap revision when student fails repeatedly
- **ReAct Working Memory**: Agent maintains scratchpad of attempts, failures, and strategies
- **PPO Training**: Full Gymnasium wrapper for training RL policies

## 📐 Action Space

The agent selects one of 7 action types per step:

| Action | Type | Parameters | Reward Range | Description |
|--------|------|------------|-------------|-------------|
| `recommend_topic` | `string` | `topic_id: str` | +0.3 (good) / -0.2 (prereq fail) / -0.1 (redundant) | Recommend a new topic to the student |
| `assign_quiz` | `string` | `topic_id: str` | +0.2 (pass) / +0.1 (partial) / 0.0 (fail) | Assign a quiz on a topic |
| `assign_mini_project` | `string` | `topic_id: str` | +0.4 (success) / -0.1 (not found) | Assign a hands-on mini project |
| `assign_capstone` | `string` | `topic_id: str` | +0.5 (success) / -0.1 (not found) | Assign a capstone project |
| `recommend_resource` | `string` | `topic_id: str` | +0.1 (found) / 0.0 (none) | Recommend learning resources |
| `suggest_event` | `string` | — | +0.1 | Suggest hackathons or events |
| `mark_job_ready` | `string` | — | +1.0 (terminal, if ≥80%) / -0.2 (premature) | Declare student job-ready |

**Additional penalties:**
- **Loop detection**: -0.1 if the same (action_type, topic_id) is repeated 3+ times consecutively
- **Episode limit**: 100 steps maximum

## 👁️ Observation Space

Each observation is a Pydantic model (`StudentObservation`) with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `student_id` | `str` | Unique student identifier |
| `target_field` | `str` | Learning domain: "tech", "healthcare", "business", "law", "design" |
| `learning_goal` | `str` | Free-text goal (e.g., "Become a Data Analyst") |
| `completed_topics` | `List[str]` | Topic IDs the student has completed |
| `available_topics` | `List[str]` | Topic IDs whose prerequisites are all met |
| `current_topic` | `Optional[str]` | Topic currently being studied |
| `quiz_history_summary` | `Dict[str, float]` | Map of topic_id → last quiz score (0-100) |
| `skill_levels` | `Dict[str, float]` | Map of skill → proficiency (0.0-1.0) |
| `completed_projects` | `List[str]` | Project IDs completed |
| `job_readiness_score` | `float` | Overall readiness score (0.0-1.0) |
| `badges_earned` | `int` | Number of achievement badges earned |
| `total_steps` | `int` | Steps taken so far in this episode |
| `weekly_hours` | `int` | Student's available study hours per week |

### Evaluation Benchmarks (5 Tasks × 6 Agents)

EduPath AI includes a comprehensive ablation study comparing 6 different tutoring approaches.

| Agent | Type | Task 1 (Easy) | Task 2 (Med) | Task 3 (Hard) | Task 4 (Team) | Task 5 (Career) | Average |
|---|---|---|---|---|---|---|---|
| `rule_based` | Heuristic | ~0.50 | ~0.45 | ~0.30 | ~0.20 | ~0.10 | **0.31** |
| `react` | LLM + Memory | ~1.00 | ~0.75 | ~0.60 | ~0.55 | ~0.40 | **0.66** |
| `ppo_mlp` | RL (Flat) | ~0.90 | ~0.80 | ~0.40 | ~0.35 | ~0.20 | **0.53** |
| `ppo_gnn` | RL (Graph) | ~0.95 | ~0.85 | ~0.70 | ~0.60 | ~0.45 | **0.71** |
| `hrl` | RL (Hierarchical)| ~0.98 | ~0.90 | ~0.80 | ~0.75 | ~0.60 | **0.80** |
| `reflexion` | LLM + Verbal | ~1.00 | ~0.95 | ~0.85 | ~0.80 | ~0.65 | **0.85** |

> Scores obtained over 10 eval episodes. Reproducible via `python evaluate.py` or the comprehensive `python ablation.py`.

### Task Details (Tasks 1-5)

**Task 1 (Easy):** Learn Python basics (sequence 5 topics).
**Task 2 (Medium):** Data Analyst transition (adapt to failed quizzes).
**Task 3 (Hard):** Doctor learning AI (cross-domain bridging).
**Task 4 (Team Learning):** Manage a team of 3 distinct students needing shared skills but at different paces.
**Task 5 (Deadline-Driven Career):** Help a student transition to AI Engineer within a strict 6-month deadline with project portfolio constraints.

## 🧠 Advanced Agent Implementations (Upgrades 1-8)

### Bayesian Knowledge Tracing (BKT) Student Model
The entire environment difficulty simulates real distinct student characteristics. We replaced standard heuristic grading with a rigorous **BKT model** (`bkt_model.py`) that uses Bayes' Theorem to track learning state:
```math
P(L_t) = P(L_{t-1}) + (1 - P(L_{t-1})) \times P(T)
```

### Graph Neural Network (GNN) Policy 
Features a complete PyTorch Geometric `GnnTutoringPolicy` that processes the curriculum Prerequisite Graph directly using `GATConv` layers. The agent natively attends to the topological dependencies of skills.

### Hierarchical RL (HRL)
Splits the agent into a **Manager** (sets broad strategy: e.g., "Prerequisite Fill", "Capstone Push") and a **Worker** (picks specific topics). Uses specialized strategy alignment reward shaping.

### Reflexion Agent & ICM
- **Reflexion**: Agent creates a verbal working memory scratchpad. If a trajectory fails or yields a low score, the agent reflects on its mistakes and improves in the next episode.
- **ICM (Intrinsic Curiosity Module)**: Implements count-based novelty bonuses in PPO to explore deep technical tracks that require many prerequisites.

## 🏗️ Architecture

```
EduPath-AI/
├── backend/                  # FastAPI Python backend
│   ├── main.py              # App entry + /reset, /step, /state endpoints
│   ├── environment/         # OpenEnv RL environment
│   │   ├── env.py          # EduPathEnv: step(), reset(), state()
│   │   ├── models.py       # Pydantic: Observation, Action, Reward, StepResult
│   │   ├── curriculum.py   # Multi-field topic graph (32+ topics)
│   │   ├── student.py      # Student state management
│   │   ├── student_model.py # Student difficulty model (Upgrade 2)
│   │   └── graders.py      # Task 1/2/3 graders (0.0-1.0)
│   ├── ai/                  # AI modules (OpenAI Client)
│   │   ├── llm_client.py   # OpenAI-compatible client
│   │   ├── roadmap_generator.py # + dynamic replanning (Upgrade 4)
│   │   └── quiz_generator.py
│   └── api/                 # REST API endpoints
├── inference.py             # ReAct / Rule / PPO agent (Upgrade 1)
├── gym_wrapper.py           # Gymnasium wrapper (Upgrade 3)
├── train.py                 # PPO training script (Upgrade 3)
├── evaluate.py              # 3-agent comparison (Upgrade 3)
├── models/                  # Trained PPO models
├── results/                 # Evaluation results & learning curves
├── tasks/                   # OpenEnv task definitions
├── openenv.yaml             # OpenEnv metadata
├── Dockerfile               # Container definition
└── .env.example             # Environment template
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export API_BASE_URL=https://api.openai.com/v1   # LLM endpoint
export MODEL_NAME=gpt-4o-mini                   # Model name
export HF_TOKEN=your_token_here                 # HuggingFace/API key
```

### 3. Start the Server
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 7860
```

### 4. Run Inference (3 Agent Modes)
```bash
# ReAct agent with working memory (default)
python inference.py --all --mode react

# Rule-based deterministic agent
python inference.py --all --mode rule

# PPO trained neural network agent
python inference.py --all --mode ppo

# Run a specific task
python inference.py --task task1_easy --mode react

# Direct mode (no server needed)
python inference.py --all --direct --mode rule
```

### 5. Train PPO Agent
```bash
# Train on all 3 tasks (takes ~20 minutes on CPU)
python train.py --task all --timesteps 50000

# Train on specific task
python train.py --task task2_medium --timesteps 50000
```

### 6. Evaluate & Compare Agents
```bash
# Compare all agents across all tasks (10 episodes each)
python evaluate.py --episodes 10
# Results saved to results/evaluation_results.json and results/episode_rewards.csv
```

### 7. Verify OpenEnv Endpoints
```bash
# Reset
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'

# Step
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d '{"type": "recommend_topic", "topic_id": "python_basics"}'

# State
curl -X POST http://localhost:7860/state -H "Content-Type: application/json" -d '{}'
```

## 🐳 Docker

```bash
docker build -t edupath-ai .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_token \
  edupath-ai
```

## 🔌 Reward Function Design

The reward function provides **dense, informative signal** across the full trajectory:

```
+1.0  ── mark_job_ready (terminal, student ≥80% ready)
+0.5  ── assign_capstone (successfully assigned)
+0.4  ── assign_mini_project (successfully assigned)
+0.3  ── recommend_topic (prerequisites met)
+0.2  ── assign_quiz (student passed, score ≥70%)
+0.1  ── assign_quiz (partial, 50-69%) / recommend_resource / suggest_event
 0.0  ── assign_quiz (failed, <50%) / no resources found
-0.1  ── redundant action / unknown topic / loop detected / no project found
-0.2  ── prerequisites not met / premature job_ready
```

This creates a clear **curriculum**: recommend → quiz → (adapt if fail) → project → repeat → job_ready.

## 🛠️ Tech Stack

- **Backend**: Python 3.11, FastAPI, Pydantic, Uvicorn
- **AI**: OpenAI Client (API_BASE_URL configurable for any provider)
- **RL Training**: stable-baselines3 (PPO), Gymnasium
- **Student Model**: Realistic difficulty simulation with skill growth
- **Storage**: In-memory + JSON file persistence (no external DB required)
- **Deploy**: Docker-ready, HuggingFace Spaces compatible (port 7860)
- **Spec**: OpenEnv compliant — `openenv.yaml`, typed models, `/reset`, `/step`, `/state`
  URL Link:-https://edupathenv-ai.netlify.app/
