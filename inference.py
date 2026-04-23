"""
EduPath AI — OpenEnv Inference Script
Team KRIYA | Meta Hackathon 2026

Runs trained RL agents (PPO, GNN-PPO, HRL, Reflexion) against all 5
tasks and prints results in the mandatory [START]/[STEP]/[END] log
format required by the OpenEnv automated evaluator.
"""
import os
import sys
import json
import time
import random
import logging
import argparse
import requests
from datetime import datetime, timezone
from typing import List, Optional, Dict

from openai import OpenAI

# Add backend to path for direct mode
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Seed for reproducibility ──
SEED = 42
random.seed(SEED)

# ── Environment configuration ──
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("IMAGE_NAME")


# ═══════════════════════════════════════════════════════════
#  Structured Logging — [START], [STEP], [END]
# ═══════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] structured log to stdout."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Emit [STEP] structured log to stdout."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] structured log to stdout."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    # CRITICAL: Use :.4f (not :.2f) to prevent rounding 0.999→1.00 or 0.001→0.00
    print(f"[END] success={success_val} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


# ═══════════════════════════════════════════════════════════
#  OpenAI Client — LLM Decision Making
# ═══════════════════════════════════════════════════════════

def _get_openai_client():
    """Get OpenAI-compatible client using required env vars."""
    try:
        if not API_BASE_URL:
            return None, MODEL_NAME  # Fallback to rule-based

        api_key = HF_TOKEN or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, MODEL_NAME  # No API key available

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=api_key,
        )
        return client, MODEL_NAME
    except Exception as e:
        logger.warning(f"OpenAI client creation failed: {e}")
        return None, MODEL_NAME


# ═══════════════════════════════════════════════════════════
#  ReAct Agent with Working Memory (Upgrade 1)
# ═══════════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = """You are an expert AI tutoring agent using the ReAct framework.
You observe the student state, think step-by-step, then decide the best tutoring action.

You have a SCRATCHPAD of memory from previous steps. Use it to track:
- What topics have been tried and their outcomes
- Which quiz scores showed weakness
- Your current tutoring strategy
- Why you are choosing the next action

CRITICAL REASONING RULES:
1. If quiz score < 50% twice on same topic → assign_resource before next quiz
2. NEVER recommend a topic whose prerequisites are NOT in completed_topics
3. NEVER recommend an already completed topic (redundant, penalized -0.1)
4. After every 2-3 completed topics → consider assign_mini_project
5. Only mark_job_ready when job_readiness_score >= 0.8
6. For cross-domain students (e.g. doctor learning AI) → bridge topics first

RESPONSE FORMAT — Return ONLY valid JSON:
{
  "thought": "Your step-by-step reasoning about what to do next and why",
  "strategy": "Brief description of your current overall tutoring strategy",
  "action": {
    "type": "recommend_topic|assign_quiz|assign_mini_project|assign_capstone|recommend_resource|mark_job_ready",
    "topic_id": "optional topic id",
    "project_id": "optional project id"
  }
}

ACTION TYPES:
- recommend_topic: Recommend a new topic to study (requires topic_id)
- assign_quiz: Test the student on a topic (requires topic_id)
- assign_mini_project: Assign a hands-on project (requires topic_id for context)
- assign_capstone: Major capstone project (requires topic_id for context)
- recommend_resource: Recommend learning materials (requires topic_id)
- mark_job_ready: Declare student job-ready (no topic_id needed)
"""


class ReActScratchpad:
    """Working memory for the ReAct agent across an episode."""

    def __init__(self):
        self.entries: List[Dict] = []
        self.topic_attempts: Dict[str, List[float]] = {}  # topic_id -> [scores]
        self.topics_recommended: List[str] = []
        self.topics_completed: List[str] = []
        self.failed_topics: List[str] = []
        self.current_strategy: str = "Initial assessment"
        self.consecutive_failures: int = 0
        self.projects_assigned: int = 0
        self.topics_since_project: int = 0

    def record_step(self, step: int, action: dict, reward: float, observation: dict, thought: str = ""):
        """Record a step in the scratchpad."""
        entry = {
            "step": step,
            "action_type": action.get("type", "unknown"),
            "topic_id": action.get("topic_id"),
            "reward": reward,
            "thought": thought,
        }
        self.entries.append(entry)

        # Track topic-specific data
        action_type = action.get("type", "")
        topic_id = action.get("topic_id")

        if action_type == "recommend_topic" and topic_id:
            if topic_id not in self.topics_recommended:
                self.topics_recommended.append(topic_id)
                self.topics_since_project += 1

        if action_type == "assign_quiz" and topic_id:
            quiz_score = observation.get("quiz_history_summary", {}).get(topic_id, 0)
            if topic_id not in self.topic_attempts:
                self.topic_attempts[topic_id] = []
            self.topic_attempts[topic_id].append(quiz_score)

            if quiz_score >= 70:
                if topic_id not in self.topics_completed:
                    self.topics_completed.append(topic_id)
                self.consecutive_failures = 0
            else:
                if topic_id not in self.failed_topics:
                    self.failed_topics.append(topic_id)
                self.consecutive_failures += 1

        if action_type in ("assign_mini_project", "assign_capstone"):
            self.projects_assigned += 1
            self.topics_since_project = 0

    def get_summary(self, max_recent: int = 8) -> str:
        """Generate a compact summary of the scratchpad for the LLM context."""
        lines = []
        lines.append(f"Strategy: {self.current_strategy}")
        lines.append(f"Topics recommended: {len(self.topics_recommended)}")
        lines.append(f"Topics completed (quiz passed): {len(self.topics_completed)}")
        lines.append(f"Projects assigned: {self.projects_assigned}")
        lines.append(f"Topics since last project: {self.topics_since_project}")

        if self.failed_topics:
            lines.append(f"Failed topics: {', '.join(self.failed_topics[-5:])}")

        # Topic attempts with scores
        if self.topic_attempts:
            attempt_strs = []
            for tid, scores in list(self.topic_attempts.items())[-5:]:
                attempt_strs.append(f"{tid}: {[int(s) for s in scores]}")
            lines.append(f"Quiz attempts: {'; '.join(attempt_strs)}")

        # Recent actions (last N)
        if self.entries:
            recent = self.entries[-max_recent:]
            lines.append(f"\nRecent actions (last {len(recent)}):")
            for e in recent:
                lines.append(
                    f"  Step {e['step']}: {e['action_type']}({e.get('topic_id', '')}) "
                    f"→ reward={e['reward']:.2f}"
                )
                if e.get('thought'):
                    lines.append(f"    Thought: {e['thought'][:100]}")

        return "\n".join(lines)

    def needs_resource_before_quiz(self, topic_id: str) -> bool:
        """Check if the student has failed a topic twice and needs resources first."""
        scores = self.topic_attempts.get(topic_id, [])
        low_scores = [s for s in scores if s < 50]
        return len(low_scores) >= 2


class ReActAgent:
    """ReAct agent with working memory for tutoring decisions."""

    def __init__(self):
        self.client, self.model_name = _get_openai_client()
        self.scratchpad = ReActScratchpad()
        self._step_count = 0

    def reset(self):
        """Reset agent state for a new episode."""
        self.scratchpad = ReActScratchpad()
        self._step_count = 0

    def decide(self, observation: dict) -> dict:
        """Make a tutoring decision using ReAct reasoning."""
        self._step_count += 1

        # First, check if we should use LLM or fall back to enhanced rules
        if self.client and API_BASE_URL:
            result = self._llm_react_decision(observation)
            if result:
                return result

        # Fallback to enhanced rule-based with scratchpad intelligence
        return self._enhanced_rule_decision(observation)

    def record(self, action: dict, reward: float, observation: dict, thought: str = ""):
        """Record a step result in the scratchpad."""
        self.scratchpad.record_step(self._step_count, action, reward, observation, thought)

    def _llm_react_decision(self, observation: dict) -> Optional[dict]:
        """Use LLM with ReAct prompting and working memory."""
        scratchpad_summary = self.scratchpad.get_summary()

        user_prompt = f"""Current student state:
{json.dumps(observation, indent=2)}

Working Memory (Scratchpad):
{scratchpad_summary}

Step {self._step_count}: Analyze the state and decide the next action.
Remember the reasoning rules. Think carefully before acting."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": REACT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)

            thought = result.get("thought", "")
            strategy = result.get("strategy", "")
            action = result.get("action", {})

            if strategy:
                self.scratchpad.current_strategy = strategy

            return {
                "type": action.get("type", "recommend_topic"),
                "topic_id": action.get("topic_id"),
                "project_id": action.get("project_id"),
                "_thought": thought,
            }
        except Exception as e:
            logger.warning(f"ReAct LLM decision failed: {e}, falling back to rules")
            return None

    def _enhanced_rule_decision(self, observation: dict) -> dict:
        """Enhanced rule-based agent — same core logic as original + scratchpad intelligence."""
        available = observation.get("available_topics", [])
        completed = observation.get("completed_topics", [])
        current = observation.get("current_topic")
        quiz_history = observation.get("quiz_history_summary", {})
        projects_done = observation.get("completed_projects", [])
        job_readiness = observation.get("job_readiness_score", 0)
        total_steps = observation.get("total_steps", 0)
        topics_since_project = len(completed) - len(projects_done) * 3

        # 1. If current topic not yet quizzed → assign quiz
        if current and current not in quiz_history:
            return {"type": "assign_quiz", "topic_id": current}

        # 2. Every 3 topics → assign mini project
        if topics_since_project >= 3 and current:
            return {"type": "assign_mini_project", "topic_id": current}

        # 3. If current topic quizzed and passed → recommend next topic
        if current and quiz_history.get(current, 0) >= 70 and available:
            return {"type": "recommend_topic", "topic_id": available[0]}

        # 4. If current topic failed → resource if struggled badly, else retry
        if current and current in quiz_history and quiz_history.get(current, 0) < 70:
            # Enhanced: if failed twice with <50%, try resource first (only early in episode)
            if self.scratchpad.needs_resource_before_quiz(current) and total_steps < 60:
                return {"type": "recommend_resource", "topic_id": current}
            return {"type": "assign_quiz", "topic_id": current}

        # 5. If no current topic → recommend first available
        if available:
            return {"type": "recommend_topic", "topic_id": available[0]}

        # 6. If job readiness high → mark ready
        if job_readiness >= 0.8:
            return {"type": "mark_job_ready"}

        # 7. No available topics → assign capstone
        if not available and len(completed) > 0:
            return {"type": "assign_capstone", "topic_id": completed[-1] if completed else "python_basics"}

        # Default
        return {"type": "recommend_resource", "topic_id": current or "python_basics"}


# ═══════════════════════════════════════════════════════════
#  Original Rule-Based Agent (kept for --mode rule)
# ═══════════════════════════════════════════════════════════

def _rule_based_decision(observation: dict) -> dict:
    """Deterministic rule-based fallback agent for reproducible scores."""
    available = observation.get("available_topics", [])
    completed = observation.get("completed_topics", [])
    current = observation.get("current_topic")
    quiz_history = observation.get("quiz_history_summary", {})
    projects_done = observation.get("completed_projects", [])
    topics_since_project = len(completed) - len(projects_done) * 3

    # 1. If current topic not yet quizzed → assign quiz
    if current and current not in quiz_history:
        return {"type": "assign_quiz", "topic_id": current}

    # 2. Every 3 topics → assign mini project
    if topics_since_project >= 3 and current:
        return {"type": "assign_mini_project", "topic_id": current}

    # 3. If current topic quizzed and passed → recommend next topic
    if current and quiz_history.get(current, 0) >= 70 and available:
        return {"type": "recommend_topic", "topic_id": available[0]}

    # 4. If current topic failed → retry the quiz
    if current and current in quiz_history and quiz_history.get(current, 0) < 70:
        return {"type": "assign_quiz", "topic_id": current}

    # 5. If no current topic → recommend first available
    if available:
        return {"type": "recommend_topic", "topic_id": available[0]}

    # 6. If job readiness high → mark ready
    if observation.get("job_readiness_score", 0) >= 0.8:
        return {"type": "mark_job_ready"}

    # 7. No available topics → assign capstone
    if not available and len(completed) > 0:
        return {"type": "assign_capstone", "topic_id": completed[-1] if completed else "python_basics"}

    # Default
    return {"type": "recommend_resource", "topic_id": current or "python_basics"}


# ═══════════════════════════════════════════════════════════
#  PPO Agent Mode
# ═══════════════════════════════════════════════════════════

class PPOAgent:
    """PPO trained policy agent (no LLM needed)."""

    def __init__(self):
        self.model = None
        self._gym_env = None

    def load(self, task_id: str):
        """Load the trained PPO model for a task."""
        try:
            from stable_baselines3 import PPO
            from gym_wrapper import EduPathGymEnv

            model_path = os.path.join("models", f"ppo_edupath_{task_id}")
            if os.path.exists(model_path + ".zip"):
                self.model = PPO.load(model_path)
                self._gym_env = EduPathGymEnv(task_id=task_id)
                logger.info(f"Loaded PPO model from {model_path}")
                return True
            else:
                logger.warning(f"PPO model not found at {model_path}.zip")
                return False
        except ImportError:
            logger.warning("stable-baselines3 not installed, cannot use PPO mode")
            return False

    def decide(self, observation: dict) -> dict:
        """Use PPO model to decide action."""
        if not self.model or not self._gym_env:
            return _rule_based_decision(observation)

        self._gym_env._observation = observation
        obs_array = self._gym_env._encode_observation(observation)
        action_int, _ = self.model.predict(obs_array, deterministic=True)
        action = self._gym_env._decode_action(int(action_int))

        return {
            "type": action.type.value,
            "topic_id": action.topic_id,
            "project_id": action.project_id,
        }


# ═══════════════════════════════════════════════════════════
#  Agent Decision Router
# ═══════════════════════════════════════════════════════════

def get_agent_decision(observation: dict, mode: str = "react", agent=None) -> dict:
    """Route to the appropriate agent based on mode."""
    if mode == "ppo" and agent:
        return agent.decide(observation)
    elif mode == "react" and agent:
        return agent.decide(observation)
    elif mode == "hrl" and agent:
        return agent.decide(observation)
    elif mode == "reflexion" and agent:
        return agent.decide(observation)
    else:
        return _rule_based_decision(observation)


# ═══════════════════════════════════════════════════════════
#  HTTP Client — Talks to the running server
# ═══════════════════════════════════════════════════════════

class EnvHTTPClient:
    """HTTP client for interacting with the OpenEnv server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, student_profile: dict = None, seed: int = 42) -> dict:
        """POST /reset → returns observation."""
        payload = {"seed": seed}
        if student_profile:
            payload["student_profile"] = student_profile
        resp = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        """POST /step → returns observation, reward, done, info."""
        resp = requests.post(f"{self.base_url}/step", json=action, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        """POST /state → returns current state."""
        resp = requests.post(f"{self.base_url}/state", json={}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def grade(self, task_id: str) -> float:
        """POST /grade → returns score for the task."""
        resp = requests.post(
            f"{self.base_url}/grade",
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("score", 0.001)


# ═══════════════════════════════════════════════════════════
#  Direct Mode — Import env class directly (fallback)
# ═══════════════════════════════════════════════════════════

class EnvDirectClient:
    """Direct client that imports the environment class."""

    def __init__(self):
        from environment.env import EduPathEnv
        from environment.models import Action, ActionType
        from environment.student import student_manager
        self.EduPathEnv = EduPathEnv
        self.Action = Action
        self.ActionType = ActionType
        self.student_manager = student_manager
        self.env = None
        self._current_student_id = None

    def reset(self, student_profile: dict = None, seed: int = 42) -> dict:
        random.seed(seed)
        self.env = self.EduPathEnv()

        if student_profile:
            student = self.student_manager.create(name=student_profile.get("name", "Agent"))
            self.student_manager.update_from_onboarding(student.id, student_profile)
            student_id = student.id
        else:
            student = self.student_manager.create(name="Test Agent")
            self.student_manager.update_from_onboarding(student.id, {
                "target_field": "tech",
                "learning_goal": "Learn Python",
                "weekly_hours": 10,
            })
            student_id = student.id

        self._current_student_id = student_id
        obs = self.env.reset(student_id)
        return {
            "observation": obs.model_dump(),
            "info": {"student_id": student_id},
        }

    def step(self, action_dict: dict) -> dict:
        from environment.models import QuizDifficulty
        action_type = self.ActionType(action_dict["type"])
        difficulty = None
        if action_dict.get("difficulty"):
            try:
                difficulty = QuizDifficulty(action_dict["difficulty"])
            except ValueError:
                pass
        action = self.Action(
            type=action_type,
            topic_id=action_dict.get("topic_id"),
            project_id=action_dict.get("project_id"),
            difficulty=difficulty,
        )
        result = self.env.step(action)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward.model_dump(),
            "done": result.done,
            "info": result.info,
        }

    def state(self) -> dict:
        return self.env.state() if self.env else {}

    def grade(self, task_id: str) -> float:
        """Grade using direct grader import."""
        from environment.graders import grade_task1, grade_task2, grade_task3, grade_task4, grade_task5
        student = self.student_manager.get(self._current_student_id)
        if not student:
            return 0.001
        graders = {
            "task1_easy": lambda s: grade_task1(s),
            "task2_medium": lambda s: grade_task2(s),
            "task3_hard": lambda s: grade_task3(s),
            "task4_team": lambda s: grade_task4([s], steps_used=self.env.total_steps if self.env else 100),
            "task5_deadline": lambda s: grade_task5(s, steps_used=self.env.total_steps if self.env else 100),
        }
        grader = graders.get(task_id)
        if grader:
            score = grader(student)
            # CRITICAL: Ensure score is strictly within (0, 1)
            if score <= 0:
                score = 0.001
            elif score >= 1:
                score = 0.999
            return score
        return 0.001


def get_client(use_http: bool = True) -> object:
    """Get the appropriate client (HTTP or direct)."""
    if use_http:
        try:
            # Try to connect to running server
            resp = requests.get(f"{SERVER_URL}/health", timeout=5)
            if resp.status_code == 200:
                logger.info(f"Connected to server at {SERVER_URL}")
                return EnvHTTPClient(SERVER_URL)
        except Exception:
            logger.info("Server not running, falling back to direct mode")

    logger.info("Using direct environment mode")
    return EnvDirectClient()


# ═══════════════════════════════════════════════════════════
#  Episode Runner
# ═══════════════════════════════════════════════════════════

TASK_PROFILES = {
    "task1_easy": {
        "name": "Alex Beginner",
        "target_field": "tech",
        "learning_goal": "Learn Python programming from scratch",
        "weekly_hours": 10,
    },
    "task2_medium": {
        "name": "Jordan Analyst",
        "target_field": "tech",
        "learning_goal": "Become a Data Analyst in 3 months",
        "weekly_hours": 15,
        "skills": [{"skill": "Python", "level": "Intermediate", "proficiency": 0.4}],
        "resume_skills": ["python", "excel"],
    },
    "task3_hard": {
        "name": "Dr. Priya Shah",
        "target_field": "healthcare",
        "learning_goal": "Apply AI to medical imaging and clinical data analysis",
        "weekly_hours": 10,
        "skills": [
            {"skill": "Biology", "level": "Expert", "proficiency": 0.9},
            {"skill": "Statistics", "level": "Beginner", "proficiency": 0.3},
        ],
        "resume_skills": ["medicine", "clinical research", "biology"],
    },
    "task4_team": {
        "name": "Sam Engineer",
        "target_field": "business",
        "learning_goal": "Cross-train from tech into business strategy",
        "weekly_hours": 10,
        "skills": [
            {"skill": "Python", "level": "Advanced", "proficiency": 0.8},
            {"skill": "Data Structures", "level": "Intermediate", "proficiency": 0.6},
        ],
        "resume_skills": ["python", "data_structures", "web_development"],
    },
    "task5_deadline": {
        "name": "Nurse Taylor",
        "target_field": "healthcare",
        "learning_goal": "Healthcare AI Product Manager",
        "weekly_hours": 7,
        "skills": [
            {"skill": "Biology", "level": "Expert", "proficiency": 0.85},
            {"skill": "Healthcare", "level": "Advanced", "proficiency": 0.8},
            {"skill": "Statistics", "level": "Basic", "proficiency": 0.2},
        ],
        "resume_skills": ["nursing", "healthcare", "biology"],
    },
}

TASK_MAX_STEPS = {
    "task1_easy": 30,
    "task2_medium": 50,
    "task3_hard": 80,
    "task4_team": 100,
    "task5_deadline": 100,
}


def run_task(task_id: str, client, mode: str = "react", episodes: int = 1) -> float:
    """Run a task episode and return graded score."""
    profile = TASK_PROFILES[task_id]
    max_steps = TASK_MAX_STEPS[task_id]

    # Reflexion mode: run multiple episodes with reflection between them
    if mode == "reflexion":
        return _run_reflexion_task(task_id, client, profile, max_steps, episodes)

    # Reset environment
    reset_result = client.reset(student_profile=profile, seed=SEED)
    observation = reset_result["observation"]
    student_id = reset_result.get("info", {}).get("student_id", "unknown")

    # Initialize agent based on mode
    agent = None
    if mode == "react":
        agent = ReActAgent()
        agent.reset()
    elif mode == "ppo":
        agent = PPOAgent()
        if not agent.load(task_id):
            logger.warning(f"PPO model not found for {task_id}, falling back to rule-based")
            mode = "rule"
            agent = None
    elif mode == "hrl":
        agent = _HRLInferenceAgent(task_id)

    # Emit [START]
    model_labels = {
        "react": f"{MODEL_NAME}(ReAct)",
        "ppo": "PPO-MlpPolicy",
        "hrl": "HRL-Manager+Worker",
        "rule": f"{MODEL_NAME}(Rule)",
    }
    log_start(task=task_id, env="edupath-ai", model=model_labels.get(mode, mode))

    rewards_list: List[float] = []
    step_count = 0
    score = 0.001
    success = False

    try:
        for step in range(max_steps):
            # Agent decides action
            action = get_agent_decision(observation, mode=mode, agent=agent)
            step_count = step + 1

            # Execute action via environment
            result = client.step(action)

            reward_val = result.get("reward", {}).get("value", 0)
            done = result.get("done", False)

            rewards_list.append(reward_val)

            # Build action string representation for logs
            action_str = f"{action.get('type')}('{action.get('topic_id', '')}')"

            # Record in scratchpad if using ReAct
            if mode == "react" and agent:
                thought = action.get("_thought", "")
                new_observation = result.get("observation", observation)
                agent.record(action, reward_val, new_observation, thought)

            # Emit [STEP]
            log_step(
                step=step_count,
                action=action_str,
                reward=reward_val,
                done=done,
                error=None
            )

            observation = result.get("observation", observation)

            if done:
                break

        # Grade the task
        score = client.grade(task_id)
        # CRITICAL: Ensure score is strictly within (0, 1)
        if score <= 0:
            score = 0.001
        elif score >= 1:
            score = 0.999
        success = score >= 0.8
    finally:
        # Always emit [END] even if an exception occurs
        log_end(success=success, steps=step_count, score=score, rewards=rewards_list)

    return score


def _run_reflexion_task(task_id: str, client, profile: dict,
                        max_steps: int, episodes: int = 3) -> float:
    """Run multiple episodes with Reflexion agent."""
    from ai.reflexion_agent import ReflexionAgent

    agent = ReflexionAgent(max_reflections=5)
    best_score = 0.001

    for ep in range(episodes):
        agent.new_episode()
        reset_result = client.reset(student_profile=profile, seed=SEED + ep)
        observation = reset_result["observation"]

        model_label = f"Reflexion(ep={ep+1}/{episodes})"
        log_start(task=task_id, env="edupath-ai", model=model_label)

        rewards_list = []
        step_count = 0

        for step in range(max_steps):
            action = agent.decide(observation)
            step_count = step + 1

            result = client.step(action)
            reward_val = result.get("reward", {}).get("value", 0)
            done = result.get("done", False)
            rewards_list.append(reward_val)

            new_obs = result.get("observation", observation)
            agent.record_step(action, reward_val, new_obs, done)

            action_str = f"{action.get('type')}('{action.get('topic_id', '')}')"
            log_step(step=step_count, action=action_str, reward=reward_val, done=done)

            observation = new_obs
            if done:
                break

        score = client.grade(task_id)
        # CRITICAL: Ensure score is strictly within (0, 1)
        if score <= 0:
            score = 0.001
        elif score >= 1:
            score = 0.999
        success = score >= 0.8
        log_end(success=success, steps=step_count, score=score, rewards=rewards_list)

        # Reflect on the episode
        reflection = agent.reflect(final_score=score)
        logger.info(f"  Reflexion ep{ep+1}: score={score:.4f} | {reflection[:100]}...")

        if score > best_score:
            best_score = score

    # Save reflection memory
    os.makedirs("results", exist_ok=True)
    agent.memory.save(f"results/reflexion_memory_{task_id}.json")

    return best_score


class _HRLInferenceAgent:
    """HRL agent for inference using trained model."""
    def __init__(self, task_id: str):
        self.model = None
        self._env = None
        try:
            from stable_baselines3 import PPO
            from environment.hierarchical_env import HierarchicalEduPathEnv
            model_path = os.path.join("models", f"hrl_{task_id}")
            if os.path.exists(model_path + ".zip"):
                self.model = PPO.load(model_path)
                self._env = HierarchicalEduPathEnv(task_id=task_id)
                logger.info(f"Loaded HRL model from {model_path}")
        except Exception as e:
            logger.warning(f"HRL model not found: {e}")

    def decide(self, observation: dict) -> dict:
        if not self.model:
            return _rule_based_decision(observation)
        self._env._observation = observation
        obs_array = self._env._encode_observation(observation)
        action, _ = self.model.predict(obs_array, deterministic=True)
        decoded = self._env._decode_action(int(action[0]))
        return {
            "type": decoded.type.value,
            "topic_id": decoded.topic_id,
            "project_id": decoded.project_id,
        }


# ═══════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EduPath AI Inference")
    parser.add_argument("--task", type=str,
                        choices=["task1_easy", "task2_medium", "task3_hard", "task4_team", "task5_deadline"],
                        help="Run specific task")
    parser.add_argument("--all", action="store_true", help="Run all tasks and print scores")
    parser.add_argument("--direct", action="store_true",
                        help="Use direct env import instead of HTTP client")
    parser.add_argument("--mode", type=str, default="react",
                        choices=["react", "rule", "ppo", "reflexion", "hrl"],
                        help="Agent mode: react, rule, ppo, reflexion, hrl")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run (reflexion uses multiple)")
    args = parser.parse_args()

    client = get_client(use_http=not args.direct)

    if args.all:
        scores = {}
        all_tasks = ["task1_easy", "task2_medium", "task3_hard", "task4_team", "task5_deadline"]
        for task_id in all_tasks:
            logger.info(f"\n{'#' * 60}")
            logger.info(f"  RUNNING {task_id.upper()} (mode={args.mode})")
            logger.info(f"{'#' * 60}")
            scores[task_id] = run_task(task_id, client, mode=args.mode)

        print(f"\n{'='*50}")
        print(f"  SCORES (mode={args.mode})")
        print(f"{'='*50}")
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")
        avg = sum(scores.values()) / len(scores)
        print(f"  {'─'*30}")
        print(f"  Average: {avg:.4f}")
        print(f"{'='*50}")

    elif args.task:
        score = run_task(args.task, client, mode=args.mode)
        print(f"\n{args.task} score: {score:.4f}")

    else:
        # Default: run all tasks
        scores = {}
        all_tasks = ["task1_easy", "task2_medium", "task3_hard", "task4_team", "task5_deadline"]
        for task_id in all_tasks:
            scores[task_id] = run_task(task_id, client, mode=args.mode)

        print(f"\n{'='*50}")
        print(f"  SCORES (mode={args.mode})")
        print(f"{'='*50}")
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")
        avg = sum(scores.values()) / len(scores)
        print(f"  {'─'*30}")
        print(f"  Average: {avg:.4f}")
        print(f"{'='*50}")
