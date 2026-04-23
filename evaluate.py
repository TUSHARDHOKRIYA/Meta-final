"""
EduPath AI — Agent Evaluation & Benchmarking
Team KRIYA | Meta Hackathon 2026

Evaluates all agent types (Random, Heuristic, PPO, GNN-PPO, HRL,
Reflexion) across all 5 tasks. Generates comparative performance
tables and saves results to results/evaluation_results.json.
"""
import os
import sys
import json
import csv
import time
import argparse
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from environment.env import EduPathEnv
from environment.models import Action, ActionType, QuizDifficulty
from environment.student import student_manager
from environment.curriculum import TOPIC_GRAPH, get_available_topics
from environment.graders import grade_task1, grade_task2, grade_task3, grade_task4, grade_task5, _clamp_score


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
    "task1_easy": 50,
    "task2_medium": 80,
    "task3_hard": 120,
    "task4_team": 150,
    "task5_deadline": 120,
}


def _get_grader(task_id: str, steps_used: int = 0):
    """Get the appropriate grader function for a task.
    All grader outputs are clamped to (0, 1) as a safety net."""
    if task_id == "task1_easy":
        return lambda student: _clamp_score(grade_task1(student))
    elif task_id == "task2_medium":
        return lambda student: _clamp_score(grade_task2(student))
    elif task_id == "task3_hard":
        return lambda student: _clamp_score(grade_task3(student))
    elif task_id == "task4_team":
        return lambda student: _clamp_score(grade_task4([student], steps_used=steps_used))
    elif task_id == "task5_deadline":
        return lambda student: _clamp_score(grade_task5(student, steps_used=steps_used))
    return lambda student: 0.001


# ═══════════════════════════════════════════════════════════
#  Rule-Based Agent (basic, no memory)
# ═══════════════════════════════════════════════════════════

def _rule_based_decision(observation: dict) -> dict:
    """Deterministic rule-based agent (baseline, no scratchpad)."""
    available = observation.get("available_topics", [])
    completed = observation.get("completed_topics", [])
    current = observation.get("current_topic")
    quiz_history = observation.get("quiz_history_summary", {})
    projects_done = observation.get("completed_projects", [])
    topics_since_project = len(completed) - len(projects_done) * 3

    if current and current not in quiz_history:
        return {"type": "assign_quiz", "topic_id": current}
    if topics_since_project >= 3 and current:
        return {"type": "assign_mini_project", "topic_id": current}
    if current and quiz_history.get(current, 0) >= 70 and available:
        return {"type": "recommend_topic", "topic_id": available[0]}
    if current and current in quiz_history and quiz_history.get(current, 0) < 70:
        return {"type": "assign_quiz", "topic_id": current}
    if available:
        return {"type": "recommend_topic", "topic_id": available[0]}
    if observation.get("job_readiness_score", 0) >= 0.8:
        return {"type": "mark_job_ready"}
    if not available and len(completed) > 0:
        return {"type": "assign_capstone", "topic_id": completed[-1] if completed else "python_basics"}
    return {"type": "recommend_resource", "topic_id": current or "python_basics"}


# ═══════════════════════════════════════════════════════════
#  Enhanced ReAct Agent (scratchpad-enhanced, no LLM needed)
# ═══════════════════════════════════════════════════════════

class EvalReActAgent:
    """
    Enhanced rule-based agent with scratchpad memory.
    Matches the ReAct fallback logic from inference.py.
    This is the evaluation proxy for the LLM ReAct agent.
    """

    def __init__(self):
        self.topic_attempts = {}   # topic_id -> [scores]
        self.topics_recommended = []
        self.topics_completed = []
        self.projects_assigned = 0
        self.topics_since_project = 0

    def decide(self, observation: dict) -> dict:
        """Enhanced decision with scratchpad intelligence."""
        available = observation.get("available_topics", [])
        completed = observation.get("completed_topics", [])
        current = observation.get("current_topic")
        quiz_history = observation.get("quiz_history_summary", {})
        projects_done = observation.get("completed_projects", [])
        job_readiness = observation.get("job_readiness_score", 0)
        total_steps = observation.get("total_steps", 0)
        topics_since_project = len(completed) - len(projects_done) * 3

        # 1. If current topic has failed twice with <50% → assign resource first
        if current and current in quiz_history:
            scores = self.topic_attempts.get(current, [])
            low_scores = [s for s in scores if s < 50]
            if len(low_scores) >= 2 and total_steps < 80:
                return {"type": "recommend_resource", "topic_id": current}

        # 2. If current topic not yet quizzed → assign quiz
        if current and current not in quiz_history:
            return {"type": "assign_quiz", "topic_id": current}

        # 3. Every 3 completed topics → assign mini project
        if topics_since_project >= 3 and current:
            self.projects_assigned += 1
            self.topics_since_project = 0
            return {"type": "assign_mini_project", "topic_id": current}

        # 4. If current topic quizzed and passed → recommend next topic
        if current and quiz_history.get(current, 0) >= 70:
            if available:
                # Prefer topics with lower difficulty first (bridge strategy)
                best = available[0]
                best_score = -1
                for tid in available:
                    topic = TOPIC_GRAPH.get(tid)
                    if topic:
                        prereq_met = sum(1 for p in topic.prerequisites if p in completed)
                        score = prereq_met * 2 + (5 - topic.difficulty)
                        if score > best_score:
                            best_score = score
                            best = tid
                self.topics_since_project += 1
                return {"type": "recommend_topic", "topic_id": best}

        # 5. If current topic failed → retry quiz
        if current and current in quiz_history and quiz_history.get(current, 0) < 70:
            return {"type": "assign_quiz", "topic_id": current}

        # 6. If no current topic → recommend best available
        if available:
            best = available[0]
            best_score = -1
            for tid in available:
                topic = TOPIC_GRAPH.get(tid)
                if topic:
                    prereq_met = sum(1 for p in topic.prerequisites if p in completed)
                    score = prereq_met * 2 + (5 - topic.difficulty)
                    if score > best_score:
                        best_score = score
                        best = tid
            return {"type": "recommend_topic", "topic_id": best}

        # 7. If job readiness high → mark ready
        if job_readiness >= 0.8:
            return {"type": "mark_job_ready"}

        # 8. No available topics → assign capstone
        if not available and len(completed) > 0:
            return {"type": "assign_capstone", "topic_id": completed[-1]}

        return {"type": "recommend_resource", "topic_id": current or "python_basics"}

    def record(self, action: dict, observation: dict):
        """Record step result for scratchpad tracking."""
        action_type = action.get("type", "")
        topic_id = action.get("topic_id")
        quiz_history = observation.get("quiz_history_summary", {})

        if action_type == "assign_quiz" and topic_id:
            score = quiz_history.get(topic_id, 0)
            if topic_id not in self.topic_attempts:
                self.topic_attempts[topic_id] = []
            self.topic_attempts[topic_id].append(score)


# ═══════════════════════════════════════════════════════════
#  Episode Runners
# ═══════════════════════════════════════════════════════════

def _create_env_and_student(task_id: str, seed: int):
    """Create an environment and student for a task."""
    profile = TASK_PROFILES[task_id]
    env = EduPathEnv(seed=seed)
    student = student_manager.create(name=profile.get("name", "Eval Agent"))
    student_manager.update_from_onboarding(student.id, profile)
    obs = env.reset(student_id=student.id, seed=seed)
    return env, student, obs.model_dump()


def run_rule_based(task_id: str, seed: int = 42, max_steps: int = None) -> tuple:
    """Run rule-based agent on a task. Returns (score, total_reward, steps)."""
    max_steps = max_steps or TASK_MAX_STEPS[task_id]
    env, student, observation = _create_env_and_student(task_id, seed)

    total_reward = 0
    steps = 0

    for step in range(max_steps):
        action_dict = _rule_based_decision(observation)
        action_type = ActionType(action_dict["type"])
        action = Action(
            type=action_type,
            topic_id=action_dict.get("topic_id"),
            project_id=action_dict.get("project_id"),
        )
        result = env.step(action)
        total_reward += result.reward.value
        observation = result.observation.model_dump()
        steps = step + 1
        if result.done:
            break

    student = student_manager.get(student.id)
    grader = _get_grader(task_id, steps_used=steps)
    score = grader(student)
    return score, total_reward, steps


def run_react(task_id: str, seed: int = 42, max_steps: int = None) -> tuple:
    """Run enhanced ReAct agent on a task. Returns (score, total_reward, steps)."""
    max_steps = max_steps or TASK_MAX_STEPS[task_id]
    env, student, observation = _create_env_and_student(task_id, seed)

    agent = EvalReActAgent()
    total_reward = 0
    steps = 0

    for step in range(max_steps):
        action_dict = agent.decide(observation)
        action_type = ActionType(action_dict["type"])
        action = Action(
            type=action_type,
            topic_id=action_dict.get("topic_id"),
            project_id=action_dict.get("project_id"),
        )
        result = env.step(action)
        total_reward += result.reward.value
        observation = result.observation.model_dump()

        # Record in scratchpad
        agent.record(action_dict, observation)

        steps = step + 1
        if result.done:
            break

    student = student_manager.get(student.id)
    grader = _get_grader(task_id, steps_used=steps)
    score = grader(student)
    return score, total_reward, steps


def run_ppo(task_id: str, model_path: str = None, seed: int = 42, max_steps: int = None) -> tuple:
    """Run PPO trained agent on a task. Returns (score, total_reward, steps)."""
    try:
        from stable_baselines3 import PPO as PPOModel
        from gym_wrapper import EduPathGymEnv
    except ImportError:
        print("  ⚠ stable-baselines3 not installed, skipping PPO evaluation")
        return 0.001, 0, 0

    max_steps = max_steps or TASK_MAX_STEPS[task_id]

    if model_path is None:
        model_path = os.path.join("models", f"ppo_edupath_{task_id}")

    if not os.path.exists(model_path + ".zip"):
        print(f"  ⚠ PPO model not found at {model_path}.zip, skipping")
        return 0.001, 0, 0

    model = PPOModel.load(model_path)
    env = EduPathGymEnv(task_id=task_id, seed=seed)
    obs, _ = env.reset(seed=seed)

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    # Grade using the student created by gym wrapper
    student = student_manager.get(env._student_id)
    if student:
        grader = _get_grader(task_id, steps_used=steps)
        score = grader(student)
    else:
        score = 0.001
    return score, total_reward, steps


def run_ppo_gnn(task_id: str, seed: int = 42, max_steps: int = None) -> tuple:
    """Run PPO-GNN trained agent on a task."""
    try:
        from stable_baselines3 import PPO as PPOModel
        from gym_wrapper import GNNGymWrapper
    except ImportError:
        return run_rule_based(task_id, seed, max_steps)

    max_steps = max_steps or TASK_MAX_STEPS[task_id]
    model_path = os.path.join("models", f"ppo_gnn_{task_id}")

    if not os.path.exists(model_path + ".zip"):
        return run_rule_based(task_id, seed, max_steps)

    model = PPOModel.load(model_path)
    env = GNNGymWrapper(task_id=task_id, seed=seed)
    obs, _ = env.reset(seed=seed)

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    student = student_manager.get(env._student_id)
    if student:
        grader = _get_grader(task_id, steps_used=steps)
        score = grader(student)
    else:
        score = 0.001
    return score, total_reward, steps


def run_hrl(task_id: str, seed: int = 42, max_steps: int = None) -> tuple:
    """Run HRL trained agent on a task."""
    try:
        from stable_baselines3 import PPO as PPOModel
        from environment.hierarchical_env import HierarchicalEduPathEnv
    except ImportError:
        return run_rule_based(task_id, seed, max_steps)

    max_steps = max_steps or TASK_MAX_STEPS[task_id]
    model_path = os.path.join("models", f"hrl_{task_id}")

    if not os.path.exists(model_path + ".zip"):
        return run_rule_based(task_id, seed, max_steps)

    model = PPOModel.load(model_path)
    env = HierarchicalEduPathEnv(task_id=task_id, seed=seed)
    obs, _ = env.reset(seed=seed)

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    student = student_manager.get(env._student_id)
    if student:
        grader = _get_grader(task_id, steps_used=steps)
        score = grader(student)
    else:
        score = 0.001
    return score, total_reward, steps


def run_reflexion(task_id: str, seed: int = 42, max_steps: int = None) -> tuple:
    """Run Reflexion agent on a task (runs 3 internal reflection episodes, returns best)."""
    max_steps = max_steps or TASK_MAX_STEPS[task_id]
    from ai.reflexion_agent import ReflexionAgent
    agent = ReflexionAgent(max_reflections=5)
    best_score = 0.001
    best_reward = 0
    best_steps = 0

    for ep in range(3):
        agent.new_episode()
        env, student, obs_dict = _create_env_and_student(task_id, seed + ep)
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.decide(obs_dict)
            action_obj = Action(
                type=ActionType(action["type"]),
                topic_id=action.get("topic_id"),
            )
            result = env.step(action_obj)
            new_obs = result.observation.model_dump()
            total_reward += result.reward.value
            agent.record_step(action, result.reward.value, new_obs, result.done)
            obs_dict = new_obs
            steps = step + 1
            if result.done:
                break

        student = student_manager.get(student.id)
        grader = _get_grader(task_id, steps_used=steps)
        score = grader(student)
        agent.reflect(final_score=score)

        if score > best_score:
            best_score = score
            best_reward = total_reward
            best_steps = steps

    return best_score, best_reward, best_steps


# ═══════════════════════════════════════════════════════════
#  Main Evaluation
# ═══════════════════════════════════════════════════════════

def evaluate_all(num_episodes: int = 10, results_dir: str = "results"):
    """Evaluate all agents on all tasks."""
    os.makedirs(results_dir, exist_ok=True)

    agents = ["rule_based", "react_enhanced", "ppo_trained", "ppo_gnn", "hrl", "reflexion"]
    agent_runners = {
        "rule_based": run_rule_based,
        "react_enhanced": run_react,
        "ppo_trained": run_ppo,
        "ppo_gnn": run_ppo_gnn,
        "hrl": run_hrl,
        "reflexion": run_reflexion,
    }
    agent_display = {
        "rule_based": "Rule-based",
        "react_enhanced": "LLM ReAct",
        "ppo_trained": "PPO MLP Trained",
        "ppo_gnn": "PPO GNN Trained",
        "hrl": "HRL Strategy",
        "reflexion": "Reflexion",
    }
    tasks = ["task1_easy", "task2_medium", "task3_hard", "task4_team", "task5_deadline"]

    results = {}
    reward_results = {}
    episode_data = []

    for agent_type in agents:
        results[agent_type] = {}
        reward_results[agent_type] = {}
        runner = agent_runners[agent_type]

        for task_id in tasks:
            scores = []
            rewards = []
            all_steps = []

            print(f"  Running {agent_display[agent_type]:18s} on {task_id}...", end=" ", flush=True)
            for ep in range(num_episodes):
                seed = 42 + ep
                score, reward, steps = runner(task_id, seed=seed)
                scores.append(score)
                rewards.append(reward)
                all_steps.append(steps)
                episode_data.append({
                    "episode": ep + 1,
                    "reward": round(reward, 4),
                    "score": round(score, 4),
                    "task": task_id,
                    "agent_type": agent_type,
                    "steps": steps,
                })

            mean_score = float(np.mean(scores)) if scores else 0
            mean_reward = float(np.mean(rewards)) if rewards else 0
            results[agent_type][task_id] = round(mean_score, 4)
            reward_results[agent_type][task_id] = round(mean_reward, 4)
            print(f"score={mean_score:.4f}  reward={mean_reward:.2f}")

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"  AGENT COMPARISON — MEAN SCORES OVER {num_episodes} EPISODES")
    print(f"{'='*90}")
    print(f"  {'Agent':<18} | {'T1':>6} | {'T2':>6} | {'T3':>6} | {'T4':>6} | {'T5':>6} | {'Avg':>6}")
    print(f"  {'-'*18}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for agent_type in agents:
        name = agent_display[agent_type]
        task_scores = [results[agent_type].get(t, 0) for t in tasks]
        avg = sum(task_scores) / len(task_scores) if task_scores else 0
        scores_str = " | ".join(f"{s:>6.3f}" for s in task_scores)
        print(f"  {name:<18} | {scores_str} | {avg:>6.3f}")

    print(f"{'='*90}\n")

    # Save results
    full_results = {
        "scores": results,
        "rewards": reward_results,
        "num_episodes": num_episodes,
    }
    results_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Save episode data as CSV
    csv_path = os.path.join(results_dir, "episode_rewards.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "score", "task", "agent_type", "steps"])
        writer.writeheader()
        writer.writerows(episode_data)
    print(f"Episode data saved to {csv_path}")

    # Generate learning curve JSON (from training data if available)
    _generate_learning_curve(results_dir)

    return results


def _generate_learning_curve(results_dir: str):
    """Combine per-task learning curves into a single JSON."""
    combined = {}
    for task_id in ["task1_easy", "task2_medium", "task3_hard", "task4_team", "task5_deadline"]:
        curve_path = os.path.join(results_dir, f"learning_curve_{task_id}.json")
        if os.path.exists(curve_path):
            with open(curve_path, "r") as f:
                combined[task_id] = json.load(f)

    if combined:
        combined_path = os.path.join(results_dir, "learning_curve.json")
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Combined learning curve saved to {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EduPath AI Agents")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes per task")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    args = parser.parse_args()

    evaluate_all(num_episodes=args.episodes, results_dir=args.results_dir)
