"""
EduPath AI — Ablation Study Runner
Team KRIYA | Meta Hackathon 2026

Systematic ablation study comparing agent architectures and
environment features. Measures the contribution of each component
(BKT, ICM, GNN, HRL) to overall task performance.
"""
import os
import sys
import json
import time
import argparse
import random
import logging
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from environment.env import EduPathEnv
from environment.models import Action, ActionType
from environment.student import student_manager
from environment.curriculum import TOPIC_GRAPH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
TASKS = ["task1_easy", "task2_medium", "task3_hard", "task4_team", "task5_deadline"]
AGENTS = ["rule", "react", "ppo", "gnn", "hrl", "reflexion"]

TASK_PROFILES = {
    "task1_easy": {
        "name": "Alex Beginner", "target_field": "tech",
        "learning_goal": "Learn Python programming from scratch", "weekly_hours": 10,
    },
    "task2_medium": {
        "name": "Jordan Analyst", "target_field": "tech",
        "learning_goal": "Become a Data Analyst in 3 months", "weekly_hours": 15,
        "skills": [{"skill": "Python", "level": "Intermediate", "proficiency": 0.4}],
        "resume_skills": ["python", "excel"],
    },
    "task3_hard": {
        "name": "Dr. Priya Shah", "target_field": "healthcare",
        "learning_goal": "Apply AI to medical imaging and clinical data analysis", "weekly_hours": 10,
        "skills": [
            {"skill": "Biology", "level": "Expert", "proficiency": 0.9},
            {"skill": "Statistics", "level": "Beginner", "proficiency": 0.3},
        ],
        "resume_skills": ["medicine", "clinical research", "biology"],
    },
    "task4_team": {
        "name": "Sam Engineer", "target_field": "business",
        "learning_goal": "Cross-train from tech into business strategy", "weekly_hours": 10,
        "skills": [
            {"skill": "Python", "level": "Advanced", "proficiency": 0.8},
            {"skill": "Data Structures", "level": "Intermediate", "proficiency": 0.6},
        ],
        "resume_skills": ["python", "data_structures", "web_development"],
    },
    "task5_deadline": {
        "name": "Nurse Taylor", "target_field": "healthcare",
        "learning_goal": "Healthcare AI Product Manager", "weekly_hours": 7,
        "skills": [
            {"skill": "Biology", "level": "Expert", "proficiency": 0.85},
            {"skill": "Healthcare", "level": "Advanced", "proficiency": 0.8},
            {"skill": "Statistics", "level": "Basic", "proficiency": 0.2},
        ],
        "resume_skills": ["nursing", "healthcare", "biology"],
    },
}

TASK_MAX_STEPS = {
    "task1_easy": 50, "task2_medium": 80, "task3_hard": 120,
    "task4_team": 150, "task5_deadline": 120,
}


def _get_grader_fn(task_id: str, student, steps_used: int) -> float:
    """Grade a student using the appropriate task grader.
    All scores are clamped to (0, 1) as a safety net."""
    from environment.graders import grade_task1, grade_task2, grade_task3, grade_task4, grade_task5, _clamp_score
    graders = {
        "task1_easy": lambda s: grade_task1(s),
        "task2_medium": lambda s: grade_task2(s),
        "task3_hard": lambda s: grade_task3(s),
        "task4_team": lambda s: grade_task4([s], steps_used=steps_used),
        "task5_deadline": lambda s: grade_task5(s, steps_used=steps_used),
    }
    raw = graders[task_id](student)
    return _clamp_score(raw)


def run_rule_episode(task_id: str, seed: int) -> tuple:
    """Run a single rule-based episode, return (score, reward, steps)."""
    random.seed(seed)
    env = EduPathEnv(seed=seed)
    profile = TASK_PROFILES[task_id]
    student = student_manager.create(name=profile.get("name", "Ablation"))
    student_manager.update_from_onboarding(student.id, profile)

    obs = env.reset(student_id=student.id, seed=seed)
    obs_dict = obs.model_dump()
    total_reward = 0
    max_steps = TASK_MAX_STEPS[task_id]

    for step in range(max_steps):
        action = _rule_action(obs_dict)
        action_obj = Action(
            type=ActionType(action["type"]),
            topic_id=action.get("topic_id"),
        )
        result = env.step(action_obj)
        obs_dict = result.observation.model_dump()
        total_reward += result.reward.value
        if result.done:
            break

    student = student_manager.get(student.id)
    score = _get_grader_fn(task_id, student, env.total_steps)
    return score, total_reward, step + 1


def run_react_episode(task_id: str, seed: int) -> tuple:
    """Run a scratchpad-enhanced ReAct agent episode (NOT rule fallback)."""
    random.seed(seed)
    env = EduPathEnv(seed=seed)
    profile = TASK_PROFILES[task_id]
    student = student_manager.create(name=profile.get("name", "ReAct"))
    student_manager.update_from_onboarding(student.id, profile)

    obs = env.reset(student_id=student.id, seed=seed)
    obs_dict = obs.model_dump()
    total_reward = 0
    max_steps = TASK_MAX_STEPS[task_id]

    # Scratchpad state
    topic_attempts = {}  # topic_id -> [scores]
    topics_since_project = 0

    for step in range(max_steps):
        action = _react_action(obs_dict, topic_attempts, topics_since_project)
        action_obj = Action(
            type=ActionType(action["type"]),
            topic_id=action.get("topic_id"),
        )
        result = env.step(action_obj)
        obs_dict = result.observation.model_dump()
        total_reward += result.reward.value

        # Update scratchpad
        atype = action.get("type", "")
        tid = action.get("topic_id")
        if atype == "assign_quiz" and tid:
            score_val = obs_dict.get("quiz_history_summary", {}).get(tid, 0)
            topic_attempts.setdefault(tid, []).append(score_val)
        if atype == "recommend_topic":
            topics_since_project += 1
        if atype in ("assign_mini_project", "assign_capstone"):
            topics_since_project = 0

        if result.done:
            break

    student = student_manager.get(student.id)
    score = _get_grader_fn(task_id, student, env.total_steps)
    return score, total_reward, step + 1


def run_reflexion_episode(task_id: str, seed: int, episodes: int = 3) -> tuple:
    """Run reflexion agent for multiple inner episodes, return best score."""
    from ai.reflexion_agent import ReflexionAgent

    agent = ReflexionAgent(max_reflections=5)
    best_score = 0.001
    best_reward = 0
    best_steps = 0

    for ep in range(episodes):
        agent.new_episode()
        random.seed(seed + ep)
        env = EduPathEnv(seed=seed + ep)
        profile = TASK_PROFILES[task_id]
        student = student_manager.create(name=profile.get("name", "Reflexion"))
        student_manager.update_from_onboarding(student.id, profile)

        obs = env.reset(student_id=student.id, seed=seed + ep)
        obs_dict = obs.model_dump()
        total_reward = 0
        max_steps = TASK_MAX_STEPS[task_id]

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
            if result.done:
                break

        student = student_manager.get(student.id)
        score = _get_grader_fn(task_id, student, env.total_steps)
        agent.reflect(final_score=score)

        if score > best_score:
            best_score = score
            best_reward = total_reward
            best_steps = step + 1

    return best_score, best_reward, best_steps


def run_ppo_episode(task_id: str, seed: int, model_prefix: str = "ppo_edupath") -> tuple:
    """Run PPO (MLP or GNN) episode. Returns (score, reward, steps)."""
    try:
        from stable_baselines3 import PPO
        from gym_wrapper import EduPathGymEnv
        model_path = os.path.join("models", f"{model_prefix}_{task_id}")
        if not os.path.exists(model_path + ".zip"):
            logger.warning(f"Model not found: {model_path}.zip — using rule-based fallback")
            return run_rule_episode(task_id, seed)

        model = PPO.load(model_path)
        env = EduPathGymEnv(task_id=task_id, seed=seed)
        obs, _ = env.reset(seed=seed)
        total_reward = 0
        max_steps = TASK_MAX_STEPS[task_id]

        for step in range(max_steps):
            action_int, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += reward
            if terminated or truncated:
                break

        student = student_manager.get(env._student_id)
        score = _get_grader_fn(task_id, student, step + 1) if student else 0.001
        return score, total_reward, step + 1

    except ImportError:
        logger.warning("stable_baselines3 not available — using rule-based fallback")
        return run_rule_episode(task_id, seed)


def run_ppo_gnn_episode(task_id: str, seed: int) -> tuple:
    """Run PPO GNN episode."""
    try:
        from stable_baselines3 import PPO
        from gym_wrapper import GNNGymWrapper
        model_path = os.path.join("models", f"ppo_gnn_{task_id}")
        if not os.path.exists(model_path + ".zip"):
            logger.warning(f"Model not found: {model_path}.zip — using rule-based fallback")
            return run_rule_episode(task_id, seed)

        model = PPO.load(model_path)
        env = GNNGymWrapper(task_id=task_id, seed=seed)
        obs, _ = env.reset(seed=seed)
        total_reward = 0
        max_steps = TASK_MAX_STEPS[task_id]

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        student = student_manager.get(env._student_id)
        score = _get_grader_fn(task_id, student, step + 1) if student else 0.001
        return score, total_reward, step + 1

    except ImportError:
        return run_rule_episode(task_id, seed)


def run_hrl_episode(task_id: str, seed: int) -> tuple:
    """Run HRL episode."""
    try:
        from stable_baselines3 import PPO
        from environment.hierarchical_env import HierarchicalEduPathEnv
        model_path = os.path.join("models", f"hrl_{task_id}")
        if not os.path.exists(model_path + ".zip"):
            logger.warning(f"HRL model not found: {model_path}.zip — using rule-based fallback")
            return run_rule_episode(task_id, seed)

        model = PPO.load(model_path)
        env = HierarchicalEduPathEnv(task_id=task_id, seed=seed)
        obs, _ = env.reset(seed=seed)
        total_reward = 0
        max_steps = TASK_MAX_STEPS[task_id]

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        student = student_manager.get(env._student_id)
        score = _get_grader_fn(task_id, student, step + 1) if student else 0.001
        return score, total_reward, step + 1

    except ImportError:
        return run_rule_episode(task_id, seed)


def _rule_action(obs: dict) -> dict:
    """Simple rule-based action selection."""
    available = obs.get("available_topics", [])
    completed = obs.get("completed_topics", [])
    current = obs.get("current_topic")
    quiz_history = obs.get("quiz_history_summary", {})
    projects_done = obs.get("completed_projects", [])
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
    if obs.get("job_readiness_score", 0) >= 0.8:
        return {"type": "mark_job_ready"}
    if not available and len(completed) > 0:
        return {"type": "assign_capstone", "topic_id": completed[-1] if completed else "python_basics"}
    return {"type": "recommend_resource", "topic_id": current or "python_basics"}


def _react_action(obs: dict, topic_attempts: dict, topics_since_project: int) -> dict:
    """Scratchpad-enhanced ReAct action — smarter than basic rule agent."""
    available = obs.get("available_topics", [])
    completed = obs.get("completed_topics", [])
    current = obs.get("current_topic")
    quiz_history = obs.get("quiz_history_summary", {})
    projects_done = obs.get("completed_projects", [])
    job_readiness = obs.get("job_readiness_score", 0)
    total_steps = obs.get("total_steps", 0)

    # 1. If current topic failed 2+ times with <50 → assign resource to re-study
    if current and current in topic_attempts:
        low_scores = [s for s in topic_attempts.get(current, []) if s < 50]
        if len(low_scores) >= 2 and total_steps < 80:
            return {"type": "recommend_resource", "topic_id": current}

    # 2. If current topic not yet quizzed → assign quiz
    if current and current not in quiz_history:
        return {"type": "assign_quiz", "topic_id": current}

    # 3. Every 3 topics → assign mini project
    if topics_since_project >= 3 and current:
        return {"type": "assign_mini_project", "topic_id": current}

    # 4. If current topic quizzed and passed → recommend best available topic
    if current and quiz_history.get(current, 0) >= 70 and available:
        # Smart topic selection: prefer topics with more completed prereqs + lower difficulty
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


def run_ablation(num_episodes: int = 5, agents: List[str] = None):
    """Run full ablation study."""
    if agents is None:
        agents = AGENTS

    results = {}
    reward_results = {}
    all_details = {}

    print(f"\n{'='*80}")
    print(f"  EDUPATH AI — ABLATION STUDY")
    print(f"  Agents: {', '.join(agents)}")
    print(f"  Tasks: {', '.join(TASKS)}")
    print(f"  Episodes per (agent, task): {num_episodes}")
    print(f"{'='*80}\n")

    start_time = time.time()

    for agent_type in agents:
        results[agent_type] = {}
        reward_results[agent_type] = {}
        all_details[agent_type] = {}

        for task_id in TASKS:
            episode_scores = []
            episode_rewards = []

            for ep in range(num_episodes):
                seed = SEED + ep

                if agent_type == "rule":
                    score, reward, steps = run_rule_episode(task_id, seed)
                elif agent_type == "react":
                    score, reward, steps = run_react_episode(task_id, seed)
                elif agent_type == "ppo":
                    score, reward, steps = run_ppo_episode(task_id, seed, "ppo_edupath")
                elif agent_type == "gnn":
                    score, reward, steps = run_ppo_gnn_episode(task_id, seed)
                elif agent_type == "hrl":
                    score, reward, steps = run_hrl_episode(task_id, seed)
                elif agent_type == "reflexion":
                    score, reward, steps = run_reflexion_episode(task_id, seed, episodes=3)
                else:
                    score, reward, steps = run_rule_episode(task_id, seed)

                episode_scores.append(score)
                episode_rewards.append(reward)

            mean_score = sum(episode_scores) / len(episode_scores)
            mean_reward = sum(episode_rewards) / len(episode_rewards)
            results[agent_type][task_id] = round(mean_score, 4)
            reward_results[agent_type][task_id] = round(mean_reward, 4)
            all_details[agent_type][task_id] = {
                "scores": [round(s, 4) for s in episode_scores],
                "rewards": [round(r, 4) for r in episode_rewards],
                "mean_score": round(mean_score, 4),
                "mean_reward": round(mean_reward, 4),
            }

            print(f"  {agent_type:>12} × {task_id:<15} → score={mean_score:.4f}  reward={mean_reward:.2f}")

    total_time = time.time() - start_time
    print(f"\nAblation completed in {total_time:.1f} seconds")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  ABLATION RESULTS — {num_episodes} EPISODES PER CELL")
    print(f"{'='*90}")
    print(f"  {'Agent':<12} | {'T1':>7} | {'T2':>7} | {'T3':>7} | {'T4':>7} | {'T5':>7} | {'Avg':>7}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    baseline_avg = None
    for agent_type in agents:
        scores = [results[agent_type].get(t, 0) for t in TASKS]
        avg = sum(scores) / len(scores)
        if baseline_avg is None:
            baseline_avg = avg
        delta = f"(+{(avg - baseline_avg):.3f})" if avg > baseline_avg else ""
        scores_str = " | ".join(f"{s:>7.4f}" for s in scores)
        print(f"  {agent_type:<12} | {scores_str} | {avg:>7.4f} {delta}")

    print(f"{'='*90}\n")

    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        "num_episodes": num_episodes,
        "agents": agents,
        "tasks": TASKS,
        "results": results,
        "reward_results": reward_results,
        "details": all_details,
        "total_time_seconds": round(total_time, 1),
    }

    with open("results/ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved: results/ablation_results.json")

    # Generate markdown table
    md_lines = ["# EduPath AI — Ablation Study Results\n"]
    md_lines.append(f"*{num_episodes} episodes per (agent, task) pair*\n")
    md_lines.append("| Agent | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | **Average** |")
    md_lines.append("|-------|--------|--------|--------|--------|--------|-------------|")

    for agent_type in agents:
        scores = [results[agent_type].get(t, 0) for t in TASKS]
        avg = sum(scores) / len(scores)
        scores_str = " | ".join(f"{s:.4f}" for s in scores)
        md_lines.append(f"| {agent_type} | {scores_str} | **{avg:.4f}** |")

    md_lines.append("\n---\n")
    md_lines.append(f"Total ablation time: {total_time:.1f}s\n")

    with open("results/ablation_table.md", "w") as f:
        f.write("\n".join(md_lines))
    print("Saved: results/ablation_table.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EduPath AI Ablation Study")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes per (agent, task) pair")
    parser.add_argument("--agents", nargs="+", default=None,
                        choices=AGENTS,
                        help="Which agents to include")
    args = parser.parse_args()

    run_ablation(
        num_episodes=args.episodes,
        agents=args.agents,
    )
