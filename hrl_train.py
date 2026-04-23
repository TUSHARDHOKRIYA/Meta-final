"""
EduPath AI — Hierarchical RL Training Script
Team KRIYA | Meta Hackathon 2026

Trains the hierarchical RL agent (meta-controller + sub-controller)
using PPO on both levels. The meta-controller selects goals while
the sub-controller executes curriculum actions within each goal.
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))


def train_hrl(
    task_id: str = "task2_medium",
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    seed: int = 42,
    save_dir: str = "models",
    results_dir: str = "results",
    verbose: int = 1,
):
    """Train HRL agent on EduPath environment."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from environment.hierarchical_env import HierarchicalEduPathEnv

    print(f"\n{'='*60}")
    print(f"  EduPath AI — HRL Training")
    print(f"  Task: {task_id}")
    print(f"  Timesteps: {total_timesteps}")
    print(f"{'='*60}\n")

    env = HierarchicalEduPathEnv(task_id=task_id, seed=seed)

    print("Checking environment...")
    try:
        check_env(env, warn=True)
        print("✓ Environment check passed!\n")
    except Exception as e:
        print(f"⚠ Warning: {e}\n")

    # Train unified HRL agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=verbose,
        seed=seed,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Higher entropy for strategy exploration
        vf_coef=0.5,
    )

    print(f"Starting HRL training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"hrl_{task_id}")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Evaluate
    print("\nEvaluating HRL agent...")
    eval_rewards = []
    strategy_usage = {}
    for ep in range(10):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = 0
        done = False
        steps = 0
        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
            # Track strategy usage
            strategy = info.get("strategy", "unknown")
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        eval_rewards.append(episode_reward)

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"Mean eval reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Strategy usage: {json.dumps(strategy_usage, indent=2)}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results = {
        "task_id": task_id,
        "agent": "hrl",
        "total_timesteps": total_timesteps,
        "training_time_seconds": round(train_time, 1),
        "eval_mean_reward": round(float(mean_reward), 4),
        "eval_std_reward": round(float(std_reward), 4),
        "eval_rewards": [round(float(r), 4) for r in eval_rewards],
        "strategy_usage": strategy_usage,
    }

    results_path = os.path.join(results_dir, f"hrl_results_{task_id}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRL on EduPath AI")
    parser.add_argument("--task", type=str, default="task2_medium",
                        choices=["task1_easy", "task2_medium", "task3_hard",
                                 "task4_team", "task5_deadline", "all"],
                        help="Task to train on")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if args.task == "all":
        for task_id in ["task1_easy", "task2_medium", "task3_hard", "task4_team", "task5_deadline"]:
            train_hrl(task_id=task_id, total_timesteps=args.timesteps, verbose=args.verbose)
    else:
        train_hrl(task_id=args.task, total_timesteps=args.timesteps, verbose=args.verbose)
