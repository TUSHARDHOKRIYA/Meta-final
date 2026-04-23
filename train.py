"""
EduPath AI — PPO Training Script
Team KRIYA | Meta Hackathon 2026

Trains Stable-Baselines3 PPO agents on the EduPath Gym environment
across all 5 task configurations. Saves trained models and learning
curves to the models/ and results/ directories.
"""
import os
import sys
import json
import time
import argparse
import logging
import numpy as np

# Suppress noisy loggers during training (Supabase, etc.)
logging.basicConfig(level=logging.WARNING)
for _name in ["backend.db", "db.supabase_client", "environment.student", "__main__"]:
    logging.getLogger(_name).setLevel(logging.ERROR)

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from gym_wrapper import EduPathGymEnv


def train_ppo(
    task_id: str = "task2_medium",
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    seed: int = 42,
    save_dir: str = "models",
    results_dir: str = "results",
    verbose: int = 1,
    policy_type: str = "mlp",
    use_curiosity: bool = False,
):
    """Train a PPO agent on EduPath environment."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback

    is_gnn = policy_type == "gnn"
    policy_label = "GNN" if is_gnn else "MLP"
    curiosity_label = " + ICM" if use_curiosity else ""

    print(f"\n{'='*60}")
    print(f"  EduPath AI — PPO Training ({policy_label}{curiosity_label})")
    print(f"  Task: {task_id}")
    print(f"  Timesteps: {total_timesteps}")
    print(f"{'='*60}\n")

    # Create environment
    if is_gnn:
        from gym_wrapper import GNNGymWrapper
        env = GNNGymWrapper(task_id=task_id, seed=seed, use_curiosity=use_curiosity)
    else:
        env = EduPathGymEnv(task_id=task_id, seed=seed)

    # Check environment
    print("Checking environment compatibility...")
    try:
        check_env(env, warn=True)
        print("✓ Environment check passed!\n")
    except Exception as e:
        print(f"⚠ Environment check warning: {e}\n")

    # Custom callback to track rewards per episode
    class RewardTracker(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.current_rewards = []

        def _on_step(self) -> bool:
            if self.locals.get("dones") is not None:
                for i, done in enumerate(self.locals["dones"]):
                    if done:
                        ep_info = self.locals.get("infos", [{}])[i].get("episode")
                        if ep_info:
                            self.episode_rewards.append(ep_info["r"])
                            self.episode_lengths.append(ep_info["l"])
            return True

    reward_tracker = RewardTracker(verbose=verbose)

    # Create PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=verbose,
        seed=seed,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=None,
    )

    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=reward_tracker)
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    if is_gnn:
        model_path = os.path.join(save_dir, f"ppo_gnn_{task_id}")
    else:
        model_path = os.path.join(save_dir, f"ppo_edupath_{task_id}")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Evaluate trained model
    print("\nEvaluating trained model...")
    eval_rewards = []
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
        eval_rewards.append(episode_reward)
        print(f"  Episode {ep+1}: reward={episode_reward:.3f}, steps={steps}")

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"\nMean eval reward: {mean_reward:.3f} ± {std_reward:.3f}")

    # Save learning curve data
    os.makedirs(results_dir, exist_ok=True)
    learning_curve = {
        "task_id": task_id,
        "policy_type": policy_type,
        "curiosity": use_curiosity,
        "total_timesteps": total_timesteps,
        "training_time_seconds": round(train_time, 1),
        "eval_mean_reward": round(float(mean_reward), 4),
        "eval_std_reward": round(float(std_reward), 4),
        "episode_rewards": [round(float(r), 4) for r in reward_tracker.episode_rewards],
        "eval_rewards": [round(float(r), 4) for r in eval_rewards],
        "hyperparameters": {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
    }

    suffix = f"_gnn" if is_gnn else ""
    curve_path = os.path.join(results_dir, f"learning_curve_{task_id}{suffix}.json")
    with open(curve_path, "w") as f:
        json.dump(learning_curve, f, indent=2)
    print(f"Learning curve saved to {curve_path}")

    return model, learning_curve


def train_all_tasks(total_timesteps: int = 50000, policy_type: str = "mlp",
                    use_curiosity: bool = False):
    """Train PPO on all tasks."""
    results = {}
    all_tasks = ["task1_easy", "task2_medium", "task3_hard", "task4_team", "task5_deadline"]
    for task_id in all_tasks:
        print(f"\n{'#'*60}")
        print(f"  Training on {task_id} (policy={policy_type})")
        print(f"{'#'*60}")
        _, curve = train_ppo(
            task_id=task_id, total_timesteps=total_timesteps,
            policy_type=policy_type, use_curiosity=use_curiosity,
        )
        results[task_id] = curve

    # Save combined results
    suffix = "_gnn" if policy_type == "gnn" else ""
    combined_path = os.path.join("results", f"learning_curve{suffix}.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY ({policy_type.upper()})")
    print(f"{'='*60}")
    for task_id, curve in results.items():
        print(f"  {task_id}: mean_reward={curve['eval_mean_reward']:.3f} "
              f"± {curve['eval_std_reward']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on EduPath AI")
    parser.add_argument("--task", type=str, default="task2_medium",
                        choices=["task1_easy", "task2_medium", "task3_hard", "task4_team", "task5_deadline", "all"],
                        help="Task to train on (or 'all')")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level")
    parser.add_argument("--policy", type=str, default="mlp",
                        choices=["mlp", "gnn"],
                        help="Policy type: mlp (default) or gnn (graph neural network)")
    parser.add_argument("--curiosity", action="store_true",
                        help="Enable Intrinsic Curiosity Module (ICM)")
    args = parser.parse_args()

    if args.task == "all":
        train_all_tasks(
            total_timesteps=args.timesteps,
            policy_type=args.policy,
            use_curiosity=args.curiosity,
        )
    else:
        train_ppo(
            task_id=args.task,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            verbose=args.verbose,
            policy_type=args.policy,
            use_curiosity=args.curiosity,
        )
