"""
EduPath AI — TRL/GRPO Training Script
Team KRIYA | Meta Hackathon 2026

Trains an LLM-based tutoring agent using Group Relative Policy
Optimization (GRPO) from HuggingFace TRL. The EduPath environment
serves as the verifier — each candidate action is scored by the
environment's reward function.

Usage:
    python train_trl.py                    # Quick test (100 steps)
    python train_trl.py --steps 1000       # Full training
    python train_trl.py --model unsloth/Llama-3.2-1B-Instruct

Requirements:
    pip install trl transformers torch accelerate peft
    # Optional for fast training: pip install unsloth
"""
import os
import sys
import json
import argparse
import logging
from typing import List, Dict

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Environment Reward Function (Verifier)
# ═══════════════════════════════════════════════════════════

def get_env_reward(action_text: str, observation: dict) -> float:
    """
    Use the EduPath environment as a verifier to score an LLM action.

    The LLM outputs a JSON action like:
        {"type": "recommend_topic", "topic_id": "python_basics"}

    We parse it than run it through the environment's reward function.

    Returns:
        float: Reward value from the environment (-0.2 to +1.0)
    """
    from environment.env import EduPathEnv
    from environment.models import Action, ActionType

    try:
        # Parse the LLM's action
        action_dict = json.loads(action_text.strip())
        action_type = ActionType(action_dict.get("type", "recommend_resource"))
        action = Action(
            type=action_type,
            topic_id=action_dict.get("topic_id"),
            project_id=action_dict.get("project_id"),
        )

        # Create a temporary env to calculate the reward
        env = EduPathEnv(seed=42)
        reward = env._calculate_reward(action)
        return reward.value

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Invalid action format → negative reward
        return -0.1


def build_prompts_from_env(num_prompts: int = 50, task_id: str = "task1_easy") -> List[str]:
    """
    Generate training prompts by running the environment and collecting
    observation states. Each prompt asks the LLM to decide the next
    tutoring action given the current student state.
    """
    from environment.env import EduPathEnv
    from environment.models import Action, ActionType
    from environment.student import student_manager
    from environment.curriculum import TOPIC_GRAPH

    TASK_PROFILES = {
        "task1_easy": {
            "name": "Alex Beginner", "target_field": "tech",
            "learning_goal": "Learn Python programming from scratch",
            "weekly_hours": 10,
        },
        "task2_medium": {
            "name": "Jordan Analyst", "target_field": "tech",
            "learning_goal": "Become a Data Analyst in 3 months",
            "weekly_hours": 15,
            "skills": [{"skill": "Python", "level": "Intermediate", "proficiency": 0.4}],
            "resume_skills": ["python", "excel"],
        },
    }

    profile = TASK_PROFILES.get(task_id, TASK_PROFILES["task1_easy"])
    prompts = []

    for seed in range(num_prompts):
        env = EduPathEnv(seed=seed)
        student = student_manager.create(name=f"TRL_Student_{seed}")
        student_manager.update_from_onboarding(student.id, profile)
        obs = env.reset(student_id=student.id, seed=seed)
        obs_dict = obs.model_dump()

        prompt = _format_observation_as_prompt(obs_dict)
        prompts.append(prompt)

    return prompts


def _format_observation_as_prompt(obs: dict) -> str:
    """Format an environment observation as an LLM prompt."""
    return f"""You are an AI tutoring agent for EduPath AI. Your job is to guide
a student through a personalized learning curriculum.

CURRENT STATE:
- Student ID: {obs.get('student_id', 'unknown')}
- Completed Topics: {obs.get('completed_topics', [])}
- Current Topic: {obs.get('current_topic', 'None')}
- Available Topics: {obs.get('available_topics', [])}
- Quiz History: {obs.get('quiz_history_summary', {})}
- Job Readiness: {obs.get('job_readiness_score', 0)}
- Step: {obs.get('total_steps', 0)}
- Target Field: {obs.get('target_field', 'tech')}
- Learning Goal: {obs.get('learning_goal', '')}

VALID ACTIONS:
1. recommend_topic - Recommend a new topic for the student to study
2. assign_quiz - Assign a quiz on the current topic
3. assign_mini_project - Assign a hands-on project
4. assign_capstone - Assign a capstone project
5. recommend_resource - Recommend a learning resource
6. suggest_event - Suggest a relevant event/hackathon
7. mark_job_ready - Mark the student as job-ready

Choose the BEST next action. Respond with ONLY a JSON object:
{{"type": "<action_type>", "topic_id": "<topic_id_if_needed>"}}"""


# ═══════════════════════════════════════════════════════════
#  GRPO Reward Function
# ═══════════════════════════════════════════════════════════

def reward_function(completions, **kwargs) -> List[float]:
    """
    GRPO reward function — scores each LLM completion using the
    EduPath environment as a verifier.

    Args:
        completions: List of LLM-generated action strings/dicts.
        **kwargs: Extra args from TRL (prompts, completion_ids, etc.)

    Returns:
        List of reward values.
    """
    rewards = []
    for i, completion in enumerate(completions):
        # Handle various completion formats from TRL
        if isinstance(completion, list):
            text = ''
            for msg in completion:
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    text = msg.get('content', '')
                    break
            if not text and completion:
                last = completion[-1]
                text = last.get('content', '') if isinstance(last, dict) else str(last)
        elif isinstance(completion, dict):
            text = completion.get('content', str(completion))
        else:
            text = str(completion)

        # Extract JSON from the completion
        action_text = _extract_json(text)
        reward = get_env_reward(action_text, {})

        # Bonus for valid JSON format
        try:
            parsed = json.loads(action_text)
            if "type" in parsed:
                reward += 0.1  # Format bonus
        except (json.JSONDecodeError, TypeError):
            reward -= 0.1  # Format penalty

        rewards.append(reward)

    return rewards


def _extract_json(text: str) -> str:
    """Extract JSON object from LLM output."""
    import re
    # Try to find JSON in the text
    match = re.search(r'\{[^}]+\}', text)
    if match:
        return match.group()
    return text.strip()


# ═══════════════════════════════════════════════════════════
#  Training with TRL GRPO
# ═══════════════════════════════════════════════════════════

def train_grpo(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_steps: int = 100,
    num_prompts: int = 50,
    task_id: str = "task1_easy",
    output_dir: str = "models/grpo_tutor",
    learning_rate: float = 1e-5,
):
    """
    Train a tutoring agent using GRPO from TRL.

    The environment serves as the verifier:
    - LLM generates candidate actions
    - Environment scores them via reward function
    - GRPO updates policy to maximize environment reward

    This demonstrates the core RL loop:
    prompt → LLM generates action → environment verifies → reward → update
    """
    try:
        from trl import GRPOTrainer, GRPOConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        logger.error(
            "TRL not installed. Install with:\n"
            "  pip install trl transformers accelerate peft\n"
            "Or for fast training:\n"
            "  pip install unsloth"
        )
        return _fallback_demonstration(num_prompts, task_id)

    print(f"\n{'='*60}")
    print(f"  EduPath AI — GRPO Training")
    print(f"  Model: {model_name}")
    print(f"  Steps: {num_steps}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}\n")

    # Try Unsloth first for 2x speed
    model = None
    tokenizer = None
    use_unsloth = False

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
        )
        use_unsloth = True
        print("✓ Using Unsloth for 2x faster training")
    except ImportError:
        print("ℹ Unsloth not available, using standard HuggingFace")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Generate training prompts from the environment
    print(f"Generating {num_prompts} training prompts from EduPath env...")
    prompts = build_prompts_from_env(num_prompts, task_id)
    print(f"✓ Generated {len(prompts)} prompts")

    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})

    # Configure GRPO
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=num_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=50,
        max_completion_length=128,
        num_generations=4,  # GRPO generates 4 candidates per prompt
        report_to="none",
    )

    # Train
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
    )

    print(f"\nStarting GRPO training for {num_steps} steps...")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✓ Model saved to {output_dir}")

    # Save training metrics
    log_history = trainer.state.log_history
    os.makedirs("results", exist_ok=True)
    with open("results/grpo_training_log.json", "w") as f:
        json.dump(log_history, f, indent=2, default=str)
    print(f"✓ Training log saved to results/grpo_training_log.json")

    return model, tokenizer


def _fallback_demonstration(num_prompts: int, task_id: str):
    """
    Demonstrate the GRPO reward loop without actual LLM training.
    Shows that the environment can serve as a verifier.
    """
    print("\n" + "=" * 60)
    print("  GRPO Reward Loop Demonstration (no GPU)")
    print("=" * 60)

    prompts = build_prompts_from_env(min(num_prompts, 5), task_id)

    # Simulate LLM generating candidate actions
    candidate_actions = [
        '{"type": "recommend_topic", "topic_id": "python_basics"}',
        '{"type": "assign_quiz", "topic_id": "python_basics"}',
        '{"type": "recommend_resource", "topic_id": "python_basics"}',
        '{"type": "mark_job_ready"}',
        '{"type": "invalid_action"}',
    ]

    print(f"\nScoring {len(candidate_actions)} candidate actions using EduPath verifier:\n")

    rewards = reward_function(candidate_actions)
    for action, reward in zip(candidate_actions, rewards):
        status = "✓" if reward > 0 else "✗"
        print(f"  {status} Action: {action:<55} → Reward: {reward:+.2f}")

    print(f"\n  Best action: {candidate_actions[rewards.index(max(rewards))]}")
    print(f"  Best reward: {max(rewards):+.2f}")
    print(f"\n  This demonstrates the core GRPO loop:")
    print(f"    prompt → LLM generates candidates → env verifies → reward → update")
    print("=" * 60)

    # Save demo results
    os.makedirs("results", exist_ok=True)
    demo_results = {
        "type": "grpo_demo",
        "task_id": task_id,
        "candidates": candidate_actions,
        "rewards": rewards,
        "best_action": candidate_actions[rewards.index(max(rewards))],
        "best_reward": max(rewards),
    }
    with open("results/grpo_demo_results.json", "w") as f:
        json.dump(demo_results, f, indent=2)
    print("Saved: results/grpo_demo_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EduPath AI with TRL GRPO")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--prompts", type=int, default=50,
                        help="Number of training prompts to generate")
    parser.add_argument("--task", type=str, default="task1_easy",
                        help="Task to train on")
    parser.add_argument("--output", type=str, default="models/grpo_tutor",
                        help="Output directory for saved model")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo mode (no GPU needed)")
    args = parser.parse_args()

    if args.demo:
        _fallback_demonstration(args.prompts, args.task)
    else:
        train_grpo(
            model_name=args.model,
            num_steps=args.steps,
            num_prompts=args.prompts,
            task_id=args.task,
            output_dir=args.output,
        )
