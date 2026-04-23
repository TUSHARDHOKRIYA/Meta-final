"""
EduPath AI — Hierarchical RL Environment Wrapper
Team KRIYA | Meta Hackathon 2026

Two-level hierarchy: a meta-controller selects high-level goals
(e.g., "complete topic cluster", "boost quiz score"), and a
sub-controller executes low-level actions within each goal.
Built on top of the base EduPathEnv.
"""
import os
import sys
import json
import random
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import gymnasium
from gymnasium import spaces
from environment.env import EduPathEnv
from environment.models import Action, ActionType
from environment.student import student_manager
from environment.curriculum import TOPIC_GRAPH

logger = logging.getLogger(__name__)

# Teaching Strategies
STRATEGIES = {
    0: "prerequisite_fill",  # Focus on completing prerequisite topics
    1: "quiz_consolidate",   # Quiz to consolidate learned knowledge
    2: "project_apply",      # Apply knowledge through projects
    3: "resource_reinforce", # Reinforce weak topics with resources
    4: "exploration",        # Try new/untested topics
    5: "job_ready_push",     # Push toward job readiness (capstone, mark ready)
}

STRATEGY_NAMES = list(STRATEGIES.values())

# Import profiles from gym_wrapper
from gym_wrapper import TASK_PROFILES, ACTION_TYPE_MAP


class HierarchicalEduPathEnv(gymnasium.Env):
    """
    Hierarchical wrapper: Manager selects strategy every K steps,
    Worker executes concrete actions conditioned on the strategy.

    This wrapper exposes a FLAT interface for SB3 PPO:
      - Observation: [student_state (15) + strategy_onehot (6)] = 21 dims
      - Action: Discrete(42) = 7 action_types × 6 strategies (flattened)

    The manager/worker split is encoded in the observation/reward shaping.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_id: str = "task2_medium", seed: int = 42,
                 manager_interval: int = 5):
        super().__init__()
        self.task_id = task_id
        self.seed_val = seed
        self.manager_interval = manager_interval

        # Internal state
        self.env = None
        self._student_id = None
        self._observation = None
        self._current_strategy = 0  # Default: prerequisite_fill
        self._strategy_step_count = 0
        self._total_steps = 0

        # Topic lists for action decoding
        self._all_topic_ids = sorted(TOPIC_GRAPH.keys())
        self._num_topics = len(self._all_topic_ids)

        # Observation: 15 (student) + 6 (strategy onehot) = 21
        self.observation_space = spaces.Box(
            low=-1, high=2.0, shape=(21,), dtype=np.float32
        )

        # Action: 7 action types × 6 strategies = 42
        # Bits: [action_type (0-6)] × [strategy context (0-5)]
        # We use MultiDiscrete for cleaner decomposition
        self.action_space = spaces.MultiDiscrete([7, 6])

    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            self.seed_val = seed

        self.env = EduPathEnv(seed=self.seed_val)
        profile = TASK_PROFILES[self.task_id]

        student = student_manager.create(name=profile.get("name", "HRL Agent"))
        student_manager.update_from_onboarding(student.id, profile)
        self._student_id = student.id

        obs = self.env.reset(student_id=self._student_id, seed=self.seed_val)
        self._observation = obs.model_dump()
        self._current_strategy = 0
        self._strategy_step_count = 0
        self._total_steps = 0

        return self._encode_observation(self._observation), {}

    def step(self, action):
        """Execute action with hierarchical reward shaping."""
        action_type_idx = int(action[0])
        strategy_idx = int(action[1])

        # Manager: update strategy every K steps
        if self._strategy_step_count >= self.manager_interval:
            self._current_strategy = strategy_idx
            self._strategy_step_count = 0

        self._strategy_step_count += 1
        self._total_steps += 1

        # Worker: decode action type into environment action
        pydantic_action = self._decode_action(action_type_idx)
        result = self.env.step(pydantic_action)

        self._observation = result.observation.model_dump()
        extrinsic_reward = result.reward.value

        # Strategy alignment bonus
        alignment_bonus = self._compute_strategy_alignment(
            action_type_idx, strategy_idx, self._observation, extrinsic_reward
        )

        total_reward = extrinsic_reward + alignment_bonus

        terminated = result.done and result.reward.is_terminal
        truncated = result.done and not result.reward.is_terminal

        info = result.info
        info["strategy"] = STRATEGIES[self._current_strategy]
        info["alignment_bonus"] = alignment_bonus
        info["extrinsic_reward"] = extrinsic_reward

        return self._encode_observation(self._observation), total_reward, terminated, truncated, info

    def _encode_observation(self, obs: dict) -> np.ndarray:
        """Encode observation as 21-dim vector (15 student + 6 strategy onehot)."""
        completed = obs.get("completed_topics", [])
        available = obs.get("available_topics", [])
        quiz_history = obs.get("quiz_history_summary", {})
        mastery = obs.get("mastery_probabilities", {})

        student_features = np.array([
            len(completed) / 30.0,                        # completion progress
            len(available) / 30.0,                        # available ratio
            obs.get("job_readiness_score", 0),          # job readiness
            len(quiz_history) / 30.0,                     # quiz coverage
            obs.get("total_steps", 0) / 100.0,            # step progress
            obs.get("badges_earned", 0) / 10.0,           # badges
            obs.get("weekly_hours", 10) / 40.0,           # study hours
            len(obs.get("completed_projects", [])) / 5.0, # projects
            sum(quiz_history.values()) / max(len(quiz_history) * 100, 1),  # avg quiz score
            np.mean(list(mastery.values())) if mastery else 0.1,           # avg mastery
            min(len(completed) / 5.0, 1),               # early progress
            self._total_steps / 100.0,                    # time pressure
            1 if obs.get("current_topic") in completed else 0,  # current topic done
            1 if len(available) == 0 else 0,          # no topics left
            self._strategy_step_count / self.manager_interval,  # strategy progress
        ], dtype=np.float32)

        # Strategy one-hot (6 dims)
        strategy_onehot = np.zeros(6, dtype=np.float32)
        strategy_onehot[self._current_strategy] = 1

        return np.concatenate([student_features, strategy_onehot])

    def _decode_action(self, action_type_idx: int) -> Action:
        """Decode action type index into Action."""
        action_type = ACTION_TYPE_MAP.get(action_type_idx, ActionType.RECOMMEND_TOPIC)
        obs = self._observation

        available = obs.get("available_topics", [])
        completed = obs.get("completed_topics", [])
        current = obs.get("current_topic")

        topic_id = None
        if action_type in (ActionType.RECOMMEND_TOPIC, ActionType.RECOMMEND_RESOURCE):
            topic_id = available[0] if available else (current or "python_basics")
        elif action_type in (ActionType.ASSIGN_QUIZ,):
            topic_id = current or (available[0] if available else "python_basics")
        elif action_type in (ActionType.ASSIGN_MINI_PROJECT, ActionType.ASSIGN_CAPSTONE):
            topic_id = current or (completed[-1] if completed else "python_basics")

        return Action(type=action_type, topic_id=topic_id)

    def _compute_strategy_alignment(self, action_type_idx: int, strategy_idx: int,
                                     obs: dict, extrinsic_reward: float) -> float:
        """Compute bonus for actions aligned with the current strategy."""
        strategy = STRATEGIES.get(strategy_idx, "exploration")
        action_type_name = list(ACTION_TYPE_MAP.values())[action_type_idx].value if action_type_idx < 7 else ""

        # Strategy-action alignment rules
        alignment = {
            "prerequisite_fill": {"recommend_topic": 0.03, "recommend_resource": 0.01},
            "quiz_consolidate": {"assign_quiz": 0.03},
            "project_apply": {"assign_mini_project": 0.04, "assign_capstone": 0.05},
            "resource_reinforce": {"recommend_resource": 0.03},
            "exploration": {"recommend_topic": 0.02, "assign_quiz": 0.01},
            "job_ready_push": {"mark_job_ready": 0.05, "assign_capstone": 0.03},
        }

        return alignment.get(strategy, {}).get(action_type_name, -0.01)

    def render(self, mode="human"):
        """Render current state."""
        if self._observation:
            print(f"[{STRATEGIES[self._current_strategy]}] "
                  f"Step {self._total_steps}: "
                  f"Completed={len(self._observation.get('completed_topics', []))} "
                  f"JR={self._observation.get('job_readiness_score', 0):.2f}")


def make_hrl_env(task_id: str = "task2_medium", seed: int = 42):
    """Factory function for HRL environment."""
    def _init():
        return HierarchicalEduPathEnv(task_id=task_id, seed=seed)
    return _init
