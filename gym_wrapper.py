"""
EduPath AI — Gymnasium Wrappers
Team KRIYA | Meta Hackathon 2026

Wraps the EduPath environment as standard Gymnasium envs for
compatibility with Stable-Baselines3. Includes EduPathGymWrapper
(flat observation/action spaces) and GNNGymWrapper (graph-structured
observations for GNN policies).
"""
import os
import sys
import numpy as np
import gymnasium
from gymnasium import spaces

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from environment.env import EduPathEnv
from environment.models import Action, ActionType, QuizDifficulty
from environment.student import student_manager
from environment.curriculum import TOPIC_GRAPH, get_available_topics


# Map integer indices to ActionType enum values
ACTION_TYPE_MAP = {
    0: ActionType.RECOMMEND_TOPIC,
    1: ActionType.ASSIGN_QUIZ,
    2: ActionType.ASSIGN_MINI_PROJECT,
    3: ActionType.ASSIGN_CAPSTONE,
    4: ActionType.RECOMMEND_RESOURCE,
    5: ActionType.SUGGEST_EVENT,
    6: ActionType.MARK_JOB_READY,
}

# Task profiles for training
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


class EduPathGymEnv(gymnasium.Env):
    """
    Gymnasium-compatible wrapper for EduPath AI environment.
    Converts the Pydantic-based environment into numpy arrays for RL training.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_id: str = "task2_medium", seed: int = 42):
        super().__init__()
        self.task_id = task_id
        self.seed_val = seed

        # Action space: 7 discrete action types
        self.action_space = spaces.Discrete(7)

        # Observation space: flattened student state as float32 array
        # Fields encoded (all normalized 0-1):
        # [0] completed_topics_count (/ 32)
        # [1] available_topics_count (/ 32)
        # [2] job_readiness_score (already 0-1)
        # [3] avg_quiz_score (/ 100)
        # [4] completed_projects_count (/ 12)
        # [5] current_skill_avg (already 0-1)
        # [6] badges_earned (/ 30)
        # [7] total_steps (/ 100)
        # [8] weekly_hours (/ 20)
        # [9] has_current_topic (0 or 1)
        # [10] num_quiz_attempts (/ 50)
        # [11] quiz_pass_rate (0-1)
        # [12] steps_remaining_ratio (0-1)
        # [13] topics_since_last_project (/ 10)
        # [14] current_topic_skill (0-1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(15,), dtype=np.float32
        )

        self.env = None
        self._student_id = None
        self._observation = None
        self._all_topic_ids = sorted(TOPIC_GRAPH.keys())

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation."""
        if seed is not None:
            self.seed_val = seed

        self.env = EduPathEnv(seed=self.seed_val)
        profile = TASK_PROFILES[self.task_id]

        # Create student
        student = student_manager.create(name=profile.get("name", "PPO Agent"))
        student_manager.update_from_onboarding(student.id, profile)
        self._student_id = student.id

        obs_pydantic = self.env.reset(student_id=self._student_id, seed=self.seed_val)
        self._observation = obs_pydantic.model_dump()

        return self._encode_observation(self._observation), {}

    def step(self, action_int):
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        action_val = int(np.squeeze(action_int))
        action = self._decode_action(action_val)
        result = self.env.step(action)

        self._observation = result.observation.model_dump()
        reward = result.reward.value
        terminated = result.done and result.reward.is_terminal
        truncated = result.done and not result.reward.is_terminal

        info = result.info
        info["reward_reason"] = result.reward.reason

        obs_array = self._encode_observation(self._observation)
        return obs_array, reward, terminated, truncated, info

    def _encode_observation(self, obs: dict) -> np.ndarray:
        """Flatten StudentObservation to numpy float32 array."""
        completed = obs.get("completed_topics", [])
        available = obs.get("available_topics", [])
        quiz_history = obs.get("quiz_history_summary", {})
        skill_levels = obs.get("skill_levels", {})
        projects = obs.get("completed_projects", [])
        total_steps = obs.get("total_steps", 0)

        # Compute derived features
        avg_quiz = (sum(quiz_history.values()) / max(len(quiz_history), 1)) if quiz_history else 0
        avg_skill = (sum(skill_levels.values()) / max(len(skill_levels), 1)) if skill_levels else 0
        num_quizzes = len(quiz_history)
        quiz_pass_rate = (sum(1 for s in quiz_history.values() if s >= 70) / max(num_quizzes, 1)) if num_quizzes > 0 else 0
        topics_since_project = len(completed) - len(projects) * 3

        current_topic = obs.get("current_topic")
        current_skill = skill_levels.get(current_topic, 0) if current_topic else 0

        features = np.array([
            len(completed) / 32.0,                          # [0]
            len(available) / 32.0,                          # [1]
            obs.get("job_readiness_score", 0),            # [2]
            avg_quiz / 100.0,                               # [3]
            len(projects) / 12.0,                           # [4]
            avg_skill,                                      # [5]
            obs.get("badges_earned", 0) / 30.0,             # [6]
            total_steps / 100.0,                            # [7]
            obs.get("weekly_hours", 10) / 20.0,             # [8]
            1 if current_topic else 0,                  # [9]
            num_quizzes / 50.0,                             # [10]
            quiz_pass_rate,                                 # [11]
            max(0, (100 - total_steps)) / 100.0,            # [12]
            max(0, min(topics_since_project, 10)) / 10.0,   # [13]
            current_skill,                                  # [14]
        ], dtype=np.float32)

        return np.clip(features, 0, 1)

    def _decode_action(self, action_int: int) -> Action:
        """Convert integer action to Action object with heuristic topic/project selection."""
        action_type = ACTION_TYPE_MAP.get(action_int, ActionType.RECOMMEND_TOPIC)
        topic_id = None
        project_id = None

        obs = self._observation or {}
        available = obs.get("available_topics", [])
        completed = obs.get("completed_topics", [])
        current = obs.get("current_topic")
        quiz_history = obs.get("quiz_history_summary", {})

        if action_type == ActionType.RECOMMEND_TOPIC:
            # Pick best available topic based on prerequisites
            if available:
                # Prefer topics with more completed prereqs
                best_topic = available[0]
                best_score = -1
                for tid in available:
                    topic = TOPIC_GRAPH.get(tid)
                    if topic:
                        prereq_complete = sum(1 for p in topic.prerequisites if p in completed)
                        score = prereq_complete + (1 if topic.difficulty <= 2 else 0)
                        if score > best_score:
                            best_score = score
                            best_topic = tid
                topic_id = best_topic

        elif action_type == ActionType.ASSIGN_QUIZ:
            # Quiz the current topic, or last recommended
            topic_id = current
            if not topic_id and completed:
                # Quiz a completed topic that hasn't been quizzed
                for t in reversed(completed):
                    if t not in quiz_history:
                        topic_id = t
                        break
            if not topic_id and available:
                topic_id = available[0]

        elif action_type in (ActionType.ASSIGN_MINI_PROJECT, ActionType.ASSIGN_CAPSTONE):
            topic_id = current or (completed[-1] if completed else None)

        elif action_type == ActionType.RECOMMEND_RESOURCE:
            topic_id = current or (available[0] if available else None)

        return Action(
            type=action_type,
            topic_id=topic_id,
            project_id=project_id,
        )

    def render(self, mode="human"):
        """Optional render for debugging."""
        if self._observation:
            print(f"Step {self._observation.get('total_steps', '?')}: "
                  f"Completed={len(self._observation.get('completed_topics', []))} "
                  f"Available={len(self._observation.get('available_topics', []))} "
                  f"JR={self._observation.get('job_readiness_score', 0):.2f}")


def make_env(task_id: str = "task2_medium", seed: int = 42):
    """Factory function to create the environment."""
    def _init():
        return EduPathGymEnv(task_id=task_id, seed=seed)
    return _init


# ═══════════════════════════════════════════════════════════
#  GNN Gym Wrapper — Graph-based observation + MultiDiscrete action
# ═══════════════════════════════════════════════════════════

class GNNGymWrapper(gymnasium.Env):
    """
    Gymnasium wrapper with GNN-compatible observations.

    Observation: Dict with node_features, edge_index, scalar_features, topic_mask
    Action: MultiDiscrete([7, num_topics]) — action type + topic jointly
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_id: str = "task2_medium", seed: int = 42,
                 use_curiosity: bool = False):
        super().__init__()
        self.task_id = task_id
        self.seed_val = seed
        self.use_curiosity = use_curiosity

        # Import GNN components
        from environment.gnn_policy import (
            ALL_TOPIC_IDS, NUM_TOPICS, STATIC_EDGE_INDEX, TOPIC_TO_IDX,
            build_node_features, build_scalar_features, build_topic_mask
        )
        self._all_topic_ids = ALL_TOPIC_IDS
        self._num_topics = NUM_TOPICS
        self._static_edge_index = STATIC_EDGE_INDEX
        self._topic_to_idx = TOPIC_TO_IDX
        self._build_node_features = build_node_features
        self._build_scalar_features = build_scalar_features
        self._build_topic_mask = build_topic_mask

        # Action space: [action_type (7), topic_index (num_topics)]
        self.action_space = spaces.MultiDiscrete([7, self._num_topics])

        # Observation space: flattened for SB3 compatibility
        # node_features (num_topics * 9) + edge_index_flat (2 * num_edges) 
        # + scalar_features (4) + topic_mask (num_topics)
        # We flatten into a single Box for SB3 MlpPolicy compatibility
        num_edges = self._static_edge_index.shape[1]
        obs_dim = (self._num_topics * 9) + 4 + self._num_topics
        self.observation_space = spaces.Box(
            low=-1, high=2.0, shape=(obs_dim,), dtype=np.float32
        )

        self.env = None
        self._student_id = None
        self._observation = None

        # ICM for curiosity-driven exploration
        self._icm = None
        if self.use_curiosity:
            from environment.icm import IntrinsicCuriosityModule
            self._icm = IntrinsicCuriosityModule()

    def reset(self, seed=None, options=None):
        """Reset environment and return GNN observation."""
        if seed is not None:
            self.seed_val = seed

        self.env = EduPathEnv(seed=self.seed_val)
        profile = TASK_PROFILES[self.task_id]

        student = student_manager.create(name=profile.get("name", "GNN Agent"))
        student_manager.update_from_onboarding(student.id, profile)
        self._student_id = student.id

        obs_pydantic = self.env.reset(student_id=self._student_id, seed=self.seed_val)
        self._observation = obs_pydantic.model_dump()

        if self._icm:
            self._icm.new_episode()

        return self._encode_gnn_observation(self._observation), {}

    def step(self, action):
        """Execute MultiDiscrete action [action_type, topic_idx]."""
        action_type_idx = int(action[0])
        topic_idx = int(action[1])

        # Decode action
        pydantic_action = self._decode_gnn_action(action_type_idx, topic_idx)
        result = self.env.step(pydantic_action)

        self._observation = result.observation.model_dump()
        extrinsic_reward = result.reward.value

        # Add curiosity bonus if enabled
        intrinsic_reward = 0
        if self._icm:
            topic_id = pydantic_action.topic_id or "none"
            intrinsic_reward = self._icm.get_bonus(topic_id, pydantic_action.type.value)

        total_reward = extrinsic_reward + intrinsic_reward

        terminated = result.done and result.reward.is_terminal
        truncated = result.done and not result.reward.is_terminal

        info = result.info
        info["reward_reason"] = result.reward.reason
        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward

        obs_array = self._encode_gnn_observation(self._observation)
        return obs_array, total_reward, terminated, truncated, info

    def _encode_gnn_observation(self, obs: dict) -> np.ndarray:
        """Encode observation as flattened GNN-compatible array."""
        completed = obs.get("completed_topics", [])
        available = obs.get("available_topics", [])
        mastery = obs.get("mastery_probabilities", {})

        # Node features (num_topics * 8)
        node_feat = self._build_node_features(completed, available, mastery)
        node_flat = node_feat.flatten()

        # Scalar features (4)
        scalar_feat = self._build_scalar_features(
            obs.get("job_readiness_score", 0),
            obs.get("badges_earned", 0),
            obs.get("total_steps", 0),
            obs.get("weekly_hours", 10),
        )

        # Topic mask (num_topics)
        topic_mask = self._build_topic_mask(completed, available)

        # Concatenate all
        return np.concatenate([node_flat, scalar_feat, topic_mask]).astype(np.float32)

    def _decode_gnn_action(self, action_type_idx: int, topic_idx: int) -> Action:
        """Decode MultiDiscrete action into Action pydantic model."""
        action_type = ACTION_TYPE_MAP.get(action_type_idx, ActionType.RECOMMEND_TOPIC)

        # Get topic_id from index
        topic_id = None
        if 0 <= topic_idx < len(self._all_topic_ids):
            topic_id = self._all_topic_ids[topic_idx]

        # For actions that don't need a topic
        if action_type in (ActionType.SUGGEST_EVENT, ActionType.MARK_JOB_READY):
            topic_id = None

        return Action(
            type=action_type,
            topic_id=topic_id,
        )

    def render(self, mode="human"):
        if self._observation:
            print(f"Step {self._observation.get('total_steps', '?')}: "
                  f"Completed={len(self._observation.get('completed_topics', []))} "
                  f"Available={len(self._observation.get('available_topics', []))} "
                  f"JR={self._observation.get('job_readiness_score', 0):.2f}")


def make_gnn_env(task_id: str = "task2_medium", seed: int = 42,
                 use_curiosity: bool = False):
    """Factory function for GNN environment."""
    def _init():
        return GNNGymWrapper(task_id=task_id, seed=seed, use_curiosity=use_curiosity)
    return _init


if __name__ == "__main__":
    # Quick test - standard wrapper
    env = EduPathGymEnv(task_id="task1_easy")
    obs, info = env.reset()
    print(f"Standard obs shape: {obs.shape}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}")
        if terminated or truncated:
            break

    print("Standard gym wrapper test passed!")

    # Quick test - GNN wrapper
    try:
        gnn_env = GNNGymWrapper(task_id="task1_easy")
        obs, info = gnn_env.reset()
        print(f"\nGNN obs shape: {obs.shape}")

        for i in range(5):
            action = gnn_env.action_space.sample()
            obs, reward, terminated, truncated, info = gnn_env.step(action)
            print(f"Step {i+1}: action={action}, reward={reward:.2f}")
            if terminated or truncated:
                break

        print("GNN gym wrapper test passed!")
    except ImportError as e:
        print(f"GNN wrapper not available: {e}")

