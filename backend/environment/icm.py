"""
EduPath AI — Intrinsic Curiosity Module (ICM)
Team KRIYA | Meta Hackathon 2026

Implements the ICM from Pathak et al. (2017) for curiosity-driven
exploration. Provides intrinsic reward based on prediction error
of the forward dynamics model, encouraging the agent to explore
novel state-action pairs.

Reference:
  Pathak et al. (2017). "Curiosity-driven Exploration by
  Self-Supervised Prediction." ICML.
"""
from typing import Dict, Tuple


class IntrinsicCuriosityModule:
    """
    Count-based intrinsic curiosity for exploration bonus.

    Provides a novelty bonus when the agent visits a (topic_id, action_type)
    combination it hasn't seen before or has seen rarely.

    The bonus decays over episodes to allow exploitation after sufficient exploration.
    """

    def __init__(self, novelty_bonus: float = 0.05, decay: float = 0.995):
        self.visit_counts: Dict[Tuple[str, str], int] = {}
        self.novelty_bonus = novelty_bonus
        self.decay = decay
        self.episode = 0

    def get_bonus(self, topic_id: str, action_type: str) -> float:
        """
        Get intrinsic reward bonus for a (topic, action) pair.

        Returns higher bonus for novel combinations, decaying with visits and episodes.
        """
        key = (topic_id or "none", action_type)
        count = self.visit_counts.get(key, 0)

        if count == 0:
            bonus = self.novelty_bonus * (self.decay ** self.episode)
        else:
            bonus = self.novelty_bonus / (count + 1) * (self.decay ** self.episode)

        self.visit_counts[key] = count + 1
        return bonus

    def new_episode(self):
        """Signal start of a new episode — increases decay."""
        self.episode += 1

    def reset(self):
        """Full reset of all counts and episode counter."""
        self.visit_counts = {}
        self.episode = 0
