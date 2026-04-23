"""
EduPath AI — Student Difficulty Model (BKT-Integrated)
Team KRIYA | Meta Hackathon 2026

Models realistic student learning using Bayesian Knowledge Tracing.
Quiz scores are predicted via BKT mastery probabilities combined
with prerequisite ordering bonuses and Gaussian noise. Maintains
both BKT states and legacy skill levels for backward compatibility.
"""
import random
from typing import Dict, List, Optional
from environment.curriculum import TOPIC_GRAPH
from environment.bkt_model import BKTModel


class StudentDifficultyModel:
    """
    Models realistic student learning using Bayesian Knowledge Tracing (BKT).

    Quiz outcomes depend on:
    - BKT mastery probability (replaces heuristic skill level)
    - Completion of prerequisites (order bonus)
    - Topic difficulty (via BKT's P(L0) initialization)
    - Accumulated learning (BKT tracks this automatically)

    The BKT model provides:
    - P(known) per topic — true mastery estimate
    - predict_quiz_score() — probabilistic score prediction
    - Bayesian updates after each quiz/study/project interaction
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self.skill_levels: Dict[str, float] = {}  # topic_id -> 0-1
        self.bkt = BKTModel()  # Bayesian Knowledge Tracing model

    def reset(self, seed: int = 42):
        """Reset the model for a new episode."""
        self._rng = random.Random(seed)
        self.skill_levels = {}
        self.bkt.reset()

    def initialize_from_profile(self, self_assessed_skills: list, resume_skills: list):
        """Initialize skill levels and BKT states from student's onboarding data."""
        # Initialize legacy skill levels (kept for backward compatibility)
        for skill in self_assessed_skills:
            skill_name = skill.skill if hasattr(skill, 'skill') else skill.get('skill', '')
            proficiency = skill.proficiency if hasattr(skill, 'proficiency') else skill.get('proficiency', 0)
            skill_lower = skill_name.lower().replace(' ', '_')
            self.skill_levels[skill_lower] = proficiency
            for topic_id in TOPIC_GRAPH:
                if skill_lower in topic_id or topic_id in skill_lower:
                    self.skill_levels[topic_id] = max(
                        self.skill_levels.get(topic_id, 0), proficiency
                    )

        for rs in resume_skills:
            rs_lower = rs.lower().replace(' ', '_')
            for topic_id in TOPIC_GRAPH:
                if rs_lower in topic_id or topic_id in rs_lower:
                    self.skill_levels[topic_id] = max(
                        self.skill_levels.get(topic_id, 0), 0.2
                    )

        # Initialize BKT model from the same data
        self.bkt.initialize_from_skills(self_assessed_skills, resume_skills)

    def simulate_quiz_score(
        self,
        topic_id: str,
        completed_topics: List[str],
    ) -> int:
        """
        Simulate a realistic quiz score using BKT prediction + prerequisite ordering.

        quiz_score = bkt_predicted_score × skill_multiplier + order_bonus + noise

        BKT provides the base score via P(correct) estimation.
        Prerequisite ordering and skill multiplier still apply as domain bonuses.
        Noise reduced to ±5 since BKT is already probabilistic.
        """
        topic = TOPIC_GRAPH.get(topic_id)
        if not topic:
            return self._rng.randint(40, 70)

        # BKT base score prediction (replaces heuristic formula)
        base_score = self.bkt.predict_quiz_score(topic_id)

        # Skill multiplier: boost from completed prerequisite skills
        prereq_skills = []
        for prereq_id in topic.prerequisites:
            prereq_skills.append(self.bkt.get_p_known(prereq_id))
        avg_prereq_mastery = (
            sum(prereq_skills) / max(len(prereq_skills), 1)
            if prereq_skills else 0
        )
        skill_multiplier = 1 + (avg_prereq_mastery * 0.3)

        # Order bonus: +15 if all prereqs completed in correct order
        order_bonus = 0
        all_prereqs_met = all(p in completed_topics for p in topic.prerequisites)
        if all_prereqs_met and len(topic.prerequisites) > 0:
            order_bonus = 15

        # Calculate raw score
        raw_score = base_score * skill_multiplier + order_bonus

        # Gaussian noise ±5 points (reduced from ±8 since BKT is probabilistic)
        noise = self._rng.gauss(0, 5)
        final_score = raw_score + noise

        # Clamp to [0, 100]
        return int(max(0, min(100, final_score)))

    def update_skill_after_quiz(self, topic_id: str, passed: bool):
        """Update skill levels and BKT after a quiz attempt."""
        # Update BKT model
        self.bkt.update(topic_id, correct=passed)

        # Update legacy skill levels
        if passed:
            current = self.skill_levels.get(topic_id, 0.3)
            self.skill_levels[topic_id] = min(current + 0.3, 1)
        else:
            current = self.skill_levels.get(topic_id, 0.3)
            self.skill_levels[topic_id] = min(current + 0.05, 1)

    def update_skill_after_project(self, related_topic_ids: List[str]):
        """Update skill levels and BKT after completing a mini project."""
        # Update BKT
        self.bkt.update_from_project(related_topic_ids)

        # Update legacy skill levels
        for topic_id in related_topic_ids:
            current = self.skill_levels.get(topic_id, 0.3)
            self.skill_levels[topic_id] = min(current + 0.1, 1)

    def update_skill_after_capstone(self, field: str):
        """Update skill levels and BKT after completing a capstone project."""
        # Update BKT
        self.bkt.update_from_capstone(field)

        # Update legacy skill levels
        for topic_id, topic in TOPIC_GRAPH.items():
            if topic.field == field:
                current = self.skill_levels.get(topic_id, 0)
                self.skill_levels[topic_id] = min(current + 0.2, 1)

    def update_skill_after_topic_study(self, topic_id: str):
        """Update skill and BKT when a topic is recommended/studied."""
        # Update BKT
        self.bkt.update_from_study(topic_id)

        # Update legacy skill levels
        current = self.skill_levels.get(topic_id, 0)
        self.skill_levels[topic_id] = min(current + 0.25, 1)

    def get_skill(self, topic_id: str) -> float:
        """Get current skill level for a topic."""
        return self.skill_levels.get(topic_id, 0.3)

    def get_all_skills(self) -> Dict[str, float]:
        """Get all skill levels."""
        return dict(self.skill_levels)

    def get_mastery_probability(self, topic_id: str) -> float:
        """Get BKT mastery probability P(known) for a topic."""
        return self.bkt.get_p_known(topic_id)

    def get_all_mastery_probabilities(self) -> Dict[str, float]:
        """Get BKT mastery probabilities for all tracked topics."""
        return self.bkt.get_all_mastery_probabilities()
