"""
EduPath AI — Bayesian Knowledge Tracing (BKT) Model
Team KRIYA | Meta Hackathon 2026

Implements the standard BKT Hidden Markov Model for estimating
P(student knows skill) after each interaction. Used by the student
difficulty model to predict quiz outcomes and track mastery.

References:
  Corbett & Anderson (1995). "Knowledge Tracing: Modeling the
  Acquisition of Procedural Knowledge."
  Baker, Corbett, Aleven (2008). "More Accurate Student Modeling
  through Contextual Estimation of Slip and Guess Probabilities."
"""
from typing import Dict, Optional
from environment.curriculum import TOPIC_GRAPH


class BKTModel:
    """
    Bayesian Knowledge Tracing per topic per student.

    BKT Parameters:
      P(L0): initial probability of knowing skill (mapped from topic difficulty)
      P(T):  probability of learning on each attempt (transit)
      P(G):  probability of correct answer when NOT knowing (guess)
      P(S):  probability of wrong answer when knowing (slip)
    """

    # Default BKT parameters (tuned for educational simulation)
    # P(T) 0.30: strong learning from each interaction (study/quiz)
    # P(G) 0.25: beginners can guess correctly ~25% of the time
    # P(S) 0.10: mastered topics rarely produce wrong answers
    DEFAULT_P_T = 0.30   # Transit probability
    DEFAULT_P_G = 0.25   # Guess probability
    DEFAULT_P_S = 0.10   # Slip probability

    def __init__(self):
        self.knowledge_states: Dict[str, float] = {}  # topic_id -> P(known)

    def reset(self):
        """Reset all knowledge states for a new episode."""
        self.knowledge_states = {}

    def _get_p_l0(self, topic_id: str) -> float:
        """
        Get initial probability of knowing a skill based on topic difficulty.
        Maps difficulty 1-5 to P(L0) range 0.50-0.15 (easier = more likely known).
        """
        topic = TOPIC_GRAPH.get(topic_id)
        if not topic:
            return 0.20  # Default for unknown topics

        # Difficulty 1 -> 0.50, Difficulty 5 -> 0.15
        difficulty = getattr(topic, 'difficulty', 3)
        p_l0 = max(0.15, 0.50 - (difficulty - 1) * 0.0875)
        return p_l0

    def get_p_known(self, topic_id: str) -> float:
        """Get current probability that the student knows this topic."""
        if topic_id in self.knowledge_states:
            return self.knowledge_states[topic_id]
        return self._get_p_l0(topic_id)

    def update(
        self,
        topic_id: str,
        correct: bool,
        p_t: Optional[float] = None,
        p_g: Optional[float] = None,
        p_s: Optional[float] = None,
    ) -> float:
        """
        Bayesian update after observing student response (correct/incorrect).

        Uses Bayes' theorem:
          P(known | observation) = P(obs | known) * P(known) / P(obs)

        Then applies learning transition:
          P(known_next) = P(known | obs) + (1 - P(known | obs)) * P(T)

        Returns updated P(known) for the topic.
        """
        p_t = p_t or self.DEFAULT_P_T
        p_g = p_g or self.DEFAULT_P_G
        p_s = p_s or self.DEFAULT_P_S

        p_known = self.get_p_known(topic_id)

        if correct:
            # P(correct | known) = 1 - P(S)
            # P(correct | unknown) = P(G)
            p_obs_given_known = 1 - p_s
            p_obs_given_unknown = p_g
        else:
            # P(incorrect | known) = P(S)
            # P(incorrect | unknown) = 1 - P(G)
            p_obs_given_known = p_s
            p_obs_given_unknown = 1 - p_g

        # Total probability of observation
        p_obs = (p_obs_given_known * p_known +
                 p_obs_given_unknown * (1 - p_known))

        # Bayesian posterior: P(known | observation)
        if p_obs > 0:
            p_known_given_obs = (p_obs_given_known * p_known) / p_obs
        else:
            p_known_given_obs = p_known

        # Apply learning transition
        p_known_next = p_known_given_obs + (1 - p_known_given_obs) * p_t

        # Clamp to [0.01, 0.99] to avoid degenerate states
        self.knowledge_states[topic_id] = max(0.01, min(p_known_next, 0.99))
        return self.knowledge_states[topic_id]

    def update_from_study(self, topic_id: str) -> float:
        """
        Update knowledge after studying a topic (recommend_topic action).
        Studying gives a strong boost — the student is actively learning.
        """
        p_known = self.get_p_known(topic_id)
        # Studying boosts knowledge by full transit probability
        p_known_next = p_known + (1 - p_known) * self.DEFAULT_P_T
        self.knowledge_states[topic_id] = min(p_known_next, 0.99)
        return self.knowledge_states[topic_id]

    def update_from_project(self, topic_ids: list) -> None:
        """
        Update knowledge for topics related to a completed project.
        Projects provide strong evidence of learning.
        """
        for topic_id in topic_ids:
            p_known = self.get_p_known(topic_id)
            # Project completion is strong evidence of mastery
            p_known_next = p_known + (1 - p_known) * 0.30
            self.knowledge_states[topic_id] = min(p_known_next, 0.99)

    def update_from_capstone(self, field: str) -> None:
        """
        Update knowledge for all topics in a field after capstone completion.
        Capstone is strong evidence of broad field mastery.
        """
        for topic_id, topic in TOPIC_GRAPH.items():
            if topic.field == field:
                p_known = self.get_p_known(topic_id)
                p_known_next = p_known + (1 - p_known) * 0.25
                self.knowledge_states[topic_id] = min(p_known_next, 0.99)

    def predict_quiz_score(self, topic_id: str) -> float:
        """
        Predict quiz score as probability of correct response × 100.

        Uses the BKT observation model:
          P(correct) = P(known) × (1 - P(S)) + (1 - P(known)) × P(G)

        Topics that have been studied (i.e. have a tracked state above
        their default P(L0)) get a +10 study bonus.
        """
        p_known = self.get_p_known(topic_id)
        p_correct = (p_known * (1 - self.DEFAULT_P_S) +
                     (1 - p_known) * self.DEFAULT_P_G)
        base_score = p_correct * 100

        # Study bonus: if the student has studied this topic (P(known) > P(L0)),
        # they get a bonus from having engaged with the material
        p_l0 = self._get_p_l0(topic_id)
        if p_known > p_l0 + 0.05:
            base_score += 10

        return min(base_score, 99)

    def get_all_mastery_probabilities(self) -> Dict[str, float]:
        """Get mastery probabilities for all tracked topics."""
        result = dict(self.knowledge_states)
        # Also include defaults for topics in the graph not yet tracked
        for topic_id in TOPIC_GRAPH:
            if topic_id not in result:
                result[topic_id] = self._get_p_l0(topic_id)
        return result

    def initialize_from_skills(
        self,
        self_assessed_skills: list,
        resume_skills: list,
    ) -> None:
        """
        Initialize BKT knowledge states from student's onboarding data.
        Maps self-assessed proficiency to initial P(known).
        """
        for skill in self_assessed_skills:
            skill_name = skill.skill if hasattr(skill, 'skill') else skill.get('skill', '')
            proficiency = skill.proficiency if hasattr(skill, 'proficiency') else skill.get('proficiency', 0)
            skill_lower = skill_name.lower().replace(' ', '_')

            # Map proficiency directly to knowledge state
            for topic_id in TOPIC_GRAPH:
                if skill_lower in topic_id or topic_id in skill_lower:
                    self.knowledge_states[topic_id] = max(
                        self.knowledge_states.get(topic_id, 0),
                        min(proficiency, 0.99)
                    )

        # Resume skills give a moderate base
        for rs in resume_skills:
            rs_lower = rs.lower().replace(' ', '_')
            for topic_id in TOPIC_GRAPH:
                if rs_lower in topic_id or topic_id in rs_lower:
                    self.knowledge_states[topic_id] = max(
                        self.knowledge_states.get(topic_id, 0),
                        0.25
                    )
