"""
EduPath AI — Reflexion Agent
Team KRIYA | Meta Hackathon 2026

Implements Verbal Reinforcement Learning (Reflexion) for tutoring
policy improvement across episodes. The agent reflects on episode
trajectories, stores up to K=5 reflections, and conditions future
decisions on past mistakes.

Reference:
  Shinn et al. (2023). "Reflexion: Language Agents with Verbal
  Reinforcement Learning." NeurIPS.
"""
import os
import sys
import json
import logging
import random
from typing import List, Dict, Optional, Tuple

# Add backend to path for direct mode
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))

logger = logging.getLogger(__name__)

# ── Reflexion System Prompts ──

REFLECTION_PROMPT = """You are analyzing a tutoring agent's episode trajectory.
The agent took a series of actions to teach a student. Look at the results and reflect on:

1. What went well? (high reward actions)
2. What went poorly? (negative rewards, repeated failures)
3. What strategy mistakes were made?
4. What should the agent do differently next episode?

Be specific and actionable. Focus on the MOST impactful changes.

Episode Summary:
- Total steps: {total_steps}
- Total reward: {total_reward:.2f}
- Final score: {score:.4f}
- Topics completed: {topics_completed}
- Topics attempted but failed: {topics_failed}
- Projects completed: {projects_completed}

Key moments (high/low reward actions):
{key_moments}

Previous reflections (if any):
{previous_reflections}

Write a concise reflection (3-5 sentences) about what to improve:"""

REFLEXION_ACTOR_PROMPT = """You are an expert AI tutoring agent using the Reflexion framework.
You learn from past mistakes by reflecting on previous episode trajectories.

PAST REFLECTIONS (most recent first):
{reflections}

CURRENT STUDENT STATE:
{observation}

WORKING MEMORY:
{scratchpad}

Based on your past reflections and current state, decide the best action.
Apply lessons learned from failures. Avoid repeating past mistakes.

Return ONLY valid JSON:
{{
  "thought": "Your reasoning incorporating past reflections",
  "action": {{
    "type": "recommend_topic|assign_quiz|assign_mini_project|assign_capstone|recommend_resource|mark_job_ready",
    "topic_id": "optional topic id"
  }}
}}"""


class EpisodeTrajectory:
    """Records a full episode trajectory for reflection."""

    def __init__(self):
        self.steps: List[Dict] = []
        self.total_reward = 0
        self.topics_completed: List[str] = []
        self.topics_failed: List[str] = []
        self.projects_completed: int = 0

    def record(self, step: int, observation: dict, action: dict,
               reward: float, done: bool):
        """Record a step in the trajectory."""
        self.steps.append({
            "step": step,
            "action_type": action.get("type", "unknown"),
            "topic_id": action.get("topic_id"),
            "reward": reward,
            "done": done,
            "completed_topics": len(observation.get("completed_topics", [])),
            "job_readiness": observation.get("job_readiness_score", 0),
        })
        self.total_reward += reward

    def get_key_moments(self, k: int = 10) -> str:
        """Get the most impactful moments (highest and lowest rewards)."""
        if not self.steps:
            return "No steps recorded."

        sorted_steps = sorted(self.steps, key=lambda s: s["reward"])
        worst = sorted_steps[:min(k // 2, len(sorted_steps))]
        best = sorted_steps[-min(k // 2, len(sorted_steps)):]

        moments = []
        for s in worst:
            moments.append(
                f"  Step {s['step']}: {s['action_type']}('{s.get('topic_id', '')}') "
                f"→ reward={s['reward']:.2f} [POOR]"
            )
        for s in best:
            moments.append(
                f"  Step {s['step']}: {s['action_type']}('{s.get('topic_id', '')}') "
                f"→ reward={s['reward']:.2f} [GOOD]"
            )
        return "\n".join(moments)

    def get_summary(self) -> Dict:
        """Get episode summary dict."""
        return {
            "total_steps": len(self.steps),
            "total_reward": self.total_reward,
            "score": self.steps[-1]["job_readiness"] if self.steps else 0,
            "topics_completed": self.topics_completed,
            "topics_failed": self.topics_failed,
            "projects_completed": self.projects_completed,
        }


class ReflexionMemory:
    """Stores reflections across episodes (max K)."""

    def __init__(self, max_reflections: int = 5):
        self.reflections: List[Dict] = []
        self.max_reflections = max_reflections

    def add_reflection(self, episode: int, reflection: str, score: float):
        """Add a new reflection."""
        self.reflections.append({
            "episode": episode,
            "reflection": reflection,
            "score": score,
        })
        # Keep only the most recent K reflections
        if len(self.reflections) > self.max_reflections:
            self.reflections = self.reflections[-self.max_reflections:]

    def get_reflections_text(self) -> str:
        """Format reflections for LLM prompt."""
        if not self.reflections:
            return "No previous reflections yet."

        lines = []
        for r in reversed(self.reflections):  # Most recent first
            lines.append(
                f"Episode {r['episode']} (score={r['score']:.3f}): {r['reflection']}"
            )
        return "\n".join(lines)

    def save(self, path: str):
        """Save reflections to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.reflections, f, indent=2)

    def load(self, path: str):
        """Load reflections from JSON file."""
        if os.path.exists(path):
            with open(path, "r") as f:
                self.reflections = json.load(f)


class ReflexionAgent:
    """
    Reflexion agent: reflects on trajectories to improve tutoring strategy.

    Flow:
    1. Act for one episode using ReAct + past reflections
    2. After episode ends, reflect on the trajectory
    3. Store reflection in memory
    4. Repeat — agent improves over episodes

    Falls back to enhanced rule-based decisions when no LLM is available.
    """

    def __init__(self, max_reflections: int = 5):
        self.memory = ReflexionMemory(max_reflections=max_reflections)
        self.trajectory = EpisodeTrajectory()
        self.episode_count = 0
        self._step_count = 0
        self._client = None
        self._model_name = None

        # Scratchpad for within-episode memory
        self.topic_attempts: Dict[str, List[float]] = {}
        self.topics_recommended: List[str] = []
        self.topics_since_project = 0

        # Try to get LLM client
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client if available."""
        try:
            from openai import OpenAI
            api_base = os.getenv("API_BASE_URL", "")
            api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_base and api_key:
                self._client = OpenAI(base_url=api_base, api_key=api_key)
                self._model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        except Exception:
            pass

    def new_episode(self):
        """Start a new episode."""
        self.episode_count += 1
        self.trajectory = EpisodeTrajectory()
        self._step_count = 0
        self.topic_attempts = {}
        self.topics_recommended = []
        self.topics_since_project = 0

    def decide(self, observation: dict) -> dict:
        """Make a tutoring decision using Reflexion + ReAct."""
        self._step_count += 1

        # Try LLM-based decision with reflections
        if self._client:
            result = self._llm_decision(observation)
            if result:
                return result

        # Fallback: enhanced rule-based with reflection conditioning
        return self._rule_decision_with_reflections(observation)

    def record_step(self, action: dict, reward: float, observation: dict, done: bool):
        """Record a step in the current trajectory."""
        self.trajectory.record(self._step_count, observation, action, reward, done)

        # Track topic-level data
        action_type = action.get("type", "")
        topic_id = action.get("topic_id")

        if action_type == "recommend_topic" and topic_id:
            if topic_id not in self.topics_recommended:
                self.topics_recommended.append(topic_id)
                self.topics_since_project += 1

        if action_type == "assign_quiz" and topic_id:
            # Track quiz scores
            quiz_summary = observation.get("quiz_history_summary", {})
            score = quiz_summary.get(topic_id, 0)
            if topic_id not in self.topic_attempts:
                self.topic_attempts[topic_id] = []
            self.topic_attempts[topic_id].append(score)

            if score >= 70:
                if topic_id not in self.trajectory.topics_completed:
                    self.trajectory.topics_completed.append(topic_id)
            else:
                if topic_id not in self.trajectory.topics_failed:
                    self.trajectory.topics_failed.append(topic_id)

        if action_type in ("assign_mini_project", "assign_capstone"):
            self.trajectory.projects_completed += 1
            self.topics_since_project = 0

    def reflect(self, final_score: float = 0):
        """Generate a reflection on the completed episode."""
        summary = self.trajectory.get_summary()
        summary["score"] = final_score

        if self._client:
            reflection = self._llm_reflect(summary)
        else:
            reflection = self._rule_reflect(summary)

        self.memory.add_reflection(self.episode_count, reflection, final_score)
        return reflection

    def _llm_decision(self, observation: dict) -> Optional[dict]:
        """Use LLM with reflexion memory for decision-making."""
        reflections_text = self.memory.get_reflections_text()

        scratchpad = (
            f"Step: {self._step_count}\n"
            f"Topics recommended: {len(self.topics_recommended)}\n"
            f"Topics since project: {self.topics_since_project}\n"
            f"Quiz attempts: {json.dumps({t: s for t, s in list(self.topic_attempts.items())[-5:]})}"
        )

        prompt = REFLEXION_ACTOR_PROMPT.format(
            reflections=reflections_text,
            observation=json.dumps(observation, indent=2),
            scratchpad=scratchpad,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            action = result.get("action", {})
            return {
                "type": action.get("type", "recommend_topic"),
                "topic_id": action.get("topic_id"),
                "project_id": action.get("project_id"),
                "_thought": result.get("thought", ""),
            }
        except Exception as e:
            logger.warning(f"Reflexion LLM decision failed: {e}")
            return None

    def _llm_reflect(self, summary: dict) -> str:
        """Use LLM to generate a reflection."""
        key_moments = self.trajectory.get_key_moments()
        previous = self.memory.get_reflections_text()

        prompt = REFLECTION_PROMPT.format(
            total_steps=summary["total_steps"],
            total_reward=summary["total_reward"],
            score=summary["score"],
            topics_completed=", ".join(summary.get("topics_completed", [])) or "none",
            topics_failed=", ".join(summary.get("topics_failed", [])) or "none",
            projects_completed=summary.get("projects_completed", 0),
            key_moments=key_moments,
            previous_reflections=previous,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Reflexion LLM reflect failed: {e}")
            return self._rule_reflect(summary)

    def _rule_reflect(self, summary: dict) -> str:
        """Generate a rule-based reflection (fallback)."""
        parts = []
        if summary["total_reward"] < 0:
            parts.append("Episode had net negative reward — too many redundant or invalid actions.")
        if len(summary.get("topics_failed", [])) > 2:
            failed = ", ".join(summary["topics_failed"][:3])
            parts.append(f"Multiple topic failures ({failed}). Should assign resources before retrying quizzes.")
        if summary.get("projects_completed", 0) == 0:
            parts.append("No projects completed. Should assign mini-projects after every 3 topics for practice.")
        if summary["score"] < 0.5:
            parts.append("Low final score. Need to focus on covering expected topics rather than repeating actions.")
        if not parts:
            parts.append("Episode went reasonably well. Continue optimizing topic sequencing and quiz timing.")
        return " ".join(parts)

    def _rule_decision_with_reflections(self, observation: dict) -> dict:
        """Enhanced rule-based agent conditioned on reflections."""
        available = observation.get("available_topics", [])
        completed = observation.get("completed_topics", [])
        current = observation.get("current_topic")
        quiz_history = observation.get("quiz_history_summary", {})
        job_readiness = observation.get("job_readiness_score", 0)

        # Apply lessons from reflections
        avoid_redundant = any("redundant" in r.get("reflection", "").lower()
                             for r in self.memory.reflections)
        use_resources = any("resource" in r.get("reflection", "").lower()
                           for r in self.memory.reflections)
        do_projects = any("project" in r.get("reflection", "").lower()
                         for r in self.memory.reflections)

        # 1. No current topic → recommend first available (top priority)
        if not current and available:
            return {"type": "recommend_topic", "topic_id": available[0]}

        # 2. If current topic not yet quizzed → assign quiz
        if current and current not in quiz_history:
            return {"type": "assign_quiz", "topic_id": current}

        # 3. Check if we need resources first (from reflections)
        if use_resources and current:
            attempts = self.topic_attempts.get(current, [])
            if len(attempts) >= 1 and all(s < 50 for s in attempts[-2:]):
                return {"type": "recommend_resource", "topic_id": current}

        # 4. Assign projects more aggressively if reflections suggest it
        project_threshold = 2 if do_projects else 3
        if self.topics_since_project >= project_threshold and current:
            return {"type": "assign_mini_project", "topic_id": current}

        # 5. If current topic passed → recommend next
        if current and quiz_history.get(current, 0) >= 70 and available:
            for topic_id in available:
                if topic_id not in self.trajectory.topics_failed or not avoid_redundant:
                    return {"type": "recommend_topic", "topic_id": topic_id}
            return {"type": "recommend_topic", "topic_id": available[0]}

        # 6. If current topic failed → retry quiz
        if current and current in quiz_history and quiz_history.get(current, 0) < 70:
            return {"type": "assign_quiz", "topic_id": current}

        # 7. Still have available topics → recommend next
        if available:
            return {"type": "recommend_topic", "topic_id": available[0]}

        # 8. Job ready check
        if job_readiness >= 0.8:
            return {"type": "mark_job_ready"}

        # 9. Capstone
        if not available and len(completed) > 0:
            return {"type": "assign_capstone", "topic_id": completed[-1] if completed else "python_basics"}

        return {"type": "recommend_resource", "topic_id": current or "python_basics"}

