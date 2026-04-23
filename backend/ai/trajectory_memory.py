"""
EduPath AI — Trajectory Memory
Team KRIYA | OpenEnv Hackathon 2026

Shared state store for inter-agent coordination. Every agent reads from
and writes to this memory. It is the source of truth for the student's
learning journey, council proposals, quiz attempts, and intervention flags.
"""
import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trajectory")
os.makedirs(DATA_DIR, exist_ok=True)


class TrajectoryMemory:
    """Per-student state store across all agents in the pipeline."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.created_at = datetime.now().isoformat()

        # Stage 0: Profiling
        self.student_profile: Dict = {}
        self.profiling_conversation: List[Dict] = []

        # Stage 1: Council proposals
        self.domain_expert_proposal: Dict = {}
        self.prereq_architect_proposal: Dict = {}
        self.feasibility_proposal: Dict = {}
        self.student_advocate_proposal: Dict = {}
        self.conflict_resolution: Dict = {}
        self.final_roadmap: Dict = {}

        # Stage 2: Learning loop
        self.section_history: List[Dict] = []  # [{topic_id, course, quiz_attempts, flags, ...}]
        self.current_topic_materials: Dict = {}  # Curator output for current topic
        self.quiz_attempts: Dict[str, List[Dict]] = {}  # topic_id -> [quiz_results]

        # Stage 3: Adaptation
        self.flags: List[Dict] = []  # [{topic_id, flag_type, timestamp}]
        self.intervention_log: List[Dict] = []  # [{level, action, timestamp}]
        self.bridging_topics_inserted: List[str] = []

        # Multi-round debate state
        self.second_round_responses: Dict = {}  # Agent name -> R2 response
        self.observer_report: Dict = {}  # Observer Agent oversight report

        # Context overflow
        self._compressed: bool = False
        self._original_chars: int = 0

    def record_profile(self, profile: Dict):
        """Record the student profile from the Profiling Agent."""
        self.student_profile = profile
        self._save()

    def record_council_proposal(self, agent_name: str, proposal: Dict):
        """Record a council agent's proposal."""
        attr_map = {
            "domain_expert": "domain_expert_proposal",
            "prereq_architect": "prereq_architect_proposal",
            "feasibility": "feasibility_proposal",
            "student_advocate": "student_advocate_proposal",
            "conflict_matcher": "conflict_resolution",
            "council_manager": "final_roadmap",
        }
        attr = attr_map.get(agent_name)
        if attr:
            setattr(self, attr, proposal)
            self._save()

    def record_section(self, topic_id: str, course_info: Dict = None,
                       quiz_result: Dict = None, flag: str = None):
        """Record the completion of a learning section."""
        entry = {
            "topic_id": topic_id,
            "timestamp": datetime.now().isoformat(),
            "course": course_info,
            "quiz_result": quiz_result,
            "flag": flag or "CLEAR",
        }
        self.section_history.append(entry)

        if flag and flag != "CLEAR":
            self.flags.append({
                "topic_id": topic_id,
                "flag_type": flag,
                "timestamp": datetime.now().isoformat(),
            })

        self._save()

    def record_quiz_attempt(self, topic_id: str, result: Dict):
        """Record a quiz attempt for a topic."""
        if topic_id not in self.quiz_attempts:
            self.quiz_attempts[topic_id] = []
        self.quiz_attempts[topic_id].append({
            **result,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def record_intervention(self, level: int, action_type: str, details: Dict = None):
        """Record an intervention decision."""
        self.intervention_log.append({
            "level": level,
            "action_type": action_type,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def get_recent_flags(self, n: int = 3) -> List[Dict]:
        """Get flags from the last n sections."""
        recent = self.section_history[-n:] if self.section_history else []
        return [s for s in recent if s.get("flag", "CLEAR") != "CLEAR"]

    def get_flag_count(self, n: int = 3) -> int:
        """Count warning flags in the last n sections."""
        return len(self.get_recent_flags(n))

    def get_quiz_attempt_count(self, topic_id: str) -> int:
        """Get the number of quiz attempts for a topic."""
        return len(self.quiz_attempts.get(topic_id, []))

    def get_latest_quiz_score(self, topic_id: str) -> Optional[float]:
        """Get the most recent quiz score for a topic."""
        attempts = self.quiz_attempts.get(topic_id, [])
        if attempts:
            return attempts[-1].get("score", 0)
        return None

    def to_context_string(self, max_sections: int = 5) -> str:
        """Convert trajectory to a context string for agent prompts."""
        parts = []

        if self.student_profile:
            parts.append(f"STUDENT PROFILE: {json.dumps(self.student_profile)}")

        if self.final_roadmap:
            parts.append(f"CURRENT ROADMAP: {json.dumps(self.final_roadmap)}")

        recent = self.section_history[-max_sections:] if self.section_history else []
        if recent:
            parts.append(f"RECENT SECTIONS ({len(recent)}):")
            for s in recent:
                flag_str = f" [FLAG: {s['flag']}]" if s.get("flag", "CLEAR") != "CLEAR" else ""
                parts.append(f"  - {s['topic_id']}{flag_str}")

        if self.flags:
            parts.append(f"WARNING FLAGS ({len(self.flags)}):")
            for f in self.flags[-5:]:
                parts.append(f"  - {f['topic_id']}: {f['flag_type']}")

        if self.intervention_log:
            parts.append(f"INTERVENTIONS ({len(self.intervention_log)}):")
            for i in self.intervention_log[-3:]:
                parts.append(f"  - Level {i['level']}: {i['action_type']}")

        return "\n".join(parts)

    # ── Context Overflow Handling ──

    MAX_CONTEXT_CHARS = 8000  # Approximate LLM context budget for trajectory

    def check_context_overflow(self):
        """Check if the context string exceeds the budget and compress if needed."""
        ctx = self.to_context_string(max_sections=100)  # Full context
        if len(ctx) <= self.MAX_CONTEXT_CHARS:
            return  # No overflow

        self._original_chars = len(ctx)
        logger.info(f"[Trajectory] Context overflow: {len(ctx)} chars > {self.MAX_CONTEXT_CHARS}. Compressing...")
        self._compress()

    def _compress(self):
        """Compress trajectory memory to fit within context budget."""
        # 1. Keep only last 3 sections (drop old history)
        if len(self.section_history) > 3:
            old_sections = self.section_history[:-3]
            self.section_history = self.section_history[-3:]
            logger.info(f"[Trajectory] Compressed: dropped {len(old_sections)} old sections, kept 3")

        # 2. Compress quiz attempts (keep last 2 per topic)
        for topic_id in self.quiz_attempts:
            attempts = self.quiz_attempts[topic_id]
            if len(attempts) > 2:
                # Keep only topic, score, passed from old attempts
                compressed = [
                    {"score": a.get("score", 0), "passed": a.get("passed", False)}
                    for a in attempts[:-2]
                ]
                self.quiz_attempts[topic_id] = compressed + attempts[-2:]
                logger.info(f"[Trajectory] Compressed quiz attempts for {topic_id}")

        # 3. Keep only last 5 flags
        if len(self.flags) > 5:
            self.flags = self.flags[-5:]

        # 4. Trim council proposals to summaries
        for attr in ["domain_expert_proposal", "prereq_architect_proposal",
                     "feasibility_proposal", "student_advocate_proposal"]:
            proposal = getattr(self, attr, {})
            if proposal and len(json.dumps(proposal, default=str)) > 1000:
                # Keep only essential keys
                compressed = {
                    "_compressed": True,
                    "topics_count": len(proposal.get("topics", proposal.get("ordered_topics", []))),
                    "key_decisions": str(proposal)[:300],
                }
                setattr(self, attr, compressed)

        # 5. Keep only last 5 intervention logs
        if len(self.intervention_log) > 5:
            self.intervention_log = self.intervention_log[-5:]

        self._compressed = True
        new_ctx = self.to_context_string()
        logger.info(f"[Trajectory] Compressed: {self._original_chars} -> {len(new_ctx)} chars")
        self._save()

    def _filepath(self) -> str:
        return os.path.join(DATA_DIR, f"trajectory_{self.student_id}.json")

    def _save(self):
        """Persist trajectory to disk."""
        try:
            data = {
                "student_id": self.student_id,
                "created_at": self.created_at,
                "student_profile": self.student_profile,
                "profiling_conversation": self.profiling_conversation,
                "domain_expert_proposal": self.domain_expert_proposal,
                "prereq_architect_proposal": self.prereq_architect_proposal,
                "feasibility_proposal": self.feasibility_proposal,
                "student_advocate_proposal": self.student_advocate_proposal,
                "conflict_resolution": self.conflict_resolution,
                "final_roadmap": self.final_roadmap,
                "section_history": self.section_history,
                "current_topic_materials": self.current_topic_materials,
                "quiz_attempts": self.quiz_attempts,
                "flags": self.flags,
                "intervention_log": self.intervention_log,
                "bridging_topics_inserted": self.bridging_topics_inserted,
            }
            with open(self._filepath(), "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save trajectory for {self.student_id}: {e}")

    @classmethod
    def load(cls, student_id: str) -> "TrajectoryMemory":
        """Load trajectory from disk, or create new if none exists."""
        mem = cls(student_id)
        filepath = mem._filepath()
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                mem.student_profile = data.get("student_profile", {})
                mem.profiling_conversation = data.get("profiling_conversation", [])
                mem.domain_expert_proposal = data.get("domain_expert_proposal", {})
                mem.prereq_architect_proposal = data.get("prereq_architect_proposal", {})
                mem.feasibility_proposal = data.get("feasibility_proposal", {})
                mem.student_advocate_proposal = data.get("student_advocate_proposal", {})
                mem.conflict_resolution = data.get("conflict_resolution", {})
                mem.final_roadmap = data.get("final_roadmap", {})
                mem.section_history = data.get("section_history", [])
                mem.current_topic_materials = data.get("current_topic_materials", {})
                mem.quiz_attempts = data.get("quiz_attempts", {})
                mem.flags = data.get("flags", [])
                mem.intervention_log = data.get("intervention_log", [])
                mem.bridging_topics_inserted = data.get("bridging_topics_inserted", [])
            except Exception as e:
                logger.warning(f"Failed to load trajectory for {student_id}: {e}")
        return mem


# Global trajectory store (in-memory cache)
# Named _trajectories (not _trajectory_cache) so analytics API can access it
_trajectories: Dict[str, TrajectoryMemory] = {}
# Keep backward-compatible alias
_trajectory_cache = _trajectories


def get_trajectory(student_id: str) -> TrajectoryMemory:
    """Get or create trajectory memory for a student."""
    if student_id not in _trajectories:
        _trajectories[student_id] = TrajectoryMemory.load(student_id)
    return _trajectories[student_id]


def clear_trajectory(student_id: str):
    """Clear trajectory memory for a student."""
    if student_id in _trajectories:
        del _trajectories[student_id]
    filepath = os.path.join(DATA_DIR, f"trajectory_{student_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
