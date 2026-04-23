"""
EduPath AI — Profiling Agent
Team KRIYA | OpenEnv Hackathon 2026

Conversational agent that builds a rich StudentProfile through
natural, warm dialogue. Probes claimed skills, detects contradictions,
and captures 5 dimensions: profession, skills, goal, time, style.
"""
import json
import logging
from typing import Dict, List, Optional

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_chat, is_api_key_set
from ai.trajectory_memory import get_trajectory

logger = logging.getLogger(__name__)

# Dimension tracking
DIMENSIONS = [
    "profession",    # Current profession and background
    "skills",        # Current skills (probed and verified)
    "goal",          # Target role and goal
    "time",          # Time constraints and deadline
    "style",         # Learning style and budget
]


class ProfilingAgent:
    """Conversational profiling agent that builds StudentProfile through dialogue."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.messages: List[Dict] = []
        self.dimensions_captured: Dict[str, bool] = {d: False for d in DIMENSIONS}
        self.extracted_data: Dict = {
            "student_id": student_id,
            "name": "",
            "profession": "",
            "domain": "",
            "target_role": "",
            "target_domain": "",
            "target_company_type": "",
            "verified_skills": {},
            "skip_topics": [],
            "weekly_hours": 10,
            "deadline_weeks": 12,
            "total_available_hours": 120,
            "learning_style": "video",
            "budget": "free_only",
            "confidence_level": "medium",
            "has_interview": False,
            "interview_weeks": 0,
        }
        self.question_count = 0
        self.is_complete = False
        self.system_prompt = get_prompt("profiling")

    def start(self) -> str:
        """Start the profiling conversation. Returns the opening message."""
        greeting = (
            "Hey there! 👋 I'm your learning mentor at EduPath. "
            "I'd love to get to know you a bit so I can build a "
            "personalised roadmap that actually fits your life.\n\n"
            "Let's start simple — what do you currently do for work? "
            "Or if you're a student, what are you studying?"
        )
        self.messages.append({"role": "assistant", "content": greeting})
        return greeting

    def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response."""
        self.messages.append({"role": "user", "content": user_message})
        self.question_count += 1

        # Try LLM-based conversation
        if is_api_key_set():
            try:
                response = generate_chat(self.system_prompt, self.messages)
                self.messages.append({"role": "assistant", "content": response})

                # Check if profiling is complete
                if "everything I need" in response.lower() or "build your" in response.lower():
                    self.is_complete = True
                    self._extract_profile_from_conversation()

                # Update dimension tracking heuristically
                self._update_dimensions(user_message)

                return response
            except Exception as e:
                logger.warning(f"LLM chat failed, using fallback: {e}")

        # Fallback: rule-based conversation
        return self._rule_based_response(user_message)

    def get_profile(self) -> Dict:
        """Get the extracted student profile."""
        return self.extracted_data

    def is_profiling_complete(self) -> bool:
        """Check if all dimensions have been captured."""
        return self.is_complete or all(self.dimensions_captured.values())

    def _update_dimensions(self, message: str):
        """Heuristically track which dimensions have been discussed."""
        msg_lower = message.lower()

        # Profession detection
        job_words = ["work", "job", "engineer", "developer", "nurse", "doctor",
                     "manager", "analyst", "student", "intern", "designer"]
        if any(w in msg_lower for w in job_words):
            self.dimensions_captured["profession"] = True

        # Skills detection
        skill_words = ["python", "java", "javascript", "sql", "excel", "ml",
                       "machine learning", "data", "coding", "programming",
                       "statistics", "math", "biology", "healthcare"]
        if any(w in msg_lower for w in skill_words):
            self.dimensions_captured["skills"] = True

        # Goal detection
        goal_words = ["want to", "become", "goal", "career", "transition",
                      "learn", "switch", "aspire", "dream", "targeting"]
        if any(w in msg_lower for w in goal_words):
            self.dimensions_captured["goal"] = True

        # Time detection
        time_words = ["hours", "week", "month", "deadline", "interview",
                      "time", "busy", "schedule", "available"]
        if any(w in msg_lower for w in time_words):
            self.dimensions_captured["time"] = True

        # Style detection
        style_words = ["video", "reading", "project", "hands-on", "free",
                       "paid", "course", "book", "tutorial", "budget"]
        if any(w in msg_lower for w in style_words):
            self.dimensions_captured["style"] = True

    def _rule_based_response(self, user_message: str) -> str:
        """Fallback rule-based conversation when LLM is unavailable."""
        uncaptured = [d for d, v in self.dimensions_captured.items() if not v]
        self._update_dimensions(user_message)

        # Extract what we can from the message
        self._extract_from_message(user_message)

        if not uncaptured or self.question_count >= 10:
            self.is_complete = True
            return (
                "Perfect — I have everything I need. "
                "Let me build your personalised roadmap now. 🚀"
            )

        next_dim = uncaptured[0]
        questions = {
            "profession": "That's great! Could you tell me a bit about your current role or what you're studying?",
            "skills": "What technical skills do you currently have? For example, any programming languages, tools, or frameworks you've used?",
            "goal": "Awesome! So what's the dream? What role or career direction are you aiming for?",
            "time": "Got it! How many hours per week can you realistically dedicate to learning? And do you have any specific deadline or interview coming up?",
            "style": "Last one — how do you prefer to learn? Videos, reading, or hands-on projects? And are you looking for free resources only, or open to paid courses too?",
        }

        response = questions.get(next_dim, "Tell me more about yourself!")
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _extract_from_message(self, message: str):
        """Extract profile data from user message heuristically."""
        msg_lower = message.lower()

        # Extract hours
        import re
        hours_match = re.search(r'(\d+)\s*hours?\s*(per|a|/)\s*week', msg_lower)
        if hours_match:
            self.extracted_data["weekly_hours"] = int(hours_match.group(1))

        # Extract deadline
        weeks_match = re.search(r'(\d+)\s*weeks?', msg_lower)
        if weeks_match:
            self.extracted_data["deadline_weeks"] = int(weeks_match.group(1))

        months_match = re.search(r'(\d+)\s*months?', msg_lower)
        if months_match:
            self.extracted_data["deadline_weeks"] = int(months_match.group(1)) * 4

        # Interview detection
        if "interview" in msg_lower:
            self.extracted_data["has_interview"] = True

        # Budget detection
        if "free" in msg_lower:
            self.extracted_data["budget"] = "free_only"
        elif "paid" in msg_lower:
            self.extracted_data["budget"] = "paid_ok"

        # Learning style
        if "video" in msg_lower:
            self.extracted_data["learning_style"] = "video"
        elif "read" in msg_lower:
            self.extracted_data["learning_style"] = "reading"
        elif "project" in msg_lower or "hands" in msg_lower:
            self.extracted_data["learning_style"] = "project_based"

        # Update total hours
        self.extracted_data["total_available_hours"] = (
            self.extracted_data["weekly_hours"] * self.extracted_data["deadline_weeks"]
        )

    def _extract_profile_from_conversation(self):
        """Use LLM to extract structured profile from the full conversation."""
        if not is_api_key_set():
            return

        try:
            from ai.llm_client import generate_json
            conversation_text = "\n".join(
                f"{'Student' if m['role'] == 'user' else 'Mentor'}: {m['content']}"
                for m in self.messages
            )

            extract_prompt = """Extract a StudentProfile JSON from this conversation.
OUTPUT ONLY VALID JSON with these fields:
{
  "name": "student name or empty string",
  "profession": "current job",
  "domain": "current domain",
  "target_role": "what they want to become",
  "target_domain": "domain of target role",
  "verified_skills": {"skill_name": "none|partial|proficient"},
  "skip_topics": ["topic_ids they already know well"],
  "weekly_hours": number,
  "deadline_weeks": number,
  "learning_style": "video|reading|project_based",
  "budget": "free_only|paid_ok",
  "confidence_level": "low|medium|high",
  "has_interview": true|false,
  "interview_weeks": number
}"""
            result = generate_json(extract_prompt, conversation_text)
            if result:
                for key, value in result.items():
                    if key in self.extracted_data and value:
                        self.extracted_data[key] = value
                # Recalculate
                self.extracted_data["total_available_hours"] = (
                    self.extracted_data["weekly_hours"] * self.extracted_data["deadline_weeks"]
                )
        except Exception as e:
            logger.warning(f"Profile extraction from conversation failed: {e}")

    def save_to_trajectory(self):
        """Save profiling data to trajectory memory."""
        memory = get_trajectory(self.student_id)
        memory.profiling_conversation = self.messages
        memory.record_profile(self.extracted_data)


# In-memory store for active profiling sessions
_active_sessions: Dict[str, ProfilingAgent] = {}


def get_or_create_session(student_id: str) -> ProfilingAgent:
    """Get or create a profiling session for a student."""
    if student_id not in _active_sessions:
        _active_sessions[student_id] = ProfilingAgent(student_id)
    return _active_sessions[student_id]


def end_session(student_id: str) -> Optional[Dict]:
    """End a profiling session and return the extracted profile."""
    session = _active_sessions.pop(student_id, None)
    if session:
        session.save_to_trajectory()
        return session.get_profile()
    return None
