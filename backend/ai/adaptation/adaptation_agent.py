"""
EduPath AI — Adaptation Agent
Team KRIYA | OpenEnv Hackathon 2026

Watchful guardian of the student's learning journey. Analyses recent
section history and makes the minimum necessary intervention at one
of four escalating levels:

  Level 1 — CONTINUE:       All clear, no action needed.
  Level 2 — REVISE_RETRY:   Quiz failed once → suggest revision.
  Level 3 — BETTER_RESOURCE: Quiz failed twice → alternative resource.
  Level 4 — INTERVENTION:   2+ flags in 3 sections → root cause diagnosis,
                             bridging topic insertion (max 1 per cycle).
"""
import logging
from typing import Dict, Optional

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(trajectory_memory, student_profile: Dict) -> Dict:
    """
    Analyse recent learning trajectory and determine intervention level.

    Args:
        trajectory_memory: TrajectoryMemory instance for the student.
        student_profile: StudentProfile dict.

    Returns:
        Dict with level, action_type, message_to_student, intervention_details, etc.
    """
    # Determine level from flag patterns
    recent_flags = trajectory_memory.get_recent_flags(n=3)
    flag_count = len(recent_flags)

    # Check latest section
    latest = trajectory_memory.section_history[-1] if trajectory_memory.section_history else None
    latest_flag = latest.get("flag", "CLEAR") if latest else "CLEAR"
    latest_topic = latest.get("topic_id", "") if latest else ""

    # Determine quiz attempts for latest topic
    attempt_count = trajectory_memory.get_quiz_attempt_count(latest_topic) if latest_topic else 0

    # ── Level determination ──
    if latest_flag == "CLEAR" and flag_count == 0:
        level = 1
        action_type = "CONTINUE"
    elif attempt_count == 1 and latest_flag != "CLEAR":
        level = 2
        action_type = "REVISE_RETRY"
    elif attempt_count == 2 and latest_flag != "CLEAR":
        level = 3
        action_type = "BETTER_RESOURCE"
    elif flag_count >= 2:
        level = 4
        action_type = "INTERVENTION"
    elif attempt_count >= 3:
        level = 4
        action_type = "INTERVENTION"
    else:
        level = 1
        action_type = "CONTINUE"

    # For Level 1, return immediately
    if level == 1:
        return {
            "level": 1,
            "action_type": "CONTINUE",
            "message_to_student": "Great progress! Keep going — you're on track. 🚀",
            "roadmap_change": False,
            "intervention_details": None,
            "flag_to_council_manager": False,
            "timeline_risk": False,
        }

    # For Levels 2-4, try LLM or fallback
    if is_api_key_set():
        try:
            return _llm_adaptation(level, action_type, trajectory_memory, student_profile)
        except Exception as e:
            logger.warning(f"LLM adaptation failed: {e}")

    return _fallback_adaptation(level, action_type, trajectory_memory, student_profile)


def _llm_adaptation(level: int, action_type: str,
                    trajectory_memory, student_profile: Dict) -> Dict:
    """Use LLM for nuanced intervention decisions."""
    system_prompt = get_prompt("adaptation")
    context = trajectory_memory.to_context_string(max_sections=5)

    user_prompt = f"""CURRENT LEVEL: {level}
ACTION TYPE: {action_type}

TRAJECTORY CONTEXT:
{context}

STUDENT PROFILE:
Target Role: {student_profile.get('target_role', 'Unknown')}
Domain: {student_profile.get('target_domain', 'tech')}
Confidence: {student_profile.get('confidence_level', 'medium')}
Weekly Hours: {student_profile.get('weekly_hours', 10)}
Deadline Weeks: {student_profile.get('deadline_weeks', 12)}

Analyse the situation and provide the minimum necessary intervention."""

    result = generate_json_with_retry(system_prompt, user_prompt)
    if result and "level" in result:
        logger.info(f"Adaptation Agent: Level {result['level']} — {result.get('action_type', '?')}")
        return result

    return _fallback_adaptation(level, action_type, trajectory_memory, student_profile)


def _fallback_adaptation(level: int, action_type: str,
                         trajectory_memory, student_profile: Dict) -> Dict:
    """Deterministic adaptation fallback."""
    latest = trajectory_memory.section_history[-1] if trajectory_memory.section_history else {}
    latest_topic = latest.get("topic_id", "unknown")
    recent_flags = trajectory_memory.get_recent_flags(n=3)

    # Collect struggling topics
    struggling = list({f.get("topic_id", "") for f in recent_flags if f.get("topic_id")})

    if level == 2:
        return {
            "level": 2,
            "action_type": "REVISE_RETRY",
            "message_to_student": (
                f"Don't worry — failing a quiz on your first try is completely normal! "
                f"Go back to the study notes for {latest_topic.replace('_', ' ').title()}, "
                f"focus on the 'Remember for Quiz' section, then try again. You've got this! 💪"
            ),
            "roadmap_change": False,
            "intervention_details": {
                "struggling_topics": [latest_topic],
                "pattern_detected": "First quiz failure",
                "root_cause": "Normal learning curve — revision needed",
                "bridging_topic_type_needed": None,
                "insert_after_topic_id": None,
            },
            "flag_to_council_manager": False,
            "timeline_risk": False,
        }

    elif level == 3:
        return {
            "level": 3,
            "action_type": "BETTER_RESOURCE",
            "message_to_student": (
                f"I see you're finding {latest_topic.replace('_', ' ').title()} challenging — "
                f"that's okay! Let me find you an alternative explanation that might click better. "
                f"Sometimes a different teaching style makes all the difference."
            ),
            "roadmap_change": False,
            "intervention_details": {
                "struggling_topics": [latest_topic],
                "pattern_detected": "Repeated quiz failure (2 attempts)",
                "root_cause": "Current resource may not match learning style",
                "bridging_topic_type_needed": None,
                "insert_after_topic_id": None,
            },
            "flag_to_council_manager": False,
            "timeline_risk": True,
        }

    else:  # level == 4
        # Root cause diagnosis
        root_cause = _diagnose_root_cause(struggling, trajectory_memory, student_profile)

        # Determine bridging topic
        bridging = _suggest_bridging(struggling, student_profile)

        return {
            "level": 4,
            "action_type": "INTERVENTION",
            "message_to_student": (
                f"I've noticed you're struggling across a few areas. "
                f"I think I've found the root cause: {root_cause}. "
                f"Let me adjust your roadmap to fill this gap before we continue. "
                f"This is exactly what the system is designed for — adaptive learning! 🎯"
            ),
            "roadmap_change": True,
            "intervention_details": {
                "struggling_topics": struggling,
                "pattern_detected": f"{len(struggling)} topics flagged in recent sections",
                "root_cause": root_cause,
                "bridging_topic_type_needed": bridging.get("type", "foundational_review"),
                "insert_after_topic_id": bridging.get("insert_after"),
            },
            "flag_to_council_manager": True,
            "timeline_risk": True,
        }


def _diagnose_root_cause(struggling: list, trajectory_memory, student_profile: Dict) -> str:
    """Simple root cause diagnosis from flag patterns."""
    from environment.curriculum import TOPIC_GRAPH

    # Check difficulty patterns
    difficulties = []
    for tid in struggling:
        topic = TOPIC_GRAPH.get(tid)
        if topic:
            difficulties.append(topic.difficulty)

    # Check if math/theoretical topics
    math_topics = [tid for tid in struggling if any(w in tid.lower() for w in ["math", "stat", "calcul", "linear", "prob"])]
    practical_topics = [tid for tid in struggling if any(w in tid.lower() for w in ["project", "build", "deploy", "implement"])]

    if math_topics:
        return "Missing mathematical foundations — need prerequisite math review"
    elif practical_topics:
        return "Theory understood but practical application skills need reinforcement"
    elif difficulties and sum(difficulties) / len(difficulties) > 3.5:
        return "Topics are too advanced — missing intermediate prerequisites"
    elif len(struggling) >= 3:
        return "Foundational knowledge gap — need to review core prerequisites"
    else:
        return "Pace/time issue — may need to slow down and consolidate"


def _suggest_bridging(struggling: list, student_profile: Dict) -> Dict:
    """Suggest a bridging topic to insert."""
    from environment.curriculum import TOPIC_GRAPH

    # Find the earliest struggling topic
    earliest = struggling[0] if struggling else ""
    topic = TOPIC_GRAPH.get(earliest)

    if topic and topic.prerequisites:
        # Insert a review of the first prerequisite
        return {
            "type": "prerequisite_review",
            "topic_id": topic.prerequisites[0],
            "insert_after": topic.prerequisites[0],
        }

    return {
        "type": "foundational_review",
        "topic_id": None,
        "insert_after": earliest,
    }
