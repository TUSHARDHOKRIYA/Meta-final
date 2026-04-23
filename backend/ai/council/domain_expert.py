"""
EduPath AI — Domain Expert Agent
Team KRIYA | OpenEnv Hackathon 2026

Council Agent 1/6: Proposes 15-25 goal-specific topics with justification,
phase assignments, and hour estimates. Considers the student's background
to tailor the topic selection.
"""
import logging
from typing import Dict

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(student_profile: Dict) -> Dict:
    """
    Propose topics for the student's learning roadmap.

    Args:
        student_profile: StudentProfile dict from the Profiling Agent.

    Returns:
        Dict with 'topics' list and 'rationale' string.
    """
    if not is_api_key_set():
        return _fallback(student_profile)

    system_prompt = get_prompt("domain_expert")

    user_prompt = f"""STUDENT PROFILE:
Name: {student_profile.get('name', 'Student')}
Current Profession: {student_profile.get('profession', 'Unknown')}
Current Domain: {student_profile.get('domain', 'Unknown')}
Target Role: {student_profile.get('target_role', 'Unknown')}
Target Domain: {student_profile.get('target_domain', 'tech')}
Verified Skills: {student_profile.get('verified_skills', {})}
Skip Topics: {student_profile.get('skip_topics', [])}
Weekly Hours: {student_profile.get('weekly_hours', 10)}
Deadline Weeks: {student_profile.get('deadline_weeks', 12)}
Total Available Hours: {student_profile.get('total_available_hours', 120)}
Has Interview: {student_profile.get('has_interview', False)}
Interview Weeks: {student_profile.get('interview_weeks', 0)}
Confidence Level: {student_profile.get('confidence_level', 'medium')}

Generate a topic proposal for this student. Remember: 15-25 topics maximum, goal-specific, justified."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result and "topics" in result:
            logger.info(f"Domain Expert proposed {len(result['topics'])} topics")
            return result
    except Exception as e:
        logger.error(f"Domain Expert failed: {e}")

    return _fallback(student_profile)


def _fallback(student_profile: Dict) -> Dict:
    """Deterministic fallback when LLM is unavailable."""
    from environment.curriculum import get_topics_for_field, TOPIC_GRAPH

    target_domain = student_profile.get("target_domain", "tech")
    skip = set(student_profile.get("skip_topics", []))
    verified = student_profile.get("verified_skills", {})

    field_topics = get_topics_for_field(target_domain)
    if not field_topics:
        field_topics = get_topics_for_field("tech")

    topics = []
    for topic in field_topics:
        if topic.id in skip:
            continue
        # Skip topics the student is already proficient in
        if verified.get(topic.id) == "proficient" or verified.get(topic.name, "").lower() == "proficient":
            continue

        phase = "foundation"
        if topic.difficulty <= 2:
            phase = "foundation"
        elif topic.difficulty <= 3:
            phase = "core"
        elif topic.difficulty <= 4:
            phase = "specialization"
        else:
            phase = "interview_prep" if student_profile.get("has_interview") else "specialization"

        topics.append({
            "id": topic.id,
            "name": topic.name,
            "why": f"Required for {student_profile.get('target_role', 'career goal')} in {target_domain}",
            "estimated_hours": topic.estimated_hours,
            "difficulty": topic.difficulty,
            "phase": phase,
            "is_interview_prep": False,
        })

    # Add interview prep if needed
    if student_profile.get("has_interview"):
        topics.append({
            "id": "interview_prep",
            "name": "Interview Preparation",
            "why": "Student has an upcoming interview",
            "estimated_hours": 8,
            "difficulty": 4,
            "phase": "interview_prep",
            "is_interview_prep": True,
        })

    return {
        "topics": topics[:25],
        "rationale": f"Deterministic topic selection for {target_domain} domain based on curriculum graph.",
    }
