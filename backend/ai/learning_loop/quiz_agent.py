"""
EduPath AI — Quiz Agent
Team KRIYA | OpenEnv Hackathon 2026

Learning Loop Agent 4/4: Generates BKT-adaptive quiz questions that test
actual understanding of the specific course completed. Calibrates
difficulty based on BKT skill level: recall → application → analysis → synthesis.
"""
import logging
from typing import Dict, Optional

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(topic_id: str, student_profile: Dict, bkt_level: float,
        course_info: Dict = None, attempt: int = 1) -> Dict:
    """
    Generate a BKT-adaptive quiz for a topic.

    Args:
        topic_id: The topic identifier.
        student_profile: StudentProfile dict.
        bkt_level: BKT mastery probability (0.0-1.0).
        course_info: Curator's selected course info (optional).
        attempt: Quiz attempt number (1, 2, 3...).

    Returns:
        Dict with topic_id, questions list, pass_threshold, etc.
    """
    topic_name = _get_topic_name(topic_id)

    if not is_api_key_set():
        return _fallback(topic_id, topic_name, student_profile, bkt_level, attempt)

    system_prompt = get_prompt("quiz")

    course_title = course_info.get("title", "the assigned course") if course_info else "the assigned course"
    course_url = course_info.get("url", "") if course_info else ""

    user_prompt = f"""TOPIC:
Topic ID: {topic_id}
Topic Name: {topic_name}

STUDENT CONTEXT:
Target Role: {student_profile.get('target_role', 'Unknown')}
Domain: {student_profile.get('target_domain', 'tech')}

BKT SKILL LEVEL: {bkt_level:.2f}
ATTEMPT NUMBER: {attempt}

COURSE COMPLETED:
Title: {course_title}
URL: {course_url}

Generate a quiz with appropriate difficulty calibration based on the BKT skill level. Mix question types. At least 2 questions must connect to the student's domain."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result and "questions" in result:
            logger.info(f"Quiz Agent generated {len(result['questions'])} questions for {topic_name} (BKT={bkt_level:.2f})")
            return result
    except Exception as e:
        logger.error(f"Quiz Agent failed for {topic_name}: {e}")

    return _fallback(topic_id, topic_name, student_profile, bkt_level, attempt)


def _get_topic_name(topic_id: str) -> str:
    """Resolve topic name from curriculum graph."""
    try:
        from environment.curriculum import TOPIC_GRAPH
        topic = TOPIC_GRAPH.get(topic_id)
        return topic.name if topic else topic_id.replace("_", " ").title()
    except Exception:
        return topic_id.replace("_", " ").title()


def _get_difficulty_type(bkt_level: float) -> str:
    """Map BKT level to question difficulty type."""
    if bkt_level < 0.3:
        return "recall"
    elif bkt_level < 0.6:
        return "application"
    elif bkt_level < 0.8:
        return "analysis"
    else:
        return "synthesis"


def _fallback(topic_id: str, topic_name: str, student_profile: Dict,
              bkt_level: float, attempt: int) -> Dict:
    """Deterministic fallback quiz generation."""
    difficulty = _get_difficulty_type(bkt_level)
    domain = student_profile.get("target_domain", "tech")
    goal = student_profile.get("target_role", "professional")

    questions = []

    if difficulty == "recall":
        questions = [
            {
                "id": "q1", "type": "multiple_choice",
                "question": f"What is the primary purpose of {topic_name}?",
                "options": [
                    f"To provide foundational understanding for {domain} professionals",
                    f"To replace existing {domain} workflows entirely",
                    f"To only serve as theoretical background",
                    f"None of the above",
                ],
                "correct_answer": f"To provide foundational understanding for {domain} professionals",
                "explanation": f"{topic_name} builds the foundation needed for {goal}.",
                "difficulty": "recall", "domain_connected": True,
            },
            {
                "id": "q2", "type": "multiple_choice",
                "question": f"Which of the following is a key concept in {topic_name}?",
                "options": ["Core principles", "Random trivia", "Unrelated theory", "None"],
                "correct_answer": "Core principles",
                "explanation": f"Core principles form the backbone of {topic_name}.",
                "difficulty": "recall", "domain_connected": False,
            },
            {
                "id": "q3", "type": "short_answer",
                "question": f"In your own words, define {topic_name} and why it matters for {domain}.",
                "options": None,
                "correct_answer": f"{topic_name} is a critical skill for {domain} that enables practical problem-solving.",
                "explanation": "Understanding the fundamentals is key.",
                "difficulty": "recall", "domain_connected": True,
            },
        ]
    elif difficulty == "application":
        questions = [
            {
                "id": "q1", "type": "scenario",
                "question": f"A {domain} company needs to implement {topic_name}. What is your recommended first step?",
                "options": None,
                "correct_answer": f"Assess the current {domain} workflows and identify where {topic_name} adds the most value.",
                "explanation": "Practical application requires understanding the context first.",
                "difficulty": "application", "domain_connected": True,
            },
            {
                "id": "q2", "type": "multiple_choice",
                "question": f"When applying {topic_name} in a {domain} project, which approach is most effective?",
                "options": [
                    "Start with a small pilot project",
                    "Implement everything at once",
                    "Skip testing entirely",
                    "Ignore domain requirements",
                ],
                "correct_answer": "Start with a small pilot project",
                "explanation": "Iterative approach reduces risk and validates understanding.",
                "difficulty": "application", "domain_connected": True,
            },
            {
                "id": "q3", "type": "short_answer",
                "question": f"Describe how you would use {topic_name} to solve a specific {domain} problem.",
                "options": None,
                "correct_answer": f"Apply {topic_name} principles to analyse, design, and implement a solution for the {domain} problem.",
                "explanation": "Application requires connecting theory to practice.",
                "difficulty": "application", "domain_connected": True,
            },
        ]
    elif difficulty == "analysis":
        questions = [
            {
                "id": "q1", "type": "scenario",
                "question": f"Compare two approaches to {topic_name} in {domain}. Which would you choose for a time-sensitive project and why?",
                "options": None,
                "correct_answer": f"Choose the approach that balances speed with reliability, considering {domain}-specific constraints.",
                "explanation": "Analysis requires weighing trade-offs.",
                "difficulty": "analysis", "domain_connected": True,
            },
            {
                "id": "q2", "type": "multiple_choice",
                "question": f"What is the most common failure mode when applying {topic_name} incorrectly?",
                "options": [
                    "Skipping prerequisite understanding",
                    "Being too thorough",
                    "Over-documenting",
                    "Testing too much",
                ],
                "correct_answer": "Skipping prerequisite understanding",
                "explanation": "Most failures come from gaps in foundational knowledge.",
                "difficulty": "analysis", "domain_connected": False,
            },
            {
                "id": "q3", "type": "short_answer",
                "question": f"Why might {topic_name} work differently in {domain} compared to other domains?",
                "options": None,
                "correct_answer": f"{domain} has unique requirements, data types, and stakeholders that influence how {topic_name} is applied.",
                "explanation": "Domain context shapes application.",
                "difficulty": "analysis", "domain_connected": True,
            },
        ]
    else:  # synthesis
        questions = [
            {
                "id": "q1", "type": "scenario",
                "question": f"Design a solution combining {topic_name} with previously learned topics to address a complex {domain} challenge.",
                "options": None,
                "correct_answer": f"Integrate {topic_name} with foundational skills to create a holistic solution for the {domain} challenge.",
                "explanation": "Synthesis requires combining multiple concepts.",
                "difficulty": "synthesis", "domain_connected": True,
            },
            {
                "id": "q2", "type": "short_answer",
                "question": f"How would you teach {topic_name} to a colleague in {domain} who has no prior experience?",
                "options": None,
                "correct_answer": f"Start with the relevance to {domain}, build intuition with examples, then progress to technical details.",
                "explanation": "Teaching requires deep understanding.",
                "difficulty": "synthesis", "domain_connected": True,
            },
            {
                "id": "q3", "type": "multiple_choice",
                "question": f"Which combination of skills creates the most value when paired with {topic_name} in {domain}?",
                "options": [
                    f"Domain expertise + {topic_name} + communication skills",
                    f"Only {topic_name}",
                    "Random unrelated skills",
                    "None of the above",
                ],
                "correct_answer": f"Domain expertise + {topic_name} + communication skills",
                "explanation": "The most impactful professionals combine technical and domain skills.",
                "difficulty": "synthesis", "domain_connected": True,
            },
        ]

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "bkt_skill_level_used": bkt_level,
        "questions": questions,
        "pass_threshold": 70,
        "attempt_number": attempt,
    }
