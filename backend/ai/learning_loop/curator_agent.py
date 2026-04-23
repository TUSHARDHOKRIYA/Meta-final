"""
EduPath AI — Curator Agent
Team KRIYA | OpenEnv Hackathon 2026

Learning Loop Agent 3/4: Selects the best course from Critic's ranking,
then generates 4 outputs: selected_course, cheat_sheet, study_notes,
and mini_project — all domain-specific to the student's goal.
"""
import logging
from typing import Dict, List

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(evaluations: List[Dict], candidates: List[Dict],
        student_profile: Dict, topic_info: Dict) -> Dict:
    """
    Select the best course and generate all learning materials.

    Args:
        evaluations: Scored evaluations from the Critic Agent.
        candidates: Original course candidates from the Scout Agent.
        student_profile: StudentProfile dict.
        topic_info: Dict with topic_id, topic_name.

    Returns:
        Dict with selected_course, cheat_sheet, study_notes, mini_project.
    """
    if not is_api_key_set():
        return _fallback(evaluations, candidates, student_profile, topic_info)

    system_prompt = get_prompt("curator")

    import json
    eval_str = json.dumps(evaluations[:5], indent=2, default=str)
    candidates_map = {c.get("id"): c for c in candidates}

    # Get top candidate details
    top_id = evaluations[0]["course_id"] if evaluations else None
    top_course = candidates_map.get(top_id, {})

    user_prompt = f"""STUDENT PROFILE:
Target Role: {student_profile.get('target_role', 'Unknown')}
Domain: {student_profile.get('target_domain', 'tech')}
Learning Style: {student_profile.get('learning_style', 'video')}

TOPIC: {topic_info.get('topic_name', topic_info.get('topic_id', '?'))}
Topic ID: {topic_info.get('topic_id', '?')}

TOP COURSE EVALUATIONS:
{eval_str}

TOP COURSE DETAILS:
{json.dumps(top_course, indent=2, default=str)}

Select the best course and generate: cheat_sheet, study_notes, and mini_project. Everything must be specific to this student's domain and goal."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result and "selected_course" in result:
            logger.info(f"Curator selected course for {topic_info.get('topic_name', '?')}")
            return result
    except Exception as e:
        logger.error(f"Curator Agent failed: {e}")

    return _fallback(evaluations, candidates, student_profile, topic_info)


def _fallback(evaluations: List[Dict], candidates: List[Dict],
              student_profile: Dict, topic_info: Dict) -> Dict:
    """Deterministic fallback with generated study materials."""
    candidates_map = {c.get("id"): c for c in candidates}

    # Select top course
    top_id = evaluations[0]["course_id"] if evaluations else (candidates[0]["id"] if candidates else "unknown")
    top_course = candidates_map.get(top_id, candidates[0] if candidates else {})

    topic_name = topic_info.get("topic_name", topic_info.get("topic_id", "Topic"))
    goal = student_profile.get("target_role", "career goal")
    domain = student_profile.get("target_domain", "tech")

    return {
        "selected_course": {
            "course_id": top_course.get("id", "course_1"),
            "title": top_course.get("title", f"{topic_name} Course"),
            "url": top_course.get("url", ""),
            "selection_rationale": f"Best match for {goal} based on relevance, difficulty, and content quality scores.",
        },
        "cheat_sheet": {
            "title": f"{topic_name} — Cheat Sheet",
            "vocabulary": [
                {"term": f"Core concept of {topic_name}", "definition": f"Fundamental principle underlying {topic_name}"},
                {"term": "Prerequisites", "definition": "Topics you should understand before diving deep"},
                {"term": "Key framework", "definition": f"Primary tool/framework used in {topic_name}"},
                {"term": "Best practice", "definition": f"Industry-standard approach to {topic_name}"},
                {"term": "Common pitfall", "definition": f"Mistake to avoid when learning {topic_name}"},
            ],
            "core_concept": f"{topic_name} is essential for anyone pursuing {goal} in {domain}. It provides the foundational understanding needed to build practical skills.",
            "why_it_matters_for_your_goal": f"As a future {goal}, mastering {topic_name} will enable you to work with real-world {domain} challenges effectively.",
            "watch_for": [
                "Understand the 'why' before the 'how'",
                "Practice with real datasets or examples",
                "Connect this to topics you've already learned",
            ],
        },
        "study_notes": {
            "title": f"Study Notes — {topic_name}",
            "what_you_should_now_know": [
                f"Core principles of {topic_name}",
                f"How {topic_name} applies to {domain}",
                "Key terminology and definitions",
                "Basic implementation patterns",
                "Common use cases and examples",
            ],
            "most_important_concept": f"The fundamental principle of {topic_name} and how it connects to your goal of becoming a {goal}.",
            "domain_connection": f"{topic_name} is directly applicable to {domain} workflows and will be built upon in later topics.",
            "remember_for_quiz": [
                f"Core definition of {topic_name}",
                "Key differences from related concepts",
                f"One practical application in {domain}",
            ],
        },
        "mini_project": {
            "title": f"Mini Project: Apply {topic_name}",
            "description": f"Build a small project demonstrating your understanding of {topic_name} using real data relevant to {domain}.",
            "dataset": f"Public dataset relevant to {domain} (e.g., from Kaggle or UCI)",
            "requirements": [
                f"Apply core {topic_name} concepts",
                "Use a real dataset",
                "Document your approach",
                "Include at least one visualization or output",
            ],
            "domain_reflection": f"How does this project connect to your goal of becoming a {goal}?",
            "estimated_hours": top_course.get("estimated_hours", 4) // 3 + 1,
            "pass_criteria": "Project demonstrates understanding of core concepts with working code and documentation.",
        },
    }
