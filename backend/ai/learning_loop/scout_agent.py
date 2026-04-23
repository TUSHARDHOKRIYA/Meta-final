"""
EduPath AI — Scout Agent
Team KRIYA | OpenEnv Hackathon 2026

Learning Loop Agent 1/4: Searches for 10 course candidates for a given topic,
filtered by budget, with diversity enforcement. Integrates with existing
resource_fetcher.py and LLM-based course discovery.
"""
import logging
from typing import Dict, List

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(topic_id: str, topic_name: str, student_profile: Dict) -> Dict:
    """
    Search for course candidates for a topic.

    Args:
        topic_id: The topic identifier.
        topic_name: Human-readable topic name.
        student_profile: StudentProfile dict.

    Returns:
        Dict with topic_id, topic_name, and candidates list.
    """
    if not is_api_key_set():
        return _fallback(topic_id, topic_name, student_profile)

    system_prompt = get_prompt("scout")

    user_prompt = f"""TOPIC TO RESEARCH:
Topic ID: {topic_id}
Topic Name: {topic_name}

STUDENT CONTEXT:
Target Role: {student_profile.get('target_role', 'Unknown')}
Domain: {student_profile.get('target_domain', 'tech')}
Learning Style: {student_profile.get('learning_style', 'video')}
Budget: {student_profile.get('budget', 'free_only')}
Confidence: {student_profile.get('confidence_level', 'medium')}

Find 10 course candidates. Ensure diversity: at least 1 project-based, 1 theoretical, 1 quick option (<5 hours)."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result and "candidates" in result:
            logger.info(f"Scout found {len(result['candidates'])} candidates for {topic_name}")
            return result
    except Exception as e:
        logger.error(f"Scout Agent failed for {topic_name}: {e}")

    return _fallback(topic_id, topic_name, student_profile)


def _fallback(topic_id: str, topic_name: str, student_profile: Dict) -> Dict:
    """Deterministic fallback using existing resource data."""
    from environment.curriculum import TOPIC_GRAPH

    topic = TOPIC_GRAPH.get(topic_id)
    candidates = []

    if topic and topic.resources:
        for i, r in enumerate(topic.resources[:10]):
            candidates.append({
                "id": f"course_{i+1}",
                "title": r.title,
                "platform": r.platform,
                "url": r.url,
                "estimated_hours": topic.estimated_hours,
                "difficulty": "beginner" if topic.difficulty <= 2 else "intermediate" if topic.difficulty <= 4 else "advanced",
                "content_type": r.type.value if hasattr(r.type, 'value') else "mixed",
                "last_updated_year": 2024,
                "is_free": True,
                "price_usd": 0,
                "has_certificate": False,
                "brief_description": f"{r.title} on {r.platform}",
            })

    # Pad with generic free resources if we have fewer than 3
    if len(candidates) < 3:
        generic = [
            {"id": f"course_{len(candidates)+1}", "title": f"{topic_name} - freeCodeCamp",
             "platform": "freeCodeCamp", "url": "https://www.freecodecamp.org/",
             "estimated_hours": 6, "difficulty": "beginner", "content_type": "interactive",
             "last_updated_year": 2024, "is_free": True, "price_usd": 0,
             "has_certificate": True, "brief_description": f"Free interactive {topic_name} course"},
            {"id": f"course_{len(candidates)+2}", "title": f"{topic_name} - Khan Academy",
             "platform": "Khan Academy", "url": "https://www.khanacademy.org/",
             "estimated_hours": 4, "difficulty": "beginner", "content_type": "video",
             "last_updated_year": 2024, "is_free": True, "price_usd": 0,
             "has_certificate": False, "brief_description": f"Khan Academy {topic_name} fundamentals"},
            {"id": f"course_{len(candidates)+3}", "title": f"{topic_name} - MIT OCW",
             "platform": "MIT OpenCourseWare", "url": "https://ocw.mit.edu/",
             "estimated_hours": 10, "difficulty": "intermediate", "content_type": "reading",
             "last_updated_year": 2024, "is_free": True, "price_usd": 0,
             "has_certificate": False, "brief_description": f"MIT open courseware for {topic_name}"},
        ]
        candidates.extend(generic[:3 - len(candidates)] if len(candidates) < 3 else [])

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "candidates": candidates,
    }
