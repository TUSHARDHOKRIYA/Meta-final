"""
EduPath AI — Critic Agent
Team KRIYA | OpenEnv Hackathon 2026

Learning Loop Agent 2/4: Scores every course candidate across 5 dimensions
(relevance, difficulty match, content quality, time efficiency, style match)
using a weighted formula. Ranks all candidates from best to worst.
"""
import logging
from typing import Dict, List

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(candidates: List[Dict], student_profile: Dict, topic_info: Dict) -> Dict:
    """
    Score and rank course candidates.

    Args:
        candidates: List of course dicts from the Scout Agent.
        student_profile: StudentProfile dict.
        topic_info: Dict with topic_id, topic_name.

    Returns:
        Dict with evaluations list and ranking.
    """
    if not is_api_key_set():
        return _fallback(candidates, student_profile, topic_info)

    system_prompt = get_prompt("critic")

    import json
    candidates_str = json.dumps(candidates[:10], indent=2, default=str)

    user_prompt = f"""STUDENT PROFILE:
Target Role: {student_profile.get('target_role', 'Unknown')}
Learning Style: {student_profile.get('learning_style', 'video')}
Confidence: {student_profile.get('confidence_level', 'medium')}
Budget: {student_profile.get('budget', 'free_only')}

TOPIC: {topic_info.get('topic_name', topic_info.get('topic_id', '?'))}

COURSE CANDIDATES:
{candidates_str}

Score each course across all 5 dimensions. Be critical — differentiate scores clearly."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result and "evaluations" in result:
            logger.info(f"Critic evaluated {len(result['evaluations'])} candidates for {topic_info.get('topic_name', '?')}")
            return result
    except Exception as e:
        logger.error(f"Critic Agent failed: {e}")

    return _fallback(candidates, student_profile, topic_info)


def _fallback(candidates: List[Dict], student_profile: Dict, topic_info: Dict) -> Dict:
    """Deterministic scoring fallback."""
    style_pref = student_profile.get("learning_style", "video")
    budget = student_profile.get("budget", "free_only")
    evaluations = []

    for c in candidates:
        # Relevance: always moderate for curriculum-sourced resources
        relevance = 7

        # Difficulty match
        diff = c.get("difficulty", "intermediate")
        confidence = student_profile.get("confidence_level", "medium")
        if diff == "beginner" and confidence == "low":
            difficulty_match = 9
        elif diff == "intermediate" and confidence == "medium":
            difficulty_match = 8
        elif diff == "advanced" and confidence == "high":
            difficulty_match = 8
        else:
            difficulty_match = 5

        # Content quality: newer = better
        year = c.get("last_updated_year", 2020)
        content_quality = min(10, max(3, (year - 2018)))

        # Time efficiency
        hours = c.get("estimated_hours", 10)
        time_efficiency = 8 if hours <= 6 else 6 if hours <= 12 else 4

        # Style match
        content_type = c.get("content_type", "mixed")
        if style_pref == content_type:
            style_match = 9
        elif content_type == "mixed":
            style_match = 7
        else:
            style_match = 5

        # Budget penalty
        if budget == "free_only" and not c.get("is_free", True):
            relevance -= 3

        total = round(
            relevance * 0.30 +
            difficulty_match * 0.25 +
            content_quality * 0.20 +
            time_efficiency * 0.15 +
            style_match * 0.10,
            2
        )

        evaluations.append({
            "course_id": c.get("id", "?"),
            "scores": {
                "relevance": relevance,
                "difficulty_match": difficulty_match,
                "content_quality": content_quality,
                "time_efficiency": time_efficiency,
                "style_match": style_match,
                "total": total,
            },
            "flags": [],
            "one_line_verdict": f"Score {total}/10 — {'good' if total >= 7 else 'acceptable' if total >= 5 else 'weak'} match",
        })

    # Sort by total score descending
    evaluations.sort(key=lambda e: e["scores"]["total"], reverse=True)
    ranking = [e["course_id"] for e in evaluations]

    return {
        "evaluations": evaluations,
        "ranking": ranking,
    }
