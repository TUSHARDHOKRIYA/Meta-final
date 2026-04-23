"""
EduPath AI — Student Advocate Agent
Team KRIYA | OpenEnv Hackathon 2026

Council Agent 4/6: Personalisation overlay — skip proficient topics,
identify confidence boosters, match learning style, flag budget
sensitivity. Champions the student's experience.
"""
import logging
from typing import Dict, List

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(feasible_topics: List[str], topics: List[Dict], student_profile: Dict) -> Dict:
    """
    Apply personalisation overlay to the feasible topic list.

    Args:
        feasible_topics: List of topic_ids that passed feasibility.
        topics: Original topic dicts from Domain Expert.
        student_profile: StudentProfile dict.

    Returns:
        Dict with skip, confidence_boost_order, depth_adjustments, style_flags.
    """
    if not is_api_key_set():
        return _fallback(feasible_topics, topics, student_profile)

    system_prompt = get_prompt("student_advocate")

    topic_map = {t["id"]: t for t in topics}
    topic_summary = "\n".join(
        f"  - {tid}: {topic_map.get(tid, {}).get('name', tid)} "
        f"(difficulty={topic_map.get(tid, {}).get('difficulty', 3)}, "
        f"phase={topic_map.get(tid, {}).get('phase', '?')})"
        for tid in feasible_topics
    )

    user_prompt = f"""STUDENT PROFILE:
Verified Skills: {student_profile.get('verified_skills', {})}
Confidence Level: {student_profile.get('confidence_level', 'medium')}
Learning Style: {student_profile.get('learning_style', 'video')}
Budget: {student_profile.get('budget', 'free_only')}
Skip Topics: {student_profile.get('skip_topics', [])}

FEASIBLE TOPICS:
{topic_summary}

Apply your personalisation lens. Skip what they know, boost confidence early, match their style."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result:
            logger.info(f"Student Advocate: skip={len(result.get('skip', []))}, boosts={len(result.get('confidence_boost_order', []))}")
            return result
    except Exception as e:
        logger.error(f"Student Advocate failed: {e}")

    return _fallback(feasible_topics, topics, student_profile)


def _fallback(feasible_topics: List[str], topics: List[Dict], student_profile: Dict) -> Dict:
    """Deterministic fallback: skip proficient, boost easy topics."""
    topic_map = {t["id"]: t for t in topics}
    verified = student_profile.get("verified_skills", {})
    skip_topics = list(student_profile.get("skip_topics", []))
    confidence = student_profile.get("confidence_level", "medium")
    style = student_profile.get("learning_style", "video")

    # Skip proficient topics
    skip = []
    for tid in feasible_topics:
        if verified.get(tid) == "proficient":
            skip.append(tid)
        topic_detail = topic_map.get(tid, {})
        if topic_detail.get("name", "") in [s for s, v in verified.items() if v == "proficient"]:
            skip.append(tid)

    skip = list(set(skip + skip_topics))

    # Confidence boosters: easiest topics first
    remaining = [tid for tid in feasible_topics if tid not in skip]
    sorted_by_difficulty = sorted(
        remaining,
        key=lambda tid: topic_map.get(tid, {}).get("difficulty", 3)
    )
    boost_count = 3 if confidence == "low" else 2
    confidence_boost = sorted_by_difficulty[:boost_count]

    # Depth adjustments
    depth_adjustments = []
    for tid in remaining:
        d = topic_map.get(tid, {}).get("difficulty", 3)
        if verified.get(tid) == "partial":
            depth_adjustments.append({"topic_id": tid, "depth": "shallow", "reason": "Student has partial knowledge"})
        elif d >= 4:
            depth_adjustments.append({"topic_id": tid, "depth": "deep", "reason": "Advanced topic requiring thorough study"})

    # Style flags
    style_flags = []
    if style == "project_based":
        for tid in remaining:
            style_flags.append({"topic_id": tid, "flag": "needs_project", "suggestion": "Include hands-on project"})
    elif style == "video":
        for tid in remaining[:3]:
            style_flags.append({"topic_id": tid, "flag": "video_heavy", "suggestion": "Prioritize video resources"})

    return {
        "skip": skip,
        "confidence_boost_order": confidence_boost,
        "depth_adjustments": depth_adjustments[:10],
        "style_flags": style_flags[:10],
        "personalization_notes": f"Skipping {len(skip)} proficient topics. {boost_count} confidence boosters placed early. Style: {style}.",
    }
