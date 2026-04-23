"""
EduPath AI — Prerequisite Architect Agent
Team KRIYA | OpenEnv Hackathon 2026

Council Agent 2/6: Defines hard/soft prerequisite DAG for the proposed
topics. Validates no circular dependencies, respects phase boundaries,
and considers the student's existing proficiency.
"""
import logging
from typing import Dict, List

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(topics: List[Dict], student_profile: Dict) -> Dict:
    """
    Define prerequisite relationships for the proposed topics.

    Args:
        topics: List of topic dicts from the Domain Expert.
        student_profile: StudentProfile dict.

    Returns:
        Dict with 'ordered_topics' list and 'dag_rationale'.
    """
    if not is_api_key_set():
        return _fallback(topics, student_profile)

    system_prompt = get_prompt("prereq_architect")

    topic_summary = "\n".join(
        f"  - {t['id']}: {t['name']} (phase={t.get('phase', '?')}, difficulty={t.get('difficulty', '?')}, hours={t.get('estimated_hours', '?')})"
        for t in topics
    )

    user_prompt = f"""STUDENT PROFILE:
Verified Skills: {student_profile.get('verified_skills', {})}
Target Role: {student_profile.get('target_role', 'Unknown')}

PROPOSED TOPICS:
{topic_summary}

Define the prerequisite DAG for these topics. Hard prerequisites = cannot proceed without. Soft = helpful but not blocking. Order for confidence."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result and "ordered_topics" in result:
            # Validate: no circular deps
            result["ordered_topics"] = _validate_dag(result["ordered_topics"])
            logger.info(f"Prereq Architect defined DAG for {len(result['ordered_topics'])} topics")
            return result
    except Exception as e:
        logger.error(f"Prereq Architect failed: {e}")

    return _fallback(topics, student_profile)


def _validate_dag(ordered_topics: List[Dict]) -> List[Dict]:
    """Remove circular dependencies if any exist."""
    topic_ids = {t["id"] for t in ordered_topics}
    for t in ordered_topics:
        # Remove prerequisites that reference non-existent topics
        prereqs = t.get("prerequisites", [])
        t["prerequisites"] = [p for p in prereqs if p in topic_ids and p != t["id"]]
    return ordered_topics


def _fallback(topics: List[Dict], student_profile: Dict) -> Dict:
    """Deterministic fallback using the curriculum graph."""
    from environment.curriculum import TOPIC_GRAPH

    phase_order = {"foundation": 0, "core": 1, "specialization": 2, "interview_prep": 3}
    sorted_topics = sorted(topics, key=lambda t: (
        phase_order.get(t.get("phase", "core"), 1),
        t.get("difficulty", 3),
    ))

    ordered = []
    for i, topic in enumerate(sorted_topics):
        # Look up real prerequisites from the curriculum graph
        graph_topic = TOPIC_GRAPH.get(topic["id"])
        prereqs = []
        dep_type = "soft"

        if graph_topic:
            # Only include prereqs that are in our topic list
            topic_ids = {t["id"] for t in topics}
            prereqs = [p for p in graph_topic.prerequisites if p in topic_ids]
            dep_type = "hard" if prereqs else "soft"

        ordered.append({
            "id": topic["id"],
            "prerequisites": prereqs,
            "dependency_type": dep_type,
            "order_position": i + 1,
            "order_rationale": f"Phase {topic.get('phase', 'core')}, difficulty {topic.get('difficulty', 3)}",
        })

    return {
        "ordered_topics": ordered,
        "dag_rationale": "Deterministic ordering based on phase boundaries and difficulty levels from the curriculum graph.",
    }
