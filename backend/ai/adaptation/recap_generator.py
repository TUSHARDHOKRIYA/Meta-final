"""
EduPath AI — Recap Generator
Team KRIYA | OpenEnv Hackathon 2026

Curator in INTERVENTION MODE. Generates smart consolidation notes
from courses ALREADY completed. Connects the dots the student is
not seeing. CRITICAL: NO new courses — consolidation only.

4 sections:
  1. What You Already Know   — affirm existing understanding
  2. The Missing Link        — the ONE connection they're missing
  3. Quick Reference Card    — 10 key concepts, one page
  4. Before You Continue     — 5 "I can..." checklist items
"""
import logging
from typing import Dict, List

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(struggling_topics: List[str], trajectory_memory,
        student_profile: Dict) -> Dict:
    """
    Generate consolidation recap for struggling topics.

    Args:
        struggling_topics: List of topic_ids the student is struggling with.
        trajectory_memory: TrajectoryMemory instance.
        student_profile: StudentProfile dict.

    Returns:
        Recap dict with what_you_know, missing_link, quick_reference, checklist.
    """
    if not is_api_key_set():
        return _fallback(struggling_topics, trajectory_memory, student_profile)

    system_prompt = get_prompt("recap_generator")
    context = trajectory_memory.to_context_string(max_sections=8)

    topic_names = [_get_topic_name(tid) for tid in struggling_topics]

    user_prompt = f"""STRUGGLING TOPICS: {', '.join(topic_names)}

STUDENT CONTEXT:
Target Role: {student_profile.get('target_role', 'Unknown')}
Domain: {student_profile.get('target_domain', 'tech')}

TRAJECTORY:
{context}

Generate consolidation recap. NO NEW COURSES. Reference only what they've already studied. Keep it under 30 minutes reading."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result and "what_you_know" in result:
            logger.info(f"Recap generated for {len(struggling_topics)} topics")
            return result
    except Exception as e:
        logger.error(f"Recap Generator failed: {e}")

    return _fallback(struggling_topics, trajectory_memory, student_profile)


def _get_topic_name(topic_id: str) -> str:
    """Resolve topic name from curriculum graph."""
    try:
        from environment.curriculum import TOPIC_GRAPH
        topic = TOPIC_GRAPH.get(topic_id)
        return topic.name if topic else topic_id.replace("_", " ").title()
    except Exception:
        return topic_id.replace("_", " ").title()


def _fallback(struggling_topics: List[str], trajectory_memory,
              student_profile: Dict) -> Dict:
    """Deterministic recap generation."""
    topic_names = [_get_topic_name(tid) for tid in struggling_topics]
    domain = student_profile.get("target_domain", "tech")
    goal = student_profile.get("target_role", "career goal")

    # Build "what you know" from completed sections
    what_you_know = []
    for tid in struggling_topics:
        attempts = trajectory_memory.quiz_attempts.get(tid, [])
        # Find concepts they got right (any attempt with score > 0)
        concepts = []
        best_score = 0
        for a in attempts:
            score = a.get("score", 0)
            if score > best_score:
                best_score = score
            if score > 0:
                concepts.append(f"Basic understanding of {_get_topic_name(tid)}")
                if score >= 40:
                    concepts.append(f"Some practical application of {_get_topic_name(tid)}")
                if score >= 60:
                    concepts.append(f"Connecting {_get_topic_name(tid)} to {domain}")

        if not concepts:
            concepts = [f"You've engaged with {_get_topic_name(tid)} content", "You've attempted the assessment"]

        what_you_know.append({
            "topic": _get_topic_name(tid),
            "concepts_you_got_right": concepts[:4],
        })

    # Missing link
    if len(struggling_topics) >= 2:
        connection = (
            f"The key connection between {' and '.join(topic_names[:3])} is that they all build on "
            f"foundational {domain} principles. Understanding how they interrelate — rather than "
            f"treating them as separate topics — is crucial for your goal of becoming a {goal}."
        )
    else:
        connection = (
            f"{topic_names[0]} builds on concepts you've already learned. "
            f"The missing link is applying these concepts specifically to {domain} contexts."
        )

    worked_example = (
        f"Imagine you're working as a {goal}. You need to use {topic_names[0]} in a real project. "
        f"Start by identifying the core problem, then apply the principles step by step. "
        f"This connects directly to what you've already learned."
    )

    # Quick reference
    quick_reference = [
        {"concept": name, "definition": f"Key skill for {goal} — review your study notes for specifics"}
        for name in topic_names
    ]
    # Pad with generic concepts
    generic_refs = [
        {"concept": "Prerequisites", "definition": "Ensure all foundational topics are solid"},
        {"concept": "Practice", "definition": "Apply concepts to small exercises before the quiz"},
        {"concept": "Domain Context", "definition": f"Always relate back to {domain} applications"},
        {"concept": "Active Recall", "definition": "Test yourself without looking at notes"},
        {"concept": "Spaced Repetition", "definition": "Review material at increasing intervals"},
    ]
    quick_reference.extend(generic_refs[:max(0, 10 - len(quick_reference))])

    # Checklist
    checklist = [
        f"I can explain {topic_names[0]} in my own words",
        f"I understand why {topic_names[0]} matters for {domain}",
        f"I know the difference between the core concepts in {topic_names[0]}",
        f"I can apply {topic_names[0]} to a simple {domain} example",
        f"I am ready to retake the assessment on {topic_names[0]}",
    ]

    return {
        "title": f"Recap: {', '.join(topic_names[:3])}",
        "root_cause_addressed": f"Consolidating understanding across {len(struggling_topics)} struggling topic(s)",
        "estimated_read_minutes": min(25, 10 + len(struggling_topics) * 5),
        "what_you_know": what_you_know,
        "missing_link": {
            "connection_statement": connection,
            "worked_example": worked_example,
        },
        "quick_reference": quick_reference[:10],
        "checklist": checklist,
    }
