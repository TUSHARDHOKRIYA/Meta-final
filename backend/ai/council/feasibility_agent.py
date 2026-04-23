"""
EduPath AI — Feasibility Agent
Team KRIYA | OpenEnv Hackathon 2026

Council Agent 3/6: Enforces time budget constraints. Computes
real_budget = weekly_hours × deadline_weeks × 0.9, cuts topics
by priority, and assigns week allocations. Never cuts hard
prerequisites or interview prep.
"""
import logging
from typing import Dict, List

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(ordered_topics: List[Dict], topics: List[Dict], student_profile: Dict) -> Dict:
    """
    Enforce time budget and assign week allocations.

    Args:
        ordered_topics: Ordered topic dicts from the Prereq Architect.
        topics: Original topic dicts from the Domain Expert (with hours).
        student_profile: StudentProfile dict.

    Returns:
        Dict with feasible_topics, cut_topics, week_allocations, etc.
    """
    if not is_api_key_set():
        return _fallback(ordered_topics, topics, student_profile)

    system_prompt = get_prompt("feasibility")

    # Build topic detail map
    topic_map = {t["id"]: t for t in topics}
    topic_summary = "\n".join(
        f"  - {t['id']}: hours={topic_map.get(t['id'], {}).get('estimated_hours', 8)}, "
        f"phase={topic_map.get(t['id'], {}).get('phase', '?')}, "
        f"is_interview_prep={topic_map.get(t['id'], {}).get('is_interview_prep', False)}, "
        f"prereqs={t.get('prerequisites', [])}, dep_type={t.get('dependency_type', 'soft')}"
        for t in ordered_topics
    )

    weekly_hours = student_profile.get("weekly_hours", 10)
    deadline_weeks = student_profile.get("deadline_weeks", 12)
    total_budget = weekly_hours * deadline_weeks
    real_budget = total_budget * 0.9

    user_prompt = f"""STUDENT PROFILE:
Weekly Hours: {weekly_hours}
Deadline Weeks: {deadline_weeks}
Total Budget: {total_budget} hours
Real Budget (90%): {real_budget} hours
Has Interview: {student_profile.get('has_interview', False)}
Interview Weeks: {student_profile.get('interview_weeks', 0)}

ORDERED TOPICS:
{topic_summary}

Enforce the time budget. Cut topics ruthlessly but never cut hard prerequisites or interview prep. Assign week allocations."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result and "feasible_topics" in result:
            logger.info(f"Feasibility Agent: {len(result['feasible_topics'])} topics feasible, {len(result.get('cut_topics', []))} cut")
            return result
    except Exception as e:
        logger.error(f"Feasibility Agent failed: {e}")

    return _fallback(ordered_topics, topics, student_profile)


def _fallback(ordered_topics: List[Dict], topics: List[Dict], student_profile: Dict) -> Dict:
    """Deterministic fallback: fit topics into budget greedily."""
    topic_map = {t["id"]: t for t in topics}

    weekly_hours = student_profile.get("weekly_hours", 10)
    deadline_weeks = student_profile.get("deadline_weeks", 12)
    real_budget = weekly_hours * deadline_weeks * 0.9

    # Separate protected vs cuttable topics
    protected_ids = set()
    for t in ordered_topics:
        tid = t["id"]
        topic_detail = topic_map.get(tid, {})
        # Protect: hard prerequisites for other topics, interview prep
        if topic_detail.get("is_interview_prep"):
            protected_ids.add(tid)
        if t.get("dependency_type") == "hard":
            protected_ids.add(tid)
            for p in t.get("prerequisites", []):
                protected_ids.add(p)

    # Greedily fit topics
    feasible = []
    cut = []
    hours_used = 0
    week_allocations = []
    current_week = 1

    for t in ordered_topics:
        tid = t["id"]
        hours = topic_map.get(tid, {}).get("estimated_hours", 8)

        if hours_used + hours <= real_budget or tid in protected_ids:
            feasible.append(tid)
            week_start = current_week
            weeks_needed = max(1, round(hours / weekly_hours))
            week_end = week_start + weeks_needed - 1
            week_allocations.append({
                "topic_id": tid,
                "week_start": week_start,
                "week_end": week_end,
                "hours_allocated": hours,
            })
            current_week = week_end + 1
            hours_used += hours
        else:
            cut.append({
                "id": tid,
                "reason": f"Budget exceeded ({hours_used + hours:.0f} > {real_budget:.0f} hours)",
            })

    return {
        "feasible_topics": feasible,
        "cut_topics": cut,
        "week_allocations": week_allocations,
        "total_hours_planned": round(hours_used, 1),
        "budget_used_percent": round((hours_used / real_budget) * 100, 1) if real_budget > 0 else 0,
        "feasibility_rationale": f"Fitted {len(feasible)} topics into {real_budget:.0f}h budget. Cut {len(cut)} lower-priority topics.",
    }
