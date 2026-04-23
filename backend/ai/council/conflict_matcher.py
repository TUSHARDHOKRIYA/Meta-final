"""
EduPath AI — Conflict Matcher Agent
Team KRIYA | OpenEnv Hackathon 2026

Council Agent 5/6: Finds and resolves disagreements between all prior
council agents. Hard rule precedence:
  - Hard prerequisites ALWAYS beat skip requests
  - Interview prep ALWAYS beats feasibility cuts  
  - Budget constraints ALWAYS beat domain expert additions
"""
import logging
from typing import Dict, List

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


def run(all_proposals: Dict, student_profile: Dict) -> Dict:
    """
    Find and resolve all conflicts between council agent proposals.

    Args:
        all_proposals: Dict with keys 'domain_expert', 'prereq_architect',
                       'feasibility', 'student_advocate' — each agent's output.
        student_profile: StudentProfile dict.

    Returns:
        Dict with conflicts, final_topic_ids, resolution_summary.
    """
    if not is_api_key_set():
        return _fallback(all_proposals, student_profile)

    system_prompt = get_prompt("conflict_matcher")

    user_prompt = f"""STUDENT PROFILE:
{_profile_summary(student_profile)}

DOMAIN EXPERT PROPOSAL:
{_safe_json(all_proposals.get('domain_expert', {}))}

PREREQ ARCHITECT PROPOSAL:
{_safe_json(all_proposals.get('prereq_architect', {}))}

FEASIBILITY PROPOSAL:
{_safe_json(all_proposals.get('feasibility', {}))}

STUDENT ADVOCATE PROPOSAL:
{_safe_json(all_proposals.get('student_advocate', {}))}

Find ALL conflicts and resolve each. Apply hard rule precedence."""

    try:
        result = generate_json_with_retry(system_prompt, user_prompt)
        if result and "final_topic_ids" in result:
            logger.info(f"Conflict Matcher: {result.get('total_conflicts_found', 0)} conflicts resolved")
            return result
    except Exception as e:
        logger.error(f"Conflict Matcher failed: {e}")

    return _fallback(all_proposals, student_profile)


def _profile_summary(profile: Dict) -> str:
    return (
        f"Target: {profile.get('target_role', '?')}\n"
        f"Has Interview: {profile.get('has_interview', False)}\n"
        f"Budget: {profile.get('budget', 'free_only')}\n"
        f"Confidence: {profile.get('confidence_level', 'medium')}"
    )


def _safe_json(data: Dict) -> str:
    import json
    try:
        return json.dumps(data, indent=2, default=str)[:3000]
    except Exception:
        return str(data)[:3000]


def _fallback(all_proposals: Dict, student_profile: Dict) -> Dict:
    """Deterministic conflict resolution using hard rule precedence."""
    domain = all_proposals.get("domain_expert", {})
    prereq = all_proposals.get("prereq_architect", {})
    feasibility = all_proposals.get("feasibility", {})
    advocate = all_proposals.get("student_advocate", {})

    # Start with feasible topics (budget wins)
    feasible_ids = set(feasibility.get("feasible_topics", []))

    # Get all proposed topic IDs
    all_topic_ids = [t["id"] for t in domain.get("topics", [])]

    # Get skip requests from advocate
    skip_ids = set(advocate.get("skip", []))

    # Get topics with hard prerequisites
    hard_prereq_ids = set()
    for t in prereq.get("ordered_topics", []):
        if t.get("dependency_type") == "hard":
            hard_prereq_ids.add(t["id"])
            for p in t.get("prerequisites", []):
                hard_prereq_ids.add(p)

    # Get interview prep topics
    interview_ids = set()
    for t in domain.get("topics", []):
        if t.get("is_interview_prep"):
            interview_ids.add(t["id"])

    # Resolve conflicts
    conflicts = []
    final_ids = []

    for tid in all_topic_ids:
        in_feasible = tid in feasible_ids
        in_skip = tid in skip_ids
        is_hard_prereq = tid in hard_prereq_ids
        is_interview = tid in interview_ids

        # Conflict: skip vs hard prerequisite → prereq wins
        if in_skip and is_hard_prereq:
            conflicts.append({
                "id": f"conflict_{len(conflicts)+1}",
                "type": "skip",
                "agents_in_conflict": ["student_advocate", "prereq_architect"],
                "description": f"{tid}: advocate wants to skip but it is a hard prerequisite",
                "topic_ids_involved": [tid],
                "resolution": "Keep — hard prerequisites cannot be skipped",
                "winning_agent": "prereq_architect",
            })
            final_ids.append(tid)
            continue

        # Conflict: feasibility cut vs interview prep → interview wins
        if not in_feasible and is_interview:
            conflicts.append({
                "id": f"conflict_{len(conflicts)+1}",
                "type": "inclusion",
                "agents_in_conflict": ["feasibility", "domain_expert"],
                "description": f"{tid}: cut by feasibility but is interview prep",
                "topic_ids_involved": [tid],
                "resolution": "Keep — interview prep cannot be cut",
                "winning_agent": "domain_expert",
            })
            final_ids.append(tid)
            continue

        # Normal: in feasible and not skipped
        if in_feasible and not in_skip:
            final_ids.append(tid)
        elif in_feasible and in_skip:
            # Skip wins for non-essential topics
            conflicts.append({
                "id": f"conflict_{len(conflicts)+1}",
                "type": "skip",
                "agents_in_conflict": ["student_advocate", "domain_expert"],
                "description": f"{tid}: student already proficient",
                "topic_ids_involved": [tid],
                "resolution": "Skip — student is proficient",
                "winning_agent": "student_advocate",
            })
        # Not feasible, not interview → stays cut

    # Respect ordering from prereq architect
    ordered = prereq.get("ordered_topics", [])
    order_map = {t["id"]: t.get("order_position", 999) for t in ordered}
    final_ids.sort(key=lambda tid: order_map.get(tid, 999))

    return {
        "conflicts": conflicts,
        "total_conflicts_found": len(conflicts),
        "final_topic_ids": final_ids,
        "resolution_summary": f"Resolved {len(conflicts)} conflicts. {len(final_ids)} topics in final roadmap.",
    }
