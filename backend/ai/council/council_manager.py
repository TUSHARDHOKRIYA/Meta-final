"""
EduPath AI — Council Manager Agent
Team KRIYA | OpenEnv Hackathon 2026

Council Agent 6/6: Orchestrates the full 6-agent council pipeline and
produces the final phased roadmap. Falls back to the existing
generate_roadmap() if any agent fails catastrophically.
"""
import logging
import json
from typing import Dict

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set
from ai.trajectory_memory import get_trajectory

logger = logging.getLogger(__name__)


def run_council(student_profile: Dict) -> Dict:
    """
    Run the full multi-round Roadmap Generation Council.

    Pipeline (Round 1):
        Domain Expert → Prereq Architect → Feasibility Agent →
        Student Advocate → Conflict Matcher → Council Manager

    Round 2 (Debate):
        Domain Expert responds to Advocate's changes
        Feasibility Agent responds to Advocate's changes

    Post-Council:
        Observer Agent produces oversight report

    Args:
        student_profile: Complete StudentProfile dict from the Profiling Agent.

    Returns:
        Final phased roadmap dict.
    """
    student_id = student_profile.get("student_id", "unknown")
    memory = get_trajectory(student_id)

    # Check for context overflow before starting
    memory.check_context_overflow()

    logger.info(f"[Council] Starting roadmap council for {student_id}")

    try:
        # ══════════════════════════════════════════════════════════
        # ROUND 1: Initial Proposals
        # ══════════════════════════════════════════════════════════

        # ── Agent 1: Domain Expert ──
        from ai.council.domain_expert import run as domain_expert_run
        domain_result = domain_expert_run(student_profile)
        memory.record_council_proposal("domain_expert", domain_result)
        logger.info(f"[Council] Domain Expert: {len(domain_result.get('topics', []))} topics proposed")

        topics = domain_result.get("topics", [])
        if not topics:
            raise ValueError("Domain Expert returned no topics")

        # ── Agent 2: Prereq Architect ──
        from ai.council.prereq_architect import run as prereq_run
        prereq_result = prereq_run(topics, student_profile)
        memory.record_council_proposal("prereq_architect", prereq_result)
        logger.info(f"[Council] Prereq Architect: DAG defined for {len(prereq_result.get('ordered_topics', []))} topics")

        # ── Agent 3: Feasibility Agent ──
        from ai.council.feasibility_agent import run as feasibility_run
        feasibility_result = feasibility_run(
            prereq_result.get("ordered_topics", []),
            topics,
            student_profile,
        )
        memory.record_council_proposal("feasibility", feasibility_result)
        logger.info(f"[Council] Feasibility: {len(feasibility_result.get('feasible_topics', []))} topics feasible")

        # ── Agent 4: Student Advocate ──
        from ai.council.student_advocate import run as advocate_run
        advocate_result = advocate_run(
            feasibility_result.get("feasible_topics", []),
            topics,
            student_profile,
        )
        memory.record_council_proposal("student_advocate", advocate_result)
        logger.info(f"[Council] Student Advocate: personalisation applied")

        # ══════════════════════════════════════════════════════════
        # ROUND 2: Second-Round Debate
        # Domain Expert and Feasibility respond to Advocate's changes
        # ══════════════════════════════════════════════════════════

        second_round_responses = _run_second_round(
            domain_result, prereq_result, feasibility_result,
            advocate_result, student_profile
        )
        if second_round_responses:
            memory.second_round_responses = second_round_responses
            logger.info(f"[Council] Round 2: {len(second_round_responses)} agents responded")

        # ── Agent 5: Conflict Matcher ──
        from ai.council.conflict_matcher import run as conflict_run
        all_proposals = {
            "domain_expert": domain_result,
            "prereq_architect": prereq_result,
            "feasibility": feasibility_result,
            "student_advocate": advocate_result,
        }
        # Include second-round responses in conflict resolution
        if second_round_responses:
            all_proposals["second_round"] = second_round_responses

        conflict_result = conflict_run(all_proposals, student_profile)
        memory.record_council_proposal("conflict_matcher", conflict_result)
        logger.info(f"[Council] Conflict Matcher: {conflict_result.get('total_conflicts_found', 0)} conflicts resolved")

        # ── Agent 6: Council Manager (Final Roadmap) ──
        final_roadmap = _produce_final_roadmap(
            conflict_result.get("final_topic_ids", []),
            topics,
            prereq_result.get("ordered_topics", []),
            feasibility_result.get("week_allocations", []),
            student_profile,
            has_second_round=bool(second_round_responses),
        )
        memory.record_council_proposal("council_manager", final_roadmap)
        logger.info(f"[Council] Final roadmap: {final_roadmap.get('total_topics', 0)} topics, confidence={final_roadmap.get('confidence_score', 0)}")

        # ══════════════════════════════════════════════════════════
        # POST-COUNCIL: Observer Agent (Oversight)
        # ══════════════════════════════════════════════════════════
        try:
            from ai.council.observer_agent import run as observer_run
            observer_report = observer_run(memory)
            memory.observer_report = observer_report
            logger.info(f"[Council] Observer: {observer_report.get('overall_assessment', '?')}")
        except Exception as e:
            logger.warning(f"[Council] Observer Agent failed (non-critical): {e}")

        return final_roadmap

    except Exception as e:
        logger.error(f"[Council] Pipeline failed: {e}. Falling back to single-agent roadmap.")
        return _fallback_to_single_agent(student_profile)


def _run_second_round(
    domain_result: Dict, prereq_result: Dict, feasibility_result: Dict,
    advocate_result: Dict, student_profile: Dict
) -> Dict:
    """
    Run the second round of the council debate.
    Domain Expert and Feasibility Agent respond to Student Advocate's changes.
    """
    responses = {}

    # Build the context of what the Advocate changed
    advocate_skips = advocate_result.get("skip_topics", [])
    advocate_boosters = advocate_result.get("confidence_boosters", [])
    advocate_adjustments = advocate_result.get("style_adjustments", [])

    if not (advocate_skips or advocate_boosters or advocate_adjustments):
        logger.info("[Council] R2: Advocate made no changes — skipping Round 2")
        return responses

    # ── Domain Expert responds ──
    try:
        de_response = _second_round_response(
            agent_name="Domain Expert",
            own_proposal=domain_result,
            advocate_changes={
                "skip_topics": advocate_skips,
                "confidence_boosters": advocate_boosters,
                "adjustments": advocate_adjustments,
            },
            all_proposals={
                "prereq": {"ordered_count": len(prereq_result.get("ordered_topics", []))},
                "feasibility": {"topics_kept": feasibility_result.get("topics_kept", 0)},
            },
            student_profile=student_profile,
        )
        responses["Domain Expert"] = de_response
        logger.info("[Council] R2: Domain Expert responded")
    except Exception as e:
        logger.warning(f"[Council] R2: Domain Expert failed: {e}")

    # ── Feasibility Agent responds ──
    try:
        fa_response = _second_round_response(
            agent_name="Feasibility Agent",
            own_proposal=feasibility_result,
            advocate_changes={
                "skip_topics": advocate_skips,
                "confidence_boosters": advocate_boosters,
                "adjustments": advocate_adjustments,
            },
            all_proposals={
                "domain_expert": {"topics_count": len(domain_result.get("topics", []))},
                "prereq": {"ordered_count": len(prereq_result.get("ordered_topics", []))},
            },
            student_profile=student_profile,
        )
        responses["Feasibility Agent"] = fa_response
        logger.info("[Council] R2: Feasibility Agent responded")
    except Exception as e:
        logger.warning(f"[Council] R2: Feasibility Agent failed: {e}")

    return responses


def _second_round_response(
    agent_name: str, own_proposal: Dict, advocate_changes: Dict,
    all_proposals: Dict, student_profile: Dict,
) -> Dict:
    """Generate a second-round debate response (deterministic)."""
    skip_topics = advocate_changes.get("skip_topics", [])
    boosters = advocate_changes.get("confidence_boosters", [])

    agreed_with = []
    disagreed_with = []
    defended = []

    # Domain Expert logic
    if agent_name == "Domain Expert":
        for skip in skip_topics:
            topic_id = skip if isinstance(skip, str) else skip.get("topic_id", "")
            # Agree with skips for proficient topics, disagree for prerequisites
            if topic_id in [t.get("id") for t in own_proposal.get("topics", [])
                           if t.get("is_prerequisite_for")]:
                disagreed_with.append({
                    "agent": "Student Advocate",
                    "point": f"Skip {topic_id}",
                    "reason": "This topic is a hard prerequisite — cannot be skipped",
                })
                defended.append(f"Maintained {topic_id} as essential prerequisite")
            else:
                agreed_with.append({
                    "agent": "Student Advocate",
                    "point": f"Skip {topic_id} (student is proficient)",
                })

        for booster in boosters:
            agreed_with.append({
                "agent": "Student Advocate",
                "point": f"Added confidence booster: {booster.get('topic_id', booster) if isinstance(booster, dict) else booster}",
            })

    # Feasibility Agent logic
    elif agent_name == "Feasibility Agent":
        budget = own_proposal.get("real_budget", own_proposal.get("budget_hours", 0))
        saved_hours = sum(
            8 for s in skip_topics  # Assume ~8h per skipped topic
        )

        if saved_hours > 0:
            agreed_with.append({
                "agent": "Student Advocate",
                "point": f"Skipping {len(skip_topics)} topics saves ~{saved_hours}h, improving budget fit",
            })

        booster_hours = len(boosters) * 4  # ~4h per booster
        if booster_hours > budget * 0.15:
            disagreed_with.append({
                "agent": "Student Advocate",
                "point": f"Too many confidence boosters ({booster_hours}h)",
                "reason": f"Exceeds 15% of budget ({budget}h). Recommend max 2 boosters.",
            })
        else:
            agreed_with.append({
                "agent": "Student Advocate",
                "point": f"Confidence boosters ({booster_hours}h) fit within budget",
            })

    return {
        "agent": agent_name,
        "round": 2,
        "agreed_with": agreed_with,
        "disagreed_with": disagreed_with,
        "revised_topics": [],  # No topic list changes in deterministic mode
        "defended_positions": defended,
        "summary": (
            f"{agent_name} reviewed Advocate's changes: "
            f"agreed with {len(agreed_with)} points, "
            f"disagreed with {len(disagreed_with)} points."
        ),
    }




def _produce_final_roadmap(
    final_topic_ids: list,
    topics: list,
    ordered_topics: list,
    week_allocations: list,
    student_profile: Dict,
    has_second_round: bool = False,
) -> Dict:
    """Produce the final structured roadmap from council results."""
    topic_map = {t["id"]: t for t in topics}
    order_map = {t["id"]: t for t in ordered_topics}
    week_map = {w["topic_id"]: w for w in week_allocations}

    # Organise into 4 phases
    phases = {
        "Foundation": {"name": "Foundation", "weeks": "Weeks 1-3", "total_hours": 0, "topics": []},
        "Core": {"name": "Core", "weeks": "Weeks 4-7", "total_hours": 0, "topics": []},
        "Specialization": {"name": "Specialization", "weeks": "Weeks 8-10", "total_hours": 0, "topics": []},
        "Interview Prep": {"name": "Interview Prep", "weeks": "Weeks 11-12", "total_hours": 0, "topics": []},
    }
    phase_key_map = {
        "foundation": "Foundation",
        "core": "Core",
        "specialization": "Specialization",
        "interview_prep": "Interview Prep",
    }

    for tid in final_topic_ids:
        detail = topic_map.get(tid, {})
        order_detail = order_map.get(tid, {})

        phase_name = phase_key_map.get(detail.get("phase", "core"), "Core")
        hours = detail.get("estimated_hours", 8)

        topic_entry = {
            "id": tid,
            "name": detail.get("name", tid),
            "why": detail.get("why", ""),
            "estimated_hours": hours,
            "difficulty": detail.get("difficulty", 3),
            "prerequisites": order_detail.get("prerequisites", []),
            "phase": detail.get("phase", "core"),
            "is_interview_prep": detail.get("is_interview_prep", False),
            "is_bridging": False,
        }

        phases[phase_name]["topics"].append(topic_entry)
        phases[phase_name]["total_hours"] += hours

    # Remove empty phases
    final_phases = [p for p in phases.values() if p["topics"]]

    # Compute confidence score
    confidence = _compute_confidence(
        total_topics_proposed=len(topics),
        total_topics_final=len(final_topic_ids),
        student_profile=student_profile,
    )

    total_hours = sum(p["total_hours"] for p in final_phases)
    total_weeks = student_profile.get("deadline_weeks", 12)

    # ── Build frontend-compatible "weeks" array from phases ──
    # The frontend renders roadmap.weeks, so we must provide it.
    weeks = []
    week_num = 1
    goal = student_profile.get("target_role", "")
    for phase in final_phases:
        for topic in phase["topics"]:
            weeks.append({
                "weekNumber": week_num,
                "title": topic.get("name", topic.get("id", f"Week {week_num}")),
                "learningObjectives": [
                    f"Master {topic.get('name', topic.get('id', ''))}",
                    topic.get("why", f"Essential for {goal}"),
                ],
                "skillsCovered": [topic.get("id", topic.get("name", ""))],
                "estimatedHours": topic.get("estimated_hours", 8),
                "actionItems": [
                    f"Study {topic.get('name', '')}",
                    "Complete practice exercises",
                    "Review key concepts",
                ],
                "resources": [],  # Will be filled by the learning loop
                "mini_project": {
                    "title": f"Apply {topic.get('name', '')}",
                    "description": f"Build a small project using {topic.get('name', '')}",
                    "requirements": [topic.get("name", "")],
                } if week_num % 3 == 0 else None,
                "phase": phase["name"],
                "is_bridge": topic.get("is_bridging", False),
            })
            week_num += 1

    # ── Build capstone projects ──
    all_skill_ids = [t.get("id", t.get("name", "")) for t in
                     [topic for phase in final_phases for topic in phase["topics"]]]
    capstone_projects = [
        {
            "title": f"Capstone: {goal} Portfolio Project",
            "description": f"Build a comprehensive project showcasing all your skills for {goal}",
            "expected_output": "Complete project with documentation and deployment",
            "requirements": all_skill_ids[:5] + ["Documentation", "Testing"],
            "skills_tested": all_skill_ids[:8],
        },
        {
            "title": f"Capstone: {goal} Interview Preparation",
            "description": f"Prepare and practice for {goal} interviews with mock projects",
            "expected_output": "Portfolio website with project demos and technical write-ups",
            "requirements": ["Portfolio", "Technical writing", "Project demos"],
            "skills_tested": all_skill_ids[:6],
        },
    ]

    return {
        "version": 1,
        "student_id": student_profile.get("student_id", "unknown"),
        "goal": goal,
        "domain": student_profile.get("target_domain", "tech"),
        "target_role": goal,
        "phases": final_phases,
        "weeks": weeks,
        "capstone_projects": capstone_projects,
        "total_hours": total_hours,
        "total_weeks": total_weeks,
        "weekly_hours": student_profile.get("weekly_hours", 10),
        "total_topics": len(final_topic_ids),
        "confidence_score": confidence,
        "interview_ready_by_week": total_weeks if student_profile.get("has_interview") else 0,
        "council_metadata": {
            "agents_used": 8 if has_second_round else 6,
            "debate_rounds": 2 if has_second_round else 1,
            "has_observer": True,
            "pipeline": (
                "domain_expert -> prereq_architect -> feasibility -> student_advocate -> "
                "[R2: domain_expert, feasibility respond] -> conflict_matcher -> council_manager -> observer"
                if has_second_round else
                "domain_expert -> prereq_architect -> feasibility -> student_advocate -> conflict_matcher -> council_manager -> observer"
            ),
        },
    }


def _compute_confidence(total_topics_proposed: int, total_topics_final: int, student_profile: Dict) -> float:
    """Compute roadmap confidence score."""
    score = 1.0

    # Subtract 0.1 if >20% topics cut
    if total_topics_proposed > 0:
        cut_pct = 1 - (total_topics_final / total_topics_proposed)
        if cut_pct > 0.2:
            score -= 0.1

    # Subtract 0.1 if low confidence
    if student_profile.get("confidence_level") == "low":
        score -= 0.1

    # Subtract 0.1 if deadline < 8 weeks
    if student_profile.get("deadline_weeks", 12) < 8:
        score -= 0.1

    # Add 0.05 if strong background match
    verified = student_profile.get("verified_skills", {})
    proficient_count = sum(1 for v in verified.values() if v == "proficient")
    if proficient_count >= 3:
        score += 0.05

    return round(max(0.1, min(1.0, score)), 2)


def _fallback_to_single_agent(student_profile: Dict) -> Dict:
    """Fall back to the existing single-agent roadmap generator."""
    from ai.roadmap_generator import generate_roadmap

    return generate_roadmap(
        target_field=student_profile.get("target_domain", "tech"),
        learning_goal=student_profile.get("target_role", ""),
        current_skills=list(student_profile.get("verified_skills", {}).keys()),
        skill_levels=student_profile.get("verified_skills", {}),
        weekly_hours=student_profile.get("weekly_hours", 10),
        duration_weeks=student_profile.get("deadline_weeks", 12),
    )
