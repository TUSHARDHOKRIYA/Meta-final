"""
EduPath AI — Analytics API
Team KRIYA | OpenEnv Hackathon 2026

REST endpoints for learning analytics and agent transparency:
  GET /api/analytics/rewards/{student_id}      — BKT improvement, quiz scores, intervention counts
  GET /api/analytics/agent-debate/{student_id} — Full council debate history with proposals and conflicts
  GET /api/analytics/observer/{student_id}     — Observer Agent's oversight report
"""
import logging
import json
from typing import Optional
from fastapi import APIRouter, HTTPException

from ai.trajectory_memory import get_trajectory, _trajectories
from db.supabase_client import get_agent_logs, is_configured

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


# ══════════════════════════════════════════════════════════════════════
# GET /rewards/{student_id} — Reward Curve Data
# ══════════════════════════════════════════════════════════════════════

@router.get("/rewards/{student_id}")
async def get_reward_curve(student_id: str):
    """
    Returns time-series data for reward curves:
    - BKT mastery probability per topic over quiz attempts
    - Quiz scores over time
    - Intervention counts and escalation levels
    - Cumulative progress metrics
    """
    mem = _trajectories.get(student_id)

    # ── BKT Improvement Curve ──
    bkt_curves = {}
    quiz_timeline = []

    if mem:
        # Extract quiz attempts with timestamps
        for topic_id, attempts in mem.quiz_attempts.items():
            if not isinstance(attempts, list):
                attempts = [attempts]
            bkt_curves[topic_id] = []
            for i, attempt in enumerate(attempts):
                if isinstance(attempt, dict):
                    score = attempt.get("score", 0)
                    passed = attempt.get("passed", False)
                    bkt_before = attempt.get("bkt_before", 0.3)
                    bkt_after = attempt.get("bkt_after", bkt_before)
                else:
                    score = attempt if isinstance(attempt, (int, float)) else 0
                    passed = score >= 70
                    bkt_before = 0.3 + (i * 0.1)
                    bkt_after = min(0.95, bkt_before + (0.15 if passed else -0.05))

                entry = {
                    "attempt": i + 1,
                    "score": score,
                    "passed": passed,
                    "bkt_before": round(bkt_before, 3),
                    "bkt_after": round(bkt_after, 3),
                }
                bkt_curves[topic_id].append(entry)
                quiz_timeline.append({
                    "topic_id": topic_id,
                    **entry,
                })

        # Sort timeline by attempt number
        quiz_timeline.sort(key=lambda x: x["attempt"])

    # ── Intervention History ──
    intervention_counts = {"CONTINUE": 0, "REVISE_RETRY": 0, "BETTER_RESOURCE": 0, "INTERVENTION": 0}
    intervention_timeline = []

    if mem:
        for entry in mem.intervention_log:
            if isinstance(entry, dict):
                level = entry.get("level", "CONTINUE")
                action = entry.get("action_type", level)
                intervention_counts[action] = intervention_counts.get(action, 0) + 1
                intervention_timeline.append({
                    "level": entry.get("level", 1),
                    "action": action,
                    "topic": entry.get("topic_id", ""),
                    "reason": entry.get("root_cause", ""),
                })

    # ── Section Progress ──
    sections_completed = 0
    topics_mastered = []
    topics_struggling = []

    if mem:
        sections_completed = len(mem.section_history)
        for section in mem.section_history:
            if isinstance(section, dict):
                tid = section.get("topic_id", "")
                result = section.get("result", "")
                if result in ["passed", "mastered"]:
                    topics_mastered.append(tid)
                elif result in ["failed", "struggling"]:
                    topics_struggling.append(tid)

    # ── Flag History ──
    flags = []
    if mem:
        flags = [f if isinstance(f, str) else str(f) for f in mem.flags]

    # ── Cumulative Metrics ──
    total_quizzes = sum(len(v) if isinstance(v, list) else 1 for v in (mem.quiz_attempts.values() if mem else []))
    total_passed = sum(
        1 for attempts in (mem.quiz_attempts.values() if mem else [])
        for a in (attempts if isinstance(attempts, list) else [attempts])
        if (isinstance(a, dict) and a.get("passed")) or (isinstance(a, (int, float)) and a >= 70)
    )

    return {
        "student_id": student_id,
        "summary": {
            "total_quizzes": total_quizzes,
            "total_passed": total_passed,
            "pass_rate": round(total_passed / max(total_quizzes, 1) * 100, 1),
            "sections_completed": sections_completed,
            "topics_mastered": len(topics_mastered),
            "topics_struggling": len(topics_struggling),
            "total_interventions": sum(intervention_counts.values()),
            "bridging_topics_inserted": mem.bridging_topics_inserted if mem else 0,
        },
        "bkt_curves": bkt_curves,
        "quiz_timeline": quiz_timeline,
        "intervention_counts": intervention_counts,
        "intervention_timeline": intervention_timeline,
        "topics_mastered": topics_mastered,
        "topics_struggling": topics_struggling,
        "flags": flags,
        "reward_signal": {
            "description": "Sparse reward: mastery measured only after quiz completion, not during study",
            "mastery_threshold": 0.7,
            "reward_per_topic_mastery": 1.0,
            "penalty_per_intervention": -0.1,
            "cumulative_reward": round(
                len(topics_mastered) * 1.0 - sum(intervention_counts.values()) * 0.1, 2
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# GET /agent-debate/{student_id} — Council Debate History
# ══════════════════════════════════════════════════════════════════════

@router.get("/agent-debate/{student_id}")
async def get_agent_debate(student_id: str):
    """
    Returns the full 6-agent council debate history:
    - Each agent's proposal
    - Conflicts identified and how they were resolved
    - Final roadmap with confidence score
    - Observer Agent's oversight report (if available)
    """
    mem = _trajectories.get(student_id)

    debate_rounds = []

    if mem:
        # ── Round 1: Initial Proposals ──
        if mem.domain_expert_proposal:
            debate_rounds.append({
                "round": 1,
                "agent": "Domain Expert",
                "role": "Proposes 15-25 goal-specific topics with justification",
                "proposal": _summarize_proposal(mem.domain_expert_proposal, "domain_expert"),
                "full_proposal": mem.domain_expert_proposal,
            })

        if mem.prereq_architect_proposal:
            debate_rounds.append({
                "round": 1,
                "agent": "Prerequisite Architect",
                "role": "Defines hard/soft prerequisite DAG",
                "proposal": _summarize_proposal(mem.prereq_architect_proposal, "prereq"),
                "full_proposal": mem.prereq_architect_proposal,
            })

        if mem.feasibility_proposal:
            debate_rounds.append({
                "round": 1,
                "agent": "Feasibility Agent",
                "role": "Enforces time budget constraints",
                "proposal": _summarize_proposal(mem.feasibility_proposal, "feasibility"),
                "full_proposal": mem.feasibility_proposal,
            })

        if mem.student_advocate_proposal:
            debate_rounds.append({
                "round": 1,
                "agent": "Student Advocate",
                "role": "Personalisation overlay — skip proficient topics, match style",
                "proposal": _summarize_proposal(mem.student_advocate_proposal, "advocate"),
                "full_proposal": mem.student_advocate_proposal,
            })

        # ── Round 1.5: Second-round responses (if 2-round debate enabled) ──
        second_round = getattr(mem, "second_round_responses", None)
        if second_round:
            for agent_name, response in second_round.items():
                debate_rounds.append({
                    "round": 2,
                    "agent": agent_name,
                    "role": "Second-round response to other agents' proposals",
                    "proposal": _summarize_proposal(response, "second_round"),
                    "full_proposal": response,
                })

        # ── Conflict Resolution ──
        conflicts = []
        if mem.conflict_resolution:
            cr = mem.conflict_resolution
            conflicts = cr.get("conflicts_found", cr.get("conflicts", []))
            debate_rounds.append({
                "round": 3,
                "agent": "Conflict Matcher",
                "role": "Finds and resolves inter-agent disagreements",
                "proposal": {
                    "conflicts_found": len(conflicts),
                    "conflicts": conflicts[:10],  # First 10
                    "resolution_rules": [
                        "Hard prerequisites ALWAYS beat skip requests",
                        "Interview prep > topic cuts",
                        "Budget constraints > new additions",
                    ],
                },
                "full_proposal": mem.conflict_resolution,
            })

        # ── Final Roadmap ──
        if mem.final_roadmap:
            fr = mem.final_roadmap
            debate_rounds.append({
                "round": 4,
                "agent": "Council Manager",
                "role": "Produces final phased roadmap from all proposals",
                "proposal": {
                    "total_topics": fr.get("total_topics", 0),
                    "total_hours": fr.get("total_hours", 0),
                    "phases": len(fr.get("phases", [])),
                    "confidence": fr.get("confidence_score", 0),
                },
                "full_proposal": mem.final_roadmap,
            })

        # ── Observer Agent Report ──
        observer_report = getattr(mem, "observer_report", None)
        if observer_report:
            debate_rounds.append({
                "round": 5,
                "agent": "Observer Agent",
                "role": "Monitors, analyzes, and explains all agent behavior",
                "proposal": observer_report,
                "full_proposal": observer_report,
            })

    # ── Agent Logs from Supabase ──
    db_logs = get_agent_logs(student_id, limit=100)

    return {
        "student_id": student_id,
        "council_active": bool(mem and mem.final_roadmap),
        "total_debate_rounds": max((r["round"] for r in debate_rounds), default=0),
        "total_agents_participated": len(set(r["agent"] for r in debate_rounds)),
        "debate_rounds": debate_rounds,
        "agent_execution_logs": db_logs[:50],
        "debate_summary": _generate_debate_summary(debate_rounds) if debate_rounds else None,
    }


# ══════════════════════════════════════════════════════════════════════
# GET /observer/{student_id} — Observer Agent Report
# ══════════════════════════════════════════════════════════════════════

@router.get("/observer/{student_id}")
async def get_observer_report(student_id: str):
    """
    Returns the Observer Agent's oversight report analyzing all agent behaviors.
    """
    mem = _trajectories.get(student_id)
    if not mem:
        return {
            "student_id": student_id,
            "status": "no_data",
            "message": "No trajectory data available. Complete profiling and council first.",
        }

    # Generate observer report on-the-fly
    from ai.council.observer_agent import run as observer_run
    report = observer_run(mem)

    return {
        "student_id": student_id,
        "status": "complete",
        "observer_report": report,
    }


# ── Helpers ──

def _summarize_proposal(proposal: dict, agent_type: str) -> dict:
    """Create a human-readable summary of an agent's proposal."""
    if not proposal:
        return {"summary": "No proposal made"}

    if agent_type == "domain_expert":
        topics = proposal.get("topics", proposal.get("proposed_topics", []))
        return {
            "topics_proposed": len(topics),
            "topic_names": [t.get("name", t.get("topic_id", "?")) for t in topics[:8]],
            "total_hours": sum(t.get("hours", 0) for t in topics),
        }
    elif agent_type == "prereq":
        deps = proposal.get("dependencies", proposal.get("prerequisite_dag", []))
        return {
            "dependencies_defined": len(deps),
            "has_circular": proposal.get("has_circular", False),
        }
    elif agent_type == "feasibility":
        return {
            "budget_hours": proposal.get("real_budget", proposal.get("budget_hours", 0)),
            "topics_kept": proposal.get("topics_kept", 0),
            "topics_cut": proposal.get("topics_cut", 0),
            "within_budget": proposal.get("within_budget", True),
        }
    elif agent_type == "advocate":
        return {
            "topics_to_skip": len(proposal.get("skip_topics", [])),
            "confidence_boosters": len(proposal.get("confidence_boosters", [])),
            "style_adjustments": len(proposal.get("style_adjustments", [])),
        }
    elif agent_type == "second_round":
        return {
            "agreed_with": proposal.get("agreed_with", []),
            "disagreed_with": proposal.get("disagreed_with", []),
            "revised_proposal": bool(proposal.get("revised_topics")),
        }
    else:
        return {"keys": list(proposal.keys())[:5]}


def _generate_debate_summary(rounds: list) -> dict:
    """Generate a natural-language summary of the debate."""
    agents = [r["agent"] for r in rounds]
    conflicts = sum(1 for r in rounds if r["agent"] == "Conflict Matcher")

    return {
        "agents_participated": list(set(agents)),
        "total_rounds": max((r["round"] for r in rounds), default=0),
        "had_conflicts": conflicts > 0,
        "narrative": (
            f"The council of {len(set(agents))} agents debated over "
            f"{max((r['round'] for r in rounds), default=0)} rounds. "
            f"{'Conflicts were identified and resolved by the Conflict Matcher. ' if conflicts else 'No major conflicts arose. '}"
            f"The Council Manager produced the final roadmap."
        ),
    }
