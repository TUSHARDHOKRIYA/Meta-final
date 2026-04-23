"""
EduPath AI — Observer Agent
Team KRIYA | OpenEnv Hackathon 2026

Fleet AI / Scalable Oversight agent that monitors, analyzes, and explains
the behavior of all other agents in the pipeline. Generates a structured
oversight report covering:
  1. Agent behavior analysis — what each agent did and why
  2. Anomaly detection — unusual patterns or potential issues
  3. Confidence assessment — how trustworthy each agent's output is
  4. Recommendations — suggestions for improving the pipeline
"""
import logging
import json
import os
from typing import Optional

logger = logging.getLogger(__name__)


def run(trajectory_memory) -> dict:
    """
    Generate an Observer Agent oversight report analyzing all agent behaviors.

    Args:
        trajectory_memory: TrajectoryMemory instance with all agent proposals.

    Returns:
        Structured oversight report dict.
    """
    # Try LLM-based analysis first
    if os.getenv("API_BASE_URL"):
        try:
            return _run_llm(trajectory_memory)
        except Exception as e:
            logger.warning(f"Observer LLM failed, using deterministic: {e}")

    return _run_deterministic(trajectory_memory)


def _run_llm(mem) -> dict:
    """LLM-powered oversight analysis."""
    from ai.llm_client import generate_json
    from ai.agent_prompts import OBSERVER_AGENT_PROMPT

    # Build context from all agent proposals
    context = {
        "student_profile": mem.student_profile,
        "domain_expert": _safe_summary(mem.domain_expert_proposal),
        "prereq_architect": _safe_summary(mem.prereq_architect_proposal),
        "feasibility": _safe_summary(mem.feasibility_proposal),
        "student_advocate": _safe_summary(mem.student_advocate_proposal),
        "conflict_resolution": _safe_summary(mem.conflict_resolution),
        "final_roadmap_topics": (
            mem.final_roadmap.get("total_topics", 0) if mem.final_roadmap else 0
        ),
        "quiz_attempts": len(mem.quiz_attempts),
        "interventions": len(mem.intervention_log),
        "flags": mem.flags,
    }

    prompt = (
        "Analyze all agent behaviors in this learning pipeline and produce an oversight report.\n\n"
        f"Agent proposals and state:\n{json.dumps(context, indent=2, default=str)}\n\n"
        "Return JSON with: agent_analyses (list of per-agent analysis), anomalies (list), "
        "confidence_scores (dict), recommendations (list), overall_assessment (str)."
    )

    result = generate_json(OBSERVER_AGENT_PROMPT, prompt)

    if result:
        result["source"] = "llm"
        return result

    return _run_deterministic(mem)


def _run_deterministic(mem) -> dict:
    """Deterministic oversight analysis based on heuristics."""
    analyses = []
    anomalies = []
    confidence_scores = {}
    recommendations = []

    # ── Analyze Domain Expert ──
    if mem.domain_expert_proposal:
        de = mem.domain_expert_proposal
        topics = de.get("topics", de.get("proposed_topics", []))
        n_topics = len(topics)
        total_hours = sum(t.get("hours", 0) for t in topics)

        confidence_scores["domain_expert"] = 0.9 if 10 <= n_topics <= 30 else 0.6
        analysis = {
            "agent": "Domain Expert",
            "action": f"Proposed {n_topics} topics totaling {total_hours}h",
            "behavior": "normal" if 10 <= n_topics <= 30 else "unusual",
            "reasoning": (
                "Topic count within expected range (10-30)"
                if 10 <= n_topics <= 30
                else f"Topic count {n_topics} is outside expected range"
            ),
        }
        analyses.append(analysis)

        if n_topics > 25:
            anomalies.append({
                "agent": "Domain Expert",
                "type": "HIGH_TOPIC_COUNT",
                "severity": "low",
                "detail": f"Proposed {n_topics} topics — may be overwhelming",
            })
        if total_hours > 200:
            anomalies.append({
                "agent": "Domain Expert",
                "type": "EXCESSIVE_HOURS",
                "severity": "medium",
                "detail": f"Total {total_hours}h exceeds typical budget",
            })
    else:
        confidence_scores["domain_expert"] = 0.0
        analyses.append({
            "agent": "Domain Expert",
            "action": "No proposal made",
            "behavior": "missing",
            "reasoning": "Agent did not produce a proposal — possible LLM failure",
        })

    # ── Analyze Prerequisite Architect ──
    if mem.prereq_architect_proposal:
        pa = mem.prereq_architect_proposal
        deps = pa.get("dependencies", pa.get("prerequisite_dag", []))
        has_circular = pa.get("has_circular", False)

        confidence_scores["prereq_architect"] = 0.3 if has_circular else 0.9
        analyses.append({
            "agent": "Prerequisite Architect",
            "action": f"Defined {len(deps)} prerequisite relationships",
            "behavior": "critical_error" if has_circular else "normal",
            "reasoning": (
                "CIRCULAR DEPENDENCY DETECTED — roadmap may be invalid"
                if has_circular
                else "DAG is well-formed with no cycles"
            ),
        })

        if has_circular:
            anomalies.append({
                "agent": "Prerequisite Architect",
                "type": "CIRCULAR_DEPENDENCY",
                "severity": "critical",
                "detail": "Circular dependency in prerequisite DAG",
            })
    else:
        confidence_scores["prereq_architect"] = 0.0

    # ── Analyze Feasibility Agent ──
    if mem.feasibility_proposal:
        fa = mem.feasibility_proposal
        kept = fa.get("topics_kept", 0)
        cut = fa.get("topics_cut", 0)
        within = fa.get("within_budget", True)

        confidence_scores["feasibility"] = 0.9 if within else 0.7
        analyses.append({
            "agent": "Feasibility Agent",
            "action": f"Kept {kept} topics, cut {cut}",
            "behavior": "normal" if within else "budget_exceeded",
            "reasoning": (
                "Roadmap fits within time budget"
                if within
                else "Budget exceeded — some topics may not be completable"
            ),
        })

        if cut > kept:
            anomalies.append({
                "agent": "Feasibility Agent",
                "type": "EXCESSIVE_CUTS",
                "severity": "medium",
                "detail": f"Cut {cut} topics vs kept {kept} — original plan may have been too ambitious",
            })
    else:
        confidence_scores["feasibility"] = 0.0

    # ── Analyze Student Advocate ──
    if mem.student_advocate_proposal:
        sa = mem.student_advocate_proposal
        skips = len(sa.get("skip_topics", []))
        boosters = len(sa.get("confidence_boosters", []))

        confidence_scores["student_advocate"] = 0.85
        analyses.append({
            "agent": "Student Advocate",
            "action": f"Recommended {skips} skips, {boosters} confidence boosters",
            "behavior": "normal",
            "reasoning": "Personalisation overlay applied successfully",
        })
    else:
        confidence_scores["student_advocate"] = 0.0

    # ── Analyze Conflict Matcher ──
    if mem.conflict_resolution:
        cr = mem.conflict_resolution
        conflicts = cr.get("conflicts_found", cr.get("conflicts", []))
        n_conflicts = len(conflicts) if isinstance(conflicts, list) else conflicts

        confidence_scores["conflict_matcher"] = 0.9
        analyses.append({
            "agent": "Conflict Matcher",
            "action": f"Resolved {n_conflicts} conflicts",
            "behavior": "normal" if n_conflicts < 10 else "high_conflict",
            "reasoning": (
                f"{n_conflicts} inter-agent disagreements resolved using hard-rule precedence"
            ),
        })

        if n_conflicts > 5:
            anomalies.append({
                "agent": "Conflict Matcher",
                "type": "HIGH_CONFLICT_COUNT",
                "severity": "low",
                "detail": f"{n_conflicts} conflicts suggest significant disagreement among agents",
            })
    else:
        confidence_scores["conflict_matcher"] = 0.0

    # ── Analyze Council Manager ──
    if mem.final_roadmap:
        fr = mem.final_roadmap
        conf = fr.get("confidence_score", 0)
        phases = len(fr.get("phases", []))

        confidence_scores["council_manager"] = conf
        analyses.append({
            "agent": "Council Manager",
            "action": f"Produced {phases}-phase roadmap with confidence {conf}",
            "behavior": "normal" if conf >= 0.7 else "low_confidence",
            "reasoning": (
                "High confidence — all agents contributed"
                if conf >= 0.7
                else "Low confidence — some agent proposals may have been missing"
            ),
        })
    else:
        confidence_scores["council_manager"] = 0.0

    # ── Analyze Adaptation System ──
    n_interventions = len(mem.intervention_log)
    if n_interventions > 0:
        levels = [e.get("level", 1) if isinstance(e, dict) else 1 for e in mem.intervention_log]
        max_level = max(levels)
        analyses.append({
            "agent": "Adaptation Agent",
            "action": f"{n_interventions} interventions, max level {max_level}",
            "behavior": "escalating" if max_level >= 3 else "normal",
            "reasoning": (
                f"Student required {n_interventions} interventions, "
                f"{'reaching critical intervention level' if max_level >= 4 else 'within normal bounds'}"
            ),
        })

        if max_level >= 4:
            anomalies.append({
                "agent": "Adaptation Agent",
                "type": "CRITICAL_INTERVENTION",
                "severity": "high",
                "detail": "Student reached intervention level 4 — fundamental understanding gaps",
            })

    # ── Generate Recommendations ──
    if not mem.domain_expert_proposal:
        recommendations.append("Run the council — no proposals exist yet")
    if anomalies:
        for a in anomalies:
            if a["severity"] == "critical":
                recommendations.append(f"CRITICAL: Fix {a['type']} in {a['agent']}")
    if n_interventions > 3:
        recommendations.append("Consider revising the roadmap — student is struggling repeatedly")
    if not mem.student_profile:
        recommendations.append("Complete student profiling before running the council")

    # ── Overall Assessment ──
    avg_confidence = (
        sum(confidence_scores.values()) / max(len(confidence_scores), 1)
    )
    n_anomalies_critical = sum(1 for a in anomalies if a["severity"] in ["critical", "high"])

    if n_anomalies_critical > 0:
        overall = "WARNING: Critical issues detected — review anomalies"
    elif avg_confidence >= 0.8:
        overall = "HEALTHY: All agents operating normally with high confidence"
    elif avg_confidence >= 0.5:
        overall = "MODERATE: Some agents produced lower-confidence results"
    else:
        overall = "LOW: Most agents did not produce proposals — pipeline may be incomplete"

    return {
        "source": "deterministic",
        "agent_analyses": analyses,
        "anomalies": anomalies,
        "confidence_scores": confidence_scores,
        "average_confidence": round(avg_confidence, 3),
        "recommendations": recommendations,
        "overall_assessment": overall,
        "pipeline_completeness": {
            "profiling": bool(mem.student_profile),
            "domain_expert": bool(mem.domain_expert_proposal),
            "prereq_architect": bool(mem.prereq_architect_proposal),
            "feasibility": bool(mem.feasibility_proposal),
            "student_advocate": bool(mem.student_advocate_proposal),
            "conflict_matcher": bool(mem.conflict_resolution),
            "council_manager": bool(mem.final_roadmap),
            "learning_active": len(mem.section_history) > 0,
            "adaptation_active": len(mem.intervention_log) > 0,
        },
    }


def _safe_summary(proposal: dict) -> str:
    """Safely summarize a proposal for LLM context."""
    if not proposal:
        return "NOT SUBMITTED"
    try:
        return json.dumps(proposal, default=str)[:500]
    except Exception:
        return str(proposal)[:500]
