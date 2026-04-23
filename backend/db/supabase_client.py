"""
EduPath AI — Supabase Client & Cloud Sync
Team KRIYA | Meta Hackathon 2026

Provides a Supabase client wrapper for cloud persistence of student
profiles, quiz history, project evaluations, and roadmaps. Handles
upserts, reads, and graceful fallback when Supabase is unavailable.
"""
import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_supabase_client = None
_supabase_checked = False


def _get_client():
    """Get Supabase client (lazy init). Caches result to avoid repeated warnings."""
    global _supabase_client, _supabase_checked
    if _supabase_checked:
        return _supabase_client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")

    if not url or not key:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set. Database sync disabled.")
        _supabase_checked = True
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        logger.info("Supabase client initialized successfully.")
        _supabase_checked = True
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        _supabase_checked = True
        return None


def is_configured() -> bool:
    """Check if Supabase is configured."""
    return bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_KEY"))


# ── Student Operations ──

def upsert_student(student_data: dict) -> bool:
    """Insert or update a student profile in Supabase."""
    client = _get_client()
    if not client:
        return False

    try:
        # Prepare data for Supabase (convert non-serializable fields)
        db_data = {
            "id": student_data.get("id"),
            "name": student_data.get("name", ""),
            "email": student_data.get("email", ""),
            "target_field": student_data.get("target_field", "tech"),
            "learning_goal": student_data.get("learning_goal", ""),
            "job_description": student_data.get("job_description", ""),
            "weekly_hours": student_data.get("weekly_hours", 10),
            "job_readiness_score": student_data.get("job_readiness_score", 0),
            "quiz_streak": student_data.get("quiz_streak", 0),
            "resume_skills": json.dumps(student_data.get("resume_skills", [])),
            "self_assessed_skills": json.dumps(student_data.get("self_assessed_skills", [])),
            "jd_required_skills": json.dumps(student_data.get("jd_required_skills", [])),
            "completed_topics": json.dumps(student_data.get("completed_topics", [])),
            "completed_projects": json.dumps(student_data.get("completed_projects", [])),
            "topics_studied": json.dumps(student_data.get("topics_studied", [])),
            "clicked_resource_links": json.dumps(student_data.get("clicked_resource_links", {})),
            "badges": json.dumps(student_data.get("badges", [])),
            "mastery_probabilities": json.dumps(student_data.get("mastery_probabilities", {})),
            "onboarding_complete": True,
        }

        client.table("students").upsert(db_data).execute()
        logger.info(f"Student {db_data['id']} synced to Supabase.")
        return True
    except Exception as e:
        logger.error(f"Failed to sync student to Supabase: {e}")
        return False


def get_student(student_id: str) -> Optional[dict]:
    """Get a student profile from Supabase."""
    client = _get_client()
    if not client:
        return None

    try:
        result = client.table("students").select("*").eq("id", student_id).execute()
        if result.data and len(result.data) > 0:
            row = result.data[0]
            # Parse JSON fields back
            for field in ["resume_skills", "self_assessed_skills", "jd_required_skills",
                          "completed_topics", "completed_projects", "topics_studied",
                          "clicked_resource_links", "badges", "mastery_probabilities"]:
                if isinstance(row.get(field), str):
                    try:
                        row[field] = json.loads(row[field])
                    except Exception:
                        pass
            return row
        return None
    except Exception as e:
        logger.error(f"Failed to get student from Supabase: {e}")
        return None


# ── Quiz Operations ──

def save_quiz_result(student_id: str, quiz_data: dict) -> bool:
    """Save a quiz result to Supabase."""
    client = _get_client()
    if not client:
        return False

    try:
        db_data = {
            "student_id": student_id,
            "topic_id": quiz_data.get("topic_id", ""),
            "score": quiz_data.get("score", 0),
            "total_questions": quiz_data.get("total_questions", 0),
            "correct_answers": quiz_data.get("correct_answers", 0),
            "passed": quiz_data.get("passed", False),
            "difficulty": quiz_data.get("difficulty", "medium"),
        }
        client.table("student_quizzes").insert(db_data).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to save quiz to Supabase: {e}")
        return False


# ── Project Operations ──

def save_project_report(report_data: dict) -> bool:
    """Save a project evaluation report to Supabase."""
    client = _get_client()
    if not client:
        return False

    try:
        db_data = {
            "id": report_data.get("project_id"),
            "student_id": report_data.get("student_id"),
            "project_title": report_data.get("project_title", ""),
            "project_type": report_data.get("project_type", "mini_project"),
            "submission_text": report_data.get("submission_text", ""),
            "score": report_data.get("evaluation", {}).get("score", 0),
            "grade": report_data.get("evaluation", {}).get("grade", "N/A"),
            "is_passing": report_data.get("evaluation", {}).get("is_passing", False),
            "evaluation_data": json.dumps(report_data.get("evaluation", {})),
        }
        client.table("student_projects").upsert(db_data).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to save project to Supabase: {e}")
        return False


# ── Roadmap Operations ──

def save_roadmap(student_id: str, roadmap_data: dict) -> bool:
    """Save a roadmap to Supabase."""
    client = _get_client()
    if not client:
        return False

    try:
        db_data = {
            "student_id": student_id,
            "roadmap_data": json.dumps(roadmap_data),
        }
        client.table("student_roadmaps").upsert(db_data).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to save roadmap to Supabase: {e}")
        return False


def get_roadmap(student_id: str) -> Optional[dict]:
    """Get a roadmap from Supabase."""
    client = _get_client()
    if not client:
        return None

    try:
        result = client.table("student_roadmaps").select("*").eq("student_id", student_id).execute()
        if result.data and len(result.data) > 0:
            data = result.data[0].get("roadmap_data")
            if isinstance(data, str):
                return json.loads(data)
            return data
        return None
    except Exception as e:
        logger.error(f"Failed to get roadmap from Supabase: {e}")
        return None


# ── Roadmap History Operations ──

def archive_roadmap(student_id: str, roadmap_data: dict, topics_covered: list = None, completion_pct: float = 0) -> bool:
    """Archive a roadmap to history before clearing it."""
    client = _get_client()
    if not client:
        return False

    try:
        db_data = {
            "student_id": student_id,
            "roadmap_data": json.dumps(roadmap_data),
            "topics_covered": topics_covered or [],
            "completion_percentage": completion_pct,
        }
        client.table("roadmap_history").insert(db_data).execute()
        logger.info(f"Roadmap archived for student {student_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to archive roadmap: {e}")
        return False


def get_roadmap_history(student_id: str) -> list:
    """Get all archived roadmaps for a student."""
    client = _get_client()
    if not client:
        return []

    try:
        result = (
            client.table("roadmap_history")
            .select("*")
            .eq("student_id", student_id)
            .order("archived_at", desc=True)
            .execute()
        )
        rows = result.data or []
        for row in rows:
            if isinstance(row.get("roadmap_data"), str):
                try:
                    row["roadmap_data"] = json.loads(row["roadmap_data"])
                except Exception:
                    pass
        return rows
    except Exception as e:
        logger.error(f"Failed to get roadmap history: {e}")
        return []


def delete_current_roadmap(student_id: str) -> bool:
    """Delete the current active roadmap from Supabase."""
    client = _get_client()
    if not client:
        return False

    try:
        client.table("student_roadmaps").delete().eq("student_id", student_id).execute()
        logger.info(f"Current roadmap deleted for student {student_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete roadmap: {e}")
        return False


# ── Progress Snapshot Operations ──

def save_progress_snapshot(student_id: str, snapshot: dict) -> bool:
    """Save a progress snapshot."""
    client = _get_client()
    if not client:
        return False

    try:
        db_data = {
            "student_id": student_id,
            "topics_completed": snapshot.get("topics_completed", []),
            "quizzes_passed": snapshot.get("quizzes_passed", 0),
            "projects_completed": snapshot.get("projects_completed", 0),
            "job_readiness_score": snapshot.get("job_readiness_score", 0),
            "total_study_hours": snapshot.get("total_study_hours", 0),
        }
        client.table("progress_snapshots").insert(db_data).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to save progress snapshot: {e}")
        return False


def get_progress_snapshots(student_id: str) -> list:
    """Get progress history for a student."""
    client = _get_client()
    if not client:
        return []

    try:
        result = (
            client.table("progress_snapshots")
            .select("*")
            .eq("student_id", student_id)
            .order("snapshot_date", desc=True)
            .limit(30)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to get progress snapshots: {e}")
        return []


def get_student_stats(student_id: str) -> dict:
    """Get aggregated stats for profile display."""
    client = _get_client()
    stats = {
        "total_roadmaps": 0,
        "total_quizzes": 0,
        "total_projects": 0,
        "quiz_results": [],
    }
    if not client:
        return stats

    try:
        # Count archived roadmaps
        history = client.table("roadmap_history").select("id").eq("student_id", student_id).execute()
        stats["total_roadmaps"] = len(history.data) if history.data else 0

        # Get quiz results
        quizzes = client.table("student_quizzes").select("*").eq("student_id", student_id).order("created_at", desc=True).limit(20).execute()
        stats["quiz_results"] = quizzes.data or []
        stats["total_quizzes"] = len(quizzes.data) if quizzes.data else 0

        # Get project count
        projects = client.table("student_projects").select("id").eq("student_id", student_id).execute()
        stats["total_projects"] = len(projects.data) if projects.data else 0

    except Exception as e:
        logger.error(f"Failed to get student stats: {e}")

    return stats


# ══════════════════════════════════════════════════════════════════════
# 12-Agent Pipeline Operations (NEW)
# ══════════════════════════════════════════════════════════════════════


# ── Trajectory Memory Operations ──

def save_trajectory(student_id: str, trajectory_data: dict) -> bool:
    """Save/update trajectory memory to Supabase."""
    client = _get_client()
    if not client:
        return False

    try:
        db_data = {
            "student_id": student_id,
            "student_profile": json.dumps(trajectory_data.get("student_profile", {})),
            "domain_expert_proposal": json.dumps(trajectory_data.get("domain_expert_proposal")) if trajectory_data.get("domain_expert_proposal") else None,
            "prereq_architect_proposal": json.dumps(trajectory_data.get("prereq_architect_proposal")) if trajectory_data.get("prereq_architect_proposal") else None,
            "feasibility_proposal": json.dumps(trajectory_data.get("feasibility_proposal")) if trajectory_data.get("feasibility_proposal") else None,
            "student_advocate_proposal": json.dumps(trajectory_data.get("student_advocate_proposal")) if trajectory_data.get("student_advocate_proposal") else None,
            "conflict_resolution": json.dumps(trajectory_data.get("conflict_resolution")) if trajectory_data.get("conflict_resolution") else None,
            "final_roadmap": json.dumps(trajectory_data.get("final_roadmap")) if trajectory_data.get("final_roadmap") else None,
            "section_history": json.dumps(trajectory_data.get("section_history", [])),
            "quiz_attempts": json.dumps(trajectory_data.get("quiz_attempts", {})),
            "flags": json.dumps(trajectory_data.get("flags", [])),
            "intervention_log": json.dumps(trajectory_data.get("intervention_log", [])),
            "bridging_topics_inserted": trajectory_data.get("bridging_topics_inserted", 0),
            "current_topic_materials": json.dumps(trajectory_data.get("current_topic_materials")) if trajectory_data.get("current_topic_materials") else None,
        }
        client.table("trajectory_memory").upsert(db_data).execute()
        logger.info(f"Trajectory synced for {student_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to sync trajectory: {e}")
        return False


def get_trajectory_data(student_id: str) -> Optional[dict]:
    """Load trajectory memory from Supabase."""
    client = _get_client()
    if not client:
        return None

    try:
        result = client.table("trajectory_memory").select("*").eq("student_id", student_id).execute()
        if result.data and len(result.data) > 0:
            row = result.data[0]
            # Parse JSON fields
            for field in ["student_profile", "domain_expert_proposal", "prereq_architect_proposal",
                          "feasibility_proposal", "student_advocate_proposal", "conflict_resolution",
                          "final_roadmap", "section_history", "quiz_attempts", "flags",
                          "intervention_log", "current_topic_materials"]:
                if isinstance(row.get(field), str):
                    try:
                        row[field] = json.loads(row[field])
                    except Exception:
                        pass
            return row
        return None
    except Exception as e:
        logger.error(f"Failed to load trajectory: {e}")
        return None


# ── Learning Materials Operations ──

def save_learning_materials(student_id: str, topic_id: str, materials: dict) -> bool:
    """Cache learning materials (Scout→Critic→Curator output) to Supabase."""
    client = _get_client()
    if not client:
        return False

    try:
        db_data = {
            "student_id": student_id,
            "topic_id": topic_id,
            "topic_name": materials.get("topic_name", ""),
            "selected_course": json.dumps(materials.get("selected_course", {})),
            "cheat_sheet": json.dumps(materials.get("cheat_sheet", {})),
            "study_notes": json.dumps(materials.get("study_notes", {})),
            "mini_project": json.dumps(materials.get("mini_project", {})),
            "scout_candidates_count": materials.get("scout_candidates", 0),
        }
        client.table("learning_materials").upsert(
            db_data, on_conflict="student_id,topic_id"
        ).execute()
        logger.info(f"Materials cached for {student_id}/{topic_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to cache materials: {e}")
        return False


def get_learning_materials(student_id: str, topic_id: str) -> Optional[dict]:
    """Load cached learning materials from Supabase."""
    client = _get_client()
    if not client:
        return None

    try:
        result = (
            client.table("learning_materials")
            .select("*")
            .eq("student_id", student_id)
            .eq("topic_id", topic_id)
            .execute()
        )
        if result.data and len(result.data) > 0:
            row = result.data[0]
            for field in ["selected_course", "cheat_sheet", "study_notes", "mini_project"]:
                if isinstance(row.get(field), str):
                    try:
                        row[field] = json.loads(row[field])
                    except Exception:
                        pass
            return row
        return None
    except Exception as e:
        logger.error(f"Failed to load materials: {e}")
        return None


# ── Agent Log Operations ──

def log_agent_execution(student_id: str, agent_name: str, stage: str,
                        input_summary: str = "", output_summary: str = "",
                        latency_ms: int = 0, used_fallback: bool = False) -> bool:
    """Log an agent execution to the audit trail."""
    client = _get_client()
    if not client:
        return False

    try:
        db_data = {
            "student_id": student_id,
            "agent_name": agent_name,
            "stage": stage,
            "input_summary": input_summary[:500],
            "output_summary": output_summary[:500],
            "latency_ms": latency_ms,
            "used_fallback": used_fallback,
        }
        client.table("agent_logs").insert(db_data).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to log agent execution: {e}")
        return False


def get_agent_logs(student_id: str, limit: int = 50) -> list:
    """Get agent execution logs for a student."""
    client = _get_client()
    if not client:
        return []

    try:
        result = (
            client.table("agent_logs")
            .select("*")
            .eq("student_id", student_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to get agent logs: {e}")
        return []


def is_configured() -> bool:
    """Check if Supabase is configured and available."""
    return _get_client() is not None


def get_progress_snapshots(student_id: str, limit: int = 30) -> list:
    """Get progress snapshots for a student."""
    client = _get_client()
    if not client:
        return []

    try:
        result = (
            client.table("progress_snapshots")
            .select("*")
            .eq("student_id", student_id)
            .order("snapshot_date", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to get progress snapshots: {e}")
        return []

