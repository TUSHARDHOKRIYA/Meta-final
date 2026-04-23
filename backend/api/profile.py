"""
EduPath AI — Student Profile API
Team KRIYA | Meta Hackathon 2026

Read-only profile endpoint providing a unified view of studentprogress, statistics, badges, and career readiness for the frontend
dashboard.
"""
from fastapi import APIRouter, HTTPException
from environment.student import student_manager

router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("/{student_id}")
async def get_profile_overview(student_id: str):
    """Get comprehensive profile overview including stats."""
    student = student_manager.get(student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    profile = student.model_dump()

    # Enrich with Supabase stats
    try:
        from db.supabase_client import get_student_stats
        stats = get_student_stats(student_id)
        profile["stats"] = stats
    except Exception:
        profile["stats"] = {
            "total_roadmaps": 0,
            "total_quizzes": 0,
            "total_projects": 0,
            "quiz_results": [],
        }

    return profile


@router.get("/{student_id}/history")
async def get_roadmap_history(student_id: str):
    """Get all archived roadmaps for a student."""
    try:
        from db.supabase_client import get_roadmap_history as sb_get_history
        history = sb_get_history(student_id)
        return {"history": history, "total": len(history)}
    except Exception as e:
        return {"history": [], "total": 0, "error": str(e)}


@router.get("/{student_id}/progress")
async def get_progress(student_id: str):
    """Get progress snapshots for a student."""
    student = student_manager.get(student_id)

    # Current progress from in-memory state
    current = {
        "topics_completed": student.completed_topics if student else [],
        "topics_studied": student.topics_studied if student else [],
        "quiz_streak": student.quiz_streak if student else 0,
        "job_readiness_score": student.job_readiness_score if student else 0,
        "badges_earned": len([b for b in (student.badges if student else []) if isinstance(b, dict) and b.get("earned")]),
    }

    # Historical progress from Supabase
    try:
        from db.supabase_client import get_progress_snapshots
        snapshots = get_progress_snapshots(student_id)
    except Exception:
        snapshots = []

    return {
        "current": current,
        "snapshots": snapshots,
    }
