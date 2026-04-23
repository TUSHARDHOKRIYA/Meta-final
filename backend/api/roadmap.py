"""
EduPath AI — Roadmap API
Team KRIYA | Meta Hackathon 2026

CRUD endpoints for learning roadmaps: generate, retrieve, quit/archive,
and dynamic replanning triggered by repeated quiz failures.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import os

from environment.student import student_manager
from ai.roadmap_generator import generate_roadmap

router = APIRouter(prefix="/api/roadmap", tags=["roadmap"])

# Store roadmaps in data directory (local cache)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


class RoadmapRequest(BaseModel):
    student_id: str
    force_regenerate: bool = False


@router.post("/generate")
async def generate_student_roadmap(data: RoadmapRequest):
    """Generate a personalized learning roadmap for the student.
    
    Uses the 6-agent Roadmap Generation Council when a student profile
    exists in trajectory memory. Falls back to single-agent LLM generation.
    """
    student = student_manager.get(data.student_id)
    if not student:
        raise HTTPException(404, "Student not found. Complete onboarding first.")

    # Check for cached roadmap (local first, then Supabase)
    roadmap_file = os.path.join(DATA_DIR, f"roadmap_{data.student_id}.json")
    if not data.force_regenerate:
        # Try local cache
        if os.path.exists(roadmap_file):
            with open(roadmap_file, "r") as f:
                return json.load(f)
        # Try Supabase
        try:
            from db.supabase_client import get_roadmap as sb_get_roadmap
            sb_roadmap = sb_get_roadmap(data.student_id)
            if sb_roadmap:
                # Save to local cache too
                with open(roadmap_file, "w") as f:
                    json.dump(sb_roadmap, f, indent=2)
                return sb_roadmap
        except Exception:
            pass

    # ── Try Council-based roadmap generation first ──
    roadmap = None
    try:
        from ai.trajectory_memory import get_trajectory
        memory = get_trajectory(data.student_id)
        if memory.student_profile:
            from ai.council.council_manager import run_council
            roadmap = run_council(memory.student_profile)
            if roadmap:
                import logging
                logging.getLogger(__name__).info(
                    f"Council roadmap generated for {data.student_id}"
                )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Council roadmap failed, using single-agent: {e}")

    # ── Fallback: single-agent LLM roadmap ──
    if not roadmap:
        # Build skill levels dict
        skill_levels = {}
        for s in student.self_assessed_skills:
            skill_levels[s.skill] = s.level

        current_skills = student.resume_skills + [s.skill for s in student.self_assessed_skills]

        roadmap = generate_roadmap(
            target_field=student.target_field or "tech",
            learning_goal=student.learning_goal,
            current_skills=current_skills,
            skill_levels=skill_levels,
            jd_skills=student.jd_required_skills,
            weekly_hours=student.weekly_hours,
            duration_weeks=max(8, 52 // max(student.weekly_hours // 5, 1)),
        )

    # Cache locally
    with open(roadmap_file, "w") as f:
        json.dump(roadmap, f, indent=2, default=str)

    # Sync to Supabase
    try:
        from db.supabase_client import save_roadmap
        save_roadmap(data.student_id, roadmap)
    except Exception:
        pass

    return roadmap



@router.get("/{student_id}")
async def get_roadmap(student_id: str):
    """Get the cached roadmap for a student."""
    # Try local cache
    roadmap_file = os.path.join(DATA_DIR, f"roadmap_{student_id}.json")
    if os.path.exists(roadmap_file):
        with open(roadmap_file, "r") as f:
            return json.load(f)

    # Try Supabase
    try:
        from db.supabase_client import get_roadmap as sb_get_roadmap
        sb_roadmap = sb_get_roadmap(student_id)
        if sb_roadmap:
            return sb_roadmap
    except Exception:
        pass

    raise HTTPException(404, "Roadmap not found. Generate one first.")


@router.post("/archive")
async def archive_roadmap(data: RoadmapRequest):
    """Archive the current roadmap to history, then clear it."""
    student = student_manager.get(data.student_id)
    if not student:
        raise HTTPException(404, "Student not found.")

    # Load current roadmap
    roadmap_file = os.path.join(DATA_DIR, f"roadmap_{data.student_id}.json")
    current_roadmap = None

    if os.path.exists(roadmap_file):
        with open(roadmap_file, "r") as f:
            current_roadmap = json.load(f)

    if not current_roadmap:
        try:
            from db.supabase_client import get_roadmap as sb_get_roadmap
            current_roadmap = sb_get_roadmap(data.student_id)
        except Exception:
            pass

    if not current_roadmap:
        raise HTTPException(404, "No active roadmap to archive.")

    # Calculate completion percentage
    total_weeks = len(current_roadmap.get("weeks", []))
    completed_topics = student.completed_topics or []
    topics_in_roadmap = []
    for week in current_roadmap.get("weeks", []):
        for skill in week.get("skillsCovered", []):
            topics_in_roadmap.append(skill.lower().replace(" ", "_"))

    covered = [t for t in topics_in_roadmap if t in completed_topics or t.replace("_", " ").title() in [ct.replace("_", " ").title() for ct in completed_topics]]
    completion_pct = (len(covered) / max(len(topics_in_roadmap), 1)) * 100

    # Archive to Supabase
    try:
        from db.supabase_client import archive_roadmap as sb_archive, delete_current_roadmap
        sb_archive(data.student_id, current_roadmap, completed_topics, completion_pct)
        delete_current_roadmap(data.student_id)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Supabase archive failed: {e}")

    # Delete local cache
    if os.path.exists(roadmap_file):
        os.remove(roadmap_file)

    return {
        "message": "Roadmap archived successfully",
        "completion_percentage": round(completion_pct, 1),
        "topics_covered": completed_topics,
    }


@router.delete("/{student_id}")
async def delete_roadmap(student_id: str):
    """Delete the current roadmap without archiving."""
    roadmap_file = os.path.join(DATA_DIR, f"roadmap_{student_id}.json")
    if os.path.exists(roadmap_file):
        os.remove(roadmap_file)

    try:
        from db.supabase_client import delete_current_roadmap
        delete_current_roadmap(student_id)
    except Exception:
        pass

    return {"message": "Roadmap deleted"}

