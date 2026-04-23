"""
EduPath AI — Career & Job Readiness API
Team KRIYA | Meta Hackathon 2026

Provides job readiness scoring, field-specific event/hackathon
recommendations, and project suggestions based on student progress
and target career field.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from environment.student import student_manager
from environment.curriculum import TOPIC_GRAPH, get_projects_for_field

router = APIRouter(prefix="/api/career", tags=["career"])


# Field-specific event/hackathon recommendations
EVENT_DB = {
    "tech": [
        {"id": "hackathon_1", "name": "MLH Global Hack Week", "type": "hackathon", "url": "https://mlh.io", "field": "tech"},
        {"id": "hackathon_2", "name": "DevPost AI Challenge", "type": "hackathon", "url": "https://devpost.com", "field": "tech"},
        {"id": "event_1", "name": "PyCon Conference", "type": "conference", "url": "https://pycon.org", "field": "tech"},
        {"id": "event_2", "name": "Google I/O", "type": "conference", "url": "https://io.google", "field": "tech"},
        {"id": "job_1", "name": "ML Engineer Roles", "type": "job_board", "url": "https://www.linkedin.com/jobs/machine-learning-engineer-jobs", "field": "tech"},
    ],
    "healthcare": [
        {"id": "hc_event_1", "name": "Health 2.0 Conference", "type": "conference", "url": "https://health2con.com", "field": "healthcare"},
        {"id": "hc_event_2", "name": "Medical AI Hackathon", "type": "hackathon", "url": "https://healthcareai.com", "field": "healthcare"},
        {"id": "hc_job_1", "name": "Health Informatics Roles", "type": "job_board", "url": "https://www.linkedin.com/jobs/health-informatics-jobs", "field": "healthcare"},
    ],
    "business": [
        {"id": "biz_event_1", "name": "TechCrunch Disrupt", "type": "conference", "url": "https://techcrunch.com/events", "field": "business"},
        {"id": "biz_event_2", "name": "Business Analytics Summit", "type": "conference", "url": "https://www.analytics-summit.com", "field": "business"},
    ],
    "law": [
        {"id": "law_event_1", "name": "Legal Tech Conference", "type": "conference", "url": "https://legaltechshow.com", "field": "law"},
        {"id": "law_event_2", "name": "LegalHack Hackathon", "type": "hackathon", "url": "https://legalhack.com", "field": "law"},
    ],
    "design": [
        {"id": "des_event_1", "name": "Config by Figma", "type": "conference", "url": "https://config.figma.com", "field": "design"},
        {"id": "des_event_2", "name": "Behance Portfolio Reviews", "type": "competition", "url": "https://www.behance.net", "field": "design"},
    ],
}


@router.get("/readiness/{student_id}")
async def get_job_readiness(student_id: str):
    """Get job readiness score and breakdown."""
    student = student_manager.get(student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    # Breakdown
    topics_progress = len(student.completed_topics)
    quiz_avg = round(
        sum(q.score for q in student.quiz_history) / max(len(student.quiz_history), 1), 1
    )
    projects_done = len(student.completed_projects)

    # Interview prep unlocked at 70%+
    interview_prep_unlocked = student.job_readiness_score >= 0.7

    return {
        "job_readiness_score": student.job_readiness_score,
        "breakdown": {
            "topics_completed": topics_progress,
            "average_quiz_score": quiz_avg,
            "projects_completed": projects_done,
            "quiz_streak": student.quiz_streak,
        },
        "interview_prep_unlocked": interview_prep_unlocked,
        "target_field": student.target_field,
    }


@router.get("/events/{student_id}")
async def get_recommended_events(student_id: str):
    """Get field-specific hackathons, competitions, jobs, and events."""
    student = student_manager.get(student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    field = student.target_field or "tech"
    events = EVENT_DB.get(field, EVENT_DB.get("tech", []))

    return {
        "field": field,
        "events": events,
    }


@router.get("/projects/{student_id}")
async def get_recommended_projects(student_id: str):
    """Get project recommendations based on progress."""
    student = student_manager.get(student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    field = student.target_field or "tech"
    all_projects = get_projects_for_field(field)

    # Filter out completed
    available = [
        p.model_dump() for p in all_projects
        if p.id not in student.completed_projects
    ]

    completed = [
        p.model_dump() for p in all_projects
        if p.id in student.completed_projects
    ]

    return {
        "available_projects": available,
        "completed_projects": completed,
    }
