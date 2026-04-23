"""
EduPath AI — Learning API
Team KRIYA | OpenEnv Hackathon 2026

REST endpoints for the full topic learning loop:
  POST /api/learning/start-topic  — Scout → Critic → Curator
  POST /api/learning/quiz         — Generate BKT-adaptive quiz
  POST /api/learning/submit-quiz  — Score + adapt
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import logging

from ai.learning_loop.orchestrator import start_topic, take_quiz, submit_quiz
from ai.trajectory_memory import get_trajectory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learning", tags=["learning"])


class StartTopicRequest(BaseModel):
    student_id: str
    topic_id: str


class QuizRequest(BaseModel):
    student_id: str
    topic_id: str


class SubmitQuizRequest(BaseModel):
    student_id: str
    topic_id: str
    answers: Dict[str, str]  # {question_id: answer}


@router.post("/start-topic")
async def api_start_topic(data: StartTopicRequest):
    """Start learning a topic — runs the Scout → Critic → Curator pipeline."""
    memory = get_trajectory(data.student_id)
    profile = memory.student_profile

    if not profile:
        raise HTTPException(400, "Student profile not found. Complete profiling first.")

    result = start_topic(data.student_id, data.topic_id, profile)
    return result


@router.post("/quiz")
async def api_take_quiz(data: QuizRequest):
    """Generate a BKT-adaptive quiz for a topic."""
    memory = get_trajectory(data.student_id)
    profile = memory.student_profile

    result = take_quiz(data.student_id, data.topic_id, profile)
    return result


@router.post("/submit-quiz")
async def api_submit_quiz(data: SubmitQuizRequest):
    """Submit quiz answers for scoring and adaptation."""
    memory = get_trajectory(data.student_id)
    profile = memory.student_profile

    result = submit_quiz(data.student_id, data.topic_id, data.answers, profile)
    return result


@router.get("/progress/{student_id}")
async def get_progress(student_id: str):
    """Get overall learning progress for a student."""
    memory = get_trajectory(student_id)

    return {
        "student_id": student_id,
        "sections_completed": len(memory.section_history),
        "topics_with_quizzes": list(memory.quiz_attempts.keys()),
        "total_quiz_attempts": sum(len(v) for v in memory.quiz_attempts.values()),
        "warning_flags": len(memory.flags),
        "interventions": len(memory.intervention_log),
        "bridging_topics_inserted": memory.bridging_topics_inserted,
        "current_topic": memory.current_topic_materials.get("topic_id") if memory.current_topic_materials else None,
        "recent_sections": memory.section_history[-5:] if memory.section_history else [],
    }


@router.get("/materials/{student_id}/{topic_id}")
async def get_materials(student_id: str, topic_id: str):
    """Get the current learning materials for a topic."""
    memory = get_trajectory(student_id)

    if memory.current_topic_materials and memory.current_topic_materials.get("topic_id") == topic_id:
        return memory.current_topic_materials

    raise HTTPException(404, f"No materials found for topic {topic_id}. Use /start-topic first.")
