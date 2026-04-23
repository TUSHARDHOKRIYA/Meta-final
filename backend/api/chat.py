"""
EduPath AI — Chat API (Profiling Agent Interface)
Team KRIYA | OpenEnv Hackathon 2026

REST endpoints for the conversational Profiling Agent:
  POST /api/chat/start   — Start a profiling conversation
  POST /api/chat/message  — Send a message, get agent response
  POST /api/chat/complete — Finalize profile, trigger roadmap council
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
import logging

from ai.profiling_agent import get_or_create_session, end_session
from environment.student import student_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatStartRequest(BaseModel):
    student_name: Optional[str] = "Student"


class ChatStartResponse(BaseModel):
    student_id: str
    message: str
    session_active: bool = True


class ChatMessageRequest(BaseModel):
    student_id: str
    message: str


class ChatMessageResponse(BaseModel):
    message: str
    is_complete: bool
    dimensions_captured: dict
    question_count: int


class ChatCompleteRequest(BaseModel):
    student_id: str


@router.post("/start", response_model=ChatStartResponse)
async def start_chat(data: ChatStartRequest):
    """Start a new profiling conversation for a student."""
    # Create student record
    student = student_manager.create(name=data.student_name or "Student")
    student_id = student.id

    # Create profiling session
    session = get_or_create_session(student_id)
    greeting = session.start()

    return ChatStartResponse(
        student_id=student_id,
        message=greeting,
        session_active=True,
    )


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(data: ChatMessageRequest):
    """Send a message to the Profiling Agent and get a response."""
    session = get_or_create_session(data.student_id)

    if session.is_profiling_complete():
        return ChatMessageResponse(
            message="Your profile is already complete! Use /api/chat/complete to finalize.",
            is_complete=True,
            dimensions_captured=session.dimensions_captured,
            question_count=session.question_count,
        )

    response = session.chat(data.message)

    return ChatMessageResponse(
        message=response,
        is_complete=session.is_profiling_complete(),
        dimensions_captured=session.dimensions_captured,
        question_count=session.question_count,
    )


@router.post("/complete")
async def complete_chat(data: ChatCompleteRequest):
    """Finalize the profiling conversation and extract the student profile."""
    profile = end_session(data.student_id)

    if not profile:
        raise HTTPException(404, "No active profiling session for this student.")

    # Update student record with extracted profile data
    student = student_manager.get(data.student_id)
    if student:
        student_manager.update_from_onboarding(data.student_id, {
            "target_field": profile.get("target_domain", profile.get("domain", "tech")),
            "learning_goal": profile.get("target_role", ""),
            "weekly_hours": profile.get("weekly_hours", 10),
        })

    # Attempt to trigger council roadmap generation
    council_result = None
    try:
        from ai.council.council_manager import run_council
        council_result = run_council(profile)
        logger.info(f"Council roadmap generated for {data.student_id}")
    except Exception as e:
        logger.warning(f"Council roadmap generation failed: {e}")

    return {
        "student_id": data.student_id,
        "profile": profile,
        "roadmap": council_result,
        "message": "Profile complete. Roadmap generated." if council_result else "Profile complete. Use /api/roadmap/generate to create your roadmap.",
    }
