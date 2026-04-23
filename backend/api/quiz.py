"""
EduPath AI — Quiz API
Team KRIYA | Meta Hackathon 2026

Generates adaptive quizzes per topic, scores submissions, and updates
student progress. Integrates with the badge system to award quiz streak
achievements.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from environment.student import student_manager
from environment.models import QuizResult, QuizDifficulty
from environment.curriculum import TOPIC_GRAPH
from ai.quiz_generator import generate_quiz, score_quiz

router = APIRouter(prefix="/api/quiz", tags=["quiz"])


class QuizRequest(BaseModel):
    student_id: str
    topic_id: str
    difficulty: str = "medium"
    num_questions: int = 5


class QuizSubmission(BaseModel):
    student_id: str
    topic_id: str
    questions: list  # The original questions
    answers: List[int]  # User's selected option indices


def _resolve_topic_name(topic_id: str) -> str:
    """Resolve a topic ID to a human-readable name. Works for any topic."""
    # Try exact match in graph
    topic = TOPIC_GRAPH.get(topic_id)
    if topic:
        return topic.name
    # Try normalized match
    normalized = topic_id.lower().replace(" ", "_").replace("-", "_")
    topic = TOPIC_GRAPH.get(normalized)
    if topic:
        return topic.name
    # Fuzzy match
    for tid, t in TOPIC_GRAPH.items():
        if normalized in tid or tid in normalized or normalized.replace("_", " ") in t.name.lower():
            return t.name
    # Fall back to converting the ID itself into a readable name
    return topic_id.replace("_", " ").replace("-", " ").title()


@router.post("/generate")
async def generate_topic_quiz(data: QuizRequest):
    """Generate an AI-powered quiz for any topic."""
    topic_name = _resolve_topic_name(data.topic_id)

    quiz = generate_quiz(topic_name, data.difficulty, data.num_questions)

    return {
        "topic_id": data.topic_id,
        "topic_name": topic_name,
        "difficulty": data.difficulty,
        "questions": quiz.get("questions", []),
    }


@router.post("/submit")
async def submit_quiz(data: QuizSubmission):
    """Submit quiz answers and get adaptive feedback."""
    student = student_manager.get(data.student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    # Score the quiz
    result = score_quiz(data.questions, data.answers)

    # Record in student history
    quiz_result = QuizResult(
        topic_id=data.topic_id,
        score=result["score"],
        total_questions=result["total_questions"],
        correct_answers=result["correct_answers"],
        passed=result["passed"],
        difficulty=QuizDifficulty.MEDIUM,
    )
    student_manager.record_quiz(data.student_id, quiz_result)

    # Get adaptive recommendation
    if result["recommendation"] == "restart_topic":
        topic = TOPIC_GRAPH.get(data.topic_id)
        if topic and topic.resources:
            result["alternative_resources"] = [
                {"title": r.title, "type": r.type.value, "url": r.url}
                for r in topic.resources[:3]
            ]

    return result


@router.get("/history/{student_id}")
async def get_quiz_history(student_id: str):
    """Get quiz history for a student."""
    student = student_manager.get(student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    return {
        "quiz_history": [q.model_dump() for q in student.quiz_history],
        "quiz_streak": student.quiz_streak,
        "total_quizzes": len(student.quiz_history),
        "pass_rate": round(
            sum(1 for q in student.quiz_history if q.passed) / max(len(student.quiz_history), 1) * 100, 1
        )
    }
