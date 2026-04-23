"""
EduPath AI — Learning Loop Orchestrator
Team KRIYA | OpenEnv Hackathon 2026

Orchestrates the full topic learning loop:
  start_topic() → Scout → Critic → Curator → materials ready
  take_quiz()   → Quiz Agent (BKT-adaptive)
  submit_quiz() → Score + flag + trigger adaptation if needed
"""
import logging
from typing import Dict, Optional

from ai.trajectory_memory import get_trajectory
from ai.learning_loop.scout_agent import run as scout_run
from ai.learning_loop.critic_agent import run as critic_run
from ai.learning_loop.curator_agent import run as curator_run
from ai.learning_loop.quiz_agent import run as quiz_run

logger = logging.getLogger(__name__)


def start_topic(student_id: str, topic_id: str,
                student_profile: Dict = None) -> Dict:
    """
    Start learning a topic: Scout → Critic → Curator pipeline.

    Args:
        student_id: Student identifier.
        topic_id: Topic to learn.
        student_profile: StudentProfile dict (loaded from trajectory if None).

    Returns:
        Dict with selected_course, cheat_sheet, study_notes, mini_project.
    """
    memory = get_trajectory(student_id)

    if not student_profile:
        student_profile = memory.student_profile or {"student_id": student_id}

    topic_name = _get_topic_name(topic_id)
    topic_info = {"topic_id": topic_id, "topic_name": topic_name}

    logger.info(f"[Learning] Starting topic {topic_name} for {student_id}")

    try:
        # Step 1: Scout — find courses
        scout_result = scout_run(topic_id, topic_name, student_profile)
        candidates = scout_result.get("candidates", [])
        logger.info(f"[Learning] Scout found {len(candidates)} candidates")

        if not candidates:
            raise ValueError("Scout returned no candidates")

        # Step 2: Critic — score and rank
        critic_result = critic_run(candidates, student_profile, topic_info)
        evaluations = critic_result.get("evaluations", [])
        logger.info(f"[Learning] Critic evaluated {len(evaluations)} courses")

        # Step 3: Curator — select and generate materials
        curator_result = curator_run(evaluations, candidates, student_profile, topic_info)
        logger.info(f"[Learning] Curator selected: {curator_result.get('selected_course', {}).get('title', '?')}")

        # Store in trajectory memory
        memory.current_topic_materials = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "scout_candidates": len(candidates),
            "selected_course": curator_result.get("selected_course"),
            "cheat_sheet": curator_result.get("cheat_sheet"),
            "study_notes": curator_result.get("study_notes"),
            "mini_project": curator_result.get("mini_project"),
        }
        memory._save()

        return {
            "status": "ready",
            "topic_id": topic_id,
            "topic_name": topic_name,
            "pipeline": "scout → critic → curator",
            **curator_result,
        }

    except Exception as e:
        logger.error(f"[Learning] Pipeline failed for {topic_name}: {e}")
        return {
            "status": "error",
            "topic_id": topic_id,
            "topic_name": topic_name,
            "error": str(e),
        }


def take_quiz(student_id: str, topic_id: str,
              student_profile: Dict = None) -> Dict:
    """
    Generate a quiz for the current topic using BKT-adaptive difficulty.

    Args:
        student_id: Student identifier.
        topic_id: Topic to quiz on.
        student_profile: StudentProfile dict.

    Returns:
        Quiz dict with questions, pass_threshold, etc.
    """
    memory = get_trajectory(student_id)

    if not student_profile:
        student_profile = memory.student_profile or {"student_id": student_id}

    # Get BKT level
    bkt_level = _get_bkt_level(student_id, topic_id)
    attempt = memory.get_quiz_attempt_count(topic_id) + 1

    # Get course info from current materials
    course_info = None
    if memory.current_topic_materials:
        course_info = memory.current_topic_materials.get("selected_course")

    logger.info(f"[Learning] Generating quiz for {topic_id} (BKT={bkt_level:.2f}, attempt={attempt})")

    quiz = quiz_run(topic_id, student_profile, bkt_level, course_info, attempt)
    return quiz


def submit_quiz(student_id: str, topic_id: str,
                answers: Dict, student_profile: Dict = None) -> Dict:
    """
    Score a quiz submission and trigger adaptation if needed.

    Args:
        student_id: Student identifier.
        topic_id: Topic the quiz is for.
        answers: Dict mapping question_id → student's answer.
        student_profile: StudentProfile dict.

    Returns:
        Dict with score, passed, flag, adaptation recommendation.
    """
    memory = get_trajectory(student_id)

    if not student_profile:
        student_profile = memory.student_profile or {"student_id": student_id}

    # Get the last quiz generated (from trajectory memory)
    attempt = memory.get_quiz_attempt_count(topic_id) + 1

    # Simple scoring: count correct answers
    # In production, the quiz questions would be stored and matched
    total_questions = len(answers)
    correct = 0

    # Basic scoring heuristic (in real use, we'd compare against stored correct answers)
    for q_id, answer in answers.items():
        if answer and len(str(answer)) > 5:  # Non-trivial answer
            correct += 1

    score = (correct / max(total_questions, 1)) * 100 if total_questions > 0 else 0
    passed = score >= 70

    # Determine flag
    flag = "CLEAR"
    if not passed and attempt >= 2:
        flag = "QUIZ_FAIL_REPEAT"
    elif not passed:
        flag = "QUIZ_FAIL"

    # Record quiz attempt
    quiz_result = {
        "score": score,
        "passed": passed,
        "correct": correct,
        "total": total_questions,
        "attempt": attempt,
    }
    memory.record_quiz_attempt(topic_id, quiz_result)

    # Record section with flag
    course_info = memory.current_topic_materials.get("selected_course") if memory.current_topic_materials else None
    memory.record_section(topic_id, course_info, quiz_result, flag)

    # Trigger adaptation check if failing
    adaptation = None
    if flag != "CLEAR":
        try:
            from ai.adaptation.adaptation_agent import run as adaptation_run
            adaptation = adaptation_run(memory, student_profile)
            if adaptation:
                memory.record_intervention(
                    adaptation.get("level", 1),
                    adaptation.get("action_type", "CONTINUE"),
                    adaptation,
                )
        except Exception as e:
            logger.warning(f"Adaptation check failed: {e}")

    return {
        "topic_id": topic_id,
        "score": score,
        "passed": passed,
        "correct": correct,
        "total": total_questions,
        "attempt": attempt,
        "flag": flag,
        "adaptation": adaptation,
    }


def _get_topic_name(topic_id: str) -> str:
    """Resolve topic name from curriculum graph."""
    try:
        from environment.curriculum import TOPIC_GRAPH
        topic = TOPIC_GRAPH.get(topic_id)
        return topic.name if topic else topic_id.replace("_", " ").title()
    except Exception:
        return topic_id.replace("_", " ").title()


def _get_bkt_level(student_id: str, topic_id: str) -> float:
    """Get BKT mastery probability for a topic."""
    try:
        from environment.bkt_model import BKTModel
        bkt = BKTModel()
        # Check previous quiz results
        memory = get_trajectory(student_id)
        attempts = memory.quiz_attempts.get(topic_id, [])
        for a in attempts:
            observed = a.get("passed", False)
            bkt.update(observed)
        return bkt.mastery_prob
    except Exception:
        # Fallback: estimate from quiz history
        memory = get_trajectory(student_id)
        last_score = memory.get_latest_quiz_score(topic_id)
        if last_score is not None:
            return min(1.0, last_score / 100)
        return 0.3  # Default for new topics
