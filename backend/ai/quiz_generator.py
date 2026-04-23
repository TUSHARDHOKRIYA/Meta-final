"""
EduPath AI — AI Quiz Generator
Team KRIYA | Meta Hackathon 2026

Generates adaptive multiple-choice quizzes per curriculum topic.
Uses LLM to produce conceptual, practical, and tricky questions with
explanations. Includes scoring logic with adaptive recommendations
(move forward / review / restart).
"""
import logging
import random
from typing import List
from ai.llm_client import generate_json, is_api_key_set

logger = logging.getLogger(__name__)

QUIZ_SYSTEM_PROMPT = """
You are a quiz generator. Output ONLY valid JSON.

Generate MCQ quiz questions adhering to this EXACT schema:

{
  "questions": [
    {
      "question": "string",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_index": 0,
      "explanation": "string",
      "type": "conceptual|practical|tricky",
      "topic": "string"
    }
  ]
}

RULES:
1. Each question must have exactly 4 options.
2. correct_index is 0-based (0=A, 1=B, 2=C, 3=D).
3. Mix question types: conceptual, practical, and tricky.
4. Provide clear explanations for each answer.
5. Output VALID JSON only.
"""


def generate_quiz(topic_name: str, difficulty: str = "medium", num_questions: int = 5) -> dict:
    """Generate a quiz for a topic via LLM."""

    if not is_api_key_set():
        return _generate_fallback_quiz(topic_name, difficulty, num_questions)

    diff_map = {
        "easy": "beginner-level, straightforward",
        "medium": "intermediate, requiring understanding of concepts",
        "hard": "advanced, with tricky edge cases"
    }

    user_prompt = f"""
Generate {num_questions} multiple-choice questions about: {topic_name}

Difficulty: {diff_map.get(difficulty, 'intermediate')}

Requirements:
- Questions should test real understanding, not just memorization.
- Include a mix of conceptual, practical, and tricky questions.
- Make wrong options plausible (not obviously wrong).
- Provide clear, educational explanations.

Return ONLY valid JSON.
"""

    try:
        result = generate_json(QUIZ_SYSTEM_PROMPT, user_prompt)
        if result and "questions" in result:
            return result
    except Exception as e:
        logger.error(f"Quiz generation failed: {e}")

    return _generate_fallback_quiz(topic_name, difficulty, num_questions)


def score_quiz(questions: list, answers: list) -> dict:
    """Score a quiz submission. Returns score details."""
    correct = 0
    total = len(questions)
    details = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        is_correct = a == q.get("correct_index", -1)
        if is_correct:
            correct += 1
        details.append({
            "question_index": i,
            "correct": is_correct,
            "selected": a,
            "correct_answer": q.get("correct_index"),
            "explanation": q.get("explanation", "")
        })

    score = round((correct / max(total, 1)) * 100, 1)
    passed = score >= 70

    # Determine adaptive path
    if score >= 70:
        recommendation = "move_forward"
        message = "Great job! You're ready to move to the next topic."
    elif score >= 50:
        recommendation = "review_and_retry"
        message = "Good attempt! Review the weak areas and try again."
    else:
        recommendation = "restart_topic"
        message = "Let's revisit this topic with different resources."

    return {
        "score": score,
        "total_questions": total,
        "correct_answers": correct,
        "passed": passed,
        "recommendation": recommendation,
        "message": message,
        "details": details,
    }


def _generate_fallback_quiz(topic_name: str, difficulty: str, num_questions: int) -> dict:
    """Generate a basic quiz when LLM is unavailable."""
    templates = [
        {
            "question": f"What is a key concept in {topic_name}?",
            "options": [
                f"A. Core principle of {topic_name}",
                f"B. Unrelated concept from a different field",
                f"C. Advanced topic beyond {topic_name}",
                f"D. Historical background only"
            ],
            "correct_index": 0,
            "explanation": f"Understanding core principles is fundamental to {topic_name}.",
            "type": "conceptual",
            "topic": topic_name
        },
        {
            "question": f"Which of the following is a practical application of {topic_name}?",
            "options": [
                "A. Theoretical proof only",
                f"B. Real-world project using {topic_name}",
                "C. Historical analysis",
                "D. None of the above"
            ],
            "correct_index": 1,
            "explanation": f"{topic_name} has many practical applications in real projects.",
            "type": "practical",
            "topic": topic_name
        },
        {
            "question": f"What is a common mistake when learning {topic_name}?",
            "options": [
                "A. Practicing too much",
                "B. Reading documentation",
                f"C. Skipping prerequisites for {topic_name}",
                "D. Building projects"
            ],
            "correct_index": 2,
            "explanation": "Skipping prerequisites leads to gaps in understanding.",
            "type": "tricky",
            "topic": topic_name
        },
        {
            "question": f"Which resource type is best for learning {topic_name}?",
            "options": [
                "A. Only textbooks",
                "B. Interactive hands-on exercises",
                "C. Only videos",
                "D. Only lectures"
            ],
            "correct_index": 1,
            "explanation": "Interactive, hands-on learning is the most effective approach.",
            "type": "practical",
            "topic": topic_name
        },
        {
            "question": f"How should you assess your progress in {topic_name}?",
            "options": [
                "A. By reading more books",
                "B. By watching videos",
                "C. By building real projects and taking quizzes",
                "D. By memorizing facts"
            ],
            "correct_index": 2,
            "explanation": "Building projects and testing yourself are the best ways to gauge progress.",
            "type": "conceptual",
            "topic": topic_name
        },
    ]

    questions = templates[:num_questions]
    random.shuffle(questions)
    return {"questions": questions}
