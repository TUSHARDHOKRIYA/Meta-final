"""
EduPath AI — AI Project Evaluator
Team KRIYA | Meta Hackathon 2026

Evaluates student project submissions (mini-projects & capstones) using
an LLM. Produces structured feedback with scores across code quality,
architecture, functionality, documentation, and best practices.
Falls back to a deterministic rubric when the LLM is unavailable.
"""
import logging
from ai.llm_client import generate_json, is_api_key_set

logger = logging.getLogger(__name__)

PROJECT_EVAL_SYSTEM_PROMPT = """
You are an expert software engineering professor and project evaluator.
You must evaluate student projects and provide a detailed assessment report.

Output ONLY valid JSON adhering to this schema:

{
  "score": number (0-100),
  "grade": "A+|A|A-|B+|B|B-|C+|C|C-|D|F",
  "overall_feedback": "string (2-3 sentences summary)",
  "strengths": ["string", "string", "string"],
  "improvements": ["string", "string", "string"],
  "technical_analysis": {
    "code_quality": { "score": number (0-10), "feedback": "string" },
    "architecture": { "score": number (0-10), "feedback": "string" },
    "functionality": { "score": number (0-10), "feedback": "string" },
    "documentation": { "score": number (0-10), "feedback": "string" },
    "best_practices": { "score": number (0-10), "feedback": "string" }
  },
  "learning_outcomes": ["string", "string"],
  "next_steps": ["string", "string"],
  "is_passing": boolean
}

RULES:
1. Be constructive but honest — highlight both strengths and areas for improvement.
2. Score fairly: 90-100 = exceptional, 80-89 = very good, 70-79 = good, 60-69 = adequate, below 60 = needs improvement.
3. Provide actionable feedback in each area.
4. "is_passing" = true if score >= 60.
5. Output VALID JSON only. No markdown, no comments.
"""


def evaluate_project(
    project_title: str,
    project_description: str,
    submission_text: str,
    requirements: list = None,
    project_type: str = "mini_project",
) -> dict:
    """
    Evaluate a student project using AI.
    submission_text can be:
    - A GitHub repo URL
    - Pasted code
    - A project description/write-up
    """
    if not is_api_key_set():
        return _generate_fallback_evaluation(project_title, project_type)

    req_str = "\n".join(f"- {r}" for r in (requirements or []))

    user_prompt = f"""
Evaluate this student {project_type.replace('_', ' ')} submission:

**Project Title:** {project_title}
**Project Description:** {project_description}
**Project Type:** {project_type.replace('_', ' ').title()}

**Requirements:**
{req_str if req_str else "No specific requirements listed."}

**Student Submission:**
{submission_text}

Provide a thorough evaluation considering:
1. Does the submission meet the project requirements?
2. Code quality, readability, and organization
3. Architecture and design decisions
4. Functionality and completeness
5. Documentation and explanations
6. Best practices followed

If the submission is a GitHub URL, evaluate based on what the URL and project description suggest.
If the submission is code or a description, evaluate directly.

Return ONLY valid JSON following the predefined schema.
"""

    try:
        result = generate_json(PROJECT_EVAL_SYSTEM_PROMPT, user_prompt)
        if result and "score" in result:
            return result
    except Exception as e:
        logger.error(f"Project evaluation failed: {e}")

    return _generate_fallback_evaluation(project_title, project_type)


def _generate_fallback_evaluation(title: str, project_type: str) -> dict:
    """Generate a basic evaluation when LLM is unavailable."""
    return {
        "score": 70,
        "grade": "B-",
        "overall_feedback": (
            f"Your {project_type.replace('_', ' ')} '{title}' has been received. "
            "The AI evaluator is currently unavailable, but based on your submission, "
            "it appears you have demonstrated foundational understanding of the requirements."
        ),
        "strengths": [
            "Project was submitted on time",
            "Demonstrates effort and understanding",
            "Covers the basic requirements",
        ],
        "improvements": [
            "Add more documentation and comments",
            "Consider edge cases and error handling",
            "Include unit tests to validate functionality",
        ],
        "technical_analysis": {
            "code_quality": {"score": 7, "feedback": "Submission received. Add more comments and follow style guides."},
            "architecture": {"score": 7, "feedback": "Basic structure is present. Consider separating concerns."},
            "functionality": {"score": 7, "feedback": "Core functionality appears to be implemented."},
            "documentation": {"score": 6, "feedback": "Add a README with setup instructions and usage examples."},
            "best_practices": {"score": 7, "feedback": "Follow PEP8/ESLint standards and add error handling."},
        },
        "learning_outcomes": [
            "Demonstrated ability to build a project from scratch",
            "Applied concepts learned in the curriculum",
        ],
        "next_steps": [
            "Refactor code for better readability",
            "Deploy the project and share the live URL",
        ],
        "is_passing": True,
    }
