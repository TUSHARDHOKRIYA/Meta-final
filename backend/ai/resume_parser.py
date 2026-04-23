"""
EduPath AI — Resume & Job Description Parsers
Team KRIYA | Meta Hackathon 2026

Extracts structured data from free-text resumes and job descriptions
using LLM. Normalises skill names and separates required vs. preferred
qualifications. Falls back to keyword matching when LLM is unavailable.
"""
import logging
from ai.llm_client import generate_json, is_api_key_set

logger = logging.getLogger(__name__)

RESUME_SYSTEM_PROMPT = """
You are a resume parser. Output ONLY valid JSON.

Extract information from the resume text and return:

{
  "skills": ["string"],
  "education": [{"degree": "string", "institution": "string", "field": "string"}],
  "experience": [{"title": "string", "company": "string", "duration": "string", "skills_used": ["string"]}],
  "certifications": ["string"],
  "summary": "string"
}

RULES:
1. Extract ALL mentioned technical and professional skills.
2. Normalize skill names (e.g., "JS" → "JavaScript", "ML" → "Machine Learning").
3. Output VALID JSON only.
"""

JD_SYSTEM_PROMPT = """
You are a job description analyzer. Output ONLY valid JSON.

Extract required skills and qualifications from the job description:

{
  "required_skills": ["string"],
  "preferred_skills": ["string"],
  "experience_level": "string",
  "role_title": "string",
  "field": "string",
  "key_responsibilities": ["string"]
}

RULES:
1. Separate required vs preferred/nice-to-have skills.
2. Normalize skill names.
3. Output VALID JSON only.
"""


def parse_resume(resume_text: str) -> dict:
    """Parse resume text and extract structured information."""
    if not resume_text or not resume_text.strip():
        return {"skills": [], "education": [], "experience": [], "certifications": [], "summary": ""}

    if not is_api_key_set():
        return _fallback_resume_parse(resume_text)

    try:
        result = generate_json(RESUME_SYSTEM_PROMPT, f"Parse this resume:\n\n{resume_text}")
        return result if result else _fallback_resume_parse(resume_text)
    except Exception as e:
        logger.error(f"Resume parsing failed: {e}")
        return _fallback_resume_parse(resume_text)


def parse_job_description(jd_text: str) -> dict:
    """Parse job description and extract required skills."""
    if not jd_text or not jd_text.strip():
        return {"required_skills": [], "preferred_skills": [], "field": "general"}

    if not is_api_key_set():
        return _fallback_jd_parse(jd_text)

    try:
        result = generate_json(JD_SYSTEM_PROMPT, f"Analyze this job description:\n\n{jd_text}")
        return result if result else _fallback_jd_parse(jd_text)
    except Exception as e:
        logger.error(f"JD parsing failed: {e}")
        return _fallback_jd_parse(jd_text)


def _fallback_resume_parse(text: str) -> dict:
    """Basic keyword extraction when LLM is unavailable."""
    text_lower = text.lower()
    known_skills = [
        "python", "javascript", "java", "c++", "c#", "sql", "html", "css",
        "react", "node.js", "django", "flask", "fastapi", "machine learning",
        "deep learning", "tensorflow", "pytorch", "pandas", "numpy",
        "docker", "kubernetes", "aws", "azure", "gcp", "git",
        "data analysis", "statistics", "excel", "tableau", "power bi",
    ]
    found = [s for s in known_skills if s in text_lower]
    return {
        "skills": found,
        "education": [],
        "experience": [],
        "certifications": [],
        "summary": f"Extracted {len(found)} skills from resume."
    }


def _fallback_jd_parse(text: str) -> dict:
    """Basic keyword extraction for JD when LLM is unavailable."""
    text_lower = text.lower()
    known_skills = [
        "python", "javascript", "java", "sql", "machine learning",
        "deep learning", "data analysis", "statistics", "communication",
        "project management", "leadership", "react", "node.js",
        "cloud computing", "aws", "docker", "agile",
    ]
    found = [s for s in known_skills if s in text_lower]
    return {
        "required_skills": found,
        "preferred_skills": [],
        "experience_level": "mid-level",
        "role_title": "Professional",
        "field": "tech",
        "key_responsibilities": []
    }
