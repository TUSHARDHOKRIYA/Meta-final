"""
EduPath AI — Student Onboarding API
Team KRIYA | Meta Hackathon 2026

Handles the 4-step onboarding flow: resume upload, skill self-assessment,
job description analysis, and time availability. Creates the student
profile and triggers initial roadmap generation.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from environment.models import OnboardingComplete, SkillLevel
from environment.student import student_manager
from ai.resume_parser import parse_resume, parse_job_description

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])


class Step1Request(BaseModel):
    student_id: str
    resume_text: Optional[str] = None
    linkedin_url: Optional[str] = None


class Step2Request(BaseModel):
    student_id: str
    target_field: str
    skills: List[dict] = []  # [{skill, level, proficiency}]


class Step3Request(BaseModel):
    student_id: str
    job_description: Optional[str] = None


class Step4Request(BaseModel):
    student_id: str
    weekly_hours: int = 10
    learning_goal: str = ""


class FullOnboardingRequest(BaseModel):
    name: str = ""
    email: str = ""
    resume_text: Optional[str] = None
    target_field: str
    skills: List[dict] = []
    job_description: Optional[str] = None
    weekly_hours: int = 10
    learning_goal: str = ""
    user_id: Optional[str] = None  # Supabase auth user ID


# ── Step 1: Resume/LinkedIn (Optional) ──

@router.post("/step1")
async def onboarding_step1(data: Step1Request):
    """Step 1: Upload resume or LinkedIn profile. AI extracts skills."""
    student = student_manager.get(data.student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    extracted = {"skills": []}
    if data.resume_text:
        extracted = parse_resume(data.resume_text)

    student_manager.update_from_onboarding(data.student_id, {
        "resume_skills": extracted.get("skills", [])
    })

    return {"message": "Step 1 complete", "extracted": extracted}


# ── Step 2: Skill Self-Assessment ──

@router.post("/step2")
async def onboarding_step2(data: Step2Request):
    """Step 2: Student enters target field and self-assesses skills with levels."""
    student = student_manager.get(data.student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    student_manager.update_from_onboarding(data.student_id, {
        "target_field": data.target_field,
        "skills": data.skills,
    })

    return {"message": "Step 2 complete", "target_field": data.target_field, "skills_count": len(data.skills)}


# ── Step 3: Job Description (Optional) ──

@router.post("/step3")
async def onboarding_step3(data: Step3Request):
    """Step 3: Paste job description. AI extracts required skills and maps gaps."""
    student = student_manager.get(data.student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    jd_result = {"required_skills": []}
    if data.job_description:
        jd_result = parse_job_description(data.job_description)

    student_manager.update_from_onboarding(data.student_id, {
        "job_description": data.job_description,
        "jd_required_skills": jd_result.get("required_skills", [])
    })

    return {"message": "Step 3 complete", "jd_analysis": jd_result}


# ── Step 4: Time Availability ──

@router.post("/step4")
async def onboarding_step4(data: Step4Request):
    """Step 4: Set weekly hours and learning goal."""
    student = student_manager.get(data.student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    student_manager.update_from_onboarding(data.student_id, {
        "weekly_hours": data.weekly_hours,
        "learning_goal": data.learning_goal,
    })

    return {"message": "Onboarding complete!", "student_id": data.student_id}


# ── Full Onboarding (All-in-One) ──

@router.post("/complete")
async def full_onboarding(data: FullOnboardingRequest):
    """Complete onboarding in a single request."""
    import traceback
    try:
        # Use Supabase user_id if provided, otherwise generate
        if data.user_id:
            student = student_manager.get(data.user_id)
            if not student:
                student = student_manager.create(name=data.name, email=data.email, student_id=data.user_id)
        else:
            student = student_manager.create(name=data.name, email=data.email)

        # Parse resume if provided
        resume_skills = []
        if data.resume_text:
            parsed = parse_resume(data.resume_text)
            resume_skills = parsed.get("skills", [])

        # Parse JD if provided
        jd_skills = []
        if data.job_description:
            jd_result = parse_job_description(data.job_description)
            jd_skills = jd_result.get("required_skills", [])

        # Update student
        student_manager.update_from_onboarding(student.id, {
            "name": data.name,
            "email": data.email,
            "resume_skills": resume_skills,
            "target_field": data.target_field,
            "skills": data.skills,
            "job_description": data.job_description,
            "jd_required_skills": jd_skills,
            "weekly_hours": data.weekly_hours,
            "learning_goal": data.learning_goal,
        })

        return {
            "message": "Onboarding complete!",
            "student_id": student.id,
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Backend Error: {str(e)}\n\nTrace: {error_trace}")


# ── Get Student Profile ──

@router.get("/profile/{student_id}")
async def get_profile(student_id: str):
    """Get student profile."""
    student = student_manager.get(student_id)
    if not student:
        raise HTTPException(404, "Student not found")
    return student.model_dump()
