"""
EduPath AI — Project Submission & Evaluation API
Team KRIYA | Meta Hackathon 2026

Endpoints for submitting mini-projects and capstone projects, receiving
AI-powered evaluations, and tracking submission history.
"""
import os
import json
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from environment.student import student_manager
from ai.project_evaluator import evaluate_project

router = APIRouter(prefix="/api/projects", tags=["projects"])
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "projects")
os.makedirs(DATA_DIR, exist_ok=True)


class ProjectSubmission(BaseModel):
    student_id: str
    project_title: str
    project_description: str
    project_type: str = "mini_project"  # "mini_project" or "capstone"
    submission_text: str  # GitHub URL, code, or description
    requirements: List[str] = []


class ProjectReport(BaseModel):
    project_id: str
    student_id: str
    project_title: str
    project_type: str
    submission_text: str
    submitted_at: str
    evaluation: dict


@router.post("/submit")
async def submit_project(data: ProjectSubmission):
    """Submit a project for AI evaluation."""
    student = student_manager.get(data.student_id)
    if not student:
        raise HTTPException(404, "Student not found. Complete onboarding first.")

    if not data.submission_text.strip():
        raise HTTPException(400, "Submission cannot be empty. Please provide a GitHub URL or project description.")

    # Generate project ID
    project_id = f"{data.project_type}_{data.project_title.lower().replace(' ', '_')}_{data.student_id[:8]}"

    logger.info(f"Evaluating project: {data.project_title} for student {data.student_id}")

    # Run AI evaluation
    evaluation = evaluate_project(
        project_title=data.project_title,
        project_description=data.project_description,
        submission_text=data.submission_text,
        requirements=data.requirements,
        project_type=data.project_type,
    )

    # Build report
    report = {
        "project_id": project_id,
        "student_id": data.student_id,
        "project_title": data.project_title,
        "project_type": data.project_type,
        "submission_text": data.submission_text,
        "submitted_at": datetime.utcnow().isoformat(),
        "evaluation": evaluation,
    }

    # Save report to disk
    report_file = os.path.join(DATA_DIR, f"{project_id}.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Sync to Supabase
    try:
        from db.supabase_client import save_project_report
        save_project_report(report)
    except Exception:
        pass

    # Record completed project in student profile
    if evaluation.get("is_passing", False):
        student_manager.complete_project(data.student_id, project_id)

    return {
        "message": "Project evaluated successfully!",
        "project_id": project_id,
        "evaluation": evaluation,
    }


@router.get("/report/{student_id}/{project_id}")
async def get_project_report(student_id: str, project_id: str):
    """Get the evaluation report for a specific project."""
    report_file = os.path.join(DATA_DIR, f"{project_id}.json")
    if not os.path.exists(report_file):
        raise HTTPException(404, "Project report not found.")

    with open(report_file, "r") as f:
        report = json.load(f)

    if report.get("student_id") != student_id:
        raise HTTPException(403, "You don't have permission to view this report.")

    return report


@router.get("/reports/{student_id}")
async def get_all_project_reports(student_id: str):
    """Get all project reports for a student."""
    student = student_manager.get(student_id)
    if not student:
        raise HTTPException(404, "Student not found.")

    reports = []
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(DATA_DIR, filename)
                with open(filepath, "r") as f:
                    report = json.load(f)
                if report.get("student_id") == student_id:
                    reports.append({
                        "project_id": report["project_id"],
                        "project_title": report["project_title"],
                        "project_type": report["project_type"],
                        "submitted_at": report["submitted_at"],
                        "score": report["evaluation"].get("score", 0),
                        "grade": report["evaluation"].get("grade", "N/A"),
                        "is_passing": report["evaluation"].get("is_passing", False),
                    })

    return {
        "student_id": student_id,
        "total_submitted": len(reports),
        "reports": sorted(reports, key=lambda x: x["submitted_at"], reverse=True),
    }
