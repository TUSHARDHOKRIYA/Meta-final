"""
EduPath AI — Core Data Models
Team KRIYA | Meta Hackathon 2026

Pydantic models defining the OpenEnv interface (Observation, Action,
Reward, StepResult), student profile schema, curriculum data structures
(Topic, Resource, ProjectMilestone), quiz/onboarding request types,
and resource page response models.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


# ─── Enums ──────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    RECOMMEND_TOPIC = "recommend_topic"
    ASSIGN_QUIZ = "assign_quiz"
    ASSIGN_MINI_PROJECT = "assign_mini_project"
    ASSIGN_CAPSTONE = "assign_capstone"
    RECOMMEND_RESOURCE = "recommend_resource"
    SUGGEST_EVENT = "suggest_event_or_hackathon"
    MARK_JOB_READY = "mark_job_ready"


class ResourceType(str, Enum):
    FREE_COURSE = "free_course"
    INTERACTIVE_NOTEBOOK = "interactive_notebook"
    OFFICIAL_DOCS = "official_docs"
    OPEN_COURSEWARE = "open_courseware"
    DOMAIN_ARTICLE = "domain_article"
    PLATFORM_REDIRECT = "platform_redirect"


class QuizDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BadgeType(str, Enum):
    TOPIC_COMPLETION = "topic_completion"
    QUIZ_STREAK = "quiz_streak"
    PROJECT_SUCCESS = "project_success"
    MILESTONE = "milestone"
    JOB_READY = "job_ready"


# ─── Topic & Curriculum ────────────────────────────────────────────────────────

class Resource(BaseModel):
    title: str
    type: ResourceType
    url: str
    description: str = ""
    platform: str = ""


class Topic(BaseModel):
    id: str
    name: str
    field: str  # e.g., "tech", "healthcare", "law", "business", "design"
    prerequisites: List[str] = []
    resources: List[Resource] = []
    estimated_hours: float = 2.0
    difficulty: int = 1  # 1-5


class ProjectMilestone(BaseModel):
    id: str
    title: str
    description: str
    requirements: List[str] = []
    expected_output: str = ""
    skills_tested: List[str] = []
    is_capstone: bool = False
    field: str = "general"


# ─── Student State ──────────────────────────────────────────────────────────────

class SkillLevel(BaseModel):
    skill: str
    level: str  # e.g., "Know till Loops", "Beginner", "Advanced"
    proficiency: float = 0  # 0 to 1


class QuizResult(BaseModel):
    topic_id: str
    score: float  # 0-100
    total_questions: int
    correct_answers: int
    passed: bool
    difficulty: QuizDifficulty = QuizDifficulty.MEDIUM


class Badge(BaseModel):
    id: str
    name: str
    description: str
    type: BadgeType
    icon: str = "🏆"
    earned_at: Optional[str] = None


class StudentProfile(BaseModel):
    id: str = ""
    name: str = ""
    email: str = ""
    # Onboarding data
    resume_skills: List[str] = []
    self_assessed_skills: List[SkillLevel] = []
    target_field: str = ""
    learning_goal: str = ""
    job_description: Optional[str] = None
    jd_required_skills: List[str] = []
    weekly_hours: int = 10
    # State
    completed_topics: List[str] = []
    current_topic: Optional[str] = None
    quiz_history: List[QuizResult] = []
    completed_projects: List[str] = []
    badges: List[Badge] = []
    job_readiness_score: float = 0.001
    total_study_hours: float = 0
    quiz_streak: int = 0
    # Resource tracking
    clicked_resource_links: Dict[str, List[str]] = {}  # topic_id -> [urls]
    topics_studied: List[str] = []  # topics where link clicked + marked complete


# ─── OpenEnv Interface Models ──────────────────────────────────────────────────

class Observation(BaseModel):
    """What the AI agent sees about the current student + environment state."""
    student_id: str
    completed_topics: List[str]
    current_topic: Optional[str]
    available_topics: List[str]
    skill_levels: Dict[str, float]
    quiz_history_summary: Dict[str, float]  # topic_id -> last score
    completed_projects: List[str]
    job_readiness_score: float
    total_steps: int
    badges_earned: int
    weekly_hours: int
    target_field: str
    learning_goal: str
    mastery_probabilities: Dict[str, float] = {}  # BKT P(known) per topic


class Action(BaseModel):
    """A decision the AI agent makes."""
    type: ActionType
    topic_id: Optional[str] = None
    difficulty: Optional[QuizDifficulty] = None
    project_id: Optional[str] = None
    resource_type: Optional[ResourceType] = None
    event_id: Optional[str] = None


class Reward(BaseModel):
    """Feedback signal after an agent action."""
    value: float
    reason: str
    is_terminal: bool = False


class StepResult(BaseModel):
    """Result of env.step(action) — OpenEnv compliant."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict = {}


# ─── Quiz Models ────────────────────────────────────────────────────────────────

class MCQQuestion(BaseModel):
    question: str
    options: List[str]  # ["A. ...", "B. ...", "C. ...", "D. ..."]
    correct_index: int  # 0-3
    explanation: str = ""
    type: str = "conceptual"  # conceptual, practical, tricky
    topic: str = ""


class QuizSubmission(BaseModel):
    topic_id: str
    answers: List[int]  # indices of selected options


# ─── Onboarding Models ─────────────────────────────────────────────────────────

class OnboardingStep1(BaseModel):
    """Resume/LinkedIn upload (optional)."""
    resume_text: Optional[str] = None
    linkedin_url: Optional[str] = None


class OnboardingStep2(BaseModel):
    """Skill self-assessment."""
    target_field: str  # free text: what they want to learn
    skills: List[SkillLevel]  # existing skills with levels


class OnboardingStep3(BaseModel):
    """Job description (optional)."""
    job_description: Optional[str] = None


class OnboardingStep4(BaseModel):
    """Time availability."""
    weekly_hours: int = 10  # 5, 10, 15, 20+


class OnboardingComplete(BaseModel):
    """Full onboarding data."""
    resume_text: Optional[str] = None
    linkedin_url: Optional[str] = None
    target_field: str
    skills: List[SkillLevel] = []
    job_description: Optional[str] = None
    weekly_hours: int = 10
    learning_goal: str = ""


# ─── Resource Page Models ──────────────────────────────────────────────────────

class ResourceCard(BaseModel):
    title: str
    url: str
    source: str  # "Kaggle" | "freeCodeCamp" | "fast.ai" | "HuggingFace" | "Other"
    description: str
    duration_estimate: str  # e.g. "~2 hours"
    resource_type: str  # "course" | "article" | "notebook"


class TopicPageResponse(BaseModel):
    topic_id: str
    topic_name: str
    ai_summary: str  # 300-500 word AI-generated explanation
    resources: List[ResourceCard]
    can_mark_complete: bool  # true if student already clicked a link
    quiz_unlocked: bool  # true if already marked complete


class LinkClickRequest(BaseModel):
    student_id: str
    resource_url: str


class MarkCompleteRequest(BaseModel):
    student_id: str
