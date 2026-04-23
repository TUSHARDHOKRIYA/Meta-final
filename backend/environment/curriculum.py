"""
EduPath AI — Curriculum Graph & Topic Database
Team KRIYA | Meta Hackathon 2026

Defines the multi-field curriculum as a directed prerequisite graph.
Contains 70+ topics across tech, healthcare, business, law, and design
domains, each with curated learning resources, difficulty ratings, and
estimated study hours. Also includes the project database for mini and
capstone projects.
"""
from typing import Dict, List
from environment.models import Topic, Resource, ResourceType, ProjectMilestone


# ─── Resource Database ──────────────────────────────────────────────────────────

RESOURCE_DB: Dict[str, List[Resource]] = {
    # TECH / CS
    "python_basics": [
        Resource(title="Python for Everybody (Coursera - Audit)", type=ResourceType.PLATFORM_REDIRECT, url="https://www.coursera.org/specializations/python", platform="Coursera"),
        Resource(title="Python Official Tutorial", type=ResourceType.OFFICIAL_DOCS, url="https://docs.python.org/3/tutorial/", platform="Python.org"),
        Resource(title="Kaggle Python Course", type=ResourceType.FREE_COURSE, url="https://www.kaggle.com/learn/python", platform="Kaggle"),
    ],
    "data_structures": [
        Resource(title="MIT 6.006 Introduction to Algorithms", type=ResourceType.OPEN_COURSEWARE, url="https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/", platform="MIT OCW"),
        Resource(title="freeCodeCamp DSA Article", type=ResourceType.DOMAIN_ARTICLE, url="https://www.freecodecamp.org/news/data-structures-101/", platform="freeCodeCamp"),
    ],
    "machine_learning": [
        Resource(title="Stanford CS229 Machine Learning", type=ResourceType.OPEN_COURSEWARE, url="https://cs229.stanford.edu/", platform="Stanford"),
        Resource(title="Kaggle Intro to Machine Learning", type=ResourceType.FREE_COURSE, url="https://www.kaggle.com/learn/intro-to-machine-learning", platform="Kaggle"),
        Resource(title="Scikit-learn Official Tutorials", type=ResourceType.OFFICIAL_DOCS, url="https://scikit-learn.org/stable/tutorial/", platform="Scikit-learn"),
    ],
    "deep_learning": [
        Resource(title="fast.ai Practical Deep Learning", type=ResourceType.FREE_COURSE, url="https://course.fast.ai/", platform="fast.ai"),
        Resource(title="PyTorch Tutorials", type=ResourceType.OFFICIAL_DOCS, url="https://pytorch.org/tutorials/", platform="PyTorch"),
        Resource(title="Deep Learning Specialization (Coursera - Audit)", type=ResourceType.PLATFORM_REDIRECT, url="https://www.coursera.org/specializations/deep-learning", platform="Coursera"),
    ],
    "web_development": [
        Resource(title="freeCodeCamp Full Stack", type=ResourceType.FREE_COURSE, url="https://www.freecodecamp.org/learn/", platform="freeCodeCamp"),
        Resource(title="MDN Web Docs", type=ResourceType.OFFICIAL_DOCS, url="https://developer.mozilla.org/en-US/docs/Learn", platform="MDN"),
    ],
    "databases": [
        Resource(title="Khan Academy SQL", type=ResourceType.PLATFORM_REDIRECT, url="https://www.khanacademy.org/computing/computer-programming/sql", platform="Khan Academy"),
        Resource(title="PostgreSQL Official Docs", type=ResourceType.OFFICIAL_DOCS, url="https://www.postgresql.org/docs/current/tutorial.html", platform="PostgreSQL"),
    ],
    "statistics": [
        Resource(title="Khan Academy Statistics", type=ResourceType.PLATFORM_REDIRECT, url="https://www.khanacademy.org/math/statistics-probability", platform="Khan Academy"),
        Resource(title="MIT 18.05 Probability and Statistics", type=ResourceType.OPEN_COURSEWARE, url="https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", platform="MIT OCW"),
    ],
    "data_analysis": [
        Resource(title="Kaggle Data Analysis Course", type=ResourceType.FREE_COURSE, url="https://www.kaggle.com/learn/pandas", platform="Kaggle"),
        Resource(title="Towards Data Science - Data Analysis Guide", type=ResourceType.DOMAIN_ARTICLE, url="https://towardsdatascience.com/", platform="TDS"),
    ],
    # HEALTHCARE
    "medical_ai": [
        Resource(title="AI for Medicine (Coursera - Audit)", type=ResourceType.PLATFORM_REDIRECT, url="https://www.coursera.org/specializations/ai-for-medicine", platform="Coursera"),
        Resource(title="HuggingFace Medical NLP", type=ResourceType.FREE_COURSE, url="https://huggingface.co/blog/medical-nlp", platform="HuggingFace"),
    ],
    "clinical_data": [
        Resource(title="MIT Clinical Data Science", type=ResourceType.OPEN_COURSEWARE, url="https://ocw.mit.edu/courses/hst-953-collaborative-data-science-for-healthcare-fall-2020/", platform="MIT OCW"),
    ],
    # BUSINESS
    "business_analytics": [
        Resource(title="HBR Analytics Articles", type=ResourceType.DOMAIN_ARTICLE, url="https://hbr.org/topic/analytics", platform="HBR"),
        Resource(title="edX Business Analytics (Audit)", type=ResourceType.PLATFORM_REDIRECT, url="https://www.edx.org/learn/business-analytics", platform="edX"),
    ],
    # LAW
    "legal_tech": [
        Resource(title="Stanford Legal Tech", type=ResourceType.OPEN_COURSEWARE, url="https://law.stanford.edu/codex-the-stanford-center-for-legal-informatics/", platform="Stanford Law"),
    ],
    # DESIGN
    "design_fundamentals": [
        Resource(title="Google UX Design (Coursera - Audit)", type=ResourceType.PLATFORM_REDIRECT, url="https://www.coursera.org/professional-certificates/google-ux-design", platform="Coursera"),
    ],
    "api_development": [
        Resource(title="FastAPI Official Tutorial", type=ResourceType.OFFICIAL_DOCS, url="https://fastapi.tiangolo.com/tutorial/", platform="FastAPI"),
    ],
    "version_control": [
        Resource(title="Git Official Docs", type=ResourceType.OFFICIAL_DOCS, url="https://git-scm.com/doc", platform="Git"),
    ],
    "cloud_computing": [
        Resource(title="AWS Free Tier Training", type=ResourceType.FREE_COURSE, url="https://aws.amazon.com/training/digital/", platform="AWS"),
    ],
}


# ─── Topic Graph (Multi-Field) ─────────────────────────────────────────────────

def build_topic_graph() -> Dict[str, Topic]:
    """Build the full topic graph with prerequisites for all supported fields."""
    topics = {}

    # ══════ TECH / CS TOPICS ══════
    tech_topics = [
        Topic(id="python_basics", name="Python Fundamentals", field="tech", prerequisites=[], estimated_hours=8, difficulty=1,
              resources=RESOURCE_DB.get("python_basics", [])),
        Topic(id="python_control_flow", name="Python Control Flow & Functions", field="tech", prerequisites=["python_basics"], estimated_hours=6, difficulty=1,
              resources=RESOURCE_DB.get("python_basics", [])),
        Topic(id="python_oop", name="Object-Oriented Python", field="tech", prerequisites=["python_control_flow"], estimated_hours=8, difficulty=2,
              resources=RESOURCE_DB.get("python_basics", [])),
        Topic(id="data_structures", name="Data Structures & Algorithms", field="tech", prerequisites=["python_oop"], estimated_hours=12, difficulty=3,
              resources=RESOURCE_DB.get("data_structures", [])),
        Topic(id="version_control", name="Git & Version Control", field="tech", prerequisites=["python_basics"], estimated_hours=4, difficulty=1,
              resources=RESOURCE_DB.get("version_control", [])),
        Topic(id="databases", name="SQL & Databases", field="tech", prerequisites=["python_basics"], estimated_hours=8, difficulty=2,
              resources=RESOURCE_DB.get("databases", [])),
        Topic(id="statistics", name="Statistics & Probability", field="tech", prerequisites=[], estimated_hours=10, difficulty=2,
              resources=RESOURCE_DB.get("statistics", [])),
        Topic(id="data_analysis", name="Data Analysis with Pandas", field="tech", prerequisites=["python_control_flow", "statistics"], estimated_hours=8, difficulty=2,
              resources=RESOURCE_DB.get("data_analysis", [])),
        Topic(id="data_visualization", name="Data Visualization", field="tech", prerequisites=["data_analysis"], estimated_hours=6, difficulty=2,
              resources=RESOURCE_DB.get("data_analysis", [])),
        Topic(id="machine_learning", name="Machine Learning Fundamentals", field="tech", prerequisites=["data_analysis", "statistics"], estimated_hours=15, difficulty=3,
              resources=RESOURCE_DB.get("machine_learning", [])),
        Topic(id="deep_learning", name="Deep Learning & Neural Networks", field="tech", prerequisites=["machine_learning"], estimated_hours=15, difficulty=4,
              resources=RESOURCE_DB.get("deep_learning", [])),
        Topic(id="nlp", name="Natural Language Processing", field="tech", prerequisites=["deep_learning"], estimated_hours=12, difficulty=4,
              resources=RESOURCE_DB.get("deep_learning", [])),
        Topic(id="computer_vision", name="Computer Vision", field="tech", prerequisites=["deep_learning"], estimated_hours=12, difficulty=4,
              resources=RESOURCE_DB.get("deep_learning", [])),
        Topic(id="web_development", name="Web Development (HTML/CSS/JS)", field="tech", prerequisites=["python_basics"], estimated_hours=10, difficulty=2,
              resources=RESOURCE_DB.get("web_development", [])),
        Topic(id="api_development", name="API Development with FastAPI", field="tech", prerequisites=["python_oop", "web_development"], estimated_hours=8, difficulty=3,
              resources=RESOURCE_DB.get("api_development", [])),
        Topic(id="cloud_computing", name="Cloud Computing Basics", field="tech", prerequisites=["api_development"], estimated_hours=8, difficulty=3,
              resources=RESOURCE_DB.get("cloud_computing", [])),
        Topic(id="mlops", name="MLOps & Model Deployment", field="tech", prerequisites=["machine_learning", "cloud_computing"], estimated_hours=10, difficulty=4,
              resources=RESOURCE_DB.get("cloud_computing", [])),
    ]

    # ══════ HEALTHCARE / MEDICAL AI TOPICS ══════
    healthcare_topics = [
        Topic(id="hc_biology_basics", name="Biology & Anatomy Refresher", field="healthcare", prerequisites=[], estimated_hours=6, difficulty=1),
        Topic(id="hc_medical_terminology", name="Medical Terminology for AI", field="healthcare", prerequisites=[], estimated_hours=4, difficulty=1),
        Topic(id="hc_python_for_health", name="Python for Healthcare Data", field="healthcare", prerequisites=["python_basics"], estimated_hours=8, difficulty=2,
              resources=RESOURCE_DB.get("python_basics", [])),
        Topic(id="hc_clinical_data", name="Clinical Data & EHR Systems", field="healthcare", prerequisites=["hc_medical_terminology", "hc_python_for_health"], estimated_hours=10, difficulty=3,
              resources=RESOURCE_DB.get("clinical_data", [])),
        Topic(id="hc_medical_imaging", name="Medical Imaging & AI", field="healthcare", prerequisites=["hc_clinical_data", "deep_learning"], estimated_hours=12, difficulty=4,
              resources=RESOURCE_DB.get("medical_ai", [])),
        Topic(id="hc_drug_discovery", name="AI for Drug Discovery", field="healthcare", prerequisites=["hc_clinical_data", "machine_learning"], estimated_hours=10, difficulty=4,
              resources=RESOURCE_DB.get("medical_ai", [])),
        Topic(id="hc_medical_nlp", name="Medical NLP & Clinical Text", field="healthcare", prerequisites=["hc_clinical_data", "nlp"], estimated_hours=10, difficulty=4,
              resources=RESOURCE_DB.get("medical_ai", [])),
    ]

    # ══════ BUSINESS TOPICS ══════
    business_topics = [
        Topic(id="biz_fundamentals", name="Business Fundamentals", field="business", prerequisites=[], estimated_hours=6, difficulty=1),
        Topic(id="biz_analytics", name="Business Analytics", field="business", prerequisites=["biz_fundamentals", "statistics"], estimated_hours=10, difficulty=2,
              resources=RESOURCE_DB.get("business_analytics", [])),
        Topic(id="biz_data_driven", name="Data-Driven Decision Making", field="business", prerequisites=["biz_analytics", "data_analysis"], estimated_hours=8, difficulty=3,
              resources=RESOURCE_DB.get("business_analytics", [])),
        Topic(id="biz_strategy", name="AI Strategy for Business", field="business", prerequisites=["biz_data_driven"], estimated_hours=8, difficulty=3),
    ]

    # ══════ LAW TOPICS ══════
    law_topics = [
        Topic(id="law_fundamentals", name="Legal Reasoning Basics", field="law", prerequisites=[], estimated_hours=6, difficulty=1),
        Topic(id="law_legal_tech", name="Legal Tech & AI in Law", field="law", prerequisites=["law_fundamentals"], estimated_hours=8, difficulty=2,
              resources=RESOURCE_DB.get("legal_tech", [])),
        Topic(id="law_contract_ai", name="Contract Analysis with AI", field="law", prerequisites=["law_legal_tech", "nlp"], estimated_hours=10, difficulty=3),
        Topic(id="law_compliance", name="AI Compliance & Ethics", field="law", prerequisites=["law_legal_tech"], estimated_hours=6, difficulty=2),
    ]

    # ══════ DESIGN TOPICS ══════
    design_topics = [
        Topic(id="des_fundamentals", name="Design Thinking", field="design", prerequisites=[], estimated_hours=6, difficulty=1,
              resources=RESOURCE_DB.get("design_fundamentals", [])),
        Topic(id="des_ux_research", name="UX Research Methods", field="design", prerequisites=["des_fundamentals"], estimated_hours=8, difficulty=2,
              resources=RESOURCE_DB.get("design_fundamentals", [])),
        Topic(id="des_ui_design", name="UI Design & Prototyping", field="design", prerequisites=["des_ux_research"], estimated_hours=10, difficulty=2),
        Topic(id="des_ai_tools", name="AI-Powered Design Tools", field="design", prerequisites=["des_ui_design"], estimated_hours=6, difficulty=3),
    ]

    all_topics = tech_topics + healthcare_topics + business_topics + law_topics + design_topics
    for t in all_topics:
        topics[t.id] = t

    return topics


# ─── Project Milestones ─────────────────────────────────────────────────────────

def build_project_milestones() -> Dict[str, ProjectMilestone]:
    """Build field-specific project milestones."""
    projects = {}
    project_list = [
        # Tech mini projects
        ProjectMilestone(id="proj_calculator", title="CLI Calculator", description="Build a command-line calculator with basic operations", requirements=["Python basics", "Control flow"], skills_tested=["python_basics", "python_control_flow"], field="tech"),
        ProjectMilestone(id="proj_todo_app", title="To-Do List App", description="Create a to-do list with file persistence", requirements=["OOP", "File handling"], skills_tested=["python_oop"], field="tech"),
        ProjectMilestone(id="proj_data_dashboard", title="Data Analysis Dashboard", description="Analyze a real dataset and create visualizations", requirements=["Pandas", "Matplotlib"], skills_tested=["data_analysis", "data_visualization"], field="tech"),
        ProjectMilestone(id="proj_ml_classifier", title="ML Classification Model", description="Build and evaluate a classification model on a real dataset", requirements=["Scikit-learn", "Pandas"], skills_tested=["machine_learning"], field="tech", is_capstone=True),
        ProjectMilestone(id="proj_web_api", title="REST API Project", description="Build a full REST API with authentication", requirements=["FastAPI", "Database"], skills_tested=["api_development", "databases"], field="tech"),
        ProjectMilestone(id="proj_full_stack", title="Full-Stack Application", description="Build a complete web application with frontend and backend", requirements=["Frontend", "Backend", "Database"], skills_tested=["web_development", "api_development", "databases"], field="tech", is_capstone=True),
        # Healthcare mini projects
        ProjectMilestone(id="proj_ehr_analysis", title="EHR Data Analysis", description="Analyze electronic health records dataset", requirements=["Python", "Pandas", "Medical knowledge"], skills_tested=["hc_clinical_data"], field="healthcare"),
        ProjectMilestone(id="proj_medical_image", title="Medical Image Classifier", description="Build an AI model to classify medical images", requirements=["Deep Learning", "Medical imaging"], skills_tested=["hc_medical_imaging"], field="healthcare", is_capstone=True),
        # Business mini projects
        ProjectMilestone(id="proj_market_analysis", title="Market Analysis Report", description="Conduct data-driven market analysis", requirements=["Analytics", "Visualization"], skills_tested=["biz_analytics"], field="business"),
        ProjectMilestone(id="proj_biz_case_study", title="AI Strategy Case Study", description="Develop an AI strategy for a business scenario", requirements=["Business fundamentals", "AI knowledge"], skills_tested=["biz_strategy"], field="business", is_capstone=True),
        # Law mini projects
        ProjectMilestone(id="proj_contract_review", title="Contract Review Assistant", description="Build a simple contract analysis tool", requirements=["NLP", "Legal knowledge"], skills_tested=["law_contract_ai"], field="law"),
        # Design mini projects
        ProjectMilestone(id="proj_ux_audit", title="UX Audit Report", description="Conduct a UX audit of an existing product", requirements=["UX Research", "Design Thinking"], skills_tested=["des_ux_research"], field="design"),
    ]

    for p in project_list:
        projects[p.id] = p

    return projects


# ─── Utility Functions ──────────────────────────────────────────────────────────

TOPIC_GRAPH = build_topic_graph()
PROJECT_DB = build_project_milestones()


def get_topics_for_field(field: str) -> List[Topic]:
    """Get all topics relevant for a given field."""
    field_topics = [t for t in TOPIC_GRAPH.values() if t.field == field]
    # Also include relevant tech foundations
    if field != "tech":
        tech_foundations = [t for t in TOPIC_GRAPH.values() if t.field == "tech" and t.difficulty <= 2]
        field_topics = tech_foundations + field_topics
    return field_topics


def get_available_topics(completed: List[str], field: str) -> List[str]:
    """Get topics whose prerequisites are all completed."""
    available = []
    for topic_id, topic in TOPIC_GRAPH.items():
        if topic_id in completed:
            continue
        if topic.field != field and topic.field != "tech":
            continue
        if all(p in completed for p in topic.prerequisites):
            available.append(topic_id)
    return available


def get_projects_for_field(field: str) -> List[ProjectMilestone]:
    """Get project milestones for a given field."""
    return [p for p in PROJECT_DB.values() if p.field == field or p.field == "general"]


def get_resources_for_topic(topic_id: str) -> List[Resource]:
    """Get resources for a specific topic."""
    topic = TOPIC_GRAPH.get(topic_id)
    if topic and topic.resources:
        return topic.resources
    return RESOURCE_DB.get(topic_id, [])
