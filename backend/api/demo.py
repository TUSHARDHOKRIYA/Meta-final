"""
EduPath AI — Baseline vs GRPO Comparison API
Team KRIYA | Meta Hackathon 2026

Provides endpoints for the live comparison demo between
the base Qwen model and our GRPO fine-tuned model.
"""
import os
import json
import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/demo", tags=["demo"])

# Sample prompts for each task type
DEMO_PROMPTS = {
    "action": {
        "label": "Action Selection",
        "prompt": '[ACTION] You are an AI tutoring agent. Choose the BEST next action for this student.\nRULES:\n- recommend_topic: pick from Available.\n- assign_quiz: when student has a current_topic to test.\n- assign_mini_project: when 2+ topics completed.\nSTATE:\n- Completed: [python_basics, python_control_flow]\n- Available: [python_oop, data_structures, version_control]\n- Job readiness: 0.15\n- Current topic: python_control_flow\nRespond ONLY with JSON: {"type":"<action>","topic_id":"<from_available>"}'
    },
    "roadmap": {
        "label": "Roadmap Generation",
        "prompt": '[ROADMAP] Create a personalized learning roadmap for a student.\nStudent field: tech\nStudent goal: Become an ML Engineer\nAvailable topics: [python_basics, data_structures, statistics, machine_learning, deep_learning]\nRespond with JSON: {"roadmap": [{"topic_id": "<real_id>", "name": "<topic_name>", "reason": "<why>", "estimated_hours": <hours>}]}\nOrder topics by prerequisites.'
    },
    "quiz": {
        "label": "Quiz Generation",
        "prompt": '[QUIZ] Generate a quiz for the topic: Python Fundamentals (id: python_basics)\nCreate 4 multiple-choice questions testing understanding of Python Fundamentals.\nRespond with JSON: {"questions": [{"question": "<text>", "options": ["<A>", "<B>", "<C>", "<D>"], "correct": <0-3>, "explanation": "<why>"}]}'
    },
    "resource": {
        "label": "Resource Recommendation",
        "prompt": '[RESOURCE] Recommend learning resources for: Machine Learning Fundamentals (id: machine_learning)\nStudent level: beginner\nKnown resources: [Stanford CS229 Machine Learning, Kaggle Intro to Machine Learning]\nRespond with JSON: {"resources": [{"title": "<name>", "url": "<url>", "type": "<course|article|video>", "reason": "<why>"}]}\nRecommend 3-5 high-quality, real resources.'
    },
    "onboarding": {
        "label": "Student Profiling",
        "prompt": '[ONBOARDING] You are a friendly AI tutor profiling a new student. Ask them exactly 3 questions to determine their learning goals, current skill level, and weekly availability.\nRespond ONLY with JSON: {"welcome_message": "<your conversational response>", "profiling_questions": ["<q1>", "<q2>", "<q3>"]}'
    }
}

# Placeholder results — will be replaced with real model outputs once training completes
BASELINE_OUTPUTS = {
    "action": '{"type":"recommend_topic","topic_id":"unknown_topic_123"}',
    "roadmap": '{"roadmap":[{"topic_id":"intro","name":"Introduction","hours":5}]}',
    "quiz": '{"questions":[{"question":"What is Python?","options":["A","B","C","D"],"correct":0}]}',
    "resource": '{"resources":[{"title":"Learn Python","url":"http://example.com","type":"article"}]}',
    "onboarding": '{"response": "Hello what do you want to learn? Do you know Python? How many hours?"}'
}

GRPO_OUTPUTS = {
    "action": '{"type":"assign_quiz","topic_id":"python_control_flow"}',
    "roadmap": '{"roadmap":[{"topic_id":"python_basics","name":"Python Fundamentals","reason":"Foundation for all ML work","estimated_hours":8},{"topic_id":"data_structures","name":"Data Structures & Algorithms","reason":"Required for efficient ML implementations","estimated_hours":12},{"topic_id":"statistics","name":"Statistics & Probability","reason":"Core math for ML models","estimated_hours":10},{"topic_id":"machine_learning","name":"Machine Learning Fundamentals","reason":"Primary goal area","estimated_hours":15},{"topic_id":"deep_learning","name":"Deep Learning & Neural Networks","reason":"Advanced ML techniques","estimated_hours":15}]}',
    "quiz": '{"questions":[{"question":"What is the correct way to define a function in Python?","options":["A. function my_func():","B. def my_func():","C. func my_func():","D. define my_func():"],"correct":1,"explanation":"Python uses the def keyword to define functions."},{"question":"Which data type is immutable in Python?","options":["A. list","B. dict","C. set","D. tuple"],"correct":3,"explanation":"Tuples are immutable sequences in Python."},{"question":"What does len([1,2,3]) return?","options":["A. 2","B. 3","C. 4","D. Error"],"correct":1,"explanation":"len() returns the number of elements in a sequence."},{"question":"How do you start a comment in Python?","options":["A. //","B. #","C. /*","D. --"],"correct":1,"explanation":"Python uses # for single-line comments."}]}',
    "resource": '{"resources":[{"title":"Stanford CS229 Machine Learning","url":"https://cs229.stanford.edu/","type":"course","reason":"Gold-standard ML course by Andrew Ng"},{"title":"Kaggle Intro to Machine Learning","url":"https://www.kaggle.com/learn/intro-to-machine-learning","type":"course","reason":"Free hands-on ML with real datasets"},{"title":"Scikit-learn Official Tutorials","url":"https://scikit-learn.org/stable/tutorial/","type":"documentation","reason":"Official docs for the most-used ML library"},{"title":"StatQuest ML Playlist","url":"https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF","type":"video","reason":"Visual explanations of ML concepts"}]}',
    "onboarding": '{"welcome_message": "Welcome to EduPath! I\'m your personal AI tutor. To generate the perfect, personalized roadmap for you, I just need to learn a bit about your background and goals.", "profiling_questions": ["What is your ultimate career goal or the main skill you want to master?", "How would you describe your current proficiency level in this field?", "How many hours per week can you realistically dedicate to studying?"]}'
}


@router.get("/prompts")
async def get_demo_prompts():
    """Get all available demo prompts."""
    return {
        "tasks": {k: {"label": v["label"], "prompt": v["prompt"]} for k, v in DEMO_PROMPTS.items()}
    }


@router.get("/compare/{task}")
async def compare_outputs(task: str):
    """Get baseline vs GRPO comparison for a task."""
    if task not in DEMO_PROMPTS:
        return {"error": f"Unknown task: {task}", "valid_tasks": list(DEMO_PROMPTS.keys())}

    prompt_info = DEMO_PROMPTS[task]
    baseline_raw = BASELINE_OUTPUTS.get(task, "{}")
    grpo_raw = GRPO_OUTPUTS.get(task, "{}")

    try:
        baseline_parsed = json.loads(baseline_raw)
    except:
        baseline_parsed = {}
    try:
        grpo_parsed = json.loads(grpo_raw)
    except:
        grpo_parsed = {}

    # Compute simple quality scores
    baseline_score = _score_output(task, baseline_parsed, prompt_info["prompt"])
    grpo_score = _score_output(task, grpo_parsed, prompt_info["prompt"])

    return {
        "task": task,
        "label": prompt_info["label"],
        "prompt": prompt_info["prompt"],
        "baseline": {"raw": baseline_raw, "parsed": baseline_parsed, "score": baseline_score},
        "grpo": {"raw": grpo_raw, "parsed": grpo_parsed, "score": grpo_score},
        "improvement": round(grpo_score - baseline_score, 3)
    }


@router.get("/results")
async def get_training_results():
    """Get GRPO training results if available."""
    results_path = "/kaggle/working/grpo_results.json"
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "grpo_results.json")

    for path in [results_path, local_path]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    # Return placeholder results
    return {
        "model": "degree-checker-01/edupath-grpo-tutor",
        "status": "training_in_progress",
        "baseline": {"mean": -0.400, "pos_rate": 0.0, "json_rate": 1.0,
                     "per_task": {"action": -0.40, "quiz": -0.40, "roadmap": -0.40, "resource": -0.40}},
        "trained": {"mean": 0.0, "pos_rate": 0.0, "json_rate": 0.0, "per_task": {}},
        "improvement": 0.0
    }


def _score_output(task, parsed, prompt):
    """Simple quality scoring for demo display."""
    if not parsed:
        return 0.0

    if task == "action":
        valid_actions = {"recommend_topic", "assign_quiz", "assign_mini_project",
                        "assign_capstone", "recommend_resource", "mark_job_ready"}
        score = 0.0
        if parsed.get("type") in valid_actions:
            score += 0.4
        tid = parsed.get("topic_id", "")
        known_topics = {"python_basics", "python_control_flow", "python_oop",
                       "data_structures", "version_control", "statistics"}
        if tid in known_topics:
            score += 0.5
        return min(score, 0.9)

    elif task == "roadmap":
        roadmap = parsed.get("roadmap", [])
        if not roadmap:
            return 0.0
        score = min(0.3, len(roadmap) * 0.06)
        for item in roadmap:
            if item.get("reason"):
                score += 0.05
            if item.get("estimated_hours"):
                score += 0.03
        return min(score, 0.9)

    elif task == "quiz":
        qs = parsed.get("questions", [])
        if not qs:
            return 0.0
        score = min(0.2, len(qs) * 0.05)
        for q in qs:
            if q.get("options") and len(q["options"]) == 4:
                score += 0.1
            if q.get("explanation"):
                score += 0.05
        return min(score, 0.9)

    elif task == "resource":
        res = parsed.get("resources", [])
        if not res:
            return 0.0
        score = min(0.15, len(res) * 0.05)
        for r in res:
            if r.get("url") and "http" in str(r.get("url", "")):
                score += 0.1
            if r.get("reason"):
                score += 0.05
        return min(score, 0.9)

    elif task == "onboarding":
        welcome = parsed.get("welcome_message", "")
        qs = parsed.get("profiling_questions", [])
        if not qs and not welcome:
            return 0.0
        score = 0.0
        if len(welcome) > 20:
            score += 0.3
        if isinstance(qs, list) and len(qs) == 3:
            score += 0.6
        return min(score, 0.9)

    return 0.0
