"""
EduPath AI — Full Pipeline End-to-End Test
Tests EVERY component: Auth → Onboarding → Roadmap → Quiz → Projects → Badges → Career → Events
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "backend")

os.environ.setdefault("API_BASE_URL", "https://api-inference.huggingface.co/v1")
os.environ.setdefault("MODEL_NAME", "degree-checker-01/edupath-grpo-tutor")


PASS = 0
FAIL = 0
WARN = 0

def check(name, condition, detail=""):
    global PASS, FAIL, WARN
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  ⚠️  {name} — {detail}")

# ═══════════════════════════════════════════════════════════════
# 1. STUDENT CREATION & ONBOARDING
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("1. STUDENT CREATION & ONBOARDING")
print("="*65)

from environment.student import student_manager

# Test with different profiles
profiles = [
    {"name": "Alice", "field": "tech", "goal": "Become a Full Stack Developer", "skills": ["HTML", "CSS"], "hours": 15},
    {"name": "Bob", "field": "healthcare", "goal": "Learn AI for Medical Imaging", "skills": ["Biology", "Statistics"], "hours": 10},
    {"name": "Carol", "field": "business", "goal": "Start a SaaS Startup", "skills": ["Marketing", "Excel"], "hours": 8},
    {"name": "Dave", "field": "design", "goal": "Become a UX Designer", "skills": ["Photoshop"], "hours": 12},
]

students = {}
for p in profiles:
    s = student_manager.create(name=p["name"])
    student_manager.update_from_onboarding(s.id, {
        "target_field": p["field"],
        "learning_goal": p["goal"],
        "skills": [{"skill": sk, "level": "beginner"} for sk in p["skills"]],
        "weekly_hours": p["hours"],
    })
    students[p["name"]] = s.id
    fetched = student_manager.get(s.id)
    check(f"{p['name']} created ({p['field']})",
          fetched is not None and fetched.target_field == p["field"],
          f"Got field={getattr(fetched, 'target_field', None)}")

check("All 4 students created", len(students) == 4)

# ═══════════════════════════════════════════════════════════════
# 2. ROADMAP GENERATION (different fields)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("2. ROADMAP GENERATION")
print("="*65)

from ai.roadmap_generator import generate_roadmap

# Clear any cached roadmaps
import glob
for f in glob.glob("backend/data/roadmap_*.json"):
    os.remove(f)

test_roadmaps = {}
for name, sid in students.items():
    s = student_manager.get(sid)
    try:
        roadmap = generate_roadmap(
            target_field=s.target_field or "tech",
            learning_goal=s.learning_goal,
            current_skills=[sk.skill for sk in s.self_assessed_skills],
            skill_levels={},
            jd_skills=[],
            weekly_hours=s.weekly_hours,
            duration_weeks=8,
        )
        test_roadmaps[name] = roadmap
        
        has_weeks = "weeks" in roadmap and len(roadmap["weeks"]) > 0
        has_title = "title" in roadmap or "roadmapTitle" in roadmap
        check(f"{name} roadmap generated ({s.target_field})",
              has_weeks,
              f"weeks={len(roadmap.get('weeks', []))}, keys={list(roadmap.keys())[:5]}")
        
        # Check roadmap weeks have content
        if has_weeks:
            first_week = roadmap["weeks"][0]
            has_skills = "skillsCovered" in first_week or "topics" in first_week
            check(f"  {name} week 1 has skills/topics", has_skills,
                  f"week keys={list(first_week.keys())}")
    except Exception as e:
        check(f"{name} roadmap generated", False, str(e)[:100])

# ═══════════════════════════════════════════════════════════════
# 3. QUIZ GENERATION & SCORING
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("3. QUIZ GENERATION & SCORING")
print("="*65)

from ai.quiz_generator import generate_quiz, score_quiz

quiz_topics = ["Python Basics", "Data Visualization", "Machine Learning", "UX Design", "Business Strategy"]
for topic in quiz_topics:
    try:
        quiz = generate_quiz(topic, "medium", 5)
        questions = quiz.get("questions", [])
        check(f"Quiz for '{topic}' generated",
              len(questions) >= 3,
              f"Got {len(questions)} questions")
        
        if questions:
            # Check question structure
            q = questions[0]
            has_options = "options" in q and len(q["options"]) == 4
            has_correct = "correct_index" in q
            has_explanation = "explanation" in q
            check(f"  Question structure valid",
                  has_options and has_correct and has_explanation,
                  f"options={len(q.get('options', []))}, correct_idx={'correct_index' in q}, expl={'explanation' in q}")
            
            # Test scoring — all correct
            correct_answers = [q["correct_index"] for q in questions]
            result = score_quiz(questions, correct_answers)
            check(f"  Perfect score = 100%", result["score"] == 100.0,
                  f"Got {result['score']}")
            check(f"  Perfect score passes", result["passed"] == True)
            check(f"  Recommendation = move_forward", result["recommendation"] == "move_forward",
                  f"Got {result['recommendation']}")
            
            # Test scoring — all wrong
            wrong_answers = [(q["correct_index"] + 1) % 4 for q in questions]
            result2 = score_quiz(questions, wrong_answers)
            check(f"  Zero score = 0%", result2["score"] == 0.0,
                  f"Got {result2['score']}")
            check(f"  Zero score fails", result2["passed"] == False)
            check(f"  Recommendation = restart_topic", result2["recommendation"] == "restart_topic",
                  f"Got {result2['recommendation']}")
    except Exception as e:
        check(f"Quiz for '{topic}'", False, str(e)[:100])

# ═══════════════════════════════════════════════════════════════
# 4. PROJECT EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("4. PROJECT EVALUATION")
print("="*65)

from ai.project_evaluator import evaluate_project

project_tests = [
    ("Todo App", "mini_project", "Build a todo app with CRUD operations", "https://github.com/example/todo-app"),
    ("E-Commerce Platform", "capstone", "Full-stack e-commerce with payments", "Built using React + Node.js + Stripe API. Includes user auth, product catalog, cart, and checkout."),
]

for title, ptype, desc, submission in project_tests:
    try:
        result = evaluate_project(title, desc, submission, [], ptype)
        check(f"Project '{title}' evaluated",
              "score" in result and "grade" in result,
              f"keys={list(result.keys())[:5]}")
        check(f"  Score is numeric (0-100)", 
              isinstance(result.get("score"), (int, float)) and 0 <= result["score"] <= 100,
              f"score={result.get('score')}")
        check(f"  Has strengths & improvements",
              len(result.get("strengths", [])) > 0 and len(result.get("improvements", [])) > 0)
        check(f"  is_passing flag present", "is_passing" in result)
    except Exception as e:
        check(f"Project '{title}'", False, str(e)[:100])

# ═══════════════════════════════════════════════════════════════
# 5. BADGE SYSTEM
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("5. BADGE SYSTEM")
print("="*65)

from environment.models import QuizResult, QuizDifficulty

# Simulate progress for Alice
alice_id = students["Alice"]
alice = student_manager.get(alice_id)

# Complete some topics
for t in ["python_basics", "python_control_flow", "html_css_basics"]:
    student_manager.complete_topic(alice_id, t)

# Record some quiz passes
for t in ["python_basics", "python_control_flow", "html_css_basics"]:
    qr = QuizResult(topic_id=t, score=85, total_questions=5, correct_answers=4, passed=True, difficulty=QuizDifficulty.MEDIUM)
    student_manager.record_quiz(alice_id, qr)

alice = student_manager.get(alice_id)
check("Alice completed 3 topics", len(alice.completed_topics) >= 3,
      f"completed={alice.completed_topics}")
check("Alice has quiz streak", alice.quiz_streak >= 3,
      f"streak={alice.quiz_streak}")

# Check badge awards
from api.badges import BADGE_CATALOG
topics_done = len(alice.completed_topics)
quizzes_passed = len([q for q in alice.quiz_history if q.score >= 70])

earned = set()
topic_thresholds = {"first_step": 1, "topics_3": 3}
for badge_id, threshold in topic_thresholds.items():
    if topics_done >= threshold:
        earned.add(badge_id)

quiz_thresholds = {"first_quiz": 1}
for badge_id, threshold in quiz_thresholds.items():
    if quizzes_passed >= threshold:
        earned.add(badge_id)

streak_thresholds = {"streak_3": 3}
for badge_id, threshold in streak_thresholds.items():
    if alice.quiz_streak >= threshold:
        earned.add(badge_id)

check("'first_step' badge earned", "first_step" in earned)
check("'topics_3' badge earned", "topics_3" in earned)
check("'first_quiz' badge earned", "first_quiz" in earned)
check("'streak_3' badge earned", "streak_3" in earned)

# Test job readiness badges
check("Job readiness score calculated", alice.job_readiness_score >= 0,
      f"readiness={alice.job_readiness_score}")

# ═══════════════════════════════════════════════════════════════
# 6. CAREER & EVENTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("6. CAREER, EVENTS & JOB READINESS")
print("="*65)

from api.career import EVENT_DB
from environment.curriculum import get_projects_for_field

# Test events for each field
for field in ["tech", "healthcare", "business", "law", "design"]:
    events = EVENT_DB.get(field, [])
    check(f"Events exist for '{field}'", len(events) > 0, f"count={len(events)}")
    if events:
        e = events[0]
        check(f"  Event has url & name", "url" in e and "name" in e,
              f"keys={list(e.keys())}")

# Test projects for each field
for field in ["tech", "healthcare", "business"]:
    projects = get_projects_for_field(field)
    check(f"Projects exist for '{field}'", len(projects) > 0, f"count={len(projects)}")

# ═══════════════════════════════════════════════════════════════
# 7. RESOURCE FETCHER (already tested but verify it's not broken)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("7. RESOURCE FETCHER (quick check)")
print("="*65)

from ai.resource_fetcher import fetch_resources_for_topic

# Clear cache
for f in ["backend/cache/topic_resources.json"]:
    if os.path.exists(f): os.remove(f)

try:
    results = fetch_resources_for_topic("Python Basics", "python_basics")
    check("Resource fetcher returns results", len(results) >= 1, f"count={len(results)}")
    check("No Google search fallback", 
          not any("google.com/search" in r.get("url", "") for r in results))
    if results:
        r = results[0]
        check("Resource has url & title", "url" in r and "title" in r)
except Exception as e:
    check("Resource fetcher works", False, str(e)[:100])

# ═══════════════════════════════════════════════════════════════
# 8. GRADERS (Task 1-5)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("8. GRADERS (Task 1-5)")
print("="*65)

from environment.graders import grade_task1, grade_task2, grade_task3, grade_task4, grade_task5

# Task 1: Python beginner
alice = student_manager.get(alice_id)
score1 = grade_task1(alice)
check(f"Task 1 grader returns (0,1)", 0 < score1 < 1, f"score={score1}")

# Task 2: Data analyst
bob_id = students["Bob"]
bob = student_manager.get(bob_id)
for t in ["statistics", "data_analysis", "data_visualization"]:
    student_manager.complete_topic(bob_id, t)
bob = student_manager.get(bob_id)
score2 = grade_task2(bob)
check(f"Task 2 grader returns (0,1)", 0 < score2 < 1, f"score={score2}")

# Task 3: Cross-domain
score3 = grade_task3(bob)
check(f"Task 3 grader returns (0,1)", 0 < score3 < 1, f"score={score3}")

# Task 4: Team learning
all_students = [student_manager.get(sid) for sid in students.values()]
score4 = grade_task4(all_students[:3], steps_used=50)
check(f"Task 4 grader returns (0,1)", 0 < score4 < 1, f"score={score4}")

# Task 5: Career transition
score5 = grade_task5(alice, steps_used=50)
check(f"Task 5 grader returns (0,1)", 0 < score5 < 1, f"score={score5}")

# Edge case: empty student
empty = student_manager.create(name="Empty")
empty_student = student_manager.get(empty.id)
score_empty = grade_task1(empty_student)
check(f"Task 1 empty student returns (0,1)", 0 < score_empty < 1, f"score={score_empty}")

# ═══════════════════════════════════════════════════════════════
# 9. CURRICULUM GRAPH
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("9. CURRICULUM GRAPH")
print("="*65)

from environment.curriculum import TOPIC_GRAPH, PROJECT_DB

check("Topic graph loaded", len(TOPIC_GRAPH) > 10, f"topics={len(TOPIC_GRAPH)}")
check("Project DB loaded", len(PROJECT_DB) > 0, f"projects={len(PROJECT_DB)}")

# Check prerequisite integrity
bad_prereqs = []
for tid, topic in TOPIC_GRAPH.items():
    for prereq in topic.prerequisites:
        if prereq not in TOPIC_GRAPH:
            bad_prereqs.append(f"{tid} requires unknown '{prereq}'")

check("All prerequisites exist in graph", len(bad_prereqs) == 0,
      f"broken: {bad_prereqs[:3]}")

# Check fields
fields = set(t.field for t in TOPIC_GRAPH.values())
check("Multiple fields in curriculum", len(fields) >= 3, f"fields={fields}")

# ═══════════════════════════════════════════════════════════════
# 10. LLM CLIENT
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("10. LLM CLIENT")
print("="*65)

from ai.llm_client import is_api_key_set, generate_json

check("API key is set", is_api_key_set())

try:
    result = generate_json(
        "You output valid JSON only.",
        'Return {"status": "ok", "value": 42}'
    )
    check("LLM returns valid JSON", isinstance(result, dict) and "status" in result,
          f"result={result}")
except Exception as e:
    check("LLM call works", False, str(e)[:100])

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"FINAL RESULTS: {PASS} PASS | {FAIL} FAIL | {WARN} WARN")
print("="*65)

if FAIL == 0:
    print("🎉 ALL TESTS PASSED! Pipeline is ready for submission.")
else:
    print(f"⚠️  {FAIL} tests failed. Review and fix before submission.")
