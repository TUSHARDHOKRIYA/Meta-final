"""
Comprehensive API smoke test for the deployed EduPath AI backend.
Tests all endpoint categories against the live HF Space.
"""
import requests
import json
import sys
import time

BASE = "https://degree-checker-01-meta-new-space.hf.space"

def test(method, path, body=None, expect_status=None):
    url = f"{BASE}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=30)
        elif method == "POST":
            r = requests.post(url, json=body, timeout=30)
        status = r.status_code
        ok = status < 500
        if expect_status:
            ok = status == expect_status
        try:
            data = r.json()
            summary = str(data)[:120]
        except Exception:
            summary = r.text[:120]
        symbol = "PASS" if ok else "FAIL"
        print(f"  [{symbol}] {method} {path} -> {status} | {summary}")
        return ok, status, data if ok else None
    except Exception as e:
        print(f"  [FAIL] {method} {path} -> ERROR: {e}")
        return False, 0, None

results = {"pass": 0, "fail": 0}

def record(ok):
    if ok:
        results["pass"] += 1
    else:
        results["fail"] += 1

print("=" * 70)
print("EduPath AI - Deployed API Smoke Test")
print(f"Base URL: {BASE}")
print("=" * 70)

# ── 1. Environment Info ──
print("\n[1] Environment Info")
ok, _, _ = test("GET", "/api/env/info")
record(ok)

# ── 2. Topics ──
print("\n[2] Curriculum Topics")
ok, _, data = test("GET", "/api/topics")
record(ok)

# ── 3. OpenEnv RL Endpoints ──
print("\n[3] OpenEnv RL Endpoints")
ok, _, reset_data = test("POST", "/reset", {"task_id": "task1_easy"})
record(ok)

if reset_data:
    ok, _, _ = test("GET", "/state")
    record(ok)

    ok, _, _ = test("POST", "/step", {
        "action": {"type": "recommend_topic", "topic_id": "python_basics"}
    })
    record(ok)

    ok, _, _ = test("POST", "/grade")
    record(ok)
else:
    print("  [SKIP] Skipping /state, /step, /grade (reset failed)")
    results["fail"] += 3

# ── 4. Onboarding ──
print("\n[4] Onboarding Flow")
onboard_student = {
    "student_id": "smoke_test_001",
    "name": "Smoke Test Student",
    "email": "test@example.com",
    "target_field": "tech",
    "learning_goal": "ML Engineer",
}

ok, _, _ = test("POST", "/api/onboarding/step1", {
    "name": onboard_student["name"],
    "email": onboard_student["email"],
})
record(ok)

ok, _, _ = test("POST", "/api/onboarding/step2", {
    "student_id": onboard_student["student_id"],
    "target_field": "tech",
    "learning_goal": "ML Engineer",
})
record(ok)

ok, _, _ = test("POST", "/api/onboarding/step3", {
    "student_id": onboard_student["student_id"],
    "skills": [
        {"skill": "Python", "level": "intermediate", "proficiency": 0.7},
    ],
})
record(ok)

ok, _, _ = test("POST", "/api/onboarding/step4", {
    "student_id": onboard_student["student_id"],
    "weekly_hours": 15,
    "job_description": "Machine Learning Engineer position",
})
record(ok)

# ── 5. Chat API (Profiling Agent) ──
print("\n[5] Chat API (Profiling Agent)")
ok, _, chat_data = test("POST", "/api/chat/start", {
    "student_id": "smoke_test_001",
})
record(ok)

ok, _, _ = test("POST", "/api/chat/message", {
    "student_id": "smoke_test_001",
    "message": "I want to learn machine learning"
})
record(ok)

# ── 6. Roadmap ──
print("\n[6] Roadmap Generation")
ok, _, roadmap_data = test("POST", "/api/roadmap/generate", {
    "student_id": "smoke_test_001",
    "field": "tech",
    "goal": "ML Engineer",
    "weekly_hours": 15,
})
record(ok)

ok, _, _ = test("GET", "/api/roadmap/smoke_test_001")
record(ok)

# ── 7. Quiz ──
print("\n[7] Quiz Generation")
ok, _, quiz_data = test("POST", "/api/quiz/generate", {
    "student_id": "smoke_test_001",
    "topic_id": "python_basics",
})
record(ok)

# ── 8. Learning Loop ──
print("\n[8] Learning Loop API")
ok, _, _ = test("POST", "/api/learning/start-topic", {
    "student_id": "smoke_test_001",
    "topic_id": "python_basics",
})
record(ok)

ok, _, _ = test("POST", "/api/learning/quiz", {
    "student_id": "smoke_test_001",
    "topic_id": "python_basics",
})
record(ok)

ok, _, _ = test("GET", "/api/learning/progress/smoke_test_001")
record(ok)

# ── 9. Resources ──
print("\n[9] Resources")
ok, _, _ = test("POST", "/api/resources/search", {
    "student_id": "smoke_test_001",
    "topic_id": "python_basics",
})
record(ok)

# ── 10. Badges ──
print("\n[10] Badges")
ok, _, _ = test("GET", "/api/badges/smoke_test_001")
record(ok)

# ── 11. Career ──
print("\n[11] Career")
ok, _, _ = test("GET", "/api/career/smoke_test_001")
record(ok)

# ── 12. Projects ──
print("\n[12] Projects")
ok, _, _ = test("POST", "/api/projects/evaluate", {
    "student_id": "smoke_test_001",
    "project_id": "test_project",
    "project_title": "Test ML Project",
    "project_type": "mini_project",
    "submission_text": "https://github.com/test/ml-project",
    "topic_id": "python_basics",
})
record(ok)

# ── Summary ──
print("\n" + "=" * 70)
total = results["pass"] + results["fail"]
print(f"RESULTS: {results['pass']}/{total} passed, {results['fail']}/{total} failed")
print("=" * 70)

if results["fail"] > 0:
    sys.exit(1)
