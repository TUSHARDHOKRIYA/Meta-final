"""
Run a full E2E pipeline test to populate debate logs and reward curves.
Creates a student, runs council, takes quizzes, and triggers adaptation.
"""
import requests
import json
import sys
import time

BASE = "https://degree-checker-01-meta-new-space.hf.space"
STUDENT_ID = "demo_student_hackathon"


def post(path, body=None):
    r = requests.post(f"{BASE}{path}", json=body or {}, timeout=60)
    print(f"  POST {path} -> {r.status_code}")
    try:
        return r.json()
    except:
        return {"error": r.text[:200]}


def get(path):
    r = requests.get(f"{BASE}{path}", timeout=60)
    print(f"  GET {path} -> {r.status_code}")
    try:
        return r.json()
    except:
        return {"error": r.text[:200]}


print("=" * 60)
print("EduPath AI — Full Pipeline E2E Test")
print("=" * 60)

# ── Step 1: Onboarding ──
print("\n[1] Onboarding...")
result = post("/api/onboarding/step1", {
    "student_id": STUDENT_ID,
    "name": "Hackathon Demo Student",
    "email": "demo@hackathon.ai",
})
print(f"    Created: {result.get('student_id', result)}")

result = post("/api/onboarding/step2", {
    "student_id": STUDENT_ID,
    "target_field": "tech",
    "learning_goal": "Machine Learning Engineer",
})

result = post("/api/onboarding/step3", {
    "student_id": STUDENT_ID,
    "skills": [
        {"skill": "Python", "level": "intermediate", "proficiency": 0.7},
        {"skill": "Statistics", "level": "beginner", "proficiency": 0.3},
        {"skill": "SQL", "level": "intermediate", "proficiency": 0.6},
    ],
})

result = post("/api/onboarding/step4", {
    "student_id": STUDENT_ID,
    "weekly_hours": 15,
    "job_description": "ML Engineer building recommendation systems at a tech startup",
})
print(f"    Onboarding complete")

# ── Step 2: Chat with Profiling Agent ──
print("\n[2] Chat with Profiling Agent...")
chat_result = post("/api/chat/start", {"student_id": STUDENT_ID})
print(f"    Chat started")

msg_result = post("/api/chat/message", {
    "student_id": STUDENT_ID,
    "message": "I am a computer science graduate wanting to become an ML engineer. I know Python well and have some SQL experience. I want to learn ML, deep learning, and NLP within 12 weeks.",
})
print(f"    Bot response: {str(msg_result.get('message', ''))[:100]}...")

# ── Step 3: Generate Roadmap (triggers Council Debate!) ──
print("\n[3] Generating Roadmap (Council Debate)...")
print("    This triggers all 6+ agents. Please wait ~30s...")
roadmap = post("/api/roadmap/generate", {
    "student_id": STUDENT_ID,
    "field": "tech",
    "goal": "Machine Learning Engineer",
    "weekly_hours": 15,
})
if "phases" in roadmap:
    print(f"    Roadmap: {roadmap.get('total_topics', '?')} topics, {len(roadmap.get('phases', []))} phases")
    print(f"    Confidence: {roadmap.get('confidence_score', '?')}")
    meta = roadmap.get("council_metadata", {})
    print(f"    Agents used: {meta.get('agents_used', '?')}, Debate rounds: {meta.get('debate_rounds', '?')}")
else:
    print(f"    Result: {str(roadmap)[:200]}")

# ── Step 4: Start Learning & Take Quiz ──
print("\n[4] Starting Learning Loop...")
learn_result = post("/api/learning/start-topic", {
    "student_id": STUDENT_ID,
    "topic_id": "python_basics",
})
print(f"    Learning started: {str(learn_result)[:100]}")

print("\n[5] Taking Quiz (generates BKT data)...")
quiz = post("/api/learning/quiz", {
    "student_id": STUDENT_ID,
    "topic_id": "python_basics",
})
if "questions" in quiz:
    print(f"    Quiz generated: {len(quiz['questions'])} questions, BKT: {quiz.get('bkt_skill_level_used', '?')}")
else:
    print(f"    Quiz: {str(quiz)[:200]}")

# Simulate submitting quiz answers
quiz_submit = post("/api/quiz/submit", {
    "student_id": STUDENT_ID,
    "topic_id": "python_basics",
    "answers": {"q1": "a", "q2": "b", "q3": "c", "q4": "a", "q5": "b"},
    "score": 80,
    "total_questions": 5,
    "correct_answers": 4,
})
print(f"    Quiz submitted: {str(quiz_submit)[:100]}")

# Take a second quiz attempt
quiz2 = post("/api/learning/quiz", {
    "student_id": STUDENT_ID,
    "topic_id": "python_basics",
})

quiz_submit2 = post("/api/quiz/submit", {
    "student_id": STUDENT_ID,
    "topic_id": "python_basics",
    "answers": {"q1": "a", "q2": "b", "q3": "c", "q4": "a", "q5": "b"},
    "score": 90,
    "total_questions": 5,
    "correct_answers": 5,
})
print(f"    Second quiz submitted: {str(quiz_submit2)[:100]}")

# ── Step 6: Check Analytics ──
print("\n" + "=" * 60)
print("ANALYTICS RESULTS")
print("=" * 60)

print("\n[A] Reward Curve:")
rewards = get(f"/api/analytics/rewards/{STUDENT_ID}")
print(json.dumps(rewards, indent=2)[:800])

print("\n[B] Agent Debate Log:")
debate = get(f"/api/analytics/agent-debate/{STUDENT_ID}")
print(f"    Council active: {debate.get('council_active')}")
print(f"    Total debate rounds: {debate.get('total_debate_rounds')}")
print(f"    Agents participated: {debate.get('total_agents_participated')}")
if debate.get("debate_rounds"):
    for r in debate["debate_rounds"]:
        print(f"      Round {r['round']}: {r['agent']} - {r['role'][:60]}")
if debate.get("debate_summary"):
    print(f"    Summary: {debate['debate_summary'].get('narrative', '')[:200]}")

print("\n[C] Observer Report:")
observer = get(f"/api/analytics/observer/{STUDENT_ID}")
if observer.get("observer_report"):
    report = observer["observer_report"]
    print(f"    Overall: {report.get('overall_assessment', '?')}")
    print(f"    Avg confidence: {report.get('average_confidence', '?')}")
    if report.get("agent_analyses"):
        for a in report["agent_analyses"][:5]:
            print(f"      {a['agent']}: {a['behavior']} - {a['reasoning'][:60]}")
    if report.get("anomalies"):
        print(f"    Anomalies: {len(report['anomalies'])}")
        for an in report["anomalies"][:3]:
            print(f"      [{an['severity']}] {an['agent']}: {an['type']}")
    if report.get("recommendations"):
        print(f"    Recommendations: {report['recommendations'][:3]}")
else:
    print(f"    {observer}")

print("\n[D] Learning Progress:")
progress = get(f"/api/learning/progress/{STUDENT_ID}")
print(json.dumps(progress, indent=2)[:500])

print("\n" + "=" * 60)
print("URLs to see the full data:")
print(f"  Rewards:  {BASE}/api/analytics/rewards/{STUDENT_ID}")
print(f"  Debate:   {BASE}/api/analytics/agent-debate/{STUDENT_ID}")
print(f"  Observer: {BASE}/api/analytics/observer/{STUDENT_ID}")
print(f"  Swagger:  {BASE}/docs")
print("=" * 60)
