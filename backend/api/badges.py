"""
EduPath AI — Gamification & Badge System API
Team KRIYA | Meta Hackathon 2026

Awards achievement badges based on student milestones: first topic
studied, quiz streaks, project completions, job readiness thresholds,
and roadmap progress. Checks and awards are idempotent.
"""
from fastapi import APIRouter, HTTPException

from environment.student import student_manager

router = APIRouter(prefix="/api/badges", tags=["badges"])


# Badge definitions with criteria — 30 unique badges!
BADGE_CATALOG = [
    # ══════ TOPIC COMPLETION BADGES ══════
    {"id": "first_step", "name": "Baby's First Step 👶", "description": "Complete your very first topic", "icon": "🐣", "criteria": "1 topic completed", "type": "topic_completion"},
    {"id": "topics_3", "name": "Curious Cat 🐱", "description": "Complete 3 topics — curiosity didn't kill this cat!", "icon": "🐱", "criteria": "3 topics completed", "type": "topic_completion"},
    {"id": "topics_5", "name": "Knowledge Goblin 👹", "description": "Complete 5 topics — you hoard knowledge like gold!", "icon": "👹", "criteria": "5 topics completed", "type": "topic_completion"},
    {"id": "topics_7", "name": "Lucky Learner 🍀", "description": "Complete 7 topics — feeling lucky!", "icon": "🍀", "criteria": "7 topics completed", "type": "topic_completion"},
    {"id": "topics_10", "name": "Brain Goes Brrr 🧊", "description": "Complete 10 topics — your brain is a machine!", "icon": "🧠", "criteria": "10 topics completed", "type": "topic_completion"},
    {"id": "topics_15", "name": "Sigma Scholar 💀", "description": "Complete 15 topics — absolute sigma energy", "icon": "💀", "criteria": "15 topics completed", "type": "topic_completion"},
    {"id": "topics_20", "name": "Galaxy Brain 🌌", "description": "Complete 20 topics — your brain contains universes", "icon": "🌌", "criteria": "20 topics completed", "type": "topic_completion"},

    # ══════ QUIZ BADGES ══════
    {"id": "first_quiz", "name": "Quiz Rookie 📋", "description": "Pass your first quiz", "icon": "📋", "criteria": "1 quiz passed", "type": "quiz_pass"},
    {"id": "quiz_5", "name": "Quiz Snacker 🍿", "description": "Pass 5 quizzes — you eat quizzes for breakfast", "icon": "🍿", "criteria": "5 quizzes passed", "type": "quiz_pass"},
    {"id": "quiz_10", "name": "Quiz Terminator 🤖", "description": "Pass 10 quizzes — I'll be back... for more", "icon": "🤖", "criteria": "10 quizzes passed", "type": "quiz_pass"},
    {"id": "perfect_score", "name": "100% Flex 💯", "description": "Score 100% on any quiz — absolute perfection", "icon": "💯", "criteria": "100% on a quiz", "type": "quiz_perfect"},

    # ══════ STREAK BADGES ══════
    {"id": "streak_3", "name": "Hat Trick Hero 🎩", "description": "Pass 3 quizzes in a row — *tips hat*", "icon": "🎩", "criteria": "3 quiz streak", "type": "quiz_streak"},
    {"id": "streak_5", "name": "On Fire 🔥", "description": "Pass 5 quizzes in a row — somebody call the fire dept!", "icon": "🔥", "criteria": "5 quiz streak", "type": "quiz_streak"},
    {"id": "streak_7", "name": "Unstoppable Force 🏋️", "description": "Pass 7 quizzes in a row — nothing can stop you", "icon": "🏋️", "criteria": "7 quiz streak", "type": "quiz_streak"},
    {"id": "streak_10", "name": "Legendary Mode 🐉", "description": "Pass 10 quizzes in a row — you've become a legend", "icon": "🐉", "criteria": "10 quiz streak", "type": "quiz_streak"},

    # ══════ CONSISTENCY BADGES ══════
    {"id": "early_bird", "name": "Early Bird 🐦", "description": "Complete a topic within 24 hours of starting", "icon": "🐦", "criteria": "Fast topic completion", "type": "speed"},
    {"id": "weekend_warrior", "name": "Weekend Warrior ⚔️", "description": "Study on both Saturday and Sunday", "icon": "⚔️", "criteria": "Weekend studying", "type": "consistency"},
    {"id": "night_owl", "name": "Night Owl 🦉", "description": "Study past midnight — sleep is for the weak!", "icon": "🦉", "criteria": "Late night study", "type": "consistency"},

    # ══════ SPECIAL ACHIEVEMENT BADGES ══════
    {"id": "resource_hunter", "name": "Resource Hunter 🔍", "description": "Click 10+ learning resources — you explore everything!", "icon": "🔍", "criteria": "10 resources clicked", "type": "resource_explorer"},
    {"id": "resource_hoarder", "name": "Resource Hoarder 📦", "description": "Click 25+ learning resources — you save EVERYTHING", "icon": "📦", "criteria": "25 resources clicked", "type": "resource_explorer"},
    {"id": "comeback_kid", "name": "Comeback Kid 💪", "description": "Pass a quiz after failing it once — never give up!", "icon": "💪", "criteria": "Retry and pass", "type": "resilience"},
    {"id": "cross_domain", "name": "Renaissance Human 🎨", "description": "Complete topics from 2+ different fields", "icon": "🎨", "criteria": "Multi-field learning", "type": "cross_domain"},

    # ══════ CAPSTONE & PROJECT BADGES ══════
    {"id": "capstone_1", "name": "Mini Boss Defeated 🎮", "description": "Complete your first capstone project", "icon": "🎮", "criteria": "1 capstone completed", "type": "capstone"},
    {"id": "capstone_2", "name": "Final Boss Mode 👑", "description": "Complete both capstone projects — you're a king/queen", "icon": "👑", "criteria": "2 capstones completed", "type": "capstone"},

    # ══════ JOB READINESS BADGES ══════
    {"id": "job_50", "name": "Getting There 🚗", "description": "Reach 50% job readiness — halfway to greatness!", "icon": "🚗", "criteria": "50% job readiness", "type": "job_ready"},
    {"id": "job_70", "name": "Almost Ready 🚀", "description": "Reach 70% job readiness — launch sequence initiated", "icon": "🚀", "criteria": "70% job readiness", "type": "job_ready"},
    {"id": "job_80", "name": "Job Ready! 💼", "description": "Reach 80% job readiness — you're employable!", "icon": "💼", "criteria": "80% job readiness", "type": "job_ready"},
    {"id": "job_95", "name": "Overqualified 🏆", "description": "Reach 95% job readiness — you might be overqualified!", "icon": "🏆", "criteria": "95% job readiness", "type": "job_ready"},

    # ══════ FUN / EASTER EGG BADGES ══════
    {"id": "speed_demon", "name": "Speed Demon ⚡", "description": "Complete a quiz in under 60 seconds", "icon": "⚡", "criteria": "Fast quiz completion", "type": "speed"},
    {"id": "completionist", "name": "The Completionist 🎯", "description": "Complete ALL available topics in your field", "icon": "🎯", "criteria": "All field topics done", "type": "completionist"},
]


@router.get("/catalog")
async def get_badge_catalog():
    """Get all available badges with criteria."""
    return {"badges": BADGE_CATALOG}


@router.get("/{student_id}")
async def get_student_badges(student_id: str):
    """Get badges earned by a student."""
    student = student_manager.get(student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    earned_ids = {b.id for b in student.badges}

    # ── Auto-award badges based on current student state ──
    topics_done = len(student.completed_topics)
    quizzes_passed = len([q for q in student.quiz_history if q.score >= 70])
    streak = student.quiz_streak
    readiness = student.job_readiness_score
    total_clicks = sum(len(v) for v in student.clicked_resource_links.values()) if hasattr(student, 'clicked_resource_links') else 0

    # Topic completion badges
    topic_thresholds = {"first_step": 1, "topics_3": 3, "topics_5": 5, "topics_7": 7, "topics_10": 10, "topics_15": 15, "topics_20": 20}
    for badge_id, threshold in topic_thresholds.items():
        if topics_done >= threshold:
            earned_ids.add(badge_id)

    # Quiz pass badges
    quiz_thresholds = {"first_quiz": 1, "quiz_5": 5, "quiz_10": 10}
    for badge_id, threshold in quiz_thresholds.items():
        if quizzes_passed >= threshold:
            earned_ids.add(badge_id)

    # Perfect score badge
    if any(q.score >= 100 for q in student.quiz_history):
        earned_ids.add("perfect_score")

    # Streak badges
    streak_thresholds = {"streak_3": 3, "streak_5": 5, "streak_7": 7, "streak_10": 10}
    for badge_id, threshold in streak_thresholds.items():
        if streak >= threshold:
            earned_ids.add(badge_id)

    # Resource badges
    resource_thresholds = {"resource_hunter": 10, "resource_hoarder": 25}
    for badge_id, threshold in resource_thresholds.items():
        if total_clicks >= threshold:
            earned_ids.add(badge_id)

    # Job readiness badges
    readiness_thresholds = {"job_50": 0.5, "job_70": 0.7, "job_80": 0.8, "job_95": 0.95}
    for badge_id, threshold in readiness_thresholds.items():
        if readiness >= threshold:
            earned_ids.add(badge_id)

    # Combine catalog with earned status
    badge_status = []
    for badge in BADGE_CATALOG:
        b = badge.copy()
        b["earned"] = badge["id"] in earned_ids
        badge_status.append(b)

    return {
        "earned_count": len([b for b in badge_status if b["earned"]]),
        "total_count": len(BADGE_CATALOG),
        "badges": badge_status,
    }
