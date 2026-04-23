"""
EduPath AI — Deterministic Task Graders
Team KRIYA | Meta Hackathon 2026

Five task graders (Task 1–5) that score agent performance on a strictly
(0, 1) scale — every returned score satisfies 0 < score < 1.
Each grader evaluates a different tutoring scenario:
  Task 1 (Easy):   Python beginner — topic sequencing & prerequisite ordering
  Task 2 (Medium): Data Analyst prep — topic coverage + quiz performance
  Task 3 (Hard):   Cross-domain (Doctor → AI) — job readiness + bridging
  Task 4 (Hard):   Team learning — min/avg readiness across 3 learners
  Task 5 (Expert): Career transition under deadline — milestones + efficiency
"""
from typing import List, Dict, Optional
from environment.models import StudentProfile
from environment.curriculum import TOPIC_GRAPH

# ── Score boundary helpers ──
# The OpenEnv validator requires every task score to be STRICTLY inside (0, 1).
# These two constants are the bounds we clamp to.
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp_score(raw: float) -> float:
    """Clamp a raw score to the open interval (0, 1).

    Any value <= 0 becomes _SCORE_MIN (0.001).
    Any value >= 1 becomes _SCORE_MAX (0.999).
    Rounding happens BEFORE boundary check to catch edge cases
    like round(0.99996, 4) == 1.0.
    """
    score = round(raw, 4)
    if score <= 0:
        return _SCORE_MIN
    if score >= 1:
        return _SCORE_MAX
    return score


def grade_task1(student: StudentProfile) -> float:
    """
    Task 1 (Easy): Complete beginner wants to learn Python from scratch.
    Agent must sequence 5 topics correctly, respecting all prerequisites.
    Grader: % of expected topics completed (0 – 1)
    """
    expected = [
        "python_basics",
        "python_control_flow",
        "python_oop",
        "data_structures",
        "version_control",
    ]

    if not student.completed_topics:
        return _SCORE_MIN

    # Score: how many expected topics were completed
    correct = sum(1 for t in expected if t in student.completed_topics)

    # Bonus: check prerequisite ordering was respected in completion order
    order_bonus = 0
    completed_set = set()
    for topic_id in student.completed_topics:
        topic = TOPIC_GRAPH.get(topic_id)
        if topic and topic_id in expected:
            prereqs_met = all(p in completed_set for p in topic.prerequisites)
            if prereqs_met:
                order_bonus += 0.1
        completed_set.add(topic_id)

    score = (correct / len(expected)) * 0.7 + min(order_bonus, 0.3)
    return _clamp_score(score)


def grade_task2(student: StudentProfile) -> float:
    """
    Task 2 (Medium): Knows Python basics, goal = Data Analyst job in 3 months.
    Grader: Topic coverage (50%) + quiz performance (50%).
    """
    data_analyst_topics = [
        "statistics", "data_analysis", "data_visualization",
        "databases", "machine_learning"
    ]

    # 1. Topic coverage (50%): how many DA topics completed
    covered = sum(1 for t in data_analyst_topics if t in student.completed_topics)
    coverage_score = min(covered / len(data_analyst_topics), 0.99)

    # 2. Quiz performance (50%): average quiz score and adaptation
    quiz_score = 0.0
    if student.quiz_history:
        avg = sum(q.score for q in student.quiz_history) / len(student.quiz_history)
        quiz_score = min(avg / 100.0, 0.99)

        # Bonus for retries (adaptation evidence)
        topics_quizzed = {}
        for q in student.quiz_history:
            topics_quizzed[q.topic_id] = topics_quizzed.get(q.topic_id, 0) + 1
        retried = sum(1 for count in topics_quizzed.values() if count > 1)
        if retried > 0:
            quiz_score = min(quiz_score + 0.1, 0.99)

    final = (coverage_score * 0.5) + (quiz_score * 0.5)
    return _clamp_score(final)


def grade_task3(student: StudentProfile) -> float:
    """
    Task 3 (Hard): Doctor learning AI for healthcare (cross-domain).
    Grader: Job readiness (40%) + topic efficiency (30%) + cross-domain bridging (30%).
    """
    jd_skills = [
        "python", "machine_learning", "deep_learning",
        "medical_imaging", "clinical_data", "nlp"
    ]

    # 1. Job readiness (40%): JD skills covered via completed topics
    jd_covered = 0
    for skill in jd_skills:
        for topic in student.completed_topics:
            if skill.lower().replace("_", " ") in topic.lower().replace("_", " "):
                jd_covered += 1
                break
    job_readiness = min(jd_covered / max(len(jd_skills), 1), 0.99)

    # 2. Topic efficiency (30%): more completed topics = better
    topic_count = len(student.completed_topics)
    efficiency = min(topic_count / 8.0, 0.99)  # cap below 1

    # 3. Cross-domain bridging (30%): both tech and healthcare topics?
    tech_completed = [t for t in student.completed_topics if TOPIC_GRAPH.get(t) and TOPIC_GRAPH[t].field == "tech"]
    domain_completed = [t for t in student.completed_topics if TOPIC_GRAPH.get(t) and TOPIC_GRAPH[t].field == "healthcare"]

    cross_domain = 0.0
    if tech_completed and domain_completed:
        balance = min(len(tech_completed), len(domain_completed)) / max(len(tech_completed), len(domain_completed))
        cross_domain = min(balance, 0.99)
    elif tech_completed or domain_completed:
        cross_domain = 0.3

    final = (job_readiness * 0.4) + (efficiency * 0.3) + (cross_domain * 0.3)
    return _clamp_score(final)


def grade_task4(students: List[StudentProfile], steps_used: int = 300) -> float:
    """
    Task 4 (Hard): Team Learning — 3 employees with different backgrounds.
    Grader: min readiness (30%) + avg readiness (20%) + efficiency (20%) +
            cross-domain bridging (15%) + all-complete bonus (15%).
    """
    if not students:
        return _SCORE_MIN

    # 1. Min job readiness across all students (30%) — weakest link
    readiness_scores = [s.job_readiness_score for s in students]
    min_readiness = min(readiness_scores) if readiness_scores else 0

    # 2. Average job readiness (20%)
    avg_readiness = sum(readiness_scores) / len(readiness_scores) if readiness_scores else 0

    # 3. Efficiency (20%): 1 - steps_used/300
    efficiency = min(max(0, 1.0 - steps_used / 300.0), 0.99)

    # 4. Cross-domain bridging quality (15%): count unique cross-domain topics
    cross_domain_topics = set()
    for student in students:
        target = student.target_field or "tech"
        for topic_id in student.completed_topics:
            topic = TOPIC_GRAPH.get(topic_id)
            if topic and topic.field != target and topic.field != "tech":
                cross_domain_topics.add(topic_id)
            # Also count tech topics for non-tech students as cross-domain
            elif topic and topic.field == "tech" and target != "tech":
                cross_domain_topics.add(topic_id)
    cross_domain_score = min(len(cross_domain_topics) / 9.0, 0.99)

    # 5. Binary completion bonus (15%): all 3 students reach 0.7 readiness
    all_complete = 1 if all(r >= 0.7 for r in readiness_scores) else 0

    final = (
        min_readiness * 0.30 +
        avg_readiness * 0.20 +
        efficiency * 0.20 +
        cross_domain_score * 0.15 +
        all_complete * 0.15
    )
    return _clamp_score(final)


def grade_task5(student: StudentProfile, steps_used: int = 100) -> float:
    """
    Task 5 (Expert): Career Transition Under Deadline.
    Nurse → Healthcare AI Product Manager in 8 weeks (56 hours).
    Grader: job readiness (25%) + milestone completion (25%) +
            skill coverage (20%) + efficiency (20%) + early completion (10%).
    """
    # 1. Job readiness at episode end (25%)
    job_readiness = student.job_readiness_score

    # 2. Milestone completion rate (25%) — check 4 milestones
    milestones_hit = 0
    total_milestones = 4

    # Milestone 1 (week 2): >= 3 topics completed
    if len(student.completed_topics) >= 3:
        milestones_hit += 1

    # Milestone 2 (week 4): quiz passed on statistics and data_analysis
    quiz_passed_topics = {q.topic_id for q in student.quiz_history if q.passed}
    if "statistics" in quiz_passed_topics or "data_analysis" in quiz_passed_topics:
        milestones_hit += 1

    # Milestone 3 (week 6): >= 1 project completed
    if len(student.completed_projects) >= 1:
        milestones_hit += 1

    # Milestone 4 (week 8): job readiness >= 0.7
    if student.job_readiness_score >= 0.7:
        milestones_hit += 1

    milestone_rate = min(milestones_hit / total_milestones, 0.99)

    # 3. Skill coverage (20%): healthcare + tech + business all > 0.5 mastery
    fields_covered = set()
    for topic_id in student.completed_topics:
        topic = TOPIC_GRAPH.get(topic_id)
        if topic:
            fields_covered.add(topic.field)
    # Need healthcare, tech, and business
    required_fields = {"healthcare", "tech", "business"}
    coverage = min(len(fields_covered.intersection(required_fields)) / len(required_fields), 0.99)

    # 4. Efficiency (20%): topics per step vs optimal (1 topic per 5 steps)
    if steps_used > 0:
        topics_per_step = len(student.completed_topics) / steps_used
        optimal_rate = 1.0 / 5.0  # 1 topic per 5 steps is optimal
        efficiency = min(topics_per_step / optimal_rate, 0.99)
    else:
        efficiency = 0

    # 5. Early completion bonus (10%): job_ready triggered before step 80
    early_bonus = 1 if (student.job_readiness_score >= 0.7 and steps_used < 80) else 0

    final = (
        job_readiness * 0.25 +
        milestone_rate * 0.25 +
        coverage * 0.20 +
        efficiency * 0.20 +
        early_bonus * 0.10
    )
    return _clamp_score(final)
