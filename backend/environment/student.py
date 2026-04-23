"""
EduPath AI — Student Profile Manager
Team KRIYA | Meta Hackathon 2026

In-memory student profile store with file-backed persistence.
Handles CRUD operations for student profiles, quiz recording,
project completion, badge awards, and job readiness score
computation. Supports both local JSON storage and optional
Supabase cloud sync.
"""
import json
import os
import uuid
from typing import Dict, Optional, List
from environment.models import StudentProfile, Badge, BadgeType, QuizResult, SkillLevel


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


class StudentManager:
    """Manages student profiles with JSON file persistence."""

    def __init__(self):
        self.students: Dict[str, StudentProfile] = {}
        self._load_all()

    def _filepath(self, student_id: str) -> str:
        return os.path.join(DATA_DIR, f"student_{student_id}.json")

    def _load_all(self):
        """Load all student profiles from disk."""
        if not os.path.exists(DATA_DIR):
            return
        for f in os.listdir(DATA_DIR):
            if f.startswith("student_") and f.endswith(".json"):
                try:
                    with open(os.path.join(DATA_DIR, f), "r") as fh:
                        data = json.load(fh)
                        profile = StudentProfile(**data)
                        self.students[profile.id] = profile
                except Exception:
                    pass

    def save(self, student: StudentProfile):
        """Persist student profile to disk AND sync to Supabase."""
        self.students[student.id] = student
        with open(self._filepath(student.id), "w") as f:
            json.dump(student.model_dump(), f, indent=2, default=str)

        # Sync to Supabase (non-blocking, fails gracefully)
        try:
            from db.supabase_client import upsert_student
            upsert_student(student.model_dump())
        except Exception:
            pass  # Supabase sync failure shouldn't break local operation

    def get(self, student_id: str) -> Optional[StudentProfile]:
        """Get student by ID."""
        return self.students.get(student_id)

    def create(self, name: str = "", email: str = "", student_id: str = "") -> StudentProfile:
        """Create a new student profile. Uses provided ID or generates one."""
        sid = student_id or str(uuid.uuid4())[:8]
        student = StudentProfile(
            id=sid,
            name=name,
            email=email,
        )
        self.save(student)
        return student

    def update_from_onboarding(self, student_id: str, data: dict) -> StudentProfile:
        """Update student profile from onboarding data."""
        student = self.get(student_id)
        if not student:
            student = StudentProfile(id=student_id)

        if data.get("resume_skills"):
            student.resume_skills = data["resume_skills"]
        if data.get("skills"):
            student.self_assessed_skills = [
                SkillLevel(**s) if isinstance(s, dict) else s
                for s in data["skills"]
            ]
        if data.get("target_field"):
            student.target_field = data["target_field"]
        if data.get("learning_goal"):
            student.learning_goal = data["learning_goal"]
        if data.get("job_description"):
            student.job_description = data["job_description"]
        if data.get("jd_required_skills"):
            student.jd_required_skills = data["jd_required_skills"]
        if data.get("weekly_hours"):
            student.weekly_hours = data["weekly_hours"]
        if data.get("name"):
            student.name = data["name"]
        if data.get("email"):
            student.email = data["email"]

        self.save(student)
        return student

    def complete_topic(self, student_id: str, topic_id: str) -> StudentProfile:
        """Mark a topic as completed."""
        student = self.get(student_id)
        if student and topic_id not in student.completed_topics:
            student.completed_topics.append(topic_id)
            self._check_badges(student)
            self._update_job_readiness(student)
            self.save(student)
        return student

    def record_quiz(self, student_id: str, result: QuizResult) -> StudentProfile:
        """Record a quiz result."""
        student = self.get(student_id)
        if student:
            student.quiz_history.append(result)
            if result.passed:
                student.quiz_streak += 1
                if result.topic_id not in student.completed_topics:
                    student.completed_topics.append(result.topic_id)
            else:
                student.quiz_streak = 0
            self._check_badges(student)
            self._update_job_readiness(student)
            self.save(student)

            # Sync quiz to Supabase
            try:
                from db.supabase_client import save_quiz_result
                save_quiz_result(student_id, result.model_dump())
            except Exception:
                pass
        return student

    def complete_project(self, student_id: str, project_id: str) -> StudentProfile:
        """Mark a project as completed."""
        student = self.get(student_id)
        if student and project_id not in student.completed_projects:
            student.completed_projects.append(project_id)
            self._check_badges(student)
            self._update_job_readiness(student)
            self.save(student)
        return student

    def _check_badges(self, student: StudentProfile):
        """Check and award badges based on current progress — synced with full 30-badge catalog."""
        earned_ids = {b.id for b in student.badges}

        # ── Topic completion badges (7 badges) ──
        topic_milestones = {
            1: ("first_step", "Baby's First Step 👶", "Complete your very first topic"),
            3: ("topics_3", "Curious Cat 🐱", "Complete 3 topics"),
            5: ("topics_5", "Knowledge Goblin 👹", "Complete 5 topics"),
            7: ("topics_7", "Lucky Learner 🍀", "Complete 7 topics"),
            10: ("topics_10", "Brain Goes Brrr 🧊", "Complete 10 topics"),
            15: ("topics_15", "Sigma Scholar 💀", "Complete 15 topics"),
            20: ("topics_20", "Galaxy Brain 🌌", "Complete 20 topics"),
        }
        for count, (badge_id, name, desc) in topic_milestones.items():
            if len(student.topics_studied) >= count and badge_id not in earned_ids:
                student.badges.append(Badge(
                    id=badge_id, name=name,
                    description=desc,
                    type=BadgeType.TOPIC_COMPLETION, icon="📚"
                ))

        # ── Quiz pass badges (3 badges) ──
        quizzes_passed = len([q for q in student.quiz_history if q.score >= 70])
        quiz_milestones = {
            1: ("first_quiz", "Quiz Rookie 📋", "Pass your first quiz"),
            5: ("quiz_5", "Quiz Snacker 🍿", "Pass 5 quizzes"),
            10: ("quiz_10", "Quiz Terminator 🤖", "Pass 10 quizzes"),
        }
        for count, (badge_id, name, desc) in quiz_milestones.items():
            if quizzes_passed >= count and badge_id not in earned_ids:
                student.badges.append(Badge(
                    id=badge_id, name=name,
                    description=desc,
                    type=BadgeType.QUIZ_STREAK, icon="📋"
                ))

        # ── Perfect score badge ──
        if any(q.score >= 100 for q in student.quiz_history) and "perfect_score" not in earned_ids:
            student.badges.append(Badge(
                id="perfect_score", name="100% Flex 💯",
                description="Score 100% on any quiz",
                type=BadgeType.QUIZ_STREAK, icon="💯"
            ))

        # ── Quiz streak badges (4 badges) ──
        streak_milestones = {
            3: ("streak_3", "Hat Trick Hero 🎩", "Pass 3 quizzes in a row"),
            5: ("streak_5", "On Fire 🔥", "Pass 5 quizzes in a row"),
            7: ("streak_7", "Unstoppable Force 🏋️", "Pass 7 quizzes in a row"),
            10: ("streak_10", "Legendary Mode 🐉", "Pass 10 quizzes in a row"),
        }
        for count, (badge_id, name, desc) in streak_milestones.items():
            if student.quiz_streak >= count and badge_id not in earned_ids:
                student.badges.append(Badge(
                    id=badge_id, name=name,
                    description=desc,
                    type=BadgeType.QUIZ_STREAK, icon="🔥"
                ))

        # ── Resource badges ──
        total_clicks = sum(len(v) for v in student.clicked_resource_links.values()) if student.clicked_resource_links else 0
        if total_clicks >= 10 and "resource_hunter" not in earned_ids:
            student.badges.append(Badge(
                id="resource_hunter", name="Resource Hunter 🔍",
                description="Click 10+ learning resources",
                type=BadgeType.MILESTONE, icon="🔍"
            ))
        if total_clicks >= 25 and "resource_hoarder" not in earned_ids:
            student.badges.append(Badge(
                id="resource_hoarder", name="Resource Hoarder 📦",
                description="Click 25+ learning resources",
                type=BadgeType.MILESTONE, icon="📦"
            ))

        # ── Comeback kid badge ──
        topics_quizzed = {}
        for q in student.quiz_history:
            if q.topic_id not in topics_quizzed:
                topics_quizzed[q.topic_id] = []
            topics_quizzed[q.topic_id].append(q)
        for topic_id, quizzes in topics_quizzed.items():
            if len(quizzes) >= 2 and not quizzes[0].passed and quizzes[-1].passed and "comeback_kid" not in earned_ids:
                student.badges.append(Badge(
                    id="comeback_kid", name="Comeback Kid 💪",
                    description="Pass a quiz after failing it once",
                    type=BadgeType.MILESTONE, icon="💪"
                ))
                break

        # ── Cross-domain badge ──
        if "cross_domain" not in earned_ids:
            from environment.curriculum import TOPIC_GRAPH
            fields = set()
            for t in student.completed_topics:
                topic = TOPIC_GRAPH.get(t)
                if topic:
                    fields.add(topic.field)
            if len(fields) >= 2:
                student.badges.append(Badge(
                    id="cross_domain", name="Renaissance Human 🎨",
                    description="Complete topics from 2+ different fields",
                    type=BadgeType.MILESTONE, icon="🎨"
                ))

        # ── Project badges (2 badges) ──
        capstone_count = 0
        for pid in student.completed_projects:
            from environment.curriculum import PROJECT_DB
            proj = PROJECT_DB.get(pid)
            if proj and proj.is_capstone:
                capstone_count += 1
        if capstone_count >= 1 and "capstone_1" not in earned_ids:
            student.badges.append(Badge(
                id="capstone_1", name="Mini Boss Defeated 🎮",
                description="Complete your first capstone project",
                type=BadgeType.PROJECT_SUCCESS, icon="🎮"
            ))
        if capstone_count >= 2 and "capstone_2" not in earned_ids:
            student.badges.append(Badge(
                id="capstone_2", name="Final Boss Mode 👑",
                description="Complete both capstone projects",
                type=BadgeType.PROJECT_SUCCESS, icon="👑"
            ))

        # ── General project badges ──
        project_milestones = {1: "Builder", 3: "Creator", 5: "Architect"}
        for count, name in project_milestones.items():
            badge_id = f"projects_{count}"
            if len(student.completed_projects) >= count and badge_id not in earned_ids:
                student.badges.append(Badge(
                    id=badge_id, name=f"{name} Badge",
                    description=f"Completed {count} projects",
                    type=BadgeType.PROJECT_SUCCESS, icon="🛠️"
                ))

        # ── Job readiness badges (4 badges) ──
        readiness_milestones = {
            0.5: ("job_50", "Getting There 🚗", "Reach 50% job readiness"),
            0.7: ("job_70", "Almost Ready 🚀", "Reach 70% job readiness"),
            0.8: ("job_80", "Job Ready! 💼", "Reach 80% job readiness"),
            0.95: ("job_95", "Overqualified 🏆", "Reach 95% job readiness"),
        }
        for threshold, (badge_id, name, desc) in readiness_milestones.items():
            if student.job_readiness_score >= threshold and badge_id not in earned_ids:
                student.badges.append(Badge(
                    id=badge_id, name=name,
                    description=desc,
                    type=BadgeType.JOB_READY, icon="💼"
                ))

        # ── Completionist badge ──
        if "completionist" not in earned_ids:
            from environment.curriculum import get_topics_for_field
            if student.target_field:
                field_topics = get_topics_for_field(student.target_field)
                field_topic_ids = {t.id for t in field_topics}
                if field_topic_ids and field_topic_ids.issubset(set(student.completed_topics)):
                    student.badges.append(Badge(
                        id="completionist", name="The Completionist 🎯",
                        description="Complete ALL available topics in your field",
                        type=BadgeType.MILESTONE, icon="🎯"
                    ))

    def _update_job_readiness(self, student: StudentProfile):
        """Calculate job readiness score based on progress."""
        score = 0.0
        total_weight = 0.0

        # Topics completed (40% weight)
        if student.jd_required_skills:
            all_skills = set(student.jd_required_skills)
            matched = sum(1 for t in student.completed_topics
                          if any(s.lower() in t.lower() for s in all_skills))
            topic_score = matched / max(len(all_skills), 1)
        else:
            topic_score = len(student.completed_topics) / 15.0
        score += min(topic_score, 1.0) * 0.4
        total_weight += 0.4

        # Quiz performance (30% weight)
        if student.quiz_history:
            avg_score = sum(q.score for q in student.quiz_history) / len(student.quiz_history)
            score += (avg_score / 100.0) * 0.3
        total_weight += 0.3

        # Projects completed (30% weight)
        project_score = len(student.completed_projects) / 5.0
        score += min(project_score, 1.0) * 0.3
        total_weight += 0.3

        raw_score = round(score / total_weight, 2) if total_weight > 0 else 0.0
        # Clamp to (0, 0.99) so graders never receive exactly 0.0 or 1.0
        if raw_score <= 0:
            raw_score = 0.001
        elif raw_score >= 1:
            raw_score = 0.99
        student.job_readiness_score = raw_score

    def get_skill_levels(self, student_id: str) -> Dict[str, float]:
        """Get computed skill levels for a student."""
        student = self.get(student_id)
        if not student:
            return {}

        levels = {}
        # From self-assessed skills
        for s in student.self_assessed_skills:
            levels[s.skill] = s.proficiency

        # From completed topics (boost)
        for topic_id in student.completed_topics:
            levels[topic_id] = min(levels.get(topic_id, 0) + 0.3, 0.99)

        # From quiz scores
        for q in student.quiz_history:
            levels[q.topic_id] = max(levels.get(q.topic_id, 0), q.score / 100.0)

        return levels

    def record_link_click(self, student_id: str, topic_id: str, url: str) -> bool:
        """
        Records that student clicked a resource link.
        Returns True if this is their first click for this topic (unlock event).
        """
        student = self.get(student_id)
        if not student:
            return False

        is_first = topic_id not in student.clicked_resource_links or \
                   len(student.clicked_resource_links.get(topic_id, [])) == 0

        if topic_id not in student.clicked_resource_links:
            student.clicked_resource_links[topic_id] = []

        if url not in student.clicked_resource_links[topic_id]:
            student.clicked_resource_links[topic_id].append(url)

        self.save(student)
        return is_first

    def mark_topic_studied(self, student_id: str, topic_id: str) -> bool:
        """
        Called when student clicks Mark as Complete.
        Only works if student has at least 1 clicked link for this topic.
        Returns False if no link was clicked (guard).
        Returns True and adds to topics_studied if valid.
        """
        student = self.get(student_id)
        if not student:
            return False

        # Guard: must have clicked at least one link
        if topic_id not in student.clicked_resource_links or \
           len(student.clicked_resource_links[topic_id]) == 0:
            return False

        if topic_id not in student.topics_studied:
            student.topics_studied.append(topic_id)

        self.save(student)
        return True


# Global instance
student_manager = StudentManager()
