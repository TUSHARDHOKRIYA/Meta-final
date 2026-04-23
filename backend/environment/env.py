"""
EduPath AI — OpenEnv Environment Engine
Team KRIYA | Meta Hackathon 2026

Core RL environment with step(), reset(), state() — fully OpenEnv
compliant. The AI agent observes student state, takes tutoring actions
(recommend topic, assign quiz, assign project, etc.), and receives
shaped reward signals. Quiz outcomes are simulated via BKT.
"""
import random
from typing import Dict, Optional, List
from environment.models import (
    Action, ActionType, Observation, Reward, StepResult,
    StudentProfile, QuizResult, QuizDifficulty
)
from environment.curriculum import (
    TOPIC_GRAPH, PROJECT_DB, get_available_topics, get_resources_for_topic,
    get_projects_for_field
)
from environment.student import student_manager
from environment.student_model import StudentDifficultyModel


class EduPathEnv:
    """
    OpenEnv-compliant reinforcement learning environment for personalized tutoring.

    The AI agent observes the student state and takes tutoring actions.
    The environment responds with new state + reward signal.
    """

    def __init__(self, student_id: Optional[str] = None, seed: int = 42):
        self.student_id = student_id
        self.total_steps = 0
        self.action_history: List[Action] = []
        self.done = False
        self.max_steps = 100
        # Dedicated RNG for reproducible quiz outcomes
        self._rng = random.Random(seed)
        # Student difficulty model for realistic quiz simulation
        self.difficulty_model = StudentDifficultyModel(seed=seed)

    def reset(self, student_id: Optional[str] = None, seed: int = 42) -> Observation:
        """Reset the environment. Returns initial observation."""
        if student_id:
            self.student_id = student_id

        if not self.student_id:
            student = student_manager.create()
            self.student_id = student.id

        self.total_steps = 0
        self.action_history = []
        self.done = False
        self._rng = random.Random(seed)  # Reset RNG for reproducibility
        self.difficulty_model.reset(seed=seed)

        # Initialize difficulty model from student profile
        if self.student_id:
            student = student_manager.get(self.student_id)
            if student:
                self.difficulty_model.initialize_from_profile(
                    student.self_assessed_skills,
                    student.resume_skills
                )

        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        """Execute an agent action. Returns (observation, reward, done, info)."""
        if self.done:
            return StepResult(
                observation=self._get_observation(),
                reward=Reward(value=0, reason="Episode already done", is_terminal=True),
                done=True,
                info={"error": "Episode already done"}
            )

        self.total_steps += 1
        self.action_history.append(action)

        # Calculate reward based on action
        reward = self._calculate_reward(action)

        # Execute the action
        info = self._execute_action(action)

        # Check termination conditions
        if reward.is_terminal or self.total_steps >= self.max_steps:
            self.done = True

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=self.done,
            info=info
        )

    def state(self) -> Dict:
        """Return current environment state (for serialization)."""
        obs = self._get_observation()
        return {
            "observation": obs.model_dump(),
            "total_steps": self.total_steps,
            "done": self.done,
            "student_id": self.student_id,
        }

    def _get_observation(self) -> Observation:
        """Build the current observation for the agent."""
        student = student_manager.get(self.student_id)
        if not student:
            student = StudentProfile(id=self.student_id or "unknown")

        field = student.target_field or "tech"
        available = get_available_topics(student.completed_topics, field)
        skill_levels = student_manager.get_skill_levels(self.student_id)
        quiz_summary = {q.topic_id: q.score for q in student.quiz_history}

        # Get BKT mastery probabilities
        mastery_probs = self.difficulty_model.get_all_mastery_probabilities()

        return Observation(
            student_id=student.id,
            completed_topics=student.completed_topics,
            current_topic=student.current_topic,
            available_topics=available,
            skill_levels=skill_levels,
            quiz_history_summary=quiz_summary,
            completed_projects=student.completed_projects,
            job_readiness_score=student.job_readiness_score,
            total_steps=self.total_steps,
            badges_earned=len(student.badges),
            weekly_hours=student.weekly_hours,
            target_field=field,
            learning_goal=student.learning_goal,
            mastery_probabilities=mastery_probs,
        )

    def _calculate_reward(self, action: Action) -> Reward:
        """Calculate reward signal for the given action."""
        student = student_manager.get(self.student_id)
        if not student:
            return Reward(value=-0.1, reason="Unknown student")

        # Loop detection — check BEFORE processing action
        # Uses (type, topic_id) pairs to avoid false positives on natural
        # sequences like recommend_A → quiz_A → recommend_B → quiz_B
        if len(self.action_history) >= 3:
            last_3 = [(a.type, a.topic_id) for a in self.action_history[-3:]]
            if len(set(last_3)) == 1:
                return Reward(value=-0.1, reason="Loop detected: same action 3+ times")

        # ── recommend_topic ──
        if action.type == ActionType.RECOMMEND_TOPIC:
            topic_id = action.topic_id
            if not topic_id:
                return Reward(value=-0.1, reason="No topic specified")

            topic = TOPIC_GRAPH.get(topic_id)
            if not topic:
                return Reward(value=-0.1, reason="Unknown topic")

            # Already completed?
            if topic_id in student.completed_topics:
                return Reward(value=-0.1, reason="Redundant: topic already completed")

            # Check prerequisites
            unmet = [p for p in topic.prerequisites if p not in student.completed_topics]
            if unmet:
                return Reward(value=-0.2, reason=f"Prerequisites not met: {unmet}")

            # Good recommendation
            return Reward(value=0.3, reason="Topic recommended correctly (prerequisites met)")

        # ── assign_quiz ──
        elif action.type == ActionType.ASSIGN_QUIZ:
            topic_id = action.topic_id
            if not topic_id or topic_id not in TOPIC_GRAPH:
                return Reward(value=-0.1, reason="Invalid quiz topic")

            # Use StudentDifficultyModel for realistic quiz simulation
            simulated_score = self.difficulty_model.simulate_quiz_score(
                topic_id, student.completed_topics
            )
            # Store on action so _execute_action uses the same score
            action._quiz_score = simulated_score

            if simulated_score >= 70:
                return Reward(value=0.2, reason=f"Quiz passed (score: {simulated_score}%)")
            elif simulated_score >= 50:
                return Reward(value=0.1, reason=f"Quiz partial (score: {simulated_score}%)")
            else:
                return Reward(value=0, reason=f"Quiz failed (score: {simulated_score}%)")

        # ── assign_mini_project ──
        elif action.type == ActionType.ASSIGN_MINI_PROJECT:
            project = self._resolve_project(action, student, is_capstone=False)
            if not project:
                return Reward(value=-0.1, reason="No suitable project found")
            if project.id in student.completed_projects:
                return Reward(value=-0.1, reason="Project already completed")
            # Store resolved project_id back for _execute_action
            action.project_id = project.id
            return Reward(value=0.4, reason="Mini project assigned successfully")

        # ── assign_capstone ──
        elif action.type == ActionType.ASSIGN_CAPSTONE:
            project = self._resolve_project(action, student, is_capstone=True)
            if not project:
                return Reward(value=-0.1, reason="No suitable capstone project found")
            if project.id in student.completed_projects:
                return Reward(value=-0.1, reason="Project already completed")
            action.project_id = project.id
            return Reward(value=0.5, reason="Capstone project assigned")

        # ── recommend_resource ──
        elif action.type == ActionType.RECOMMEND_RESOURCE:
            resources = get_resources_for_topic(action.topic_id or "")
            if resources:
                return Reward(value=0.1, reason="Resource recommended")
            return Reward(value=0, reason="No resources found for topic")

        # ── suggest_event ──
        elif action.type == ActionType.SUGGEST_EVENT:
            return Reward(value=0.1, reason="Event/hackathon suggested")

        # ── mark_job_ready ──
        elif action.type == ActionType.MARK_JOB_READY:
            if student.job_readiness_score >= 0.8:
                return Reward(value=0.99, reason="Student is job-ready!", is_terminal=True)
            else:
                return Reward(value=-0.2, reason=f"Not job-ready yet (score: {student.job_readiness_score})")

        return Reward(value=0, reason="Unknown action")

    def _execute_action(self, action: Action) -> Dict:
        """Execute the action and update student state."""
        info = {"action": action.type.value}

        if action.type == ActionType.RECOMMEND_TOPIC and action.topic_id:
            student = student_manager.get(self.student_id)
            if student:
                student.current_topic = action.topic_id
                # Simulate learning: boost skill for this topic
                existing_skill = next(
                    (s for s in student.self_assessed_skills if s.skill == action.topic_id), None
                )
                from environment.models import SkillLevel
                if existing_skill:
                    existing_skill.proficiency = min(existing_skill.proficiency + 0.3, 0.99)
                else:
                    student.self_assessed_skills.append(
                        SkillLevel(skill=action.topic_id, level="Studied", proficiency=0.5)
                    )
                student_manager.save(student)
                # Update difficulty model when topic is studied
                self.difficulty_model.update_skill_after_topic_study(action.topic_id)
                info["topic"] = action.topic_id

        elif action.type == ActionType.ASSIGN_QUIZ and action.topic_id:
            # Use the score calculated in _calculate_reward (same step)
            score = getattr(action, '_quiz_score', None)
            if score is None:
                # Fallback: use difficulty model
                student = student_manager.get(self.student_id)
                score = self.difficulty_model.simulate_quiz_score(
                    action.topic_id,
                    student.completed_topics if student else []
                )
            result = QuizResult(
                topic_id=action.topic_id,
                score=score,
                total_questions=5,
                correct_answers=max(0, int(score / 20)),
                passed=score >= 70,
                difficulty=action.difficulty or QuizDifficulty.MEDIUM,
            )
            student_manager.record_quiz(self.student_id, result)
            # Update difficulty model skill after quiz
            self.difficulty_model.update_skill_after_quiz(action.topic_id, score >= 70)
            info["quiz_score"] = score
            info["passed"] = score >= 70

        elif action.type in (ActionType.ASSIGN_MINI_PROJECT, ActionType.ASSIGN_CAPSTONE):
            if action.project_id:
                student_manager.complete_project(self.student_id, action.project_id)
                # Update difficulty model for project completion
                project = PROJECT_DB.get(action.project_id)
                if project:
                    if action.type == ActionType.ASSIGN_CAPSTONE:
                        student = student_manager.get(self.student_id)
                        self.difficulty_model.update_skill_after_capstone(
                            student.target_field if student else "tech"
                        )
                    else:
                        self.difficulty_model.update_skill_after_project(
                            project.skills_tested
                        )
                info["project"] = action.project_id

        elif action.type == ActionType.MARK_JOB_READY:
            student = student_manager.get(self.student_id)
            if student:
                info["job_readiness_score"] = student.job_readiness_score

        elif action.type == ActionType.RECOMMEND_RESOURCE and action.topic_id:
            # Studying a resource boosts BKT mastery for the topic
            self.difficulty_model.update_skill_after_topic_study(action.topic_id)
            info["topic"] = action.topic_id
            info["effect"] = "bkt_study_boost"

        return info

    def _resolve_project(self, action: Action, student: StudentProfile, is_capstone: bool):
        """Resolve a project from action.project_id, or auto-find by field."""
        # Direct lookup first
        if action.project_id:
            project = PROJECT_DB.get(action.project_id)
            if project:
                return project

        # Auto-find a project for the student's field
        field = student.target_field or "tech"
        field_projects = get_projects_for_field(field)
        for p in field_projects:
            if p.is_capstone == is_capstone and p.id not in student.completed_projects:
                return p

        # Fallback: any uncompleted project
        for p in PROJECT_DB.values():
            if p.is_capstone == is_capstone and p.id not in student.completed_projects:
                return p

        return None
