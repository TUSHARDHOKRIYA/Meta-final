"""
EduPath AI — Curator Agent
Team KRIYA | OpenEnv Hackathon 2026

Learning Loop Agent 3/4: Receives the Critic's scored candidate list,
applies student-profile-aware re-ranking, and selects the top 3 courses.
Also generates cheat sheets, study notes, and mini-project specs when
invoked through the legacy orchestrator interface.

Pipeline position:  Scout → Critic → **Curator**

Selection logic:
  1. Filter to passed=True results only
  2. Apply profile-based score boosts (beginner, video preference, etc.)
  3. Sort by adjusted score descending
  4. Return top 3 with full metadata
  5. Fallback: lower threshold to 0.30 if fewer than 3 pass
"""
import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)


# ── Public API (new pipeline) ────────────────────────────────────────

def select(scored_candidates: List[Dict],
           student_profile: Optional[Dict] = None,
           top_n: int = 3,
           topic_name: str = "") -> List[Dict]:
    """Select the top N courses from the Critic's scored list, matched
    to the student's profile.

    Args:
        scored_candidates: Scored dicts from critic_agent.score_all().
                           Each has keys: url, title, platform, score,
                           passed, score_breakdown, reject_reason.
        student_profile:   Optional StudentProfile dict.  Used to
                           boost scores for beginner / video preferences.
        top_n:             Number of courses to return (default 3).

    Returns:
        List of curated course dicts ready for the frontend.
    """
    profile = student_profile or {}

    # Step 1 — filter to passing candidates (threshold 0.45)
    passing = [c for c in scored_candidates if c.get("passed")]

    # Fallback: if fewer than top_n pass, lower threshold to 0.30
    if len(passing) < top_n:
        logger.info(
            f"[Curator] Only {len(passing)} passed at 0.45 — "
            f"lowering threshold to 0.30"
        )
        passing = [
            c for c in scored_candidates
            if c.get("score", 0) >= 0.30 and not (c.get("reject_reason") or "").startswith("Hard reject")
        ]

    # Ultimate fallback: take everything that isn't hard-rejected
    if not passing:
        logger.warning("[Curator] No candidates at 0.30 — taking best available")
        passing = [
            c for c in scored_candidates
            if not (c.get("reject_reason") or "").startswith("Hard reject")
        ]


    # If still empty (all hard-rejected), leave `passing` empty
    # so the platform fallback (freeCodeCamp/Coursera/Khan) kicks in below

    # Step 2 — profile-based score adjustment
    adjusted = []
    for c in passing:
        adj_score = c.get("score", 0)
        content = (c.get("title", "") + " " + c.get("url", "")).lower()
        bd = c.get("score_breakdown", {})

        # Beginner boost
        level = profile.get("confidence_level", profile.get("level", "medium"))
        if level in ("low", "beginner"):
            if bd.get("is_course", 0) >= 2:
                adj_score += 0.05
            if any(w in content for w in ["beginner", "introduction", "intro", "101"]):
                adj_score += 0.05

        # Video preference boost
        style = profile.get("learning_style", "")
        if style == "video":
            if any(w in content for w in ["video", "watch", "lecture"]):
                adj_score += 0.03

        adjusted.append({**c, "_adj_score": adj_score})

    # Step 3 — sort by adjusted score descending
    adjusted.sort(key=lambda x: x["_adj_score"], reverse=True)

    # Step 4 — build output
    results: List[Dict] = []
    for rank, c in enumerate(adjusted[:top_n], start=1):
        results.append({
            "rank": rank,
            "title": c.get("title", ""),
            "url": c.get("url", ""),
            "platform": c.get("platform", _extract_domain(c.get("url", ""))),
            "quality_score": round(c["_adj_score"], 4),
            "why_selected": _generate_reason(c, profile, rank),
            "is_free": c.get("score_breakdown", {}).get("is_free", 1) > 0,
            "estimated_hours": _extract_hours(c),
            "description": c.get("snippet", c.get("title", "")),
            "source": c.get("platform", "Other"),
            "resource_type": "course",
            "duration_estimate": _extract_hours_str(c),
        })

    # Guarantee at least 1 result — use direct course platform links
    if not results:
        # Use provided topic_name, or try to extract from candidate titles
        topic_hint = topic_name
        if not topic_hint:
            for c in scored_candidates[:3]:
                t = c.get("title", "")
                if t and t != "Search for courses":
                    # Filter out non-English titles
                    non_ascii = sum(1 for ch in t if ord(ch) > 127)
                    if non_ascii < len(t) * 0.3:
                        topic_hint = t.split(" - ")[0].split(" | ")[0][:40]
                        break
        topic_q = topic_hint.replace(" ", "+") if topic_hint else "online+courses"

        results = [
            {
                "rank": 1,
                "title": f"Learn {topic_hint or 'Online'} - YouTube",
                "url": f"https://www.youtube.com/results?search_query={topic_q}+full+course",
                "platform": "youtube.com",
                "quality_score": 0.55,
                "why_selected": "Free full-length video courses and tutorials.",
                "is_free": True,
                "estimated_hours": None,
                "description": f"Free {topic_hint or 'online'} video courses and tutorials",
                "source": "YouTube",
                "resource_type": "video",
                "duration_estimate": "Varies",
            },
            {
                "rank": 2,
                "title": f"Learn {topic_hint or 'Online'} - Coursera",
                "url": f"https://www.coursera.org/search?query={topic_q}",
                "platform": "coursera.org",
                "quality_score": 0.5,
                "why_selected": "University-backed courses, free to audit.",
                "is_free": True,
                "estimated_hours": None,
                "description": f"Top-rated {topic_hint or 'online'} courses from universities",
                "source": "Coursera",
                "resource_type": "course",
                "duration_estimate": "~4 hours",
            },
            {
                "rank": 3,
                "title": f"{topic_hint or 'Online'} Courses - Udemy",
                "url": f"https://www.udemy.com/courses/search/?q={topic_q}&price=price-free",
                "platform": "udemy.com",
                "quality_score": 0.5,
                "why_selected": "Free community-rated courses and tutorials.",
                "is_free": True,
                "estimated_hours": None,
                "description": f"Free {topic_hint or 'online'} courses by expert instructors",
                "source": "Udemy",
                "resource_type": "course",
                "duration_estimate": "~2 hours",
            },
        ]

    logger.info(f"[Curator] Selected {len(results)} courses (from {len(scored_candidates)} candidates)")
    return results


# ── Helpers ──────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Extract a clean domain from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


def _generate_reason(candidate: Dict, profile: Dict, rank: int) -> str:
    """Generate a one-sentence reason explaining why this course was selected."""
    bd = candidate.get("score_breakdown", {})
    parts = []
    if bd.get("is_course", 0) >= 2:
        parts.append("structured course content")
    if bd.get("is_free", 0) >= 2:
        parts.append("confirmed free access")
    if bd.get("is_structured", 0) >= 2:
        parts.append("includes exercises")
    if bd.get("is_credible", 0) >= 2:
        parts.append("credible source")
    if bd.get("freshness", 0) >= 1:
        parts.append("up-to-date material")

    goal = profile.get("target_role", profile.get("learning_goal", "your goal"))

    if parts:
        return f"Ranked #{rank} for {goal}: {', '.join(parts[:3])}."
    return f"Ranked #{rank} based on overall quality score."


def _extract_hours(candidate: Dict) -> Optional[int]:
    """Try to extract estimated hours from candidate metadata."""
    # Check if the candidate already has hours
    hours = candidate.get("estimated_hours")
    if hours and isinstance(hours, (int, float)):
        return int(hours)

    # Try to extract from snippet / title
    text = candidate.get("title", "") + " " + candidate.get("snippet", "")
    match = re.search(r"(\d+)\s*hours?", text, re.I)
    if match:
        return int(match.group(1))
    return None


def _extract_hours_str(candidate: Dict) -> str:
    """Return human-readable duration estimate."""
    hours = _extract_hours(candidate)
    if hours:
        return f"~{hours} hours"
    return "~2 hours"


# ── Legacy interface (used by learning_loop/orchestrator.py) ─────────

def run(evaluations: List[Dict], candidates: List[Dict],
        student_profile: Dict, topic_info: Dict) -> Dict:
    """Select the best course and generate all learning materials.

    Preserves the original function signature so that the orchestrator
    and any other caller continue to work unchanged.

    Args:
        evaluations: Scored evaluations from the Critic Agent.
        candidates: Original course candidates from the Scout Agent.
        student_profile: StudentProfile dict.
        topic_info: Dict with topic_id, topic_name.

    Returns:
        Dict with selected_course, cheat_sheet, study_notes, mini_project.
    """
    # Try LLM-based curator first
    if is_api_key_set():
        try:
            system_prompt = get_prompt("curator")
            import json
            eval_str = json.dumps(evaluations[:5], indent=2, default=str)
            candidates_map = {c.get("id"): c for c in candidates}
            top_id = evaluations[0]["course_id"] if evaluations else None
            top_course = candidates_map.get(top_id, {})

            user_prompt = f"""STUDENT PROFILE:
Target Role: {student_profile.get('target_role', 'Unknown')}
Domain: {student_profile.get('target_domain', 'tech')}
Learning Style: {student_profile.get('learning_style', 'video')}

TOPIC: {topic_info.get('topic_name', topic_info.get('topic_id', '?'))}
Topic ID: {topic_info.get('topic_id', '?')}

TOP COURSE EVALUATIONS:
{eval_str}

TOP COURSE DETAILS:
{json.dumps(top_course, indent=2, default=str)}

Select the best course and generate: cheat_sheet, study_notes, and mini_project. Everything must be specific to this student's domain and goal."""

            result = generate_json_with_retry(system_prompt, user_prompt)
            if result and "selected_course" in result:
                logger.info(f"Curator (LLM) selected course for {topic_info.get('topic_name', '?')}")
                return result
        except Exception as e:
            logger.error(f"Curator LLM failed: {e}")

    # Fallback: deterministic material generation
    return _fallback(evaluations, candidates, student_profile, topic_info)


def _fallback(evaluations: List[Dict], candidates: List[Dict],
              student_profile: Dict, topic_info: Dict) -> Dict:
    """Deterministic fallback with generated study materials."""
    candidates_map = {c.get("id"): c for c in candidates}

    # Select top course
    top_id = evaluations[0]["course_id"] if evaluations else (
        candidates[0]["id"] if candidates else "unknown"
    )
    top_course = candidates_map.get(top_id, candidates[0] if candidates else {})

    topic_name = topic_info.get("topic_name", topic_info.get("topic_id", "Topic"))
    goal = student_profile.get("target_role", "career goal")
    domain = student_profile.get("target_domain", "tech")

    return {
        "selected_course": {
            "course_id": top_course.get("id", "course_1"),
            "title": top_course.get("title", f"{topic_name} Course"),
            "url": top_course.get("url", ""),
            "selection_rationale": (
                f"Best match for {goal} based on relevance, difficulty, "
                f"and content quality scores."
            ),
        },
        "cheat_sheet": {
            "title": f"{topic_name} — Cheat Sheet",
            "vocabulary": [
                {"term": f"Core concept of {topic_name}",
                 "definition": f"Fundamental principle underlying {topic_name}"},
                {"term": "Prerequisites",
                 "definition": "Topics you should understand before diving deep"},
                {"term": "Key framework",
                 "definition": f"Primary tool/framework used in {topic_name}"},
                {"term": "Best practice",
                 "definition": f"Industry-standard approach to {topic_name}"},
                {"term": "Common pitfall",
                 "definition": f"Mistake to avoid when learning {topic_name}"},
            ],
            "core_concept": (
                f"{topic_name} is essential for anyone pursuing {goal} in "
                f"{domain}. It provides the foundational understanding "
                f"needed to build practical skills."
            ),
            "why_it_matters_for_your_goal": (
                f"As a future {goal}, mastering {topic_name} will enable "
                f"you to work with real-world {domain} challenges effectively."
            ),
            "watch_for": [
                "Understand the 'why' before the 'how'",
                "Practice with real datasets or examples",
                "Connect this to topics you've already learned",
            ],
        },
        "study_notes": {
            "title": f"Study Notes — {topic_name}",
            "what_you_should_now_know": [
                f"Core principles of {topic_name}",
                f"How {topic_name} applies to {domain}",
                "Key terminology and definitions",
                "Basic implementation patterns",
                "Common use cases and examples",
            ],
            "most_important_concept": (
                f"The fundamental principle of {topic_name} and how it "
                f"connects to your goal of becoming a {goal}."
            ),
            "domain_connection": (
                f"{topic_name} is directly applicable to {domain} workflows "
                f"and will be built upon in later topics."
            ),
            "remember_for_quiz": [
                f"Core definition of {topic_name}",
                "Key differences from related concepts",
                f"One practical application in {domain}",
            ],
        },
        "mini_project": {
            "title": f"Mini Project: Apply {topic_name}",
            "description": (
                f"Build a small project demonstrating your understanding of "
                f"{topic_name} using real data relevant to {domain}."
            ),
            "dataset": f"Public dataset relevant to {domain} (e.g., from Kaggle or UCI)",
            "requirements": [
                f"Apply core {topic_name} concepts",
                "Use a real dataset",
                "Document your approach",
                "Include at least one visualization or output",
            ],
            "domain_reflection": (
                f"How does this project connect to your goal of becoming a {goal}?"
            ),
            "estimated_hours": top_course.get("estimated_hours", 4) // 3 + 1,
            "pass_criteria": (
                "Project demonstrates understanding of core concepts "
                "with working code and documentation."
            ),
        },
    }
