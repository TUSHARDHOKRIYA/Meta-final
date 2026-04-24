"""
EduPath AI — Smart Resource Fetcher
Team KRIYA | Meta Hackathon 2026

Unified entry point for course discovery.  Checks the file-based cache
first; on a miss, runs the full Scout→Critic→Curator pipeline:

  1. Scout  — 5 diversified DuckDuckGo queries → 15-20 raw candidates
  2. Critic — fetch each URL, score across 6 dimensions → pass/reject
  3. Curator — profile-aware re-rank → top 3 curated courses

Results are cached for 7 days in backend/cache/topic_resources.json.
"""
import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=3)

# Legacy cache path (kept for backward compat with any existing data)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "resource_cache.json")
CACHE_EXPIRY_SECONDS = 7 * 24 * 60 * 60  # 7 days


# ── Main entry point ─────────────────────────────────────────────────

def fetch_resources_for_topic(topic_name: str, topic_id: str = "",
                              student_profile: Optional[Dict] = None) -> List[Dict]:
    """Fetch curated course links for a topic.

    Checks the resource cache first.  On a miss, runs the full
    Scout→Critic→Curator pipeline and caches the result.

    Args:
        topic_name:      Human-readable topic name (e.g. "Python Basics").
        topic_id:        Optional topic identifier for cache keying.
        student_profile: Optional StudentProfile dict for personalisation.

    Returns:
        List of 1-3 curated course dicts, each with at least:
        title, url, platform, quality_score, why_selected, is_free.
    """
    from cache.resource_cache import ResourceCache

    cache = ResourceCache()
    cache_key = topic_id or topic_name

    # Step 1: Check cache
    cached = cache.get(cache_key)
    if cached:
        logger.info(f"[Fetcher] Cache HIT for '{cache_key}' ({len(cached)} courses)")
        return cached

    logger.info(f"[Fetcher] Cache MISS for '{cache_key}', running Scout→Critic→Curator pipeline...")

    # Step 2: Run Scout→Critic→Curator pipeline
    try:
        from ai.learning_loop.scout_agent import search as scout_search
        from ai.learning_loop.critic_agent import score_all as critic_score_all
        from ai.learning_loop.curator_agent import select as curator_select

        candidates = scout_search(topic_name)            # 15-20 raw URLs
        scored = critic_score_all(candidates)             # Score each URL
        curated = curator_select(scored, student_profile) # Top 3

        # Step 3: Cache result
        cache.set(cache_key, curated)

        logger.info(f"[Fetcher] Pipeline returned {len(curated)} courses for '{cache_key}'")
        return curated

    except Exception as e:
        logger.error(f"[Fetcher] Pipeline failed for '{cache_key}': {e}")
        # Fall back to legacy search if the new pipeline fails entirely
        return _legacy_fallback(topic_name, topic_id)


def get_alternative_resources(topic_name: str, topic_id: str = "",
                              offset: int = 3) -> List[Dict]:
    """Get alternative resources beyond the initially shown ones.

    Re-runs the Scout→Critic pipeline with a broader search if the
    cache has been exhausted.

    Args:
        topic_name: Human-readable topic name.
        topic_id:   Optional topic identifier.
        offset:     Number of initial results to skip.

    Returns:
        List of alternative course dicts.
    """
    from cache.resource_cache import ResourceCache

    cache = ResourceCache()
    cache_key = topic_id or topic_name
    cached = cache.get(cache_key)

    # If we have more than `offset` cached results, return the rest
    if cached and len(cached) > offset:
        return cached[offset:]

    # Otherwise run a fresh broader search
    logger.info(f"[Fetcher] Alternative resources requested for '{topic_name}'")
    try:
        from ai.learning_loop.scout_agent import search as scout_search
        from ai.learning_loop.critic_agent import score_all as critic_score_all
        from ai.learning_loop.curator_agent import select as curator_select

        candidates = scout_search(topic_name)
        scored = critic_score_all(candidates)
        # Return next 3 after the initial set
        curated = curator_select(scored, top_n=6)
        return curated[offset:] if len(curated) > offset else curated

    except Exception as e:
        logger.warning(f"[Fetcher] Alternative resource search failed: {e}")
        return [{
            "title": f"Search more: {topic_name} courses",
            "url": (
                f"https://www.google.com/search?q="
                f"{topic_name.replace(' ', '+')}+best+free+course+"
                f"{time.strftime('%Y')}"
            ),
            "description": f"Find more free courses for {topic_name}",
            "source": "Google",
            "resource_type": "search",
            "duration_estimate": "Varies",
            "is_fallback": True,
        }]


# ── Async wrappers (used by the FastAPI resource endpoints) ──────────

async def fetch_resources_async(topic_name: str, topic_id: str = "") -> List[Dict]:
    """Async wrapper for resource fetching."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        fetch_resources_for_topic,
        topic_name,
        topic_id,
    )


async def fetch_alternative_resources_async(topic_name: str, topic_id: str = "",
                                            offset: int = 3) -> List[Dict]:
    """Async wrapper for alternative resource fetching."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        get_alternative_resources,
        topic_name,
        topic_id,
        offset,
    )


# ── Legacy helpers (kept for backward compatibility) ─────────────────

def _detect_source(url: str) -> str:
    """Detect the resource source from its URL."""
    url_lower = url.lower()
    source_map = {
        "kaggle.com": "Kaggle",
        "freecodecamp.org": "freeCodeCamp",
        "fast.ai": "fast.ai",
        "huggingface.co": "HuggingFace",
        "ocw.mit.edu": "MIT OCW",
        "coursera.org": "Coursera",
        "edx.org": "edX",
        "khanacademy.org": "Khan Academy",
        "udemy.com": "Udemy",
        "youtube.com": "YouTube",
        "youtu.be": "YouTube",
        "codecademy.com": "Codecademy",
        "pluralsight.com": "Pluralsight",
    }
    for domain, name in source_map.items():
        if domain in url_lower:
            return name
    return "Other"


def _detect_resource_type(url: str, title: str) -> str:
    """Detect resource type from URL and title."""
    combined = (url + " " + title).lower()
    if "notebook" in combined or "colab" in combined or "kaggle.com/code" in combined:
        return "notebook"
    elif "youtube.com" in combined or "youtu.be" in combined or "video" in combined:
        return "video"
    elif "article" in combined or "blog" in combined or "news" in combined:
        return "article"
    elif "tutorial" in combined:
        return "tutorial"
    else:
        return "course"


def _estimate_duration(description: str, resource_type: str) -> str:
    """Estimate learning duration from description or resource type."""
    desc_lower = description.lower()
    for pattern in ["hour", "min", "week"]:
        if pattern in desc_lower:
            words = desc_lower.split()
            for i, w in enumerate(words):
                if pattern in w and i > 0:
                    try:
                        num = int(words[i - 1])
                        if "hour" in w:
                            return f"~{num} hours"
                        elif "min" in w:
                            return f"~{num} min"
                        elif "week" in w:
                            return f"~{num} weeks"
                    except ValueError:
                        pass

    defaults = {
        "notebook": "~1 hour",
        "video": "~45 min",
        "article": "~30 min",
        "tutorial": "~1.5 hours",
        "course": "~2 hours",
    }
    return defaults.get(resource_type, "~2 hours")


def _legacy_fallback(topic_name: str, topic_id: str) -> List[Dict]:
    """Fallback to curated resources from curriculum.py when the
    new pipeline fails entirely.
    """
    try:
        from environment.curriculum import TOPIC_GRAPH, RESOURCE_DB

        topic = TOPIC_GRAPH.get(topic_id)
        if topic and topic.resources:
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "description": r.description or f"{r.title} on {r.platform}",
                    "source": r.platform or _detect_source(r.url),
                    "resource_type": r.type.value.replace("_", " "),
                    "duration_estimate": "~2 hours",
                    "quality_score": 0.5,
                    "platform": r.platform or _detect_source(r.url),
                    "is_free": True,
                }
                for r in topic.resources[:3]
            ]

        if topic_id in RESOURCE_DB:
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "description": r.description or f"{r.title} on {r.platform}",
                    "source": r.platform or _detect_source(r.url),
                    "resource_type": r.type.value.replace("_", " "),
                    "duration_estimate": "~2 hours",
                    "quality_score": 0.5,
                    "platform": r.platform or _detect_source(r.url),
                    "is_free": True,
                }
                for r in RESOURCE_DB[topic_id][:3]
            ]
    except Exception as e:
        logger.warning(f"Legacy fallback lookup failed: {e}")

    # Absolute fallback: direct links to known free course platforms
    topic_q = topic_name.replace(" ", "+")
    topic_slug = topic_name.lower().replace(" ", "-")
    return [
        {
            "title": f"{topic_name} - Free Course on freeCodeCamp",
            "url": f"https://www.freecodecamp.org/news/search/?query={topic_q}",
            "description": f"Search freeCodeCamp for free {topic_name} courses, tutorials, and guides",
            "source": "freeCodeCamp",
            "resource_type": "course",
            "duration_estimate": "~2 hours",
            "quality_score": 0.55,
            "platform": "freecodecamp.org",
            "is_free": True,
        },
        {
            "title": f"Learn {topic_name} - Coursera",
            "url": f"https://www.coursera.org/search?query={topic_q}",
            "description": f"Find top-rated {topic_name} courses from universities worldwide (free to audit)",
            "source": "Coursera",
            "resource_type": "course",
            "duration_estimate": "~4 hours",
            "quality_score": 0.5,
            "platform": "coursera.org",
            "is_free": True,
        },
        {
            "title": f"{topic_name} Tutorial - W3Schools",
            "url": f"https://www.w3schools.com/{topic_slug.split('-')[0]}/",
            "description": f"Interactive {topic_name} tutorial with examples and exercises",
            "source": "W3Schools",
            "resource_type": "tutorial",
            "duration_estimate": "~1.5 hours",
            "quality_score": 0.45,
            "platform": "w3schools.com",
            "is_free": True,
        },
    ]
