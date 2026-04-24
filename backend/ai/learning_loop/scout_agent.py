"""
EduPath AI — Scout Agent
Team KRIYA | OpenEnv Hackathon 2026

Learning Loop Agent 1/4: Runs 5 diversified DuckDuckGo search queries per
topic to discover 15-20 raw course candidates.  No platform names are
hardcoded in queries — every URL that appears in search results proceeds
to the Critic Agent for merit-based scoring.

Pipeline position:  Scout → Critic → Curator
"""
import logging
from typing import Dict, List

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)

# ── Diversified query templates ──────────────────────────────────────
# These cover different search intents (beginner, certification, hands-on,
# academic, project-based) without naming any specific platform so that
# new high-quality platforms are discovered automatically.
QUERY_TEMPLATES = [
    "{topic} free course for beginners",
    "{topic} full course tutorial 2025",
    "{topic} free certification learn online",
    "{topic} open courseware lecture notes",
    "{topic} hands-on free training project",
]


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """Run a single DuckDuckGo search and return parsed results.

    Handles both the old (duckduckgo_search) and new (ddgs) package names
    so the code works regardless of which version is installed.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*renamed.*ddgs.*")
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
        return []


def search(topic: str) -> List[Dict]:
    """Run 5 diversified search queries for a topic and return 15-20
    deduplicated raw candidate URLs.

    This is the Scout Agent's primary entry point.  It fires all 5
    query templates, collects every result, and deduplicates by URL.
    No filtering by domain or platform name happens here — that is
    the Critic Agent's responsibility.

    Args:
        topic: Human-readable topic name (e.g. "python basics").

    Returns:
        List of dicts, each with keys: title, url, snippet.
    """
    all_results: List[Dict] = []
    for template in QUERY_TEMPLATES:
        query = template.format(topic=topic)
        hits = _search_duckduckgo(query, max_results=5)
        all_results.extend(hits)

    # Deduplicate by URL, preserving first occurrence
    seen_urls: set = set()
    unique: List[Dict] = []
    for r in all_results:
        url = r.get("url", "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)

    logger.info(
        f"[Scout] '{topic}' — {len(all_results)} raw hits, "
        f"{len(unique)} unique candidates after dedup"
    )
    return unique


# ── Legacy interface (used by learning_loop/orchestrator.py) ─────────

def run(topic_id: str, topic_name: str, student_profile: Dict) -> Dict:
    """Search for course candidates for a topic.

    This preserves the original function signature so that the
    orchestrator and any other caller continue to work unchanged.

    Args:
        topic_id: The topic identifier.
        topic_name: Human-readable topic name.
        student_profile: StudentProfile dict.

    Returns:
        Dict with topic_id, topic_name, and candidates list.
    """
    # Try LLM-based scout first (if API key is set)
    if is_api_key_set():
        try:
            system_prompt = get_prompt("scout")
            user_prompt = f"""TOPIC TO RESEARCH:
Topic ID: {topic_id}
Topic Name: {topic_name}

STUDENT CONTEXT:
Target Role: {student_profile.get('target_role', 'Unknown')}
Domain: {student_profile.get('target_domain', 'tech')}
Learning Style: {student_profile.get('learning_style', 'video')}
Budget: {student_profile.get('budget', 'free_only')}
Confidence: {student_profile.get('confidence_level', 'medium')}

Find 10 course candidates. Ensure diversity: at least 1 project-based, 1 theoretical, 1 quick option (<5 hours)."""

            result = generate_json_with_retry(system_prompt, user_prompt)
            if result and "candidates" in result:
                logger.info(f"Scout (LLM) found {len(result['candidates'])} candidates for {topic_name}")
                return result
        except Exception as e:
            logger.error(f"Scout LLM failed for {topic_name}: {e}")

    # Fallback: use the new diversified search pipeline
    raw_candidates = search(topic_name)

    # Convert raw search results into the candidate format expected by
    # the rest of the learning-loop pipeline
    candidates = []
    for i, r in enumerate(raw_candidates):
        candidates.append({
            "id": f"course_{i + 1}",
            "title": r["title"],
            "platform": _extract_domain(r["url"]),
            "url": r["url"],
            "estimated_hours": 6,
            "difficulty": "beginner",
            "content_type": "mixed",
            "last_updated_year": 2025,
            "is_free": True,
            "price_usd": 0,
            "has_certificate": False,
            "brief_description": r.get("snippet", "")[:250],
        })

    # Pad with generic resources if we have fewer than 3
    if len(candidates) < 3:
        candidates.extend(_generic_padding(topic_name, len(candidates)))

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "candidates": candidates,
    }


def _extract_domain(url: str) -> str:
    """Extract a readable domain name from a URL.

    e.g. 'https://www.freecodecamp.org/learn/python' → 'freecodecamp.org'
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


def _generic_padding(topic_name: str, existing_count: int) -> List[Dict]:
    """Return generic fallback candidates to ensure at least 3 results.

    These are last-resort entries when DuckDuckGo returns too few results.
    """
    generic = [
        {
            "id": f"course_{existing_count + 1}",
            "title": f"{topic_name} - Free Interactive Course",
            "platform": "freecodecamp.org",
            "url": f"https://www.google.com/search?q={topic_name.replace(' ', '+')}+free+course",
            "estimated_hours": 6,
            "difficulty": "beginner",
            "content_type": "interactive",
            "last_updated_year": 2025,
            "is_free": True,
            "price_usd": 0,
            "has_certificate": False,
            "brief_description": f"Free interactive {topic_name} course",
        },
        {
            "id": f"course_{existing_count + 2}",
            "title": f"{topic_name} - Online Lectures",
            "platform": "search",
            "url": f"https://www.google.com/search?q={topic_name.replace(' ', '+')}+lecture+notes",
            "estimated_hours": 10,
            "difficulty": "intermediate",
            "content_type": "reading",
            "last_updated_year": 2025,
            "is_free": True,
            "price_usd": 0,
            "has_certificate": False,
            "brief_description": f"Open courseware for {topic_name}",
        },
    ]
    return generic[: max(0, 3 - existing_count)]
