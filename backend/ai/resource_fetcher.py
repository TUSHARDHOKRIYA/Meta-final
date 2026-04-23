"""
EduPath AI — Smart Resource Fetcher
Team KRIYA | Meta Hackathon 2026

Multi-platform course discovery engine. Searches across Coursera, edX,
Khan Academy, freeCodeCamp, Udemy, and YouTube via DuckDuckGo, then
ranks results using platform reputation heuristics and optional LLM
quality scoring. Results are cached for 7 days.
"""
import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "resource_cache.json")
CACHE_EXPIRY_SECONDS = 7 * 24 * 60 * 60  # 7 days

_executor = ThreadPoolExecutor(max_workers=3)

# Platform quality tiers (used for scoring)
PLATFORM_SCORES = {
    "coursera.org": 95,
    "edx.org": 93,
    "khanacademy.org": 90,
    "mit.edu": 95,
    "stanford.edu": 95,
    "freecodecamp.org": 88,
    "kaggle.com": 87,
    "fast.ai": 90,
    "huggingface.co": 88,
    "udemy.com": 80,
    "youtube.com": 75,
    "pluralsight.com": 85,
    "codecademy.com": 82,
    "w3schools.com": 70,
    "geeksforgeeks.org": 72,
    "tutorialspoint.com": 65,
}


def _load_cache() -> dict:
    """Load the resource cache from disk."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    """Save the resource cache to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _is_cache_fresh(entry: dict) -> bool:
    """Check if a cache entry is still fresh (within 7 days)."""
    cached_at = entry.get("cached_at", 0)
    return (time.time() - cached_at) < CACHE_EXPIRY_SECONDS


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """Run a single DuckDuckGo search and return parsed results."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "description": r.get("body", ""),
                })
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
        return []


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


def _get_platform_score(url: str) -> int:
    """Score a URL based on platform reputation."""
    url_lower = url.lower()
    for domain, score in PLATFORM_SCORES.items():
        if domain in url_lower:
            return score
    return 50  # Unknown platform


def _extract_rating_from_text(text: str) -> Optional[float]:
    """Try to extract a star rating from description text."""
    import re
    # Look for patterns like "4.7 stars", "4.8/5", "rated 4.5"
    patterns = [
        r'(\d+\.?\d*)\s*(?:stars?|/5|out of 5)',
        r'(?:rated?|rating)\s*:?\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*⭐',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                rating = float(match.group(1))
                if 0 <= rating <= 5:
                    return rating
            except ValueError:
                pass
    return None


def _score_resource(resource: dict, topic_name: str) -> float:
    """Score a resource for relevance and quality (0-100)."""
    score = 0

    # Platform reputation (0-40 points)
    platform_score = _get_platform_score(resource["url"])
    score += (platform_score / 100) * 40

    # Title relevance to topic (0-30 points)
    topic_words = set(topic_name.lower().split())
    title_words = set(resource["title"].lower().split())
    overlap = len(topic_words & title_words)
    relevance = min(overlap / max(len(topic_words), 1), 1)
    score += relevance * 30

    # Review/rating presence (0-15 points)
    rating = _extract_rating_from_text(resource.get("description", ""))
    if rating is not None:
        score += (rating / 5) * 15
    elif any(word in resource.get("description", "").lower() for word in
             ["highly rated", "best", "top", "recommended", "popular", "bestseller"]):
        score += 10

    # Resource type bonus (0-15 points)
    rtype = resource.get("resource_type", "course")
    type_scores = {"course": 15, "tutorial": 13, "video": 10, "notebook": 12, "article": 8}
    score += type_scores.get(rtype, 5)

    return round(score, 1)


def _rank_with_ai(resources: List[Dict], topic_name: str) -> List[Dict]:
    """Use LLM to rank and enhance resource descriptions."""
    try:
        from ai.llm_client import generate_json, is_api_key_set
        if not is_api_key_set() or len(resources) == 0:
            return resources

        resource_text = "\n".join([
            f"{i+1}. [{r['source']}] {r['title']} — {r.get('description', '')[:100]}"
            for i, r in enumerate(resources[:8])
        ])

        result = generate_json(
            system_prompt=(
                "You are a course quality evaluator. Given a list of learning resources for a topic, "
                "rank them by quality and relevance. Return JSON with a 'rankings' array of objects: "
                "{'index': number (1-based), 'quality_score': number (1-10), 'reason': string (short, 15 words max)}. "
                "Prioritize: structured courses > tutorials > videos > articles. "
                "Prefer reputable platforms (Coursera, edX, MIT, Stanford, Khan Academy)."
            ),
            user_prompt=f"Topic: {topic_name}\n\nResources:\n{resource_text}\n\nRank these resources."
        )

        if result and "rankings" in result:
            ranking_map = {}
            for r in result["rankings"]:
                idx = r.get("index", 0) - 1
                if 0 <= idx < len(resources):
                    ranking_map[idx] = {
                        "ai_score": r.get("quality_score", 5),
                        "ai_reason": r.get("reason", ""),
                    }

            for i, res in enumerate(resources):
                if i in ranking_map:
                    res["ai_score"] = ranking_map[i]["ai_score"]
                    res["ai_reason"] = ranking_map[i]["ai_reason"]

            # Sort by AI score (descending)
            resources.sort(key=lambda x: x.get("ai_score", 5), reverse=True)

    except Exception as e:
        logger.warning(f"AI ranking failed, using heuristic scoring: {e}")

    return resources


def fetch_resources_for_topic(topic_name: str, topic_id: str = "") -> List[Dict]:
    """
    Fetch real course links for a topic using multi-platform DuckDuckGo Search.
    Returns up to 6 ranked resources. Top 3 shown initially, rest available via pagination.
    Results are cached for 7 days.
    """
    cache_key = topic_id or topic_name.lower().replace(" ", "_")

    # Check cache first
    cache = _load_cache()
    if cache_key in cache and _is_cache_fresh(cache[cache_key]):
        logger.info(f"Resource cache HIT for '{cache_key}'")
        return cache[cache_key]["resources"]

    logger.info(f"Resource cache MISS for '{cache_key}', fetching from multiple platforms...")

    # Multi-platform search queries with review-focused keywords
    queries = [
        f"{topic_name} best course review site:coursera.org OR site:edx.org",
        f"{topic_name} tutorial site:freecodecamp.org OR site:khanacademy.org",
        f"{topic_name} course highly rated site:udemy.com OR site:youtube.com",
        f"{topic_name} learn free course {topic_name} beginner",
    ]

    all_results = []
    for query in queries:
        results = _search_duckduckgo(query, max_results=3)
        all_results.extend(results)

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r["url"] and r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            source = _detect_source(r["url"])
            rtype = _detect_resource_type(r["url"], r["title"])
            resource = {
                "title": r["title"],
                "url": r["url"],
                "description": r["description"][:250],
                "source": source,
                "resource_type": rtype,
                "duration_estimate": _estimate_duration(r["description"], rtype),
                "rating": _extract_rating_from_text(r["description"]),
            }
            resource["quality_score"] = _score_resource(resource, topic_name)
            unique_results.append(resource)

    # Sort by heuristic quality score first
    unique_results.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # Keep top 6
    resources = unique_results[:6]

    # Try AI ranking (enhances with ai_score and ai_reason)
    if len(resources) > 1:
        resources = _rank_with_ai(resources, topic_name)

    # Mark the top result
    if resources:
        resources[0]["is_top_pick"] = True

    # If nothing found, fall back to curated resources
    if not resources:
        resources = _get_fallback_resources(topic_name, topic_id)

    # Save to cache (all 6, not just top 3)
    cache[cache_key] = {
        "resources": resources,
        "cached_at": time.time(),
    }
    _save_cache(cache)

    return resources


def get_alternative_resources(topic_name: str, topic_id: str = "", offset: int = 3) -> List[Dict]:
    """
    Get alternative resources beyond the initially shown ones.
    Returns resources from offset onwards in the cached list.
    If exhausted, performs a fresh Google-style fallback search.
    """
    cache_key = topic_id or topic_name.lower().replace(" ", "_")
    cache = _load_cache()

    if cache_key in cache and cache[cache_key].get("resources"):
        all_resources = cache[cache_key]["resources"]
        if offset < len(all_resources):
            return all_resources[offset:]

    # All cached resources exhausted — do a broader search
    logger.info(f"Alternative resources requested for '{topic_name}', performing broader search...")
    fallback_queries = [
        f"{topic_name} online course free 2024",
        f"best {topic_name} tutorial for beginners",
    ]

    results = []
    for query in fallback_queries:
        results.extend(_search_duckduckgo(query, max_results=3))

    # Deduplicate against existing cache
    existing_urls = set()
    if cache_key in cache:
        for r in cache[cache_key].get("resources", []):
            existing_urls.add(r["url"])

    new_resources = []
    for r in results:
        if r["url"] and r["url"] not in existing_urls:
            existing_urls.add(r["url"])
            source = _detect_source(r["url"])
            rtype = _detect_resource_type(r["url"], r["title"])
            new_resources.append({
                "title": r["title"],
                "url": r["url"],
                "description": r["description"][:250],
                "source": source,
                "resource_type": rtype,
                "duration_estimate": _estimate_duration(r["description"], rtype),
                "quality_score": _score_resource({"url": r["url"], "title": r["title"], "description": r["description"], "source": source, "resource_type": rtype}, topic_name),
            })

    if not new_resources:
        # Absolute fallback — Google search link
        new_resources = [{
            "title": f"Search more: {topic_name} courses",
            "url": f"https://www.google.com/search?q={topic_name.replace(' ', '+')}+best+free+course+{time.strftime('%Y')}",
            "description": f"Find more free courses and tutorials for {topic_name} on Google",
            "source": "Google",
            "resource_type": "search",
            "duration_estimate": "Varies",
            "is_fallback": True,
        }]

    return new_resources


async def fetch_resources_async(topic_name: str, topic_id: str = "") -> List[Dict]:
    """Async wrapper for resource fetching."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        fetch_resources_for_topic,
        topic_name,
        topic_id,
    )


async def fetch_alternative_resources_async(topic_name: str, topic_id: str = "", offset: int = 3) -> List[Dict]:
    """Async wrapper for alternative resource fetching."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        get_alternative_resources,
        topic_name,
        topic_id,
        offset,
    )


def _get_fallback_resources(topic_name: str, topic_id: str) -> List[Dict]:
    """Fallback to curated resources from curriculum.py when search fails."""
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
                }
                for r in RESOURCE_DB[topic_id][:3]
            ]
    except Exception as e:
        logger.warning(f"Fallback resource lookup failed: {e}")

    # Absolute fallback
    return [
        {
            "title": f"Search: {topic_name}",
            "url": f"https://www.google.com/search?q={topic_name.replace(' ', '+')}+free+course",
            "description": f"Find free courses and tutorials for {topic_name}",
            "source": "Google",
            "resource_type": "course",
            "duration_estimate": "~2 hours",
            "is_fallback": True,
        }
    ]
