"""
EduPath AI — Critic Agent
Team KRIYA | OpenEnv Hackathon 2026

Learning Loop Agent 2/4: Fetches each candidate URL and scores it
across 6 quality dimensions using page-content analysis.  This is the
quality gate — no platform name is used for scoring; only the actual
content of the page determines whether a URL passes or fails.

Pipeline position:  Scout → **Critic** → Curator

Scoring dimensions (max 9 points):
  1. Is it actually a course?   (0-2)
  2. Is it free?                (0-2)
  3. Is it structured learning? (0-2)
  4. Is it credible?            (0-2)
  5. Red flags (penalties)      (0-N subtracted)
  6. Freshness                  (0-1)

PASS threshold: normalised ≥ 0.30  (i.e. ≥ ~2.7 raw points)
"""
import asyncio
import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ai.agent_prompts import get_prompt
from ai.llm_client import generate_json_with_retry, is_api_key_set

logger = logging.getLogger(__name__)

# ── Signal word lists (no platform names anywhere) ───────────────────

_COURSE_SIGNALS = [
    "syllabus", "curriculum", "modules", "week 1", "lesson",
    "chapter", "unit", "what you will learn", "course outline",
]
_FREE_POSITIVE = [
    "free", "no cost", "audit for free", "open access", "no credit card",
]
_FREE_NEGATIVE = [
    "subscribe to unlock", "premium only", "buy now",
    "pricing", "$99", "per month",
]
_STRUCTURED_SIGNALS = [
    "quiz", "assignment", "exercise", "project",
    "practice", "hands-on", "lab",
]
_CREDIBLE_SIGNALS = [
    "university", "professor", "instructor", "certificate",
    "accredited", "peer reviewed", "students enrolled",
]
_HARD_REJECT_URL = [
    "stackoverflow.com", "reddit.com", "quora.com",
    "medium.com", "stackoverflow", "forum", "discussion",
    "zhihu.com", "support.google.com", "answers.microsoft.com",
    "superuser.com", "serverfault.com", "askubuntu.com",
    "news.ycombinator.com", "twitter.com", "x.com",
    "facebook.com", "linkedin.com", "pinterest.com",
    "amazon.com", "ebay.com", "walmart.com",
    "wikipedia.org",
]
_SOFT_PENALTY_CONTENT = [
    "sponsored", "affiliate", "buy this book", "click here to purchase",
]


# ── Page fetching ────────────────────────────────────────────────────

async def _fetch_page_text(url: str, timeout: float = 5.0) -> Optional[str]:
    """Fetch a URL and return the first 5 000 characters of visible text.

    Uses httpx for async I/O with a strict per-request timeout.
    Falls back to requests if httpx is unavailable.
    Returns None on any failure (timeout, DNS, HTTP error, etc.).
    """
    try:
        import httpx
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout),
        ) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; EduPathBot/1.0)"
            })
            if resp.status_code >= 400:
                return None
            html = resp.text
    except Exception:
        # Fallback: synchronous requests (runs in executor if needed)
        try:
            import requests as _req
            resp = _req.get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (compatible; EduPathBot/1.0)"
            })
            if resp.status_code >= 400:
                return None
            html = resp.text
        except Exception:
            return None

    # Strip HTML tags to get visible text
    text = _html_to_text(html)
    return text[:5000]


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text.

    Uses BeautifulSoup if available, otherwise falls back to a simple
    regex-based tag stripper.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except ImportError:
        # Fallback: strip tags with regex
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()


# ── Scoring engine ───────────────────────────────────────────────────

def _count_signals(text: str, signals: List[str]) -> int:
    """Count how many signal phrases appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for s in signals if s in text_lower)


def _extract_domain(url: str) -> str:
    """Extract a clean domain name from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


# Known education domains — get a baseline boost when page fetch fails
_KNOWN_EDU_DOMAINS = {
    "coursera.org": 0.6, "edx.org": 0.6, "khanacademy.org": 0.6,
    "freecodecamp.org": 0.65, "codecademy.com": 0.55, "udemy.com": 0.5,
    "kaggle.com": 0.6, "fast.ai": 0.65, "huggingface.co": 0.55,
    "ocw.mit.edu": 0.7, "learn.microsoft.com": 0.55,
    "developer.mozilla.org": 0.6, "w3schools.com": 0.5,
    "geeksforgeeks.org": 0.45, "tutorialspoint.com": 0.4,
    "realpython.com": 0.55, "docs.python.org": 0.5,
    "reactjs.org": 0.5, "vuejs.org": 0.5, "angular.io": 0.5,
    "nodejs.org": 0.5, "developer.android.com": 0.5,
    "cs50.harvard.edu": 0.7, "scrimba.com": 0.55,
    "theodinproject.com": 0.65, "fullstackopen.com": 0.65,
    "javascript.info": 0.55, "css-tricks.com": 0.45,
    "programiz.com": 0.45, "javatpoint.com": 0.4,
}


def score_url(url: str, title: str, snippet: str, page_text: Optional[str]) -> Dict:
    """Score a single candidate URL across 6 quality dimensions.

    Args:
        url: The candidate URL.
        title: Title from the search result.
        snippet: Snippet / description from the search result.
        page_text: First 5 000 chars of visible page text (may be None).

    Returns:
        Evaluation dict with score, passed flag, and breakdown.
    """
    url_lower = url.lower()
    domain = _extract_domain(url)

    # ── Hard reject: URL matches a Q&A / social / blog site ──
    for pattern in _HARD_REJECT_URL:
        if pattern in url_lower or pattern in title.lower():
            return {
                "url": url,
                "title": title,
                "platform": domain,
                "score": 0.0,
                "passed": False,
                "score_breakdown": {
                    "is_course": 0, "is_free": 0, "is_structured": 0,
                    "is_credible": 0, "freshness": 0, "penalties": 9,
                },
                "reject_reason": f"Hard reject: matched '{pattern}' in URL/title",
            }

    # ── Domain reputation shortcut ──
    # When page_text is None (fetch failed/timeout), use domain reputation
    # instead of penalising the URL unfairly.
    if page_text is None:
        baseline = _KNOWN_EDU_DOMAINS.get(domain, 0)
        if baseline > 0:
            logger.debug(f"[Critic] Page fetch failed for {domain} — using baseline {baseline}")
            return {
                "url": url,
                "title": title,
                "platform": domain,
                "score": baseline,
                "passed": baseline >= 0.30,
                "score_breakdown": {
                    "is_course": 1, "is_free": 1, "is_structured": 1,
                    "is_credible": 1, "freshness": 0, "penalties": 0,
                },
                "reject_reason": None,
            }

    # Combine all available text for signal detection
    content = " ".join(filter(None, [title, snippet, page_text or ""])).lower()

    # ── Dimension 1: Is it a course? (0-2) ──
    course_hits = _count_signals(content, _COURSE_SIGNALS)
    dim_course = 2 if course_hits >= 3 else (1 if course_hits >= 1 else 0)

    # ── Dimension 2: Is it free? (0-2) ──
    free_pos = _count_signals(content, _FREE_POSITIVE)
    free_neg = _count_signals(content, _FREE_NEGATIVE)
    if free_neg > 0:
        dim_free = 0
    elif free_pos > 0:
        dim_free = 2
    else:
        dim_free = 1  # ambiguous

    # ── Dimension 3: Structured learning? (0-2) ──
    struct_hits = _count_signals(content, _STRUCTURED_SIGNALS)
    dim_struct = 2 if struct_hits >= 2 else (1 if struct_hits >= 1 else 0)

    # ── Dimension 4: Credibility (0-2) ──
    cred_hits = _count_signals(content, _CREDIBLE_SIGNALS)
    dim_cred = 2 if cred_hits >= 2 else (1 if cred_hits >= 1 else 0)

    # ── Dimension 5: Red flags (penalties) ──
    penalties = 0
    for penalty_signal in _SOFT_PENALTY_CONTENT:
        if penalty_signal in content:
            penalties += 1

    # ── Dimension 6: Freshness (0-1) ──
    year_matches = re.findall(r"\b(202[3-9]|203\d)\b", content)
    dim_fresh = 1 if year_matches else 0

    # ── Final score ──
    raw = dim_course + dim_free + dim_struct + dim_cred + dim_fresh - penalties
    raw = max(raw, 0)
    normalised = round(raw / 9.0, 4)

    # Boost known education domains slightly
    domain_boost = _KNOWN_EDU_DOMAINS.get(domain, 0)
    if domain_boost > 0:
        normalised = round(min(1.0, normalised + 0.1), 4)

    passed = normalised >= 0.30

    return {
        "url": url,
        "title": title,
        "platform": _extract_domain(url),
        "score": normalised,
        "passed": passed,
        "score_breakdown": {
            "is_course": dim_course,
            "is_free": dim_free,
            "is_structured": dim_struct,
            "is_credible": dim_cred,
            "freshness": dim_fresh,
            "penalties": penalties,
        },
        "reject_reason": None,
    }


# ── Batch scoring (async, concurrent) ────────────────────────────────

async def score_all_async(candidates: List[Dict]) -> List[Dict]:
    """Fetch and score all candidate URLs concurrently.

    Args:
        candidates: List of dicts with at least 'url', 'title', 'snippet'.

    Returns:
        List of evaluation dicts, one per candidate, sorted by score desc.
    """
    # Fetch all pages concurrently
    async def _fetch_and_score(c: Dict) -> Dict:
        url = c.get("url", "")
        title = c.get("title", "")
        snippet = c.get("snippet", c.get("brief_description", ""))
        page_text = await _fetch_page_text(url) if url else None
        return score_url(url, title, snippet, page_text)

    results = await asyncio.gather(
        *[_fetch_and_score(c) for c in candidates],
        return_exceptions=True,
    )

    # Replace exceptions with zero-score entries
    scored: List[Dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning(f"[Critic] Scoring failed for {candidates[i].get('url', '?')}: {r}")
            scored.append({
                "url": candidates[i].get("url", ""),
                "title": candidates[i].get("title", ""),
                "platform": _extract_domain(candidates[i].get("url", "")),
                "score": 0.0,
                "passed": False,
                "score_breakdown": {
                    "is_course": 0, "is_free": 0, "is_structured": 0,
                    "is_credible": 0, "freshness": 0, "penalties": 0,
                },
                "reject_reason": f"Fetch/score error: {r}",
            })
        else:
            scored.append(r)

    # Sort best first
    scored.sort(key=lambda x: x["score"], reverse=True)

    passed_count = sum(1 for s in scored if s["passed"])
    logger.info(
        f"[Critic] Scored {len(scored)} candidates: "
        f"{passed_count} passed, {len(scored) - passed_count} rejected"
    )
    return scored


def score_all(candidates: List[Dict]) -> List[Dict]:
    """Synchronous wrapper around score_all_async.

    Detects whether an event loop is already running and handles both
    cases (called from sync code or from within an async context).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We are inside an async context — run in a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, score_all_async(candidates))
            return future.result()
    else:
        return asyncio.run(score_all_async(candidates))


# ── Legacy interface (used by learning_loop/orchestrator.py) ─────────

def run(candidates: List[Dict], student_profile: Dict, topic_info: Dict) -> Dict:
    """Score and rank course candidates.

    Preserves the original function signature so that the orchestrator
    and any other caller continue to work unchanged.

    Args:
        candidates: List of course dicts from the Scout Agent.
        student_profile: StudentProfile dict.
        topic_info: Dict with topic_id, topic_name.

    Returns:
        Dict with evaluations list and ranking.
    """
    # Try LLM-based critic first
    if is_api_key_set():
        try:
            system_prompt = get_prompt("critic")
            import json
            candidates_str = json.dumps(candidates[:10], indent=2, default=str)
            user_prompt = f"""STUDENT PROFILE:
Target Role: {student_profile.get('target_role', 'Unknown')}
Learning Style: {student_profile.get('learning_style', 'video')}
Confidence: {student_profile.get('confidence_level', 'medium')}
Budget: {student_profile.get('budget', 'free_only')}

TOPIC: {topic_info.get('topic_name', topic_info.get('topic_id', '?'))}

COURSE CANDIDATES:
{candidates_str}

Score each course across all 5 dimensions. Be critical — differentiate scores clearly."""
            result = generate_json_with_retry(system_prompt, user_prompt)
            if result and "evaluations" in result:
                logger.info(f"Critic (LLM) evaluated {len(result['evaluations'])} candidates")
                return result
        except Exception as e:
            logger.error(f"Critic LLM failed: {e}")

    # Fallback: content-based scoring via the new 6-dimension engine
    scored = score_all(candidates)

    # Convert to the evaluations format expected by the Curator
    evaluations = []
    for s in scored:
        bd = s.get("score_breakdown", {})
        evaluations.append({
            "course_id": s.get("title", "?")[:60],
            "scores": {
                "relevance": bd.get("is_course", 0) * 5,
                "difficulty_match": 7,
                "content_quality": bd.get("is_credible", 0) * 5,
                "time_efficiency": 7,
                "style_match": 7,
                "total": round(s["score"] * 10, 2),
            },
            "flags": [s["reject_reason"]] if s.get("reject_reason") else [],
            "one_line_verdict": (
                f"Score {s['score']:.2f} — "
                f"{'PASS' if s['passed'] else 'REJECT'} "
                f"({s['platform']})"
            ),
        })

    ranking = [e["course_id"] for e in evaluations]
    return {"evaluations": evaluations, "ranking": ranking}
