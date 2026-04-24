"""
EduPath AI — Resource Cache
Team KRIYA | OpenEnv Hackathon 2026

File-based JSON cache for the Scout→Critic→Curator pipeline output.
Keyed by topic name, with a 7-day TTL so stale results are automatically
refreshed.  The cache persists across server restarts.

Usage:
    cache = ResourceCache()
    result = cache.get("python basics")   # None on miss
    cache.set("python basics", courses)   # stores + writes to disk
    cache.warm(["python basics", ...])    # pre-warms multiple topics
"""
import json
import os
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
_DEFAULT_CACHE_PATH = os.path.join(_CACHE_DIR, "topic_resources.json")
_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


class ResourceCache:
    """Simple file-backed JSON cache for curated course results.

    Cache structure on disk::

        {
            "python_basics": {
                "cached_at": 1714000000.0,
                "courses": [ ... ]
            },
            ...
        }
    """

    def __init__(self, cache_path: str = _DEFAULT_CACHE_PATH):
        self._path = cache_path
        self._data: dict = {}
        self._load()

    # ── Public API ───────────────────────────────────────────────────

    def get(self, topic: str) -> Optional[List[dict]]:
        """Return cached courses for a topic, or None on miss / expiry.

        Args:
            topic: Human-readable topic name (e.g. "python basics").

        Returns:
            List of curated course dicts, or None.
        """
        key = self._normalise_key(topic)
        entry = self._data.get(key)
        if entry is None:
            return None
        if not self._is_fresh(entry):
            logger.info(f"[Cache] TTL expired for '{key}', treating as miss")
            return None
        return entry.get("courses")

    def set(self, topic: str, courses: List[dict]) -> None:
        """Store curated courses under a topic key and persist to disk.

        Args:
            topic:   Human-readable topic name.
            courses: List of curated course dicts from the Curator.
        """
        key = self._normalise_key(topic)
        self._data[key] = {
            "cached_at": time.time(),
            "courses": courses,
        }
        self._save()
        logger.info(f"[Cache] Stored {len(courses)} courses for '{key}'")

    def warm(self, topics: List[str]) -> None:
        """Pre-warm the cache for a list of topics.

        For each topic NOT already in cache (or expired), runs the full
        Scout→Critic→Curator pipeline and stores the result.

        Args:
            topics: List of topic names to warm.
        """
        total = len(topics)
        for i, topic in enumerate(topics, start=1):
            if self.get(topic) is not None:
                logger.info(f"[Cache] Warming: {i}/{total} '{topic}' -- already cached, skipping")
                print(f"  Warming cache: {i}/{total} '{topic}' -- cached [OK]")
                continue

            print(f"  Warming cache: {i}/{total} '{topic}' -- fetching...")
            try:
                courses = self._run_pipeline(topic)
                self.set(topic, courses)
                print(f"  Warming cache: {i}/{total} '{topic}' -- stored {len(courses)} courses [OK]")
            except Exception as e:
                logger.error(f"[Cache] Warming failed for '{topic}': {e}")
                print(f"  Warming cache: {i}/{total} '{topic}' -- FAILED: {e}")

    # ── Pipeline runner ──────────────────────────────────────────────

    @staticmethod
    def _run_pipeline(topic: str) -> List[dict]:
        """Execute the full Scout→Critic→Curator pipeline for a topic.

        Returns:
            List of curated course dicts.
        """
        from ai.learning_loop.scout_agent import search as scout_search
        from ai.learning_loop.critic_agent import score_all as critic_score_all
        from ai.learning_loop.curator_agent import select as curator_select

        candidates = scout_search(topic)
        scored = critic_score_all(candidates)
        curated = curator_select(scored)
        return curated

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalise_key(topic: str) -> str:
        """Normalise a topic name into a cache key.

        Lowercases, strips, replaces spaces with underscores.
        """
        return topic.strip().lower().replace(" ", "_")

    @staticmethod
    def _is_fresh(entry: dict) -> bool:
        """Check whether a cache entry is within the 7-day TTL."""
        cached_at = entry.get("cached_at", 0)
        return (time.time() - cached_at) < _TTL_SECONDS

    def _load(self) -> None:
        """Load cache data from the JSON file."""
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(f"[Cache] Loaded {len(self._data)} entries from {self._path}")
            except Exception as e:
                logger.warning(f"[Cache] Failed to load {self._path}: {e}")
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Persist cache data to the JSON file."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[Cache] Failed to save {self._path}: {e}")
