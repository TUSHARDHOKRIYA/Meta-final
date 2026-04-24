"""
EduPath AI — Cache Warmer
Team KRIYA | OpenEnv Hackathon 2026

Pre-warms the resource cache for demo topics so that the first user
interaction is instant.  Run this script before a demo or deployment:

    cd meta-hacka
    python -m backend.warm_cache

Or from the backend directory:

    python warm_cache.py
"""
import sys
import os
import logging

# Ensure backend/ is on sys.path so imports work when run as a script
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Demo topics to pre-warm ──────────────────────────────────────────
# These are the topics most likely to be shown during a hackathon demo.
DEMO_TOPICS = [
    "python basics",
    "machine learning fundamentals",
    "neural networks",
    "transformer architecture",
    "data structures and algorithms",
    "sql and databases",
]


def main():
    """Pre-warm the resource cache for all demo topics."""
    from cache.resource_cache import ResourceCache

    print("=" * 60)
    print("EduPath AI -- Cache Warmer")
    print("=" * 60)
    print(f"Topics to warm: {len(DEMO_TOPICS)}")
    print()

    cache = ResourceCache()
    cache.warm(DEMO_TOPICS)

    print()
    print("=" * 60)
    print("Cache warmed. Demo is ready. [DONE]")
    print("=" * 60)


if __name__ == "__main__":
    main()
