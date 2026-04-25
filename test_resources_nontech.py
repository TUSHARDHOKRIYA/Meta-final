"""Thorough test of the resource pipeline across multiple non-tech topics."""
import sys
import time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "backend")

# Clear ALL caches first
import os, json
cache_files = [
    "backend/cache/topic_resources.json",
    "backend/data/resource_cache.json",
]
for f in cache_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Cleared: {f}")

from ai.resource_fetcher import fetch_resources_for_topic

# Test cases: (topic_name, topic_id) across different non-tech domains
TEST_CASES = [
    ("Graphic Design Fundamentals", "graphic_design_fundamentals"),
    ("Personal Finance", "personal_finance"),
    ("Creative Writing", "creative_writing"),
    ("Digital Marketing", "digital_marketing"),
    ("Public Speaking", "public_speaking"),
    ("Photography Basics", "photography_basics"),
]

total_pass = 0
total_fail = 0

for topic_name, topic_id in TEST_CASES:
    print(f"\n{'='*60}")
    print(f"TESTING: {topic_name} ({topic_id})")
    print(f"{'='*60}")
    
    try:
        start = time.time()
        results = fetch_resources_for_topic(topic_name, topic_id)
        elapsed = time.time() - start
        
        # Check quality
        has_google = any("google.com/search" in r.get("url", "") for r in results)
        real_courses = [r for r in results if "google.com/search" not in r.get("url", "")]
        
        print(f"  Results: {len(results)} total, {len(real_courses)} real courses")
        print(f"  Time: {elapsed:.1f}s")
        
        for r in results:
            src = r.get("source", r.get("platform", "?"))
            title = r.get("title", "?")[:55]
            url = r.get("url", "?")[:70]
            score = r.get("quality_score", "?")
            is_fallback = r.get("is_fallback", False)
            marker = " [FALLBACK]" if is_fallback or "google.com/search" in url else ""
            print(f"    [{src}] {title}")
            print(f"      URL: {url}")
            print(f"      Score: {score}{marker}")
        
        if has_google and len(results) <= 1:
            print(f"  RESULT: FAIL - Only Google search fallback")
            total_fail += 1
        elif len(real_courses) >= 1:
            print(f"  RESULT: PASS - {len(real_courses)} real course(s) found")
            total_pass += 1
        else:
            print(f"  RESULT: WARN - No real courses")
            total_fail += 1
            
    except Exception as e:
        print(f"  RESULT: ERROR - {e}")
        total_fail += 1

print(f"\n{'='*60}")
print(f"SUMMARY: {total_pass} PASS, {total_fail} FAIL out of {len(TEST_CASES)}")
print(f"{'='*60}")
