"""Verify: (1) which results are dynamic vs fallback, (2) are they free."""
import sys
import time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "backend")

import os
for f in ["backend/cache/topic_resources.json", "backend/data/resource_cache.json"]:
    if os.path.exists(f): os.remove(f)

from ai.resource_fetcher import fetch_resources_for_topic

TEST_CASES = [
    ("HTML CSS Basics", "html_css_basics"),
    ("React Framework", "react_framework"),
    ("Machine Learning", "machine_learning"),
    ("Personal Finance and Budgeting", "personal_finance"),
    ("Creative Writing", "creative_writing"),
    ("Photography Basics", "photography_basics"),
    ("Business Management", "business_management"),
    ("UX UI Design", "ux_ui_design"),
]

FALLBACK_DOMAINS = ["youtube.com", "coursera.org", "udemy.com"]
dynamic_count = 0
fallback_count = 0

for topic_name, topic_id in TEST_CASES:
    print(f"\n{'='*65}")
    print(f"TOPIC: {topic_name}")
    print(f"{'='*65}")
    
    results = fetch_resources_for_topic(topic_name, topic_id)
    
    for r in results:
        platform = r.get("platform", "?")
        title = r.get("title", "?")[:55]
        url = r.get("url", "?")[:75]
        is_free = r.get("is_free", "?")
        score = r.get("quality_score", "?")
        
        # Check if this is a hardcoded fallback or dynamically discovered
        is_fallback = any(d in platform for d in FALLBACK_DOMAINS) and score in [0.55, 0.5]
        source_type = "FALLBACK" if is_fallback else "DYNAMIC"
        
        # Check free indicators in title/url
        title_lower = title.lower()
        url_lower = url.lower()
        free_evidence = []
        if "free" in title_lower: free_evidence.append("'free' in title")
        if "free" in url_lower: free_evidence.append("'free' in URL")
        if "price=price-free" in url_lower: free_evidence.append("free filter in URL")
        if any(d in platform for d in ["youtube.com", "khanacademy.org", "freecodecamp.org",
            "w3schools.com", "geeksforgeeks.org", "javascript.info", "ocw.mit.edu"]):
            free_evidence.append("platform is free")
        if "audit" in str(r.get("description", "")).lower(): free_evidence.append("audit available")
        
        free_str = ", ".join(free_evidence) if free_evidence else "NO FREE EVIDENCE"
        
        print(f"  [{source_type}] {title}")
        print(f"    URL: {url}")
        print(f"    is_free={is_free} | score={score} | Free? {free_str}")
        
        if is_fallback:
            fallback_count += 1
        else:
            dynamic_count += 1

print(f"\n{'='*65}")
print(f"TOTALS: {dynamic_count} DYNAMIC results, {fallback_count} FALLBACK results")
print(f"{'='*65}")
