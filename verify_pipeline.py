"""Quick verification of the Scout - Critic - Curator pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 60)
print("EduPath AI - Pipeline Verification")
print("=" * 60)

# --- Test Scout ---
print("\n[1/3] Testing Scout Agent...")
from ai.learning_loop.scout_agent import search
candidates = search("python basics")
print(f"  Scout found {len(candidates)} candidates")
for r in candidates[:5]:
    title = r.get("title", "?")[:65]
    url = r.get("url", "?")[:55]
    print(f"    - {title}")
    print(f"      {url}")

# --- Test Critic ---
print(f"\n[2/3] Testing Critic Agent on {len(candidates)} candidates...")
from ai.learning_loop.critic_agent import score_all
scored = score_all(candidates)
passed = [s for s in scored if s.get("passed")]
rejected = [s for s in scored if not s.get("passed")]
print(f"  Critic: {len(passed)} passed, {len(rejected)} rejected")
for s in scored[:8]:
    status = "PASS" if s["passed"] else "REJECT"
    reason = f" ({s['reject_reason'][:40]})" if s.get("reject_reason") else ""
    print(f"    [{s['score']:.2f}] {status} {s['platform'][:20]} - {s['title'][:45]}{reason}")

# --- Test Curator ---
print(f"\n[3/3] Testing Curator Agent...")
from ai.learning_loop.curator_agent import select
curated = select(scored)
print(f"  Curator selected {len(curated)} courses:")
for c in curated:
    print(f"    #{c['rank']} [{c['quality_score']:.2f}] {c['platform'][:20]} - {c['title'][:50]}")
    print(f"       URL: {c['url'][:70]}")
    print(f"       Why: {c['why_selected']}")

# --- Summary ---
print("\n" + "=" * 60)
avg_score = sum(c["quality_score"] for c in curated) / max(len(curated), 1)
print(f"Results: {len(curated)} courses, avg quality = {avg_score:.2f}")
if len(curated) >= 3 and avg_score >= 0.45:
    print("Status: PASS")
elif len(curated) >= 1:
    print("Status: PARTIAL (fewer than 3 or low avg score)")
else:
    print("Status: FAIL")
print("=" * 60)
