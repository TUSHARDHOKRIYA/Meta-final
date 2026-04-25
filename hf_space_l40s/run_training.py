"""
EduPath AI — HuggingFace Training Runner
Runs part1 (setup+rewards) then part2 (training), then serves results on port 7860.
"""
import subprocess, sys, os, http.server, threading

WORK = '/app/output'
os.makedirs(WORK, exist_ok=True)
os.environ['WORK'] = WORK

print("=" * 60)
print("  EduPath AI — GRPO Training on HuggingFace")
print("=" * 60)

# Run part1 (setup + reward functions)
print("\n[1/2] Running setup + reward functions...")
exec(open('/app/train_multitask_part1.py').read())

# Run part2 (dataset + model + training + eval + plots)
print("\n[2/2] Running training pipeline...")
exec(open('/app/train_multitask_part2.py').read())

print("\n✅ Training complete! Serving results on port 7860...")

# Create results HTML
html = f"""<!DOCTYPE html><html><head><title>EduPath GRPO Results</title>
<style>body{{font-family:sans-serif;background:#0a0e17;color:#e5e7eb;padding:2rem;max-width:900px;margin:0 auto}}
h1{{color:#3b82f6}}img{{max-width:100%;border-radius:8px}}pre{{background:#1a1f2e;padding:1rem;border-radius:8px;overflow-x:auto}}</style></head>
<body><h1>🎓 EduPath AI — GRPO Training Results</h1>
<p>Training completed successfully on HuggingFace A10G GPU.</p>
<h2>Training Plots</h2>"""

plot_path = f'{WORK}/grpo_training_results.png'
json_path = f'{WORK}/grpo_results.json'
if os.path.exists(plot_path):
    import base64
    with open(plot_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    html += f'<img src="data:image/png;base64,{img_b64}" alt="Training Results">'

if os.path.exists(json_path):
    with open(json_path) as f:
        html += f'<h2>Results JSON</h2><pre>{f.read()}</pre>'

html += '</body></html>'
with open(f'{WORK}/index.html', 'w') as f:
    f.write(html)

# Serve results
os.chdir(WORK)
server = http.server.HTTPServer(('0.0.0.0', 7860), http.server.SimpleHTTPRequestHandler)
server.serve_forever()
