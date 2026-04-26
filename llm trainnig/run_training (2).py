"""
EduPath AI — HuggingFace Training Runner
Starts HTTP server immediately (satisfies 30min health check),
then runs GRPO training in background thread.
"""
import os, sys, json, threading, http.server, time

WORK = '/app/output'
os.makedirs(WORK, exist_ok=True)
os.environ['WORK'] = WORK

# Step 1: Write initial status immediately
initial_status = {'status': 'initializing', 'step': 0, 'total': 0, 'reward': None}
with open(f'{WORK}/status.json', 'w') as f:
    json.dump(initial_status, f)

# Step 2: Write live HTML status page
html = """<!DOCTYPE html>
<html>
<head>
    <title>EduPath AI — Live Training</title>
    <meta http-equiv="refresh" content="30">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(26, 31, 46, 0.6);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.5);
            --text-main: #f3f4f6;
            --text-dim: #9ca3af;
            --success: #10b981;
            --error: #ef4444;
        }
        body { 
            font-family: 'Outfit', sans-serif; 
            background: var(--bg-color); 
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-main); 
            padding: 3rem 2rem; 
            max-width: 1000px; 
            margin: 0 auto; 
            min-height: 100vh;
        }
        .header { text-align: center; margin-bottom: 3rem; }
        h1 { 
            font-size: 2.5rem; 
            font-weight: 800; 
            background: linear-gradient(to right, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle { color: var(--text-dim); font-size: 1.1rem; }
        
        .glass-card { 
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            padding: 2rem; 
            border-radius: 16px; 
            margin-bottom: 2rem; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.3); }
        
        h2 { margin-top: 0; color: #fff; display: flex; align-items: center; gap: 0.75rem; font-size: 1.5rem; }
        .status-dot {
            width: 12px; height: 12px; border-radius: 50%; display: inline-block;
        }
        .status-pulse {
            box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
            animation: pulse 2s infinite; background: var(--accent);
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
        }

        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }
        .stat-box { 
            background: rgba(0,0,0,0.2); border-radius: 12px; padding: 1.5rem; text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .stat-label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); margin-bottom: 0.5rem; }
        .stat-value { font-size: 2.5rem; font-weight: 800; color: #fff; font-family: 'JetBrains Mono', monospace; }
        .reward-positive { color: var(--success); text-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }
        .reward-negative { color: var(--error); text-shadow: 0 0 10px rgba(239, 68, 68, 0.4); }
        
        .progress-wrapper { margin-top: 2rem; }
        .progress-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; color: var(--text-dim); font-size: 0.9rem; }
        .progress-bar { 
            background: rgba(0,0,0,0.4); height: 12px; border-radius: 6px; overflow: hidden; 
            border: 1px solid rgba(255,255,255,0.05);
        }
        .progress-fill { 
            background: linear-gradient(90deg, #3b82f6, #8b5cf6); 
            height: 100%; width: 0%; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 10px var(--accent-glow);
        }
        
        .terminal { 
            background: #0f131a; padding: 1.5rem; border-radius: 12px; overflow-x: auto; 
            border: 1px solid #1f2937; box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
        }
        pre { font-family: 'JetBrains Mono', monospace; color: #a78bfa; font-size: 0.95em; margin: 0; line-height: 1.5; }
        
        img { max-width: 100%; border-radius: 12px; margin-top: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.4); border: 1px solid var(--card-border); }
        .error-msg { color: var(--error); background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin-top: 1.5rem;}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎓 EduPath AI</h1>
        <div class="subtitle">Multi-Agent GRPO Pipeline Tracker</div>
    </div>
    
    <div class="glass-card" id="status-card">
        <h2 id="status-title"><div class="status-dot status-pulse" id="pulse-dot"></div> Initializing Environment...</h2>
        
        <div id="stats-container" style="display:none;">
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Global Step</div>
                    <div class="stat-value" id="step-text">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Mean Reward</div>
                    <div class="stat-value" id="reward-text">--</div>
                </div>
            </div>
            
            <div class="progress-wrapper">
                <div class="progress-header">
                    <span>Training Progress</span>
                    <span id="progress-pct">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
            </div>
        </div>
        <div id="error-container" style="display:none;" class="error-msg"></div>
    </div>

    <div class="glass-card" id="logs-card" style="display:none;">
        <h2 style="font-size: 1.2rem; margin-bottom: 1rem;">Terminal Output</h2>
        <div class="terminal">
            <pre id="log-text">Waiting for logs...</pre>
        </div>
    </div>

    <div class="glass-card" id="results-card" style="display:none;">
        <h2>Training Complete! 🎉</h2>
        <div class="terminal" style="margin-top: 1.5rem;">
            <pre id="final-stats"></pre>
        </div>
        <img id="results-img" src="" style="display:none;">
    </div>

    <script>
        async function fetchStatus() {
            try {
                const res = await fetch('status.json?t=' + Date.now());
                if (!res.ok) return;
                const data = await res.json();
                
                const title = document.getElementById('status-title');
                const pulse = document.getElementById('pulse-dot');
                const stats = document.getElementById('stats-container');
                const progFill = document.getElementById('progress-fill');
                const errCon = document.getElementById('error-container');
                const logsCard = document.getElementById('logs-card');
                const resCard = document.getElementById('results-card');
                
                errCon.style.display = 'none';

                if (data.status === 'initializing') {
                    title.innerHTML = '<div class="status-dot status-pulse"></div> Initializing Environment...';
                } else if (data.status === 'training' || data.status === 'evaluating') {
                    const isTrain = data.status === 'training';
                    title.innerHTML = `<div class="status-dot status-pulse" style="background: ${isTrain ? '#3b82f6' : '#8b5cf6'}"></div> ${isTrain ? 'Training in Progress' : 'Evaluating Model'}`;
                    stats.style.display = 'block';
                    logsCard.style.display = 'block';
                    
                    document.getElementById('step-text').innerText = `${data.step} / ${data.total}`;
                    const pct = data.total > 0 ? Math.min(100, (data.step / data.total) * 100) : 0;
                    progFill.style.width = `${pct}%`;
                    document.getElementById('progress-pct').innerText = `${pct.toFixed(1)}%`;
                    
                    if (data.reward !== null && data.reward !== undefined) {
                        const rText = document.getElementById('reward-text');
                        rText.innerText = data.reward > 0 ? `+${data.reward.toFixed(3)}` : data.reward.toFixed(3);
                        rText.className = 'stat-value ' + (data.reward > 0 ? 'reward-positive' : 'reward-negative');
                    }
                    
                    if (data.log_history && data.log_history.length > 0) {
                        document.getElementById('log-text').innerText = data.log_history.join('\\n');
                    }
                } else if (data.status === 'complete') {
                    title.innerHTML = '<div class="status-dot" style="background: var(--success)"></div> Training Complete!';
                    stats.style.display = 'none';
                    logsCard.style.display = 'none';
                    resCard.style.display = 'block';
                    
                    document.getElementById('final-stats').innerText = JSON.stringify(data.per_task, null, 2);
                    const img = document.getElementById('results-img');
                    img.src = 'grpo_training_results.png?t=' + Date.now();
                    img.style.display = 'block';
                } else if (data.status === 'error') {
                    title.innerHTML = '<div class="status-dot" style="background: var(--error)"></div> Training Failed';
                    errCon.style.display = 'block';
                    errCon.innerText = data.error + "\\n\\n" + (data.traceback || '');
                }
            } catch (e) {
                console.error("Status fetch failed", e);
            }
        }
        setInterval(fetchStatus, 3000); // Poll every 3s
        fetchStatus();
    </script>
</body>
</html>
"""
with open(f'{WORK}/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

# Step 3: Start HTTP server in daemon thread
import functools
def run_server():
    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WORK)
    server = http.server.HTTPServer(('0.0.0.0', 7860), Handler)
    print("🚀 HTTP Server started on port 7860 (Daemon thread)")
    server.serve_forever()

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Step 4: Run part1 + part2 in background thread
def run_training_pipeline():
    try:
        print("\n[1/2] Running setup + reward functions...")
        env_globals = globals().copy()
        exec(open('/app/train_multitask_part1.py').read(), env_globals)
        
        print("\n[2/2] Running training pipeline...")
        exec(open('/app/train_multitask_part2.py').read(), env_globals)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"PIPELINE CRASHED: {e}\n{err_msg}")
        with open(f'{WORK}/status.json', 'w') as f:
            json.dump({'status':'error', 'error': str(e), 'traceback': err_msg}, f)

train_thread = threading.Thread(target=run_training_pipeline)
train_thread.start()

# Step 5: Main thread joins background thread (keeps alive)
train_thread.join()

print("\nPipeline thread finished. Keeping HTTP server alive.")
while True:
    time.sleep(3600)
