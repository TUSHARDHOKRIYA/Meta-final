import os
import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # We want to replace all bounding fallbacks with exactly what the user requested: 0.001 and 0.999
    # Replace 0.0001 -> 0.001
    new_content = content.replace('0.0001', '0.001')
    # Replace 0.9999 -> 0.999
    new_content = new_content.replace('0.9999', '0.999')
    
    # Replace exact 0 integer assignments to score
    new_content = re.sub(r'\bscore\s*=\s*0\b', 'score = 0.001', new_content)
    new_content = re.sub(r'\bbest_score\s*=\s*0\b', 'best_score = 0.001', new_content)
    new_content = re.sub(r'\bquiz_score\s*=\s*0\b', 'quiz_score = 0.001', new_content)
    new_content = re.sub(r'\bget\(\"score\",\s*0\)', 'get("score", 0.001)', new_content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

workspace_dir = r'c:\Users\LOQ\Desktop\Meta HAckathon\EduPath-AI'
for root, dirs, files in os.walk(workspace_dir):
    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.venv', 'node_modules']]
    for file in files:
        if file.endswith('.py') and file != 'fix_bounds.py':
            process_file(os.path.join(root, file))
