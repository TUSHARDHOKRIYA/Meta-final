"""
EduPath AI — Analytics Dashboard Server
Team KRIYA | Meta Hackathon 2026

Lightweight Flask server that serves the interactive analytics
dashboard (dashboard/index.html) for visualising training results,
ablation studies, and agent comparison charts.
"""
import os
import sys
import json
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EduPath AI Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard index.html not found</h1>"


@app.get("/api/metrics")
async def get_metrics():
    """Get all available metrics from results directory."""
    metrics = {
        "learning_curves": {},
        "evaluation_results": {},
        "ablation_results": None,
        "hrl_results": {},
        "reflexion_memory": {},
    }

    if not os.path.exists(RESULTS_DIR):
        return metrics

    for filename in os.listdir(RESULTS_DIR):
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            if filename.startswith("learning_curve"):
                metrics["learning_curves"][filename] = data
            elif filename.startswith("eval"):
                metrics["evaluation_results"][filename] = data
            elif filename.startswith("ablation"):
                metrics["ablation_results"] = data
            elif filename.startswith("hrl_"):
                metrics["hrl_results"][filename] = data
            elif filename.startswith("reflexion_"):
                metrics["reflexion_memory"][filename] = data
        except Exception:
            continue

    return metrics


@app.get("/api/graph")
async def get_curriculum_graph():
    """Get the curriculum prerequisite graph for visualization."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))
    try:
        from environment.curriculum import TOPIC_GRAPH
        nodes = []
        links = []
        for topic_id, topic in TOPIC_GRAPH.items():
            nodes.append({
                "id": topic_id,
                "name": topic.name,
                "field": topic.field,
                "difficulty": topic.difficulty,
            })
            for prereq in topic.prerequisites:
                links.append({"source": prereq, "target": topic_id})
        return {"nodes": nodes, "links": links}
    except Exception as e:
        return {"error": str(e), "nodes": [], "links": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7861, reload=True)
