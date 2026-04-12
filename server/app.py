# server/app.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from models import EmailAction, EmailObservation
from server.env import EmailEnv

app = FastAPI(title="EmailEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_env = EmailEnv()


class _Action(BaseModel):
    action_type: str = "classify"
    content: Optional[str] = None

class _StepReq(BaseModel):
    action: _Action


# ── Core env endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "email_env"}


@app.post("/reset")
def reset():
    obs = _env.reset()
    return {"observation": obs.model_dump(), "reward": 0.0, "done": False}


@app.post("/step")
def step(req: _StepReq):
    action = EmailAction(action_type=req.action.action_type, content=req.action.content)
    obs = _env.step(action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.get("/state")
def state():
    s = _env.state
    return {"episode_id": s.episode_id, "step_count": s.step_count}


# ── Tasks endpoint — required by OpenEnv validator ───────────────────────────

TASKS: List[Dict[str, Any]] = [
    {
        "name": "email_classification",
        "description": "Classify emails as spam or important",
        "grader": "server.graders:grade_email_classification",
        "enabled": True,
    },
    {
        "name": "spam_detection",
        "description": "Detect and label spam emails",
        "grader": "server.graders:grade_spam_detection",
        "enabled": True,
    },
    {
        "name": "email_summarization",
        "description": "Summarize email content accurately",
        "grader": "server.graders:grade_email_summarization",
        "enabled": True,
    },
]

@app.get("/tasks")
def list_tasks():
    """OpenEnv validator calls this to discover registered tasks with graders."""
    return {"tasks": TASKS, "count": len(TASKS)}


@app.post("/grade/{task_name}")
def grade_task(task_name: str, payload: Dict[str, Any]):
    """
    OpenEnv validator may call this to grade a specific task.
    payload: {"prediction": str, "email": {...}}
    """
    from server.graders import (
        grade_email_classification,
        grade_spam_detection,
        grade_email_summarization,
    )
    prediction = payload.get("prediction", payload.get("content", ""))
    email      = payload.get("email", {})

    graders = {
        "email_classification": grade_email_classification,
        "spam_detection":       grade_spam_detection,
        "email_summarization":  grade_email_summarization,
    }

    if task_name not in graders:
        return {"error": f"unknown task: {task_name}"}, 404

    score = graders[task_name](prediction, email)
    return {"task": task_name, "score": score, "passed": score >= 0.85}


# ── Serve web UI ──────────────────────────────────────────────────────────────

_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

@app.get("/web", include_in_schema=False)
@app.get("/web/", include_in_schema=False)
def web_index():
    return FileResponse(os.path.join(_WEB, "index.html"))

@app.get("/")
def root():
    return FileResponse(os.path.join(_WEB, "index.html"))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()