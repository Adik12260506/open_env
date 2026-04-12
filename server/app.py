# server/app.py  — standalone FastAPI, no openenv dependency
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import EmailAction, EmailObservation
from server.env import EmailEnv

app = FastAPI(title="EmailEnv", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# One shared env instance per container
_env = EmailEnv()


class _Action(BaseModel):
    action_type: str = "classify"
    content: Optional[str] = None

class _StepReq(BaseModel):
    action: _Action


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


# ── Serve web UI ──────────────────────────────────────────────────────────────
_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

@app.get("/web", include_in_schema=False)
@app.get("/web/", include_in_schema=False)
def web_index():
    return FileResponse(os.path.join(_WEB, "index.html"))

@app.get("/web/{filename}", include_in_schema=False)
def web_file(filename: str):
    path = os.path.join(_WEB, filename)
    if os.path.isfile(path):
        return FileResponse(path)
    return FileResponse(os.path.join(_WEB, "index.html"))

@app.get("/")
def root():
    return FileResponse(os.path.join(_WEB, "index.html"))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()