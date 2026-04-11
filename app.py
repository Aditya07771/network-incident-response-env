"""
app.py
FastAPI server for the Network Incident Response Environment v2.

Endpoints (OpenEnv standard):
  POST /reset          Start a new episode
  POST /step           Execute an action
  GET  /state          Current environment state
  GET  /tasks          Available tasks
  GET  /health         Health check
  GET  /summary        Episode summary with graded score
  GET  /grade_breakdown  Detailed scoring breakdown (debug)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from env_models import Action, Observation
from graders import grade, grade_breakdown
from network_incident_env import SCENARIO_REGISTRY, NetworkIncidentEnv


# ──────────────────────────────────────────────
# Application state
# ──────────────────────────────────────────────

_env: NetworkIncidentEnv | None = None
_last_done: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-create a default environment on startup."""
    global _env
    _env = NetworkIncidentEnv(task_id="ssh_bruteforce")
    yield
    if _env:
        _env.close()


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

app = FastAPI(
    title="Network Incident Response Environment",
    description=(
        "An OpenEnv-compliant RL environment for training SOC analyst agents. "
        "The agent investigates network incidents using a realistic toolkit "
        "(traffic analysis, payload inspection, threat intel, event correlation) "
        "before taking containment actions (block, isolate) and submitting a report."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "network-incident-response-env", "version": "2.0.0"}


@app.get("/tasks")
async def tasks():
    """List available tasks."""
    return {
        "tasks": [
            {"id": tid, "attack_type": cls.ATTACK_TYPE}
            for tid, cls in SCENARIO_REGISTRY.items()
        ]
    }


@app.post("/reset")
async def reset(request: Request):
    """Reset the environment. Optionally accepts {"task_id": "..."} in the body."""
    global _env, _last_done

    body: Dict[str, Any] = {}
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw)
    except Exception:
        pass

    task_id = body.get("task_id", "ssh_bruteforce")
    seed = body.get("seed", 0)

    if task_id not in SCENARIO_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. "
                   f"Valid: {list(SCENARIO_REGISTRY.keys())}",
        )

    _env = NetworkIncidentEnv(task_id=task_id, seed=seed)
    obs: Observation = _env.reset()
    _last_done = False

    return {"observation": obs.model_dump(), "info": {}}


@app.post("/step")
async def step(action: Action):
    """Execute one action in the environment."""
    global _last_done

    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    if _last_done:
        raise HTTPException(status_code=400, detail="Episode is done — call /reset")

    obs, reward, done, info = _env.step(action)
    _last_done = done

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state():
    """Return the current environment state."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return _env.state()


@app.get("/summary")
async def summary():
    """Return the episode summary with graded score."""
    if _env is None:
        raise HTTPException(status_code=400, detail="No episode has been run")
    try:
        s = _env.episode_summary()
        return {
            "summary": s.model_dump(),
            "graded_score": grade(s),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/grade_breakdown")
async def breakdown():
    """Detailed scoring breakdown (debug endpoint)."""
    if _env is None:
        raise HTTPException(status_code=400, detail="No episode has been run")
    try:
        s = _env.episode_summary()
        return grade_breakdown(s)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)