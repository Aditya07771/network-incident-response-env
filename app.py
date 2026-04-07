"""
app.py
FastAPI server — exposes the environment over HTTP for the OpenEnv harness
and for manual exploration via the /docs Swagger UI.

Endpoints:
  POST /reset       → resets env, returns first Observation
  POST /step        → sends an Action, returns Observation + Reward + done + info
  GET  /state       → lightweight current-state snapshot
  GET  /tasks       → list available task IDs
  GET  /health      → liveness probe
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from pydantic import BaseModel

from env_models import Action, Observation, Reward
from network_incident_env import NetworkIncidentEnv, SCENARIO_REGISTRY


# ──────────────────────────────────────────────
# App init
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load a default environment so /state is always available."""
    app.state.env = NetworkIncidentEnv(task_id="ssh_bruteforce")
    yield


app = FastAPI(
    title="Network Incident Response Environment",
    description=(
        "OpenEnv-compatible SOC analyst training environment. "
        "Agents detect and respond to network attacks by analysing log streams."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS (needed for HF Spaces iframe / external callers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def get_env() -> NetworkIncidentEnv:
    return app.state.env   # type: ignore[attr-defined]


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {"id": k, "difficulty": cls.__name__} for k, cls in SCENARIO_REGISTRY.items()
        ]
    }


class ResetRequest(BaseModel):
    """Request body for /reset."""

    task_id: str = "ssh_bruteforce"


@app.post("/reset", response_model=Observation)
def reset(payload: Optional[ResetRequest] = None, task_id: Optional[str] = None) -> Observation:
    """
    Reset the environment for a given task.

    Accepts either:
    - JSON body: {"task_id": "..."}
    - Query param: ?task_id=...
    """
    requested_task = (
        payload.task_id
        if payload is not None and payload.task_id
        else task_id or "ssh_bruteforce"
    )
    if requested_task not in SCENARIO_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown task_id '{requested_task}'. "
                f"Valid: {list(SCENARIO_REGISTRY.keys())}"
            ),
        )
    # Reinitialise for the requested task
    app.state.env = NetworkIncidentEnv(task_id=requested_task)
    return app.state.env.reset()


@app.post("/step")
def step(action: Action) -> Dict[str, Any]:
    """
    Submit an action and advance the environment by one step.

    Body example:
    ```json
    {"action_type": "block_ip", "parameters": {"ip": "203.0.113.42"}}
    ```
    """
    env = get_env()
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return the current internal state (lightweight snapshot)."""
    return get_env().state()


@app.get("/summary")
def summary() -> Dict[str, Any]:
    """Return the full episode summary + graded score (call after done=True)."""
    from graders import grade
    env = get_env()
    try:
        ep = env.episode_summary()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    score = grade(ep)
    return {**ep.model_dump(), "graded_score": score}


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
