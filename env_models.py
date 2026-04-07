"""
env_models.py
Pydantic v2 data models for the Network Incident Response Environment.
All fields are strictly typed and validated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Core domain models
# ──────────────────────────────────────────────

class LogEntry(BaseModel):
    """A single line from the SIEM / log stream."""

    timestamp: str
    source_ip: str
    destination_ip: str
    port: int = Field(ge=1, le=65535)
    protocol: str
    message: str
    severity: Literal["INFO", "WARNING", "CRITICAL"]

    model_config = {"frozen": True}


class Observation(BaseModel):
    """What the agent sees at each step (last ≤40 logs)."""

    recent_logs: List[LogEntry] = Field(default_factory=list)
    query_results: List[LogEntry] = Field(default_factory=list)
    blocked_ips: List[str] = Field(default_factory=list)
    time_elapsed: int = Field(default=0, ge=0)

    @field_validator("recent_logs")
    @classmethod
    def cap_logs(cls, v: List[LogEntry]) -> List[LogEntry]:
        # Always give the agent at most 40 entries
        return v[-40:] if len(v) > 40 else v

    @field_validator("query_results")
    @classmethod
    def cap_query_results(cls, v: List[LogEntry]) -> List[LogEntry]:
        return v[:10]


class Action(BaseModel):
    """An action the agent can submit."""

    thought: Optional[str] = Field(
        None, description="The agent's internal reasoning before choosing an action"
    )
    action_type: Literal["query_logs", "block_ip", "no_op"]
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: Dict[str, Any], info: Any) -> Dict[str, Any]:
        # Pydantic v2: access sibling field via info.data
        action_type = info.data.get("action_type")
        if action_type == "block_ip" and "ip" not in v:
            raise ValueError("block_ip action requires 'ip' key in parameters")
        if action_type == "query_logs" and "filter" not in v:
            raise ValueError("query_logs action requires 'filter' key in parameters")
        return v


class Reward(BaseModel):
    """Outcome of a single env step."""

    score: float = Field(ge=-1.0, le=1.0)
    threat_stopped: bool = False
    collateral_damage: bool = False
    message: str = ""

    model_config = {"frozen": True}


class EpisodeSummary(BaseModel):
    """Full episode result (returned by run helpers & used by graders)."""

    task_id: str
    attacker_ip: str
    blocked_ips: List[str]
    steps_taken: int
    threat_neutralized: bool
    collateral_damage: bool
    total_reward: float
    normalized_score: float   # total_reward / max_steps, clipped [0, 1]
