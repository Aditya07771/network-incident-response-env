"""
env_models.py
Pydantic v2 data models for the Network Incident Response Environment v2.

Key design decisions:
  - NetworkEvent has NO severity labels — the agent must investigate to infer threat level.
  - 7 action types give the agent a realistic SOC analyst toolkit.
  - Observation includes raw events AND investigation results.
  - Rewards are multi-dimensional (detection, speed, investigation, collateral).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Raw network telemetry (NO severity hints)
# ──────────────────────────────────────────────

class NetworkEvent(BaseModel):
    """A single network event as seen on the wire — no analyst labels."""

    timestamp: str
    source_ip: str
    destination_ip: str
    port: int = Field(ge=1, le=65535)
    protocol: str
    payload_snippet: str          # Raw summary of what the packet contained
    bytes_transferred: int = 0
    flags: List[str] = Field(default_factory=list)   # TCP flags, HTTP methods, etc.

    model_config = {"frozen": True}


# ──────────────────────────────────────────────
# Investigation results (returned by analysis actions)
# ──────────────────────────────────────────────

class TrafficAnalysis(BaseModel):
    """Returned by analyze_traffic(<ip>)."""
    target_ip: str
    total_connections: int
    unique_destinations: int
    unique_ports: int
    port_list: List[int]
    avg_bytes: float
    connection_rate: float            # connections per step
    protocols: List[str]

class PayloadInspection(BaseModel):
    """Returned by inspect_payload(<ip>)."""
    target_ip: str
    sample_payloads: List[str]
    entropy_score: float              # 0-1, high = encrypted/encoded data
    signatures_matched: List[str]     # e.g. "sql_injection", "base64_data"
    anomaly_flags: List[str]          # e.g. "unusual_encoding", "binary_in_http"

class ReputationReport(BaseModel):
    """Returned by check_reputation(<ip>)."""
    target_ip: str
    threat_score: float               # 0-1
    categories: List[str]             # e.g. "scanner", "c2_server", "tor_exit"
    known_campaigns: List[str]
    first_reported: Optional[str] = None

class EventCorrelation(BaseModel):
    """Returned by correlate_events(<filter>)."""
    filter_used: str
    matching_events: int
    timeline: List[str]               # chronological summary
    related_ips: List[str]
    pattern_match: Optional[str] = None   # matched attack pattern name


# ──────────────────────────────────────────────
# Actions (7 types — realistic SOC toolkit)
# ──────────────────────────────────────────────

ACTION_TYPES = Literal[
    "analyze_traffic",    # traffic flow analysis for an IP
    "inspect_payload",    # deep packet inspection
    "check_reputation",   # threat intelligence lookup
    "correlate_events",   # cross-reference events
    "block_ip",           # firewall block (irreversible)
    "isolate_host",       # disconnect internal host
    "submit_report",      # declare findings, end episode
]


class Action(BaseModel):
    """An action the SOC analyst agent can take."""

    thought: Optional[str] = Field(
        None, description="Agent's reasoning before choosing an action"
    )
    action_type: ACTION_TYPES
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: Dict[str, Any], info: Any) -> Dict[str, Any]:
        action_type = info.data.get("action_type")
        if action_type == "block_ip" and "ip" not in v:
            raise ValueError("block_ip requires 'ip' in parameters")
        if action_type == "isolate_host" and "ip" not in v:
            raise ValueError("isolate_host requires 'ip' in parameters")
        if action_type == "analyze_traffic" and "ip" not in v:
            raise ValueError("analyze_traffic requires 'ip' in parameters")
        if action_type == "inspect_payload" and "ip" not in v:
            raise ValueError("inspect_payload requires 'ip' in parameters")
        if action_type == "check_reputation" and "ip" not in v:
            raise ValueError("check_reputation requires 'ip' in parameters")
        if action_type == "correlate_events" and "filter" not in v:
            raise ValueError("correlate_events requires 'filter' in parameters")
        if action_type == "submit_report" and "attack_type" not in v:
            raise ValueError("submit_report requires 'attack_type' in parameters")
        return v


# ──────────────────────────────────────────────
# Observation (raw events + investigation results)
# ──────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees each step — raw events plus any investigation output."""

    network_events: List[NetworkEvent] = Field(default_factory=list)
    analysis_result: Optional[Dict[str, Any]] = None    # last investigation output
    blocked_ips: List[str] = Field(default_factory=list)
    isolated_hosts: List[str] = Field(default_factory=list)
    time_elapsed: int = Field(default=0, ge=0)
    time_remaining: int = Field(default=30, ge=0)
    alert_summary: str = ""      # vague IDS alert — direction, not answer

    @field_validator("network_events")
    @classmethod
    def cap_events(cls, v: List[NetworkEvent]) -> List[NetworkEvent]:
        return v[-50:] if len(v) > 50 else v


# ──────────────────────────────────────────────
# Reward (multi-dimensional)
# ──────────────────────────────────────────────

class Reward(BaseModel):
    """Step reward with breakdown."""

    score: float = Field(ge=-1.0, le=1.0)
    threat_stopped: bool = False
    collateral_damage: bool = False
    investigation_credit: bool = False
    message: str = ""

    model_config = {"frozen": True}


# ──────────────────────────────────────────────
# Episode summary (for graders)
# ──────────────────────────────────────────────

class EpisodeSummary(BaseModel):
    """Full episode report — consumed by graders."""

    task_id: str
    ground_truth_attack_type: str
    ground_truth_attacker_ips: List[str]
    ground_truth_compromised_hosts: List[str]

    # Agent's actions
    blocked_ips: List[str]
    isolated_hosts: List[str]
    agent_attack_type: Optional[str] = None   # from submit_report
    steps_taken: int
    max_steps: int

    # Quality metrics
    investigation_actions: int         # how many investigate actions agent used
    unique_ips_investigated: int       # distinct IPs the agent examined
    threat_neutralized: bool
    collateral_blocks: int             # wrong IPs blocked
    collateral_isolations: int         # wrong hosts isolated
    attack_progression: float          # 0-1, how far the attack got
    total_reward: float
