"""
graders.py
Continuous multi-dimensional grading for the Network Incident Response Env.

Scoring dimensions (weighted):
  - Threat neutralization    40%   Did the agent stop the attack?
  - Speed                    25%   How quickly was it contained?
  - Investigation quality    20%   Did the agent gather evidence first?
  - Collateral avoidance     15%   Were false positives minimised?

All final scores are clamped to the open interval (0.0001, 0.9999)
to satisfy the OpenEnv requirement of strictly (0, 1).
"""

from __future__ import annotations

from env_models import EpisodeSummary


# ──────────────────────────────────────────────
# Strict open-interval clamping
# ──────────────────────────────────────────────

_EPS = 0.0001

def _clamp(x: float) -> float:
    """Clamp x to (_EPS, 1 - _EPS) — strictly inside (0, 1)."""
    return max(_EPS, min(1.0 - _EPS, x))


# ──────────────────────────────────────────────
# Dimension scorers
# ──────────────────────────────────────────────

def _threat_neutralization_score(s: EpisodeSummary) -> float:
    """
    Did the agent stop the attack AND correctly identify it?

    1.0 if threat neutralized AND attack type correct
    0.8 if threat neutralized but attack type wrong/missing
    0.3 if attack type identified but threat not neutralized
    0.0 if neither
    """
    neutralized = s.threat_neutralized
    type_correct = (
        s.agent_attack_type is not None
        and s.agent_attack_type.lower() == s.ground_truth_attack_type.lower()
    )

    if neutralized and type_correct:
        return 1.0
    if neutralized:
        return 0.8
    if type_correct:
        return 0.3
    return 0.0


def _speed_score(s: EpisodeSummary) -> float:
    """
    How quickly was the threat neutralized?

    Scored as (max_steps - steps_taken) / max_steps if neutralized.
    0.1 base if not neutralized (agent used all steps but failed).
    """
    if not s.threat_neutralized:
        return 0.1

    ratio = (s.max_steps - s.steps_taken) / max(1, s.max_steps)
    return max(0.1, ratio)


def _investigation_score(s: EpisodeSummary) -> float:
    """
    Did the agent investigate before acting?

    Rewards:
      - 0.3 per unique IP investigated, up to 3 IPs (max 0.9)
      - 0.1 base for any investigation
      - Capped at 1.0
    """
    if s.investigation_actions == 0:
        return 0.0

    ip_bonus = min(3, s.unique_ips_investigated) * 0.3
    return min(1.0, 0.1 + ip_bonus)


def _collateral_score(s: EpisodeSummary) -> float:
    """
    Were false positives avoided?

    1.0 if zero collateral
    Deduct 0.25 per wrong block/isolation, floor at 0.0
    """
    total_collateral = s.collateral_blocks + s.collateral_isolations
    return max(0.0, 1.0 - total_collateral * 0.25)


# ──────────────────────────────────────────────
# Task-specific weight profiles
# ──────────────────────────────────────────────

WEIGHT_PROFILES = {
    "ssh_bruteforce": {
        "neutralization": 0.45,
        "speed": 0.30,
        "investigation": 0.10,
        "collateral": 0.15,
    },
    "port_scan": {
        "neutralization": 0.40,
        "speed": 0.20,
        "investigation": 0.25,
        "collateral": 0.15,
    },
    "data_exfiltration": {
        "neutralization": 0.35,
        "speed": 0.25,
        "investigation": 0.25,
        "collateral": 0.15,
    },
    "lateral_movement": {
        "neutralization": 0.35,
        "speed": 0.20,
        "investigation": 0.30,
        "collateral": 0.15,
    },
    "ransomware_c2": {
        "neutralization": 0.35,
        "speed": 0.25,
        "investigation": 0.25,
        "collateral": 0.15,
    },
}

DEFAULT_WEIGHTS = {
    "neutralization": 0.40,
    "speed": 0.25,
    "investigation": 0.20,
    "collateral": 0.15,
}


# ──────────────────────────────────────────────
# Main grading function
# ──────────────────────────────────────────────

def grade(summary: EpisodeSummary) -> float:
    """
    Compute the final task score from an EpisodeSummary.

    Returns a float in the open interval (0, 1).
    """
    w = WEIGHT_PROFILES.get(summary.task_id, DEFAULT_WEIGHTS)

    neutralization = _threat_neutralization_score(summary)
    speed = _speed_score(summary)
    investigation = _investigation_score(summary)
    collateral = _collateral_score(summary)

    raw = (
        w["neutralization"] * neutralization
        + w["speed"] * speed
        + w["investigation"] * investigation
        + w["collateral"] * collateral
    )

    return _clamp(raw)


def grade_breakdown(summary: EpisodeSummary) -> dict:
    """Return the full scoring breakdown (useful for debugging)."""
    w = WEIGHT_PROFILES.get(summary.task_id, DEFAULT_WEIGHTS)
    return {
        "neutralization": round(_threat_neutralization_score(summary), 4),
        "speed": round(_speed_score(summary), 4),
        "investigation": round(_investigation_score(summary), 4),
        "collateral": round(_collateral_score(summary), 4),
        "weights": w,
        "final_score": grade(summary),
    }
