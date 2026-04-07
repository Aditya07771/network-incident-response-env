"""
graders.py
Deterministic episode graders for all three tasks.

Each grader receives an EpisodeSummary and returns a float score in [0, 1].

Grading rubric
──────────────
ssh_bruteforce   : Correct IP blocked quickly → 1.0 | slowly → 0.7 | wrong → 0.0
stealth_scan     : Correct IP blocked, no collateral → 1.0 | with collateral → 0.6 | missed → 0.0
lateral_movement : Pivot isolated early → 1.0 | late → 0.5 | DB compromised → 0.0
"""

from __future__ import annotations

from env_models import EpisodeSummary


# ──────────────────────────────────────────────
# Per-task graders
# ──────────────────────────────────────────────

def grade_ssh_bruteforce(summary: EpisodeSummary) -> float:
    """
    Easy task.
    - Correct attacker blocked within ≤10 steps → 1.00
    - Correct attacker blocked after 10 steps   → 0.70
    - Attacker NOT blocked (or wrong IP)         → 0.00
    Collateral damage subtracts 0.20 from the above.
    """
    if summary.attacker_ip not in summary.blocked_ips:
        return 0.0

    base = 1.0 if summary.steps_taken <= 10 else 0.7
    if summary.collateral_damage:
        base = max(0.0, base - 0.20)
    return round(base, 4)


def grade_stealth_scan(summary: EpisodeSummary) -> float:
    """
    Medium task.
    - Correct IP blocked, zero collateral → 1.00
    - Correct IP blocked, with collateral → 0.60
    - Attacker NOT blocked                → 0.00
    Speed bonus: if solved in ≤8 steps add 0.10 (capped at 1.0).
    """
    if summary.attacker_ip not in summary.blocked_ips:
        return 0.0

    base = 0.60 if summary.collateral_damage else 1.0

    # Speed bonus (only when no collateral)
    if not summary.collateral_damage and summary.steps_taken <= 8:
        base = min(1.0, base + 0.10)

    return round(base, 4)


def grade_lateral_movement(summary: EpisodeSummary) -> float:
    """
    Hard task.
    The agent must block either the initial attacker OR the pivot server
    (10.0.1.80) before the DB is deeply compromised.

    Scoring:
    - Pivot/attacker isolated before step 10 → 1.00
    - Pivot/attacker isolated after step 10  → 0.50
    - Neither blocked                         → 0.00
    Collateral subtracts 0.20.
    """
    PIVOT = "10.0.1.80"
    target_blocked = (
        summary.attacker_ip in summary.blocked_ips
        or PIVOT in summary.blocked_ips
    )

    if not target_blocked:
        return 0.0

    # Find the step at which the blocking occurred
    # We approximate using steps_taken (first correct block ends episode)
    base = 1.0 if summary.steps_taken < 10 else 0.5

    if summary.collateral_damage:
        base = max(0.0, base - 0.20)

    return round(base, 4)


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────

_GRADERS = {
    "ssh_bruteforce": grade_ssh_bruteforce,
    "stealth_scan": grade_stealth_scan,
    "lateral_movement": grade_lateral_movement,
}


def grade(summary: EpisodeSummary) -> float:
    """Unified entry-point: dispatch to the correct grader by task_id."""
    grader = _GRADERS.get(summary.task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task_id='{summary.task_id}'")
    return grader(summary)
