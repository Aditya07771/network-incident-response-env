"""
test_local.py
Run this BEFORE deploying to HuggingFace to catch any issues early.

Usage:
    python test_local.py

All tests must show ✅ before you deploy.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Callable, List, Tuple

# ──────────────────────────────────────────────
# Tiny test harness
# ──────────────────────────────────────────────

_results: List[Tuple[str, bool, str]] = []


def test(name: str):
    """Decorator to register a test function."""
    def decorator(fn: Callable):
        try:
            fn()
            _results.append((name, True, ""))
        except Exception:
            _results.append((name, False, traceback.format_exc()))
        return fn
    return decorator


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

@test("Import: env_models")
def _():
    from env_models import LogEntry, Observation, Action, Reward, EpisodeSummary
    assert LogEntry and Observation and Action and Reward and EpisodeSummary


@test("Import: scenarios")
def _():
    from scenarios import SSHBruteForce, StealthScan, LateralMovement
    assert SSHBruteForce and StealthScan and LateralMovement


@test("Import: network_incident_env")
def _():
    from network_incident_env import NetworkIncidentEnv, SCENARIO_REGISTRY
    assert set(SCENARIO_REGISTRY.keys()) == {
        "ssh_bruteforce", "stealth_scan", "lateral_movement"
    }


@test("Import: graders")
def _():
    from graders import grade, grade_ssh_bruteforce, grade_stealth_scan, grade_lateral_movement
    assert grade and grade_ssh_bruteforce and grade_stealth_scan and grade_lateral_movement


@test("Action validation: block_ip requires 'ip' key")
def _():
    from env_models import Action
    from pydantic import ValidationError
    try:
        Action(action_type="block_ip", parameters={})
        raise AssertionError("Should have raised ValidationError")
    except ValidationError:
        pass   # expected


@test("Action validation: query_logs requires 'filter' key")
def _():
    from env_models import Action
    from pydantic import ValidationError
    try:
        Action(action_type="query_logs", parameters={})
        raise AssertionError("Should have raised ValidationError")
    except ValidationError:
        pass


@test("Action validation: no_op passes with empty parameters")
def _():
    from env_models import Action
    a = Action(action_type="no_op", parameters={})
    assert a.action_type == "no_op"


@test("Observation: caps recent_logs at 40")
def _():
    from env_models import LogEntry, Observation
    logs = [
        LogEntry(
            timestamp="12:00:00", source_ip="1.2.3.4", destination_ip="10.0.0.1",
            port=22, protocol="SSH", message="test", severity="INFO"
        )
        for _ in range(100)
    ]
    obs = Observation(recent_logs=logs)
    assert len(obs.recent_logs) == 40, f"Expected 40, got {len(obs.recent_logs)}"


@test("Reward: score outside [-1, 1] rejected")
def _():
    from env_models import Reward
    from pydantic import ValidationError
    try:
        Reward(score=2.0)
        raise AssertionError("Should have raised")
    except ValidationError:
        pass


@test("SSHBruteForce: generates logs with attacker IP")
def _():
    from scenarios import SSHBruteForce
    s = SSHBruteForce()
    logs = s.generate_logs(step=1)
    assert len(logs) >= 1
    ips = {l.source_ip for l in logs}
    assert s.attacker_ip in ips, f"attacker_ip {s.attacker_ip} not in {ips}"


@test("StealthScan: scan log only appears every 2 steps")
def _():
    from scenarios import StealthScan
    s = StealthScan()
    # Step 1 (odd) → no scan
    logs1 = s.generate_logs(step=1)
    scan1 = [l for l in logs1 if l.source_ip == s.attacker_ip]
    assert len(scan1) == 0, "Scan should not appear on odd steps"
    # Step 2 (even) → scan
    logs2 = s.generate_logs(step=2)
    scan2 = [l for l in logs2 if l.source_ip == s.attacker_ip]
    assert len(scan2) == 1, "Scan should appear on even steps"


@test("LateralMovement: Phase 1 SQLi before step 8")
def _():
    from scenarios import LateralMovement
    s = LateralMovement()
    logs = s.generate_logs(step=3)
    assert any("SQLi" in l.message or "OR" in l.message for l in logs)


@test("LateralMovement: Phase 2 pivot after step 8")
def _():
    from scenarios import LateralMovement
    s = LateralMovement()
    logs = s.generate_logs(step=9)
    assert any(l.source_ip == "10.0.1.80" for l in logs)


@test("Env: reset() returns empty Observation")
def _():
    from network_incident_env import NetworkIncidentEnv
    env = NetworkIncidentEnv(task_id="ssh_bruteforce")
    obs = env.reset()
    assert obs.recent_logs == []
    assert obs.blocked_ips == []
    assert obs.time_elapsed == 0


@test("Env: unknown task_id raises ValueError")
def _():
    from network_incident_env import NetworkIncidentEnv
    try:
        NetworkIncidentEnv(task_id="nonexistent")
        raise AssertionError("Should have raised")
    except ValueError:
        pass


@test("Env: step() before reset() raises RuntimeError")
def _():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce")
    try:
        env.step(Action(action_type="no_op", parameters={}))
        raise AssertionError("Should have raised")
    except RuntimeError:
        pass


@test("Env: blocking attacker ends episode (ssh_bruteforce)")
def _():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce")
    env.reset()
    action = Action(action_type="block_ip",
                    parameters={"ip": env._scenario.attacker_ip})
    _, reward, done, _ = env.step(action)
    assert reward.score == 1.0
    assert reward.threat_stopped is True
    assert done is True


@test("Env: blocking legitimate IP gives -0.5 (collateral)")
def _():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce")
    env.reset()
    action = Action(action_type="block_ip",
                    parameters={"ip": "10.0.1.10"})
    _, reward, done, _ = env.step(action)
    assert reward.score == -0.5
    assert reward.collateral_damage is True
    assert done is False   # episode continues


@test("Env: query with attacker IP gives +0.10")
def _():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce")
    env.reset()
    attacker = env._scenario.attacker_ip
    action = Action(action_type="query_logs",
                    parameters={"filter": attacker})
    _, reward, _, _ = env.step(action)
    assert reward.score == 0.10


@test("Env: no_op gives 0.0")
def _():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="stealth_scan")
    env.reset()
    _, reward, _, _ = env.step(Action(action_type="no_op", parameters={}))
    assert reward.score == 0.0


@test("Env: blocking same IP twice gives -0.05 second time")
def _():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="stealth_scan")
    env.reset()
    attacker = env._scenario.attacker_ip
    action = Action(action_type="block_ip", parameters={"ip": attacker})
    _, r1, _, _ = env.step(action)
    # reset without re-resetting scenario — manually test
    env._threat_neutralized = False  # force continue
    _, r2, _, _ = env.step(action)
    assert r2.score == -0.05, "Duplicate block should be penalized slightly"


@test("Env: episode ends after max_steps")
def _():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="stealth_scan")
    env.reset()
    noop = Action(action_type="no_op", parameters={})
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(noop)
        steps += 1
        assert steps <= NetworkIncidentEnv.MAX_STEPS + 1, "Episode never ended!"
    assert steps == NetworkIncidentEnv.MAX_STEPS


@test("Env: lateral_movement — blocking pivot (10.0.1.80) wins")
def _():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="lateral_movement")
    env.reset()
    action = Action(action_type="block_ip", parameters={"ip": "10.0.1.80"})
    _, reward, done, _ = env.step(action)
    assert reward.score == 1.0
    assert done is True


@test("Grader: ssh_bruteforce — fast block scores 1.0")
def _():
    from env_models import EpisodeSummary
    from graders import grade_ssh_bruteforce
    s = EpisodeSummary(
        task_id="ssh_bruteforce", attacker_ip="1.2.3.4",
        blocked_ips=["1.2.3.4"], steps_taken=5,
        threat_neutralized=True, collateral_damage=False,
        total_reward=1.0, normalized_score=1.0,
    )
    assert grade_ssh_bruteforce(s) == 1.0


@test("Grader: ssh_bruteforce — slow block scores 0.7")
def _():
    from env_models import EpisodeSummary
    from graders import grade_ssh_bruteforce
    s = EpisodeSummary(
        task_id="ssh_bruteforce", attacker_ip="1.2.3.4",
        blocked_ips=["1.2.3.4"], steps_taken=15,
        threat_neutralized=True, collateral_damage=False,
        total_reward=1.0, normalized_score=1.0,
    )
    assert grade_ssh_bruteforce(s) == 0.7


@test("Grader: ssh_bruteforce — miss scores 0.0")
def _():
    from env_models import EpisodeSummary
    from graders import grade_ssh_bruteforce
    s = EpisodeSummary(
        task_id="ssh_bruteforce", attacker_ip="1.2.3.4",
        blocked_ips=["9.9.9.9"], steps_taken=20,
        threat_neutralized=False, collateral_damage=False,
        total_reward=0.0, normalized_score=0.0,
    )
    assert grade_ssh_bruteforce(s) == 0.0


@test("Grader: stealth_scan — clean block scores 1.0")
def _():
    from env_models import EpisodeSummary
    from graders import grade_stealth_scan
    s = EpisodeSummary(
        task_id="stealth_scan", attacker_ip="5.5.5.5",
        blocked_ips=["5.5.5.5"], steps_taken=6,
        threat_neutralized=True, collateral_damage=False,
        total_reward=1.0, normalized_score=1.0,
    )
    assert grade_stealth_scan(s) == 1.0


@test("Grader: stealth_scan — collateral reduces to 0.6")
def _():
    from env_models import EpisodeSummary
    from graders import grade_stealth_scan
    s = EpisodeSummary(
        task_id="stealth_scan", attacker_ip="5.5.5.5",
        blocked_ips=["5.5.5.5", "10.0.1.11"], steps_taken=10,
        threat_neutralized=True, collateral_damage=True,
        total_reward=0.5, normalized_score=0.5,
    )
    assert grade_stealth_scan(s) == 0.6


@test("Grader: lateral_movement — early isolation scores 1.0")
def _():
    from env_models import EpisodeSummary
    from graders import grade_lateral_movement
    s = EpisodeSummary(
        task_id="lateral_movement", attacker_ip="7.7.7.7",
        blocked_ips=["10.0.1.80"], steps_taken=5,
        threat_neutralized=True, collateral_damage=False,
        total_reward=1.0, normalized_score=1.0,
    )
    assert grade_lateral_movement(s) == 1.0


@test("Grader: lateral_movement — late isolation scores 0.5")
def _():
    from env_models import EpisodeSummary
    from graders import grade_lateral_movement
    s = EpisodeSummary(
        task_id="lateral_movement", attacker_ip="7.7.7.7",
        blocked_ips=["10.0.1.80"], steps_taken=14,
        threat_neutralized=True, collateral_damage=False,
        total_reward=1.0, normalized_score=1.0,
    )
    assert grade_lateral_movement(s) == 0.5


@test("Grader: dispatch works for all task IDs")
def _():
    from env_models import EpisodeSummary
    from graders import grade
    for task_id in ["ssh_bruteforce", "stealth_scan", "lateral_movement"]:
        s = EpisodeSummary(
            task_id=task_id, attacker_ip="1.1.1.1",
            blocked_ips=[], steps_taken=20,
            threat_neutralized=False, collateral_damage=False,
            total_reward=0.0, normalized_score=0.0,
        )
        score = grade(s)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


@test("JSON extraction: handles plain JSON")
def _():
    import json
    import re
    from env_models import Action

    raw = '{"action_type": "block_ip", "parameters": {"ip": "1.2.3.4"}}'
    data = json.loads(raw.strip())
    a = Action(**data)
    assert a.action_type == "block_ip"
    assert a.parameters["ip"] == "1.2.3.4"


@test("JSON extraction: handles markdown-fenced JSON")
def _():
    import json
    import re
    from env_models import Action

    raw = '```json\n{"action_type": "no_op", "parameters": {}}\n```'
    fence_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    m = fence_re.search(raw)
    assert m, "No fence match"
    data = json.loads(m.group(1))
    a = Action(**data)
    assert a.action_type == "no_op"


@test("Full episode: ssh_bruteforce completes with score")
def _():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    from graders import grade

    env = NetworkIncidentEnv(task_id="ssh_bruteforce")
    obs = env.reset()
    done = False
    step = 0
    while not done and step < 20:
        step += 1
        # Smart agent: block the most frequent source IP in WARNING logs
        warning_ips: dict = {}
        for log in obs.recent_logs:
            if log.severity in ("WARNING", "CRITICAL"):
                warning_ips[log.source_ip] = warning_ips.get(log.source_ip, 0) + 1
        if warning_ips:
            top_ip = max(warning_ips, key=warning_ips.__getitem__)
            action = Action(action_type="block_ip", parameters={"ip": top_ip})
        else:
            action = Action(action_type="no_op", parameters={})
        obs, _, done, _ = env.step(action)

    summary = env.episode_summary()
    score = grade(summary)
    assert 0.0 <= score <= 1.0
    # A simple heuristic agent should score at least 0.7 on easy
    assert score >= 0.7, f"Heuristic agent scored too low: {score}"


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

if __name__ == "__main__":
    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'─'*55}")
    print(f"  Network Incident Env — Local Test Suite")
    print(f"{'─'*55}")

    for name, ok, err in _results:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name}")
        if not ok:
            # Print only the last line of the traceback for brevity
            last_line = [l.strip() for l in err.strip().splitlines() if l.strip()][-1]
            print(f"        → {last_line}")

    print(f"{'─'*55}")
    print(f"  {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED ← fix before deploying")
    else:
        print("  |  All good — ready to deploy 🚀")
    print(f"{'─'*55}\n")

    sys.exit(0 if failed == 0 else 1)
