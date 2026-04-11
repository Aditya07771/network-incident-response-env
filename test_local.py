#!/usr/bin/env python3
"""
test_local.py
Comprehensive test suite for Network Incident Response Environment v2.

Tests cover:
  - Model validation (env_models.py)
  - Scenario generation (scenarios.py)
  - Environment mechanics (network_incident_env.py)
  - Grading logic (graders.py)
  - API endpoints (app.py)
  - Full episode integration
"""

from __future__ import annotations

import json
import sys
import traceback

# ──────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────

_results: list[tuple[str, bool]] = []


def run_test(label: str, fn):
    try:
        fn()
        _results.append((label, True))
        print(f"  ✅  {label}")
    except Exception as exc:
        _results.append((label, False))
        print(f"  ❌  {label}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# 1. Import tests
# ──────────────────────────────────────────────

def test_import_env_models():
    import env_models
    assert hasattr(env_models, "NetworkEvent")
    assert hasattr(env_models, "Action")
    assert hasattr(env_models, "Observation")
    assert hasattr(env_models, "Reward")
    assert hasattr(env_models, "EpisodeSummary")

def test_import_scenarios():
    import scenarios
    assert hasattr(scenarios, "SSHBruteForce")
    assert hasattr(scenarios, "PortScan")
    assert hasattr(scenarios, "DataExfiltration")
    assert hasattr(scenarios, "LateralMovement")
    assert hasattr(scenarios, "RansomwareC2")

def test_import_env():
    import network_incident_env
    assert hasattr(network_incident_env, "NetworkIncidentEnv")

def test_import_graders():
    import graders
    assert hasattr(graders, "grade")
    assert hasattr(graders, "grade_breakdown")

def test_import_app():
    import app
    assert hasattr(app, "app")


# ──────────────────────────────────────────────
# 2. Model validation tests
# ──────────────────────────────────────────────

def test_action_requires_ip_for_block():
    from env_models import Action
    try:
        Action(action_type="block_ip", parameters={})
        assert False, "Should have raised"
    except Exception:
        pass

def test_action_requires_ip_for_analyze():
    from env_models import Action
    try:
        Action(action_type="analyze_traffic", parameters={})
        assert False, "Should have raised"
    except Exception:
        pass

def test_action_requires_ip_for_inspect():
    from env_models import Action
    try:
        Action(action_type="inspect_payload", parameters={})
        assert False, "Should have raised"
    except Exception:
        pass

def test_action_requires_ip_for_reputation():
    from env_models import Action
    try:
        Action(action_type="check_reputation", parameters={})
        assert False, "Should have raised"
    except Exception:
        pass

def test_action_requires_filter_for_correlate():
    from env_models import Action
    try:
        Action(action_type="correlate_events", parameters={})
        assert False, "Should have raised"
    except Exception:
        pass

def test_action_requires_attack_type_for_report():
    from env_models import Action
    try:
        Action(action_type="submit_report", parameters={})
        assert False, "Should have raised"
    except Exception:
        pass

def test_action_isolate_host_requires_ip():
    from env_models import Action
    try:
        Action(action_type="isolate_host", parameters={})
        assert False, "Should have raised"
    except Exception:
        pass

def test_valid_actions_pass():
    from env_models import Action
    Action(action_type="block_ip", parameters={"ip": "1.2.3.4"})
    Action(action_type="isolate_host", parameters={"ip": "10.0.1.80"})
    Action(action_type="analyze_traffic", parameters={"ip": "1.2.3.4"})
    Action(action_type="inspect_payload", parameters={"ip": "1.2.3.4"})
    Action(action_type="check_reputation", parameters={"ip": "1.2.3.4"})
    Action(action_type="correlate_events", parameters={"filter": "SSH"})
    Action(action_type="submit_report", parameters={"attack_type": "ssh_bruteforce"})

def test_observation_caps_events():
    from env_models import NetworkEvent, Observation
    events = [
        NetworkEvent(timestamp="12:00:00", source_ip="10.0.1.10",
                     destination_ip="10.0.1.50", port=80, protocol="HTTP",
                     payload_snippet="GET /")
        for _ in range(60)
    ]
    obs = Observation(network_events=events)
    assert len(obs.network_events) == 50

def test_reward_bounds():
    from env_models import Reward
    try:
        Reward(score=1.5)
        assert False, "Should have raised"
    except Exception:
        pass
    Reward(score=1.0)
    Reward(score=-1.0)

def test_network_event_no_severity():
    from env_models import NetworkEvent
    event = NetworkEvent(
        timestamp="12:00:00",
        source_ip="203.0.113.50",
        destination_ip="10.0.1.50",
        port=22,
        protocol="SSH",
        payload_snippet="password auth for root",
    )
    # Ensure there's no severity field
    assert not hasattr(event, "severity")
    d = event.model_dump()
    assert "severity" not in d


# ──────────────────────────────────────────────
# 3. Scenario tests
# ──────────────────────────────────────────────

def test_ssh_bruteforce_generates_events():
    from scenarios import SSHBruteForce
    s = SSHBruteForce(seed=42)
    events = s.generate_events(1)
    assert len(events) > 0
    ssh_events = [e for e in events if e.port == 22]
    assert len(ssh_events) > 0
    attacker_events = [e for e in events if e.source_ip == s.attacker_ip]
    assert len(attacker_events) > 0

def test_port_scan_stealth():
    from scenarios import PortScan
    s = PortScan(seed=42)
    e1 = s.generate_events(1)  # odd step — no scan
    e2 = s.generate_events(2)  # even step — scan
    syn_step1 = [e for e in e1 if "SYN" in e.flags and e.source_ip == s.attacker_ip]
    syn_step2 = [e for e in e2 if "SYN" in e.flags and e.source_ip == s.attacker_ip]
    assert len(syn_step1) == 0, "No scan on odd steps"
    assert len(syn_step2) > 0, "Scan expected on even steps"

def test_data_exfiltration_dns_tunnel():
    from scenarios import DataExfiltration
    s = DataExfiltration(seed=42)
    events = s.generate_events(1)
    dns_tunnel = [e for e in events if e.protocol == "DNS"
                  and e.source_ip == s.compromised_host]
    assert len(dns_tunnel) > 0
    for e in dns_tunnel:
        assert len(e.payload_snippet) > 30, "DNS tunnel queries should be long"

def test_lateral_movement_phases():
    from scenarios import LateralMovement
    s = LateralMovement(seed=42)
    # Phase 1: SQLi
    e_early = s.generate_events(3)
    sqli = [e for e in e_early if "OR '" in e.payload_snippet or "UNION" in e.payload_snippet
            or "DROP" in e.payload_snippet or "admin'" in e.payload_snippet
            or "xp_cmd" in e.payload_snippet]
    assert len(sqli) > 0, "Phase 1 should have SQLi"
    # Phase 2: lateral movement
    e_late = s.generate_events(10)
    db_conn = [e for e in e_late if e.port == 3306]
    assert len(db_conn) > 0, "Phase 2 should have DB connections"

def test_ransomware_c2_beacon():
    from scenarios import RansomwareC2
    s = RansomwareC2(seed=42)
    events = s.generate_events(3)
    beacon = [e for e in events if "check-update" in e.payload_snippet
              and e.source_ip == s.compromised_host]
    assert len(beacon) > 0
    encrypted = [e for e in events if ".encrypted" in e.payload_snippet]
    assert len(encrypted) > 0, "Step 3 should have encrypted file events"

def test_no_severity_in_any_scenario():
    """Verify NO scenario adds severity labels to events."""
    from scenarios import SSHBruteForce, PortScan, DataExfiltration, LateralMovement, RansomwareC2
    for cls in [SSHBruteForce, PortScan, DataExfiltration, LateralMovement, RansomwareC2]:
        s = cls(seed=42)
        events = s.generate_events(5)
        for e in events:
            d = e.model_dump()
            assert "severity" not in d, f"{cls.__name__} has severity in events"

def test_traffic_analysis():
    from scenarios import SSHBruteForce
    s = SSHBruteForce(seed=42)
    events = s.generate_events(1) + s.generate_events(2)
    result = s.get_traffic_analysis(s.attacker_ip, events)
    assert result["total_connections"] > 0
    assert result["target_ip"] == s.attacker_ip

def test_payload_inspection_attacker():
    from scenarios import SSHBruteForce
    s = SSHBruteForce(seed=42)
    events = s.generate_events(1) + s.generate_events(2)
    result = s.get_payload_inspection(s.attacker_ip, events)
    assert "repeated_ssh_auth_attempts" in result["signatures_matched"]

def test_reputation_check_attacker():
    from scenarios import SSHBruteForce
    s = SSHBruteForce(seed=42)
    result = s.get_reputation(s.attacker_ip)
    assert result["threat_score"] > 0.5

def test_reputation_check_legit():
    from scenarios import SSHBruteForce
    s = SSHBruteForce(seed=42)
    result = s.get_reputation("10.0.1.10")
    assert result["threat_score"] == 0.0

def test_event_correlation():
    from scenarios import SSHBruteForce
    s = SSHBruteForce(seed=42)
    events = s.generate_events(1) + s.generate_events(2) + s.generate_events(3)
    result = s.get_event_correlation("SSH", events)
    assert result["matching_events"] > 0


# ──────────────────────────────────────────────
# 4. Environment tests
# ──────────────────────────────────────────────

def test_env_reset():
    from network_incident_env import NetworkIncidentEnv
    env = NetworkIncidentEnv(task_id="ssh_bruteforce")
    obs = env.reset()
    assert obs.time_elapsed == 0
    assert obs.time_remaining == 30
    assert len(obs.blocked_ips) == 0

def test_env_unknown_task():
    from network_incident_env import NetworkIncidentEnv
    try:
        NetworkIncidentEnv(task_id="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_env_step_before_reset():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv()
    try:
        env.step(Action(action_type="analyze_traffic", parameters={"ip": "1.2.3.4"}))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

def test_env_investigate_before_block():
    """Agent that investigates first gets a bonus."""
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce", seed=42)
    env.reset()

    # Step 1-2: investigate TWO different IPs to trigger bonus
    env.step(Action(action_type="analyze_traffic",
                    parameters={"ip": env._scenario.attacker_ip}))
    env.step(Action(action_type="check_reputation",
                    parameters={"ip": "10.0.1.50"}))
    # Step 3: block
    _, reward, done, _ = env.step(Action(action_type="block_ip",
                                          parameters={"ip": env._scenario.attacker_ip}))
    assert done
    assert reward.threat_stopped
    assert reward.score == 1.0, "Should get full score with investigation bonus"

def test_env_block_without_investigation():
    """Blocking immediately without investigation gives lower score."""
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce", seed=42)
    env.reset()
    # Immediately block
    env.step(Action(action_type="correlate_events", parameters={"filter": "x"}))  # generate events
    _, reward, done, _ = env.step(Action(action_type="block_ip",
                                          parameters={"ip": env._scenario.attacker_ip}))
    assert done
    assert reward.score == 0.9, "Should get 0.9 without investigation bonus"

def test_env_block_legit_ip():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce", seed=42)
    env.reset()
    env.step(Action(action_type="correlate_events", parameters={"filter": "x"}))
    _, reward, _, _ = env.step(Action(action_type="block_ip",
                                      parameters={"ip": "10.0.1.10"}))
    assert reward.score == -0.40
    assert reward.collateral_damage

def test_env_isolate_compromised_host():
    """Isolating a compromised host should neutralize the threat."""
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="data_exfiltration", seed=42)
    env.reset()
    # Investigate first
    env.step(Action(action_type="analyze_traffic",
                    parameters={"ip": env._scenario.compromised_host}))
    env.step(Action(action_type="inspect_payload",
                    parameters={"ip": env._scenario.compromised_host}))
    # Isolate
    _, reward, done, _ = env.step(Action(action_type="isolate_host",
                                          parameters={"ip": env._scenario.compromised_host}))
    assert done
    assert reward.threat_stopped

def test_env_submit_report():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce", seed=42)
    env.reset()
    env.step(Action(action_type="correlate_events", parameters={"filter": "x"}))
    _, reward, _, _ = env.step(Action(action_type="submit_report",
                                      parameters={"attack_type": "ssh_bruteforce"}))
    assert reward.score == 0.15, "Correct report should give +0.15"

def test_env_wrong_report():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce", seed=42)
    env.reset()
    env.step(Action(action_type="correlate_events", parameters={"filter": "x"}))
    _, reward, _, _ = env.step(Action(action_type="submit_report",
                                      parameters={"attack_type": "ransomware_c2"}))
    assert reward.score == -0.10, "Wrong report should give -0.10"

def test_env_max_steps():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce", seed=42)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(Action(action_type="correlate_events",
                                        parameters={"filter": "x"}))
        steps += 1
    assert steps == 30

def test_env_diminishing_returns():
    """Re-investigating the same IP gives diminishing rewards."""
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce", seed=42)
    env.reset()
    env.step(Action(action_type="correlate_events", parameters={"filter": "x"}))

    _, r1, _, _ = env.step(Action(action_type="analyze_traffic",
                                   parameters={"ip": env._scenario.attacker_ip}))
    _, r2, _, _ = env.step(Action(action_type="analyze_traffic",
                                   parameters={"ip": env._scenario.attacker_ip}))
    assert r1.score > r2.score, "Diminishing returns on re-investigation"

def test_env_episode_summary():
    from network_incident_env import NetworkIncidentEnv
    from env_models import Action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce", seed=42)
    env.reset()
    env.step(Action(action_type="analyze_traffic",
                    parameters={"ip": env._scenario.attacker_ip}))
    env.step(Action(action_type="block_ip",
                    parameters={"ip": env._scenario.attacker_ip}))
    summary = env.episode_summary()
    assert summary.task_id == "ssh_bruteforce"
    assert summary.threat_neutralized
    assert summary.investigation_actions == 1
    assert summary.steps_taken == 2

def test_env_five_scenarios():
    """All 5 scenarios can be created and run."""
    from network_incident_env import SCENARIO_REGISTRY, NetworkIncidentEnv
    from env_models import Action
    for task_id in SCENARIO_REGISTRY:
        env = NetworkIncidentEnv(task_id=task_id, seed=42)
        obs = env.reset()
        assert obs.time_remaining == 30
        obs, reward, done, info = env.step(Action(
            action_type="correlate_events", parameters={"filter": "test"}
        ))
        assert not done


# ──────────────────────────────────────────────
# 5. Grader tests
# ──────────────────────────────────────────────

def test_grader_perfect_fast_investigated():
    """Fast neutralization with investigation gets high score."""
    from graders import grade
    from env_models import EpisodeSummary
    s = EpisodeSummary(
        task_id="ssh_bruteforce",
        ground_truth_attack_type="ssh_bruteforce",
        ground_truth_attacker_ips=["203.0.113.42"],
        ground_truth_compromised_hosts=[],
        blocked_ips=["203.0.113.42"],
        isolated_hosts=[],
        agent_attack_type="ssh_bruteforce",
        steps_taken=5,
        max_steps=30,
        investigation_actions=4,
        unique_ips_investigated=3,
        threat_neutralized=True,
        collateral_blocks=0,
        collateral_isolations=0,
        attack_progression=0.2,
        total_reward=1.5,
    )
    score = grade(s)
    assert 0.0001 < score < 0.9999, f"Score {score} not in open interval"
    assert score > 0.85, f"Perfect run should score > 0.85, got {score}"

def test_grader_no_investigation():
    """Neutralized but no investigation gets lower score."""
    from graders import grade
    from env_models import EpisodeSummary
    s = EpisodeSummary(
        task_id="ssh_bruteforce",
        ground_truth_attack_type="ssh_bruteforce",
        ground_truth_attacker_ips=["203.0.113.42"],
        ground_truth_compromised_hosts=[],
        blocked_ips=["203.0.113.42"],
        isolated_hosts=[],
        agent_attack_type=None,
        steps_taken=2,
        max_steps=30,
        investigation_actions=0,
        unique_ips_investigated=0,
        threat_neutralized=True,
        collateral_blocks=0,
        collateral_isolations=0,
        attack_progression=0.1,
        total_reward=0.9,
    )
    score = grade(s)
    assert 0.0001 < score < 0.9999
    assert score < 0.85, f"No-investigation run should score < 0.85, got {score}"

def test_grader_total_miss():
    """Agent failed to neutralize — low score."""
    from graders import grade
    from env_models import EpisodeSummary
    s = EpisodeSummary(
        task_id="ssh_bruteforce",
        ground_truth_attack_type="ssh_bruteforce",
        ground_truth_attacker_ips=["203.0.113.42"],
        ground_truth_compromised_hosts=[],
        blocked_ips=[],
        isolated_hosts=[],
        agent_attack_type=None,
        steps_taken=30,
        max_steps=30,
        investigation_actions=0,
        unique_ips_investigated=0,
        threat_neutralized=False,
        collateral_blocks=0,
        collateral_isolations=0,
        attack_progression=1.0,
        total_reward=0.0,
    )
    score = grade(s)
    assert 0.0001 < score < 0.9999
    assert score < 0.20, f"Total miss should score < 0.20, got {score}"

def test_grader_collateral_damage():
    """Neutralized but with collateral — lower score."""
    from graders import grade
    from env_models import EpisodeSummary
    s = EpisodeSummary(
        task_id="port_scan",
        ground_truth_attack_type="port_scan",
        ground_truth_attacker_ips=["203.0.113.42"],
        ground_truth_compromised_hosts=[],
        blocked_ips=["203.0.113.42", "10.0.1.10", "10.0.1.20"],
        isolated_hosts=[],
        agent_attack_type="port_scan",
        steps_taken=10,
        max_steps=30,
        investigation_actions=3,
        unique_ips_investigated=2,
        threat_neutralized=True,
        collateral_blocks=2,
        collateral_isolations=0,
        attack_progression=0.4,
        total_reward=0.5,
    )
    score = grade(s)
    assert 0.0001 < score < 0.9999
    # Should be moderate — neutralized but with collateral
    assert score < 0.80, f"Collateral damage should reduce score, got {score}"

def test_grader_all_tasks():
    """Grade function works for all 5 task IDs."""
    from graders import grade
    from env_models import EpisodeSummary
    for tid in ["ssh_bruteforce", "port_scan", "data_exfiltration",
                "lateral_movement", "ransomware_c2"]:
        s = EpisodeSummary(
            task_id=tid,
            ground_truth_attack_type=tid,
            ground_truth_attacker_ips=["203.0.113.42"],
            ground_truth_compromised_hosts=[],
            blocked_ips=["203.0.113.42"],
            isolated_hosts=[],
            agent_attack_type=tid,
            steps_taken=8,
            max_steps=30,
            investigation_actions=3,
            unique_ips_investigated=2,
            threat_neutralized=True,
            collateral_blocks=0,
            collateral_isolations=0,
            attack_progression=0.3,
            total_reward=1.2,
        )
        score = grade(s)
        assert 0.0001 < score < 0.9999, f"Score for {tid}: {score}"

def test_grader_breakdown():
    from graders import grade_breakdown
    from env_models import EpisodeSummary
    s = EpisodeSummary(
        task_id="ssh_bruteforce",
        ground_truth_attack_type="ssh_bruteforce",
        ground_truth_attacker_ips=["203.0.113.42"],
        ground_truth_compromised_hosts=[],
        blocked_ips=["203.0.113.42"],
        isolated_hosts=[],
        agent_attack_type="ssh_bruteforce",
        steps_taken=5,
        max_steps=30,
        investigation_actions=3,
        unique_ips_investigated=2,
        threat_neutralized=True,
        collateral_blocks=0,
        collateral_isolations=0,
        attack_progression=0.2,
        total_reward=1.3,
    )
    bd = grade_breakdown(s)
    assert "neutralization" in bd
    assert "speed" in bd
    assert "investigation" in bd
    assert "collateral" in bd
    assert "final_score" in bd
    assert 0.0001 < bd["final_score"] < 0.9999

def test_grader_continuous_range():
    """Verify the grader produces a range of scores, not just a few buckets."""
    from graders import grade
    from env_models import EpisodeSummary
    scores = set()
    for steps in [2, 5, 10, 15, 25, 30]:
        for inv in [0, 1, 3, 5]:
            s = EpisodeSummary(
                task_id="ssh_bruteforce",
                ground_truth_attack_type="ssh_bruteforce",
                ground_truth_attacker_ips=["203.0.113.42"],
                ground_truth_compromised_hosts=[],
                blocked_ips=["203.0.113.42"],
                isolated_hosts=[],
                agent_attack_type="ssh_bruteforce" if inv > 0 else None,
                steps_taken=steps,
                max_steps=30,
                investigation_actions=inv,
                unique_ips_investigated=min(inv, 3),
                threat_neutralized=True,
                collateral_blocks=0,
                collateral_isolations=0,
                attack_progression=steps/30,
                total_reward=1.0,
            )
            scores.add(round(grade(s), 4))
    assert len(scores) >= 5, f"Should have 5+ distinct scores, got {len(scores)}: {scores}"


# ──────────────────────────────────────────────
# 6. API tests
# ──────────────────────────────────────────────

def test_api_reset():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.post("/reset", json={"task_id": "ssh_bruteforce"})
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "info" in data
    obs = data["observation"]
    assert "network_events" in obs
    assert "blocked_ips" in obs
    assert "time_remaining" in obs

def test_api_reset_accepts_task_id():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    for task_id in ["ssh_bruteforce", "port_scan", "data_exfiltration",
                    "lateral_movement", "ransomware_c2"]:
        resp = client.post("/reset", json={"task_id": task_id})
        assert resp.status_code == 200

def test_api_step():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    client.post("/reset", json={"task_id": "ssh_bruteforce"})
    resp = client.post("/step", json={
        "action_type": "correlate_events",
        "parameters": {"filter": "SSH"}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    assert "info" in data

def test_api_health():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_api_tasks():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.get("/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["tasks"]) == 5

def test_api_state():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    client.post("/reset", json={"task_id": "ssh_bruteforce"})
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert "investigation_actions" in data

def test_api_summary():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    client.post("/reset", json={"task_id": "ssh_bruteforce"})
    client.post("/step", json={
        "action_type": "correlate_events",
        "parameters": {"filter": "SSH"}
    })
    resp = client.get("/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "graded_score" in data
    score = data["graded_score"]
    assert 0.0001 < score < 0.9999


# ──────────────────────────────────────────────
# 7. Integration tests
# ──────────────────────────────────────────────

def test_json_extraction():
    from inference import parse_action
    raw = '{"thought":"test","action_type":"block_ip","parameters":{"ip":"1.2.3.4"}}'
    action = parse_action(raw)
    assert action is not None
    assert action.action_type == "block_ip"

def test_json_extraction_with_markdown():
    from inference import parse_action
    raw = '```json\n{"thought":"test","action_type":"analyze_traffic","parameters":{"ip":"10.0.1.50"}}\n```'
    action = parse_action(raw)
    assert action is not None
    assert action.action_type == "analyze_traffic"

def test_full_episode_ssh():
    """Run a full episode with the heuristic agent."""
    from network_incident_env import NetworkIncidentEnv
    from inference import heuristic_action
    env = NetworkIncidentEnv(task_id="ssh_bruteforce", seed=101)
    obs = env.reset()
    done = False
    step = 0
    while not done and step < 30:
        step += 1
        action = heuristic_action("ssh_bruteforce", obs.model_dump(), step)
        obs, reward, done, info = env.step(action)
    summary = env.episode_summary()
    from graders import grade
    score = grade(summary)
    assert 0.0001 < score < 0.9999
    assert summary.steps_taken > 0

def test_full_episode_all_tasks():
    """Run a full episode for every task with the heuristic agent."""
    from network_incident_env import SCENARIO_REGISTRY, NetworkIncidentEnv
    from inference import heuristic_action
    from graders import grade
    task_seeds = {
        "ssh_bruteforce": 101, "port_scan": 202, "data_exfiltration": 303,
        "lateral_movement": 404, "ransomware_c2": 505,
    }
    for task_id in SCENARIO_REGISTRY:
        env = NetworkIncidentEnv(task_id=task_id, seed=task_seeds.get(task_id, 0))
        obs = env.reset()
        done = False
        step = 0
        while not done and step < 30:
            step += 1
            action = heuristic_action(task_id, obs.model_dump(), step)
            obs, reward, done, info = env.step(action)
        summary = env.episode_summary()
        score = grade(summary)
        assert 0.0001 < score < 0.9999, f"Score for {task_id}: {score}"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("───────────────────────────────────────────────────────")
    print("  Network Incident Env v2 — Local Test Suite")
    print("───────────────────────────────────────────────────────")

    # 1. Imports
    run_test("Import: env_models", test_import_env_models)
    run_test("Import: scenarios", test_import_scenarios)
    run_test("Import: network_incident_env", test_import_env)
    run_test("Import: graders", test_import_graders)
    run_test("Import: app", test_import_app)

    # 2. Model validation
    run_test("Action: block_ip requires 'ip'", test_action_requires_ip_for_block)
    run_test("Action: analyze_traffic requires 'ip'", test_action_requires_ip_for_analyze)
    run_test("Action: inspect_payload requires 'ip'", test_action_requires_ip_for_inspect)
    run_test("Action: check_reputation requires 'ip'", test_action_requires_ip_for_reputation)
    run_test("Action: correlate_events requires 'filter'", test_action_requires_filter_for_correlate)
    run_test("Action: submit_report requires 'attack_type'", test_action_requires_attack_type_for_report)
    run_test("Action: isolate_host requires 'ip'", test_action_isolate_host_requires_ip)
    run_test("Action: all 7 valid actions pass", test_valid_actions_pass)
    run_test("Observation: caps events at 50", test_observation_caps_events)
    run_test("Reward: score outside [-1,1] rejected", test_reward_bounds)
    run_test("NetworkEvent: no severity field", test_network_event_no_severity)

    # 3. Scenarios
    run_test("SSHBruteForce: generates attack events", test_ssh_bruteforce_generates_events)
    run_test("PortScan: scan only on even steps", test_port_scan_stealth)
    run_test("DataExfiltration: DNS tunnel queries", test_data_exfiltration_dns_tunnel)
    run_test("LateralMovement: Phase 1 SQLi + Phase 2 DB", test_lateral_movement_phases)
    run_test("RansomwareC2: C2 beacon + encrypted files", test_ransomware_c2_beacon)
    run_test("All scenarios: no severity labels", test_no_severity_in_any_scenario)
    run_test("TrafficAnalysis: returns stats", test_traffic_analysis)
    run_test("PayloadInspection: detects signatures", test_payload_inspection_attacker)
    run_test("ReputationCheck: attacker score > 0.5", test_reputation_check_attacker)
    run_test("ReputationCheck: legit score = 0.0", test_reputation_check_legit)
    run_test("EventCorrelation: finds matching events", test_event_correlation)

    # 4. Environment
    run_test("Env: reset returns valid observation", test_env_reset)
    run_test("Env: unknown task raises ValueError", test_env_unknown_task)
    run_test("Env: step before reset raises RuntimeError", test_env_step_before_reset)
    run_test("Env: investigate-then-block gets full bonus", test_env_investigate_before_block)
    run_test("Env: block-without-investigation gets lower", test_env_block_without_investigation)
    run_test("Env: block legitimate IP gives -0.40", test_env_block_legit_ip)
    run_test("Env: isolate compromised host neutralizes", test_env_isolate_compromised_host)
    run_test("Env: correct submit_report +0.15", test_env_submit_report)
    run_test("Env: wrong submit_report -0.10", test_env_wrong_report)
    run_test("Env: episode ends at max_steps", test_env_max_steps)
    run_test("Env: diminishing returns on re-investigation", test_env_diminishing_returns)
    run_test("Env: episode summary has all fields", test_env_episode_summary)
    run_test("Env: all 5 task IDs run successfully", test_env_five_scenarios)

    # 5. Graders
    run_test("Grader: perfect fast investigated → high score", test_grader_perfect_fast_investigated)
    run_test("Grader: no investigation → lower score", test_grader_no_investigation)
    run_test("Grader: total miss → very low score", test_grader_total_miss)
    run_test("Grader: collateral damage reduces score", test_grader_collateral_damage)
    run_test("Grader: all 5 tasks produce valid scores", test_grader_all_tasks)
    run_test("Grader: breakdown returns all dimensions", test_grader_breakdown)
    run_test("Grader: produces continuous score range", test_grader_continuous_range)

    # 6. API
    run_test("API: POST /reset returns wrapped dict", test_api_reset)
    run_test("API: POST /reset accepts all 5 task_ids", test_api_reset_accepts_task_id)
    run_test("API: POST /step returns full response", test_api_step)
    run_test("API: GET /health returns healthy", test_api_health)
    run_test("API: GET /tasks lists 5 tasks", test_api_tasks)
    run_test("API: GET /state returns env state", test_api_state)
    run_test("API: GET /summary returns graded score", test_api_summary)

    # 7. Integration
    run_test("JSON extraction: plain JSON", test_json_extraction)
    run_test("JSON extraction: markdown-fenced JSON", test_json_extraction_with_markdown)
    run_test("Full episode: ssh_bruteforce completes", test_full_episode_ssh)
    run_test("Full episode: all 5 tasks complete with valid scores", test_full_episode_all_tasks)

    print("───────────────────────────────────────────────────────")
    passed = sum(1 for _, ok in _results if ok)
    total = len(_results)
    if passed == total:
        print(f"  {passed}/{total} passed  |  All good — ready to deploy 🚀")
    else:
        failed = [label for label, ok in _results if not ok]
        print(f"  {passed}/{total} passed  |  FAILURES: {failed}")
        sys.exit(1)
    print("───────────────────────────────────────────────────────")
