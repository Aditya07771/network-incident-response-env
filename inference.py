"""
inference.py
Runs one full episode per task using an OpenAI-compatible client, with a
deterministic investigation-then-containment heuristic fallback.

Strategy:
  1. Investigate (steps 1-3): analyze_traffic + check_reputation on suspicious IPs
  2. Deep dive (steps 4-6):  inspect_payload + correlate_events
  3. Contain (step 7+):      block_ip / isolate_host based on findings
  4. Report:                 submit_report with attack classification
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from env_models import Action
from graders import grade
from network_incident_env import NetworkIncidentEnv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

TASKS = ["ssh_bruteforce", "port_scan", "data_exfiltration",
         "lateral_movement", "ransomware_c2"]
BENCHMARK = "network-incident-response"
MAX_HISTORY = 6
TASK_SEEDS = {
    "ssh_bruteforce": 101,
    "port_scan": 202,
    "data_exfiltration": 303,
    "lateral_movement": 404,
    "ransomware_c2": 505,
}


def get_client() -> OpenAI:
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY") or "local-heuristic-fallback"
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


SYSTEM_PROMPT = """\
You are a SOC analyst investigating a network security incident.
You have 7 tools available:

1. analyze_traffic(ip) — Get traffic flow statistics for an IP
2. inspect_payload(ip) — Deep packet inspection
3. check_reputation(ip) — Threat intelligence lookup
4. correlate_events(filter) — Cross-reference events by keyword/IP/port
5. block_ip(ip) — Block an external IP at the firewall
6. isolate_host(ip) — Disconnect an internal host from the network
7. submit_report(attack_type) — Submit your findings

Strategy: INVESTIGATE FIRST, then CONTAIN.
- Spend your first actions investigating suspicious IPs and patterns.
- Only block/isolate when you have strong evidence.
- Submit a report classifying the attack type.

Return ONLY valid JSON:
{"thought":"reasoning","action_type":"...","parameters":{...}}
""".strip()


def build_user_prompt(task_id: str, observation: Dict[str, Any], step: int) -> str:
    events = observation.get("network_events", [])
    lines = []
    for e in events[-20:]:
        lines.append(
            f"{e['timestamp']} {e['source_ip']}→{e['destination_ip']}:{e['port']} "
            f"{e['protocol']} {e['payload_snippet'][:60]} ({e.get('bytes_transferred',0)}B)"
        )

    analysis = observation.get("analysis_result")
    analysis_str = json.dumps(analysis, indent=2) if analysis else "none"

    return (
        f"Task: {task_id}\n"
        f"Step: {step} / Time remaining: {observation.get('time_remaining', 30)}\n"
        f"Alert: {observation.get('alert_summary', 'none')}\n"
        f"Blocked IPs: {observation.get('blocked_ips', [])}\n"
        f"Isolated hosts: {observation.get('isolated_hosts', [])}\n"
        f"Recent events:\n" + "\n".join(lines) + "\n"
        f"Last analysis result:\n{analysis_str}\n"
        "Return the next JSON action."
    )


def parse_action(raw_text: str) -> Optional[Action]:
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
        return Action(**payload)
    except Exception:
        return None


def heuristic_action(task_id: str, observation: Dict[str, Any], step: int) -> Action:
    """Investigation-then-containment heuristic."""
    events = observation.get("network_events", [])
    blocked = set(observation.get("blocked_ips", []))
    isolated = set(observation.get("isolated_hosts", []))
    analysis = observation.get("analysis_result")

    # Collect external IPs (non-10.x.x.x) from events
    external_ips: Counter = Counter()
    internal_suspects: Counter = Counter()
    for e in events:
        src = e["source_ip"]
        dst = e["destination_ip"]
        if not src.startswith("10.") and src not in blocked:
            external_ips[src] += 1
        # Internal hosts talking to unusual ports
        if src.startswith("10.") and e["port"] not in (80, 443, 22, 53):
            internal_suspects[src] += 1

    # Phase 1: Investigate (steps 1-4)
    if step <= 2 and external_ips:
        top_ip = external_ips.most_common(1)[0][0]
        return Action(
            thought=f"Investigating top external IP: {top_ip}",
            action_type="analyze_traffic",
            parameters={"ip": top_ip},
        )

    if step == 3 and external_ips:
        top_ip = external_ips.most_common(1)[0][0]
        return Action(
            thought=f"Checking reputation of {top_ip}",
            action_type="check_reputation",
            parameters={"ip": top_ip},
        )

    if step == 4 and external_ips:
        top_ip = external_ips.most_common(1)[0][0]
        return Action(
            thought=f"Deep inspecting payloads from {top_ip}",
            action_type="inspect_payload",
            parameters={"ip": top_ip},
        )

    # Phase 2: Correlate (steps 5-6)
    if step == 5:
        # Look for patterns
        if task_id == "data_exfiltration":
            return Action(
                thought="Correlating DNS-related events",
                action_type="correlate_events",
                parameters={"filter": "DNS"},
            )
        return Action(
            thought="Correlating SSH-related events",
            action_type="correlate_events",
            parameters={"filter": external_ips.most_common(1)[0][0] if external_ips else "SSH"},
        )

    if step == 6:
        # Check internal suspects
        if internal_suspects:
            suspect = internal_suspects.most_common(1)[0][0]
            return Action(
                thought=f"Investigating internal suspect {suspect}",
                action_type="inspect_payload",
                parameters={"ip": suspect},
            )

    # Phase 3: Contain (steps 7+)
    attack_type_guess = _guess_attack_type(task_id, events, analysis)

    # Submit report first
    if step == 7:
        return Action(
            thought=f"Submitting report: {attack_type_guess}",
            action_type="submit_report",
            parameters={"attack_type": attack_type_guess},
        )

    # Block/isolate
    if task_id == "data_exfiltration":
        # Isolate the internal host doing DNS tunnelling
        for e in events:
            if e["protocol"] == "DNS" and e.get("bytes_transferred", 0) > 180:
                host = e["source_ip"]
                if host.startswith("10.") and host not in isolated:
                    return Action(
                        thought=f"Isolating DNS tunnel source {host}",
                        action_type="isolate_host",
                        parameters={"ip": host},
                    )

    if task_id in ("lateral_movement",):
        for e in events:
            if e["port"] == 3306 and e["source_ip"].startswith("10."):
                host = e["source_ip"]
                if host not in isolated:
                    return Action(
                        thought=f"Isolating pivot host {host}",
                        action_type="isolate_host",
                        parameters={"ip": host},
                    )

    if task_id == "ransomware_c2":
        for e in events:
            if ".encrypted" in e.get("payload_snippet", ""):
                host = e["source_ip"]
                if host.startswith("10.") and host not in isolated:
                    return Action(
                        thought=f"Isolating ransomware-affected host {host}",
                        action_type="isolate_host",
                        parameters={"ip": host},
                    )

    # Default: block the most frequent external IP
    if external_ips:
        target = external_ips.most_common(1)[0][0]
        if target not in blocked:
            return Action(
                thought=f"Blocking top suspect {target}",
                action_type="block_ip",
                parameters={"ip": target},
            )

    return Action(
        thought="No clear signal yet — correlating further",
        action_type="correlate_events",
        parameters={"filter": "SSH" if task_id == "ssh_bruteforce" else "443"},
    )


def _guess_attack_type(task_id: str, events: List[Dict], analysis: Optional[Dict]) -> str:
    """Heuristic to guess attack type from events."""
    payloads = " ".join(e.get("payload_snippet", "") for e in events[-30:])
    if "password auth" in payloads and "SSH" in payloads:
        return "ssh_bruteforce"
    if "SYN" in payloads and "no response" in payloads:
        return "port_scan"
    if "TXT" in payloads and len([e for e in events if e.get("protocol") == "DNS" and e.get("bytes_transferred", 0) > 180]) > 3:
        return "data_exfiltration"
    if "UNION" in payloads or "DROP" in payloads or "SQLi" in payloads:
        return "lateral_movement"
    if ".encrypted" in payloads or "check-update" in payloads:
        return "ransomware_c2"
    return task_id   # fallback


def model_action(
    task_id: str,
    observation: Dict[str, Any],
    step: int,
    history: List[Dict[str, str]],
) -> tuple[Action, Optional[str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-MAX_HISTORY:])
    user_prompt = build_user_prompt(task_id, observation, step)
    messages.append({"role": "user", "content": user_prompt})

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=250,
        )
        content = response.choices[0].message.content or ""
        history.append({"role": "user", "content": user_prompt})
        history.append({"role": "assistant", "content": content})
        parsed = parse_action(content)
        if parsed is None:
            return heuristic_action(task_id, observation, step), "failed_to_parse_json"
        return parsed, None
    except Exception as exc:
        error = str(exc).replace("\n", " ").strip() or "model_call_failed"
        return heuristic_action(task_id, observation, step), error


def format_action(action: Action) -> str:
    if action.action_type in ("block_ip", "isolate_host", "analyze_traffic",
                                "inspect_payload", "check_reputation"):
        return f"{action.action_type}('{action.parameters.get('ip', '')}')"
    if action.action_type == "correlate_events":
        return f"correlate_events('{action.parameters.get('filter', '')}')"
    if action.action_type == "submit_report":
        return f"submit_report('{action.parameters.get('attack_type', '')}')"
    return f"{action.action_type}()"


def run_episode(task_id: str) -> float:
    env = NetworkIncidentEnv(task_id=task_id, seed=TASK_SEEDS.get(task_id, 0))
    observation = env.reset()
    rewards: List[float] = []
    history: List[Dict[str, str]] = []
    done = False
    step = 0
    success = False
    score = 0.0

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while not done and step < NetworkIncidentEnv.MAX_STEPS:
            step += 1
            action, _ = model_action(task_id, observation.model_dump(), step, history)
            observation, reward, done, _info = env.step(action)
            rewards.append(reward.score)

            print(
                f"[STEP] step={step} action={format_action(action)} "
                f"reward={reward.score:.3f} done={str(done).lower()} "
                f"progression={_info.get('attack_progression', 0):.2f}",
                flush=True,
            )
        success = env.state()["threat_neutralized"]
        summary = env.episode_summary()
        score = grade(summary)
        return score
    finally:
        env.close()
        rewards_str = ",".join(f"{r:.3f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={step} "
            f"score={score:.4f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    tasks = sys.argv[1:] if len(sys.argv) > 1 else TASKS
    for task_name in tasks:
        run_episode(task_name)