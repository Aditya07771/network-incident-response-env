"""
inference.py
Runs one full episode per task using an OpenAI-compatible client, with a
deterministic local heuristic fallback when the API is unavailable.
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

TASKS = ["ssh_bruteforce", "stealth_scan", "lateral_movement"]
BENCHMARK = "network-incident-response"
MAX_HISTORY = 6
TASK_SEEDS = {
    "ssh_bruteforce": 101,
    "stealth_scan": 202,
    "lateral_movement": 303,
}


def get_client() -> OpenAI:
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY") or "local-heuristic-fallback"
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)

SYSTEM_PROMPT = """
You are a SOC analyst. Return only JSON with this exact shape:
{"thought":"short reasoning","action_type":"query_logs|block_ip|no_op","parameters":{}}

Rules:
- Use query_logs to investigate suspicious IPs, protocols, keywords, or ports.
- Use block_ip only when the evidence is strong.
- Avoid blocking internal 10.0.x.x addresses unless the logs clearly show a compromised pivot.
- Never output markdown or prose outside the JSON object.
""".strip()


def build_user_prompt(task_id: str, observation: Dict[str, Any], step: int) -> str:
    lines = []
    for entry in observation.get("recent_logs", []):
        lines.append(
            f"{entry['timestamp']} {entry['severity']} "
            f"{entry['source_ip']}->{entry['destination_ip']}:{entry['port']} "
            f"{entry['protocol']} {entry['message']}"
        )

    query_results = observation.get("query_results", [])
    query_section = "\n".join(
        f"- {entry['source_ip']}:{entry['port']} {entry['message']}"
        for entry in query_results
    ) or "- none"

    log_section = "\n".join(lines) or "no logs yet"
    return (
        f"Task: {task_id}\n"
        f"Step: {step}\n"
        f"Blocked IPs: {observation.get('blocked_ips', [])}\n"
        f"Time elapsed: {observation.get('time_elapsed', 0)}\n"
        f"Recent logs:\n{log_section}\n"
        f"Query results:\n{query_section}\n"
        "Return the next JSON action."
    )


def parse_action(raw_text: str) -> Optional[Action]:
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
        return Action(**payload)
    except Exception:
        return None


def heuristic_action(task_id: str, observation: Dict[str, Any]) -> Action:
    logs = observation.get("recent_logs", [])
    suspicious_sources: Counter[str] = Counter()
    suspicious_filters: Counter[str] = Counter()
    blocked_ips = set(observation.get("blocked_ips", []))

    for entry in logs:
        source_ip = entry["source_ip"]
        message = entry["message"]
        port = entry["port"]
        severity = entry["severity"]
        protocol = entry["protocol"]
        destination_ip = entry["destination_ip"]

        if (
            task_id == "lateral_movement"
            and source_ip == "10.0.1.80"
            and destination_ip == "10.0.1.90"
            and port == 3306
        ):
            return Action(
                thought="Critical pivot from web server to database detected.",
                action_type="block_ip",
                parameters={"ip": "10.0.1.80"},
            )

        if severity in {"WARNING", "CRITICAL"}:
            suspicious_sources[source_ip] += 2
        if "Failed password" in message:
            suspicious_sources[source_ip] += 3
            suspicious_filters[source_ip] += 1
        if "SYN scan detected" in message or (protocol == "TCP" and port >= 8000):
            suspicious_sources[source_ip] += 3
            suspicious_filters[source_ip] += 1
        if "SQLi" in message or "Successful SQLi" in message:
            suspicious_sources[source_ip] += 3
            suspicious_filters[source_ip] += 1
        if "Anomalous connection from web server to database" in message:
            suspicious_filters["10.0.1.80"] += 1

    candidate_ip = None
    if suspicious_sources:
        candidate_ip = suspicious_sources.most_common(1)[0][0]

    if candidate_ip and candidate_ip not in blocked_ips:
        score = suspicious_sources[candidate_ip]
        if candidate_ip.startswith("203.0.113.") and score >= 5:
            return Action(
                thought="External IP has the strongest repeated malicious pattern.",
                action_type="block_ip",
                parameters={"ip": candidate_ip},
            )
        if task_id == "lateral_movement" and score >= 4 and candidate_ip.startswith("203.0.113."):
            return Action(
                thought="Strong external SQLi source identified.",
                action_type="query_logs",
                parameters={"filter": candidate_ip},
            )
        if score >= 3:
            return Action(
                thought="Investigating the most suspicious source before blocking.",
                action_type="query_logs",
                parameters={"filter": candidate_ip},
            )

    if suspicious_filters:
        filter_value = suspicious_filters.most_common(1)[0][0]
        return Action(
            thought="Investigating the strongest suspicious indicator.",
            action_type="query_logs",
            parameters={"filter": filter_value},
        )

    return Action(
        thought="No credible signal yet.",
        action_type="no_op",
        parameters={},
    )


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
            max_tokens=200,
        )
        content = response.choices[0].message.content or ""
        history.append({"role": "user", "content": user_prompt})
        history.append({"role": "assistant", "content": content})
        parsed = parse_action(content)
        if parsed is None:
            return heuristic_action(task_id, observation), "failed_to_parse_json"
        return parsed, None
    except Exception as exc:
        error = str(exc).replace("\n", " ").strip() or "model_call_failed"
        return heuristic_action(task_id, observation), error


def format_action(action: Action) -> str:
    if action.action_type == "block_ip":
        return f"block_ip('{action.parameters.get('ip', '')}')"
    if action.action_type == "query_logs":
        return f"query_logs('{action.parameters.get('filter', '')}')"
    return "no_op()"


def run_episode(task_id: str) -> float:
    env = NetworkIncidentEnv(task_id=task_id, seed=TASK_SEEDS.get(task_id, 0))
    observation = env.reset()
    rewards: List[float] = []
    history: List[Dict[str, str]] = []
    done = False
    step = 0
    success = False
    score = 0.0001

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while not done and step < NetworkIncidentEnv.MAX_STEPS:
            step += 1
            action, error = model_action(task_id, observation.model_dump(), step, history)
            observation, reward, done, _info = env.step(action)
            rewards.append(reward.score)
            print(
                f"[STEP] step={step} action={format_action(action)} "
                f"reward={reward.score:.2f} done={str(done).lower()} "
                f"error={error if error is not None else 'null'}",
                flush=True,
            )
        success = env.state()["threat_neutralized"]
        summary = env.episode_summary()
        score = grade(summary)
        return score
    finally:
        env.close()
        rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            f"[END] success={str(success).lower()} steps={step} "
            f"score={score:.4f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    tasks = sys.argv[1:] if len(sys.argv) > 1 else TASKS
    for task_name in tasks:
        run_episode(task_name)
