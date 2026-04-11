"""
network_incident_env.py
Core Gym-style environment for Network Incident Response v2.

Key innovations over v1:
  - 7 action types (investigation + containment + reporting)
  - Investigation tracking — rewards thoroughness before action
  - Time pressure — attacker progresses each step
  - Multi-dimensional rewards (detection, speed, investigation, collateral)
  - No severity labels — agent must infer threats from patterns

Public API (mirrors OpenEnv spec):
  env = NetworkIncidentEnv(task_id="ssh_bruteforce")
  obs          = env.reset()
  obs, rew, done, info = env.step(action)
  summary      = env.episode_summary()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from env_models import Action, EpisodeSummary, NetworkEvent, Observation, Reward
from scenarios import (
    AttackScenario,
    DataExfiltration,
    LateralMovement,
    PortScan,
    RansomwareC2,
    SSHBruteForce,
)


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

SCENARIO_REGISTRY: Dict[str, type[AttackScenario]] = {
    "ssh_bruteforce": SSHBruteForce,
    "port_scan": PortScan,
    "data_exfiltration": DataExfiltration,
    "lateral_movement": LateralMovement,
    "ransomware_c2": RansomwareC2,
}


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class NetworkIncidentEnv:
    """
    SOC analyst RL environment with investigation-then-containment gameplay.

    Action Costs
    ─────────────
    analyze_traffic    +0.03 if IP is attacker-related, else +0.01
    inspect_payload    +0.05 if reveals attack signature, else +0.01
    check_reputation   +0.04 if threat_score > 0.5, else +0.01
    correlate_events   +0.05 if pattern matched, else +0.01
    block_ip           +1.00 if attacker IP, -0.40 if legitimate
    isolate_host       +1.00 if compromised host, -0.40 if clean
    submit_report      +0.20 if attack_type correct, -0.10 if wrong

    Investigation Bonus: Agent gets +0.10 bonus on correct containment
    if it investigated >= 2 distinct IPs before acting.

    Step Limit: MAX_STEPS (default 30).
    """

    MAX_STEPS = 30

    def __init__(self, task_id: str = "ssh_bruteforce", seed: int = 0) -> None:
        if task_id not in SCENARIO_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(SCENARIO_REGISTRY.keys())}"
            )
        self.task_id = task_id
        self.seed = seed
        self._scenario: Optional[AttackScenario] = None
        self._all_events: List[NetworkEvent] = []
        self._analysis_result: Optional[Dict[str, Any]] = None
        self._blocked_ips: List[str] = []
        self._isolated_hosts: List[str] = []
        self._step_count: int = 0
        self._threat_neutralized: bool = False
        self._agent_attack_type: Optional[str] = None
        self._total_reward: float = 0.0

        # Investigation tracking
        self._investigated_ips: set = set()
        self._investigation_actions: int = 0
        self._containment_actions: int = 0
        self._collateral_blocks: int = 0
        self._collateral_isolations: int = 0
        self._investigation_rewarded_ips: set = set()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode."""
        self._scenario = SCENARIO_REGISTRY[self.task_id](seed=self.seed)
        self._all_events = []
        self._analysis_result = None
        self._blocked_ips = []
        self._isolated_hosts = []
        self._step_count = 0
        self._threat_neutralized = False
        self._agent_attack_type = None
        self._total_reward = 0.0
        self._investigated_ips = set()
        self._investigation_actions = 0
        self._containment_actions = 0
        self._collateral_blocks = 0
        self._collateral_isolations = 0
        self._investigation_rewarded_ips = set()
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Advance the environment by one step."""
        if self._scenario is None:
            raise RuntimeError("Call reset() before step()")

        self._step_count += 1

        # Generate new network events for this step
        new_events = self._scenario.generate_events(self._step_count)
        self._all_events.extend(new_events)

        # Process action
        reward = self._evaluate(action)
        self._total_reward += reward.score

        obs = self._build_observation()
        done = self._threat_neutralized or (self._step_count >= self.MAX_STEPS)

        info: Dict[str, Any] = {
            "steps_remaining": self.MAX_STEPS - self._step_count,
            "threat_neutralized": self._threat_neutralized,
            "investigation_actions": self._investigation_actions,
            "unique_ips_investigated": len(self._investigated_ips),
            "attack_progression": round(self._scenario.attack_progression, 3),
        }

        return obs, reward, done, info

    def episode_summary(self) -> EpisodeSummary:
        """Call after done to get grader-friendly summary."""
        if self._scenario is None:
            raise RuntimeError("No episode has been run yet.")

        return EpisodeSummary(
            task_id=self.task_id,
            ground_truth_attack_type=self._scenario.ATTACK_TYPE,
            ground_truth_attacker_ips=[self._scenario.attacker_ip],
            ground_truth_compromised_hosts=self._scenario.compromised_hosts,
            blocked_ips=list(self._blocked_ips),
            isolated_hosts=list(self._isolated_hosts),
            agent_attack_type=self._agent_attack_type,
            steps_taken=self._step_count,
            max_steps=self.MAX_STEPS,
            investigation_actions=self._investigation_actions,
            unique_ips_investigated=len(self._investigated_ips),
            threat_neutralized=self._threat_neutralized,
            collateral_blocks=self._collateral_blocks,
            collateral_isolations=self._collateral_isolations,
            attack_progression=round(self._scenario.attack_progression, 4),
            total_reward=round(self._total_reward, 4),
        )

    def state(self) -> Dict[str, Any]:
        """Lightweight state snapshot for /state endpoint."""
        return {
            "task_id": self.task_id,
            "step_count": self._step_count,
            "blocked_ips": list(self._blocked_ips),
            "isolated_hosts": list(self._isolated_hosts),
            "threat_neutralized": self._threat_neutralized,
            "total_reward": round(self._total_reward, 4),
            "investigation_actions": self._investigation_actions,
            "unique_ips_investigated": len(self._investigated_ips),
            "attack_progression": round(
                self._scenario.attack_progression if self._scenario else 0, 3
            ),
        }

    def close(self) -> None:
        """Cleanup resources."""
        pass

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _build_observation(self) -> Observation:
        return Observation(
            network_events=self._all_events[-50:],
            analysis_result=self._analysis_result,
            blocked_ips=list(self._blocked_ips),
            isolated_hosts=list(self._isolated_hosts),
            time_elapsed=self._step_count,
            time_remaining=self.MAX_STEPS - self._step_count,
            alert_summary=self._scenario.alert_hint() if self._scenario else "",
        )

    def _evaluate(self, action: Action) -> Reward:
        """Map an action to a reward, updating internal state."""
        scenario = self._scenario
        self._analysis_result = None   # clear previous result

        # ── analyze_traffic ──────────────────────
        if action.action_type == "analyze_traffic":
            ip = action.parameters.get("ip", "").strip()
            if not ip:
                return Reward(score=-0.01, message="analyze_traffic: empty IP")
            self._investigation_actions += 1
            self._investigated_ips.add(ip)
            result = scenario.get_traffic_analysis(ip, self._all_events)
            self._analysis_result = {"type": "traffic_analysis", **result}
            reward_val = self._investigation_reward(ip, result)
            return Reward(
                score=reward_val,
                investigation_credit=True,
                message=f"Traffic analysis for {ip}: {result['total_connections']} connections",
            )

        # ── inspect_payload ──────────────────────
        elif action.action_type == "inspect_payload":
            ip = action.parameters.get("ip", "").strip()
            if not ip:
                return Reward(score=-0.01, message="inspect_payload: empty IP")
            self._investigation_actions += 1
            self._investigated_ips.add(ip)
            result = scenario.get_payload_inspection(ip, self._all_events)
            self._analysis_result = {"type": "payload_inspection", **result}
            reward_val = 0.05 if result.get("signatures_matched") else 0.01
            if ip not in self._investigation_rewarded_ips:
                self._investigation_rewarded_ips.add(ip)
            else:
                reward_val = 0.005   # diminishing returns
            return Reward(
                score=reward_val,
                investigation_credit=True,
                message=f"Payload inspection for {ip}: {result.get('signatures_matched', [])}",
            )

        # ── check_reputation ─────────────────────
        elif action.action_type == "check_reputation":
            ip = action.parameters.get("ip", "").strip()
            if not ip:
                return Reward(score=-0.01, message="check_reputation: empty IP")
            self._investigation_actions += 1
            self._investigated_ips.add(ip)
            result = scenario.get_reputation(ip)
            self._analysis_result = {"type": "reputation_report", **result}
            reward_val = 0.04 if result.get("threat_score", 0) > 0.5 else 0.01
            if ip not in self._investigation_rewarded_ips:
                self._investigation_rewarded_ips.add(ip)
            else:
                reward_val = 0.005
            return Reward(
                score=reward_val,
                investigation_credit=True,
                message=f"Reputation for {ip}: threat_score={result['threat_score']}",
            )

        # ── correlate_events ─────────────────────
        elif action.action_type == "correlate_events":
            filter_str = str(action.parameters.get("filter", "")).strip()
            if not filter_str:
                return Reward(score=-0.01, message="correlate_events: empty filter")
            self._investigation_actions += 1
            result = scenario.get_event_correlation(filter_str, self._all_events)
            self._analysis_result = {"type": "event_correlation", **result}
            reward_val = 0.05 if result.get("pattern_match") else 0.01
            return Reward(
                score=reward_val,
                investigation_credit=True,
                message=f"Correlation for '{filter_str}': {result['matching_events']} events",
            )

        # ── block_ip ─────────────────────────────
        elif action.action_type == "block_ip":
            ip = action.parameters.get("ip", "").strip()
            if not ip:
                return Reward(score=-0.01, message="block_ip: empty IP")
            self._containment_actions += 1

            if ip in self._blocked_ips:
                return Reward(score=-0.03, message=f"{ip} already blocked")
            self._blocked_ips.append(ip)

            # Check: is this the attacker?
            if ip == scenario.attacker_ip:
                investigation_bonus = 0.10 if len(self._investigated_ips) >= 2 else 0.0
                self._threat_neutralized = True
                return Reward(
                    score=min(1.0, 0.90 + investigation_bonus),
                    threat_stopped=True,
                    message=f"✅ Attacker {ip} blocked" +
                            (" (+investigation bonus)" if investigation_bonus else ""),
                )
            elif ip in scenario.legitimate_ips:
                self._collateral_blocks += 1
                return Reward(
                    score=-0.40,
                    collateral_damage=True,
                    message=f"❌ {ip} is legitimate — collateral damage",
                )
            else:
                return Reward(score=-0.05, message=f"{ip} is not relevant")

        # ── isolate_host ─────────────────────────
        elif action.action_type == "isolate_host":
            ip = action.parameters.get("ip", "").strip()
            if not ip:
                return Reward(score=-0.01, message="isolate_host: empty IP")
            self._containment_actions += 1

            if ip in self._isolated_hosts:
                return Reward(score=-0.03, message=f"{ip} already isolated")
            self._isolated_hosts.append(ip)

            if ip in scenario.compromised_hosts:
                investigation_bonus = 0.10 if len(self._investigated_ips) >= 2 else 0.0
                self._threat_neutralized = True
                return Reward(
                    score=min(1.0, 0.90 + investigation_bonus),
                    threat_stopped=True,
                    message=f"✅ Compromised host {ip} isolated" +
                            (" (+investigation bonus)" if investigation_bonus else ""),
                )
            elif ip in scenario.legitimate_ips:
                self._collateral_isolations += 1
                return Reward(
                    score=-0.40,
                    collateral_damage=True,
                    message=f"❌ {ip} is a clean host — collateral damage",
                )
            else:
                return Reward(score=-0.05, message=f"{ip} isolation has no effect")

        # ── submit_report ────────────────────────
        elif action.action_type == "submit_report":
            attack_type = str(action.parameters.get("attack_type", "")).strip()
            self._agent_attack_type = attack_type
            correct = attack_type.lower() == scenario.ATTACK_TYPE.lower()
            return Reward(
                score=0.15 if correct else -0.10,
                message=f"Report: {attack_type} ({'correct' if correct else 'incorrect'})",
            )

        # Unknown action
        return Reward(score=-0.01, message="Unknown action type")

    def _investigation_reward(self, ip: str, result: Dict[str, Any]) -> float:
        """Reward for traffic analysis — higher if IP is attack-related."""
        if ip in self._investigation_rewarded_ips:
            return 0.005   # diminishing returns for re-investigating same IP
        self._investigation_rewarded_ips.add(ip)

        conns = result.get("total_connections", 0)
        if ip == self._scenario.attacker_ip:
            return 0.04
        if ip in self._scenario.compromised_hosts:
            return 0.04
        if conns > 0:
            return 0.01
        return 0.005
