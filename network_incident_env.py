"""
network_incident_env.py
Core Gym-style environment for the Network Incident Response simulation.

Public API (mirrors OpenEnv spec):
  env = NetworkIncidentEnv(task_id="ssh_bruteforce")
  obs          = env.reset()
  obs, rew, done, info = env.step(action)
  summary      = env.episode_summary()   ← returns EpisodeSummary
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from env_models import Action, EpisodeSummary, LogEntry, Observation, Reward
from scenarios import AttackScenario, LateralMovement, SSHBruteForce, StealthScan


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

SCENARIO_REGISTRY: Dict[str, type[AttackScenario]] = {
    "ssh_bruteforce": SSHBruteForce,
    "stealth_scan": StealthScan,
    "lateral_movement": LateralMovement,
}


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class NetworkIncidentEnv:
    """
    Sequential decision-making environment for SOC incident response.

    Reward table
    ────────────
    Correct IP blocked (attacker)          +1.00  (episode ends)
    Legitimate IP blocked                  -0.50  (collateral damage flag set)
    Pivot server isolated (lateral task)   +1.00  (episode ends)
    Query that mentions attacker IP         +0.05
    no_op / irrelevant query                0.00

    Step limit: max_steps (default 20).
    Scores are clipped to [-1, 1] per step.
    """

    MAX_STEPS = 20

    def __init__(self, task_id: str = "ssh_bruteforce", seed: int = 0) -> None:
        if task_id not in SCENARIO_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(SCENARIO_REGISTRY.keys())}"
            )
        self.task_id = task_id
        self.seed = seed
        self._scenario: AttackScenario | None = None
        self._all_logs: List[LogEntry] = []
        self._query_results: List[LogEntry] = []
        self._blocked_ips: List[str] = []
        self._step_count: int = 0
        self._threat_neutralized: bool = False
        self._collateral_damage: bool = False
        self._total_reward: float = 0.0
        self._query_rewarded: bool = False  # only reward the FIRST good query
        self._idle_streak: int = 0
        self._last_action_error: str | None = None

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode.  Must be called before step()."""
        self._scenario = SCENARIO_REGISTRY[self.task_id](seed=self.seed)
        self._all_logs = []
        self._query_results = []
        self._blocked_ips = []
        self._step_count = 0
        self._threat_neutralized = False
        self._collateral_damage = False
        self._total_reward = 0.0
        self._query_rewarded = False
        self._idle_streak = 0
        self._last_action_error = None
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Returns
        -------
        obs   : updated Observation (latest logs visible to agent)
        reward: Reward for this step
        done  : True if episode is over
        info  : diagnostic dict (attacker IP revealed only for debugging)
        """
        if self._scenario is None:
            raise RuntimeError("Call reset() before step()")

        self._step_count += 1

        # Generate new logs for this step
        new_logs = self._scenario.generate_logs(self._step_count)
        self._all_logs.extend(new_logs)

        # Evaluate action
        reward = self._evaluate(action)
        self._total_reward += reward.score

        obs = self._build_observation()
        done = self._threat_neutralized or (self._step_count >= self.MAX_STEPS)

        info: Dict[str, Any] = {
            # Reveal attacker IP only in info dict so graders can check
            "attacker_ip": self._scenario.attacker_ip,
            "steps_remaining": self.MAX_STEPS - self._step_count,
            "threat_neutralized": self._threat_neutralized,
            "query_rewarded": self._query_rewarded,
            "last_action_error": self._last_action_error,
        }

        return obs, reward, done, info

    def episode_summary(self) -> EpisodeSummary:
        """Call after the episode is done to get a grader-friendly summary."""
        if self._scenario is None:
            raise RuntimeError("No episode has been run yet.")

        raw_score = max(0.0, self._total_reward)
        normalized = min(1.0, raw_score / self.MAX_STEPS)

        return EpisodeSummary(
            task_id=self.task_id,
            attacker_ip=self._scenario.attacker_ip,
            blocked_ips=list(self._blocked_ips),
            steps_taken=self._step_count,
            threat_neutralized=self._threat_neutralized,
            collateral_damage=self._collateral_damage,
            total_reward=self._total_reward,
            normalized_score=normalized,
        )

    def state(self) -> Dict[str, Any]:
        """Lightweight state snapshot (used by the FastAPI /state endpoint)."""
        return {
            "task_id": self.task_id,
            "step_count": self._step_count,
            "blocked_ips": list(self._blocked_ips),
            "threat_neutralized": self._threat_neutralized,
            "collateral_damage": self._collateral_damage,
            "total_reward": round(self._total_reward, 4),
            "last_action_error": self._last_action_error,
        }

    def close(self) -> None:
        """Cleanup resources if needed."""
        # No heavy resources in this local simulation, but we follow standard Gym API
        pass

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _build_observation(self) -> Observation:
        return Observation(
            recent_logs=self._all_logs[-40:],
            query_results=list(self._query_results),
            blocked_ips=list(self._blocked_ips),
            time_elapsed=self._step_count,
        )

    def _evaluate(self, action: Action) -> Reward:
        """Map an action to a Reward, updating internal state."""
        scenario = self._scenario  # guaranteed non-None here
        self._query_results = []
        self._last_action_error = None

        # ── block_ip ──────────────────────────────────────────────────
        if action.action_type == "block_ip":
            self._idle_streak = 0
            ip: str = action.parameters.get("ip", "").strip()
            if not ip:
                self._last_action_error = "empty_ip"
                return Reward(score=0.0, message="block_ip with empty IP — ignored")

            if ip in self._blocked_ips:
                self._last_action_error = "already_blocked"
                return Reward(score=-0.05, message=f"{ip} already blocked")

            self._blocked_ips.append(ip)

            # Lateral movement: accept blocking either the external attacker
            # OR the compromised pivot server (10.0.1.80)
            if isinstance(scenario, LateralMovement):
                if ip in (scenario.attacker_ip, LateralMovement.PIVOT_IP):
                    self._threat_neutralized = True
                    return Reward(
                        score=1.0,
                        threat_stopped=True,
                        message=f"✅ Correct — {ip} isolated, attack chain broken",
                    )
                elif ip in scenario.legitimate_ips:
                    self._collateral_damage = True
                    self._last_action_error = "blocked_legitimate_ip"
                    return Reward(
                        score=-0.5,
                        collateral_damage=True,
                        message=f"❌ {ip} is a legitimate host — collateral damage",
                    )
                else:
                    self._last_action_error = "blocked_irrelevant_ip"
                    self._idle_streak += 1
                    return Reward(score=0.0, message=f"{ip} is irrelevant")

            # All other scenarios: attacker IP only
            if ip == scenario.attacker_ip:
                self._threat_neutralized = True
                return Reward(
                    score=1.0,
                    threat_stopped=True,
                    message=f"✅ Correct — attacker {ip} blocked",
                )
            elif ip in scenario.legitimate_ips:
                self._collateral_damage = True
                self._last_action_error = "blocked_legitimate_ip"
                return Reward(
                    score=-0.5,
                    collateral_damage=True,
                    message=f"❌ {ip} is legitimate — collateral damage",
                )
            else:
                self._last_action_error = "blocked_irrelevant_ip"
                return Reward(score=-0.05, message=f"{ip} is unknown / irrelevant")

        # ── query_logs ────────────────────────────────────────────────
        elif action.action_type == "query_logs":
            self._idle_streak = 0
            filter_str: str = str(action.parameters.get("filter", ""))
            filter_text = filter_str.lower().strip()
            if not filter_text:
                self._last_action_error = "empty_filter"
                return Reward(score=-0.02, message="Empty query filter")

            self._query_results = [
                log for log in self._all_logs
                if filter_text in log.source_ip.lower()
                or filter_text in log.destination_ip.lower()
                or filter_text in log.protocol.lower()
                or filter_text in log.message.lower()
                or filter_text == str(log.port)
            ][:10]

            if not self._query_results:
                self._last_action_error = "query_no_hits"
                return Reward(score=-0.02, message="Query returned no relevant hits")

            suspicious_hits = [
                log for log in self._query_results
                if log.severity in ("WARNING", "CRITICAL")
            ]

            if scenario.attacker_ip in filter_str and not self._query_rewarded:
                self._query_rewarded = True
                return Reward(
                    score=0.10,
                    message="Investigation hit — attacker IP found (one-time reward)",
                )
            if suspicious_hits:
                return Reward(
                    score=0.03,
                    message=f"Investigation surfaced {len(suspicious_hits)} suspicious logs",
                )
            return Reward(score=0.01, message="Query returned benign context")

        # ── no_op ─────────────────────────────────────────────────────
        else:
            self._idle_streak += 1
            penalty = -0.01 if self._idle_streak > 2 else 0.0
            if penalty < 0.0:
                self._last_action_error = "idle_loop"
            return Reward(score=penalty, message="Agent chose to wait")
