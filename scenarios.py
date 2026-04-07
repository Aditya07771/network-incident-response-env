"""
scenarios.py
Realistic network attack scenario generators.
Each scenario exposes:
  - attacker_ip      : the IP the agent must block
  - legitimate_ips   : IPs the agent must NOT block
  - generate_logs()  : returns List[LogEntry] for the current step
"""

from __future__ import annotations

import random
from typing import List

from env_models import LogEntry


# ──────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────

class AttackScenario:
    """Abstract base for all scenarios."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        # Randomise attacker IP every episode so agents can't memorise it
        self.attacker_ip: str = f"203.0.113.{self._rng.randint(10, 250)}"
        # Massive pool of legitimate IPs for large scale testing
        self.legitimate_ips: List[str] = [f"10.0.{i}.{j}" for i in range(1, 10) for j in range(1, 250)]
        self.server_ip: str = "10.0.1.50"

    # ------------------------------------------------------------------
    def _ts(self, step: int) -> str:
        hour = 12 + ((step - 1) // 60)
        minute = (step - 1) % 60
        second = (step * 7) % 60
        return f"{hour:02d}:{minute:02d}:{second:02d}"

    def _rand_legit_ip(self) -> str:
        return self._rng.choice(self.legitimate_ips)

    # ------------------------------------------------------------------
    def generate_logs(self, step: int) -> List[LogEntry]:  # noqa: ARG002
        raise NotImplementedError("Subclasses must implement generate_logs()")


# ──────────────────────────────────────────────
# Task 1 – SSH Brute Force  (EASY)
# ──────────────────────────────────────────────

class SSHBruteForce(AttackScenario):
    """
    Obvious pattern: 3-5 failed SSH attempts from the same foreign IP
    every step, mixed with a few legitimate successful logins.
    The signal-to-noise ratio is high — any decent agent should find it.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.server_ip = "10.0.1.50"
        # Random usernames the attacker tries
        self._usernames = ["root", "admin", "ubuntu", "pi", "test", "deploy"]

    def generate_logs(self, step: int) -> List[LogEntry]:  # noqa: ARG002
        logs: List[LogEntry] = []

        # ── Attack traffic ──────────────────────────────────────────────
        burst = self._rng.randint(3, 5)
        for _ in range(burst):
            logs.append(LogEntry(
                timestamp=self._ts(step),
                source_ip=self.attacker_ip,
                destination_ip=self.server_ip,
                port=22,
                protocol="SSH",
                message=f"Failed password for {self._rng.choice(self._usernames)} "
                        f"from {self.attacker_ip} port {self._rng.randint(40000, 60000)}",
                severity="WARNING",
            ))

        # ── Massive legitimate traffic (representing 12k req/min scale) ───
        for _ in range(self._rng.randint(40, 60)):
            logs.append(LogEntry(
                timestamp=self._ts(step),
                source_ip=self._rand_legit_ip(),
                destination_ip=self.server_ip,
                port=22,
                protocol="SSH",
                message="Accepted publickey for " + self._rng.choice(["devops", "admin", "user", "service"]),
                severity="INFO",
            ))

        self._rng.shuffle(logs)
        return logs


# ──────────────────────────────────────────────
# Task 2 – Stealth Port Scan  (MEDIUM)
# ──────────────────────────────────────────────

class StealthScan(AttackScenario):
    """
    Attacker slowly probes TCP ports 8000-9000.
    The scan packets arrive only every 2 steps, buried inside
    heavy legitimate HTTP/HTTPS traffic.  The agent must distinguish
    the low-rate unusual port hits from normal web traffic.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.server_ip = "10.0.1.80"
        self._scanned_ports: set[int] = set()

    def _next_scan_port(self) -> int:
        port = self._rng.randint(8000, 9000)
        # Avoid repeating ports (realistic nmap behaviour)
        while port in self._scanned_ports:
            port = self._rng.randint(8000, 9000)
        self._scanned_ports.add(port)
        return port

    def generate_logs(self, step: int) -> List[LogEntry]:
        logs: List[LogEntry] = []
        methods = ["GET", "POST", "GET", "GET", "PUT"]  # weighted GET

        # ── Heavy legitimate web traffic (representing 12k req/min scale) ───
        for _ in range(self._rng.randint(40, 60)):
            logs.append(LogEntry(
                timestamp=self._ts(step),
                source_ip=self._rand_legit_ip(),
                destination_ip=self.server_ip,
                port=self._rng.choice([80, 443]),
                protocol="HTTP" if self._rng.random() < 0.4 else "HTTPS",
                message=f"{self._rng.choice(methods)} /api/v1/{self._rng.choice(['data','users','health'])} 200",
                severity="INFO",
            ))

        # ── Stealth scan: one probe every 2 steps ───────────────────────
        if step % 2 == 0:
            port = self._next_scan_port()
            logs.append(LogEntry(
                timestamp=self._ts(step),
                source_ip=self.attacker_ip,
                destination_ip=self.server_ip,
                port=port,
                protocol="TCP",
                message=f"SYN scan detected on port {port} - no service banner",
                severity="WARNING",
            ))

        self._rng.shuffle(logs)
        return logs


# ──────────────────────────────────────────────
# Task 3 – Lateral Movement  (HARD)
# ──────────────────────────────────────────────

class LateralMovement(AttackScenario):
    """
    Multi-stage APT chain:
      Phase 1 (steps 1-7)  : SQL injection attempts on the web server.
      Phase 2 (steps 8+)   : Compromised web server pivots to the DB.

    The agent must correlate both phases and isolate the web server
    (10.0.1.80) before the DB (10.0.1.90) is accessed at depth.
    The 'attacker_ip' is the initial external attacker;
    after compromise the web server BECOMES the pivot — the grader
    checks for isolation of 10.0.1.80 as well.
    """

    PIVOT_IP = "10.0.1.80"
    DB_IP = "10.0.1.90"
    COMPROMISE_STEP = 8

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.server_ip = self.PIVOT_IP
        # Track when (if ever) the agent isolates the pivot
        self.pivot_isolated_at: int | None = None
        self.db_accessed_at: int | None = None
        self._sqli_payloads = [
            "' OR '1'='1",
            "1; DROP TABLE users--",
            "' UNION SELECT username,password FROM users--",
            "admin'--",
        ]

    def generate_logs(self, step: int) -> List[LogEntry]:
        logs: List[LogEntry] = []

        if step < self.COMPROMISE_STEP:
            # ── Phase 1: SQL Injection probing ──────────────────────────
            payload = self._rng.choice(self._sqli_payloads)
            logs.append(LogEntry(
                timestamp=self._ts(step),
                source_ip=self.attacker_ip,
                destination_ip=self.PIVOT_IP,
                port=443,
                protocol="HTTPS",
                message=f"POST /login HTTP/1.1 - SQLi attempt: {payload}",
                severity="WARNING",
            ))
            # Mix in massive normal web traffic
            for _ in range(self._rng.randint(40, 60)):
                logs.append(LogEntry(
                    timestamp=self._ts(step),
                    source_ip=self._rand_legit_ip(),
                    destination_ip=self.PIVOT_IP,
                    port=443,
                    protocol="HTTPS",
                    message="GET /dashboard 200",
                    severity="INFO",
                ))

        else:
            # ── Phase 2: Pivot → DB lateral movement ────────────────────
            if step == self.COMPROMISE_STEP:
                logs.append(LogEntry(
                    timestamp=self._ts(step),
                    source_ip=self.attacker_ip,
                    destination_ip=self.PIVOT_IP,
                    port=443,
                    protocol="HTTPS",
                    message="POST /login HTTP/1.1 200 - Successful SQLi login as admin",
                    severity="CRITICAL",
                ))

            logs.append(LogEntry(
                timestamp=self._ts(step),
                source_ip=self.PIVOT_IP,
                destination_ip=self.DB_IP,
                port=3306,
                protocol="MySQL",
                message="Anomalous connection from web server to database - SELECT * FROM customers",
                severity="CRITICAL",
            ))

            if self.db_accessed_at is None:
                self.db_accessed_at = step
                
            for _ in range(self._rng.randint(40, 60)):
                logs.append(LogEntry(
                    timestamp=self._ts(step),
                    source_ip=self._rand_legit_ip(),
                    destination_ip=self.PIVOT_IP,
                    port=443,
                    protocol="HTTPS",
                    message="GET /dashboard 200",
                    severity="INFO",
                ))

        self._rng.shuffle(logs)
        return logs
