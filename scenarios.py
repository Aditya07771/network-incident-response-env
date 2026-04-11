"""
scenarios.py
Five realistic attack scenarios — each with distinct patterns the agent
must discover via investigation, NOT via pre-labelled severity fields.

Design principles:
  - NetworkEvent objects carry NO severity / threat labels.
  - The attack signal is embedded in traffic patterns, payload content,
    timing, and port usage — the agent must actively investigate.
  - Each scenario tracks attack progression so graders can assess speed.
  - Legitimate traffic is high-volume and realistic to obscure the signal.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from env_models import NetworkEvent


# ──────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────

class AttackScenario:
    """Abstract base for all scenarios."""

    ATTACK_TYPE: str = "unknown"          # ground-truth label for grader

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.attacker_ip: str = f"203.0.113.{self._rng.randint(10, 250)}"
        self.legitimate_ips: List[str] = [
            f"10.0.{i}.{j}" for i in range(1, 6) for j in range(10, 60)
        ]
        self.server_ip: str = "10.0.1.50"
        self.compromised_hosts: List[str] = []           # ground truth
        self.attack_progression: float = 0.0             # 0-1

    # ── Helpers ───────────────────────────────────
    def _ts(self, step: int) -> str:
        hour = 12 + ((step - 1) // 60)
        minute = (step - 1) % 60
        second = (step * 7) % 60
        return f"{hour:02d}:{minute:02d}:{second:02d}"

    def _rand_legit_ip(self) -> str:
        return self._rng.choice(self.legitimate_ips)

    def _legit_web_traffic(self, step: int, dest: str, count: int) -> List[NetworkEvent]:
        """Generate realistic background web traffic."""
        methods = ["GET", "POST", "GET", "GET", "PUT", "GET"]
        paths = ["/api/v1/data", "/api/v1/users", "/health", "/dashboard",
                 "/api/v2/metrics", "/login", "/api/v1/search"]
        events: List[NetworkEvent] = []
        for _ in range(count):
            method = self._rng.choice(methods)
            path = self._rng.choice(paths)
            code = self._rng.choice([200, 200, 200, 200, 201, 301, 404])
            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self._rand_legit_ip(),
                destination_ip=dest,
                port=self._rng.choice([80, 443]),
                protocol=self._rng.choice(["HTTP", "HTTPS"]),
                payload_snippet=f"{method} {path} HTTP/1.1 → {code}",
                bytes_transferred=self._rng.randint(200, 15000),
            ))
        return events

    def _legit_ssh_traffic(self, step: int, dest: str, count: int) -> List[NetworkEvent]:
        """Generate realistic background SSH traffic."""
        users = ["devops", "admin", "deploy", "ci-runner", "monitoring"]
        events: List[NetworkEvent] = []
        for _ in range(count):
            user = self._rng.choice(users)
            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self._rand_legit_ip(),
                destination_ip=dest,
                port=22,
                protocol="SSH",
                payload_snippet=f"publickey accepted for {user}",
                bytes_transferred=self._rng.randint(500, 3000),
                flags=["ACK"],
            ))
        return events

    def _legit_dns_traffic(self, step: int, count: int) -> List[NetworkEvent]:
        """Generate realistic DNS queries."""
        domains = ["api.internal.corp", "cdn.example.com", "ntp.ubuntu.com",
                    "updates.vendor.io", "telemetry.app.local",
                    "auth.identity.corp", "logs.splunk.internal"]
        events: List[NetworkEvent] = []
        for _ in range(count):
            domain = self._rng.choice(domains)
            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self._rand_legit_ip(),
                destination_ip="10.0.0.2",
                port=53,
                protocol="DNS",
                payload_snippet=f"A {domain} → 10.0.{self._rng.randint(1,5)}.{self._rng.randint(10,99)}",
                bytes_transferred=self._rng.randint(60, 200),
            ))
        return events

    # ── Interface ─────────────────────────────────
    def generate_events(self, step: int) -> List[NetworkEvent]:
        raise NotImplementedError

    def get_traffic_analysis(self, ip: str, all_events: List[NetworkEvent]) -> Dict[str, Any]:
        """Compute traffic stats for a given IP across accumulated events."""
        related = [e for e in all_events if e.source_ip == ip or e.destination_ip == ip]
        if not related:
            return {"target_ip": ip, "total_connections": 0,
                    "unique_destinations": 0, "unique_ports": 0,
                    "port_list": [], "avg_bytes": 0.0,
                    "connection_rate": 0.0, "protocols": []}
        dests = {e.destination_ip for e in related if e.source_ip == ip}
        ports = sorted({e.port for e in related})
        protocols = sorted({e.protocol for e in related})
        total_bytes = sum(e.bytes_transferred for e in related)
        return {
            "target_ip": ip,
            "total_connections": len(related),
            "unique_destinations": len(dests),
            "unique_ports": len(ports),
            "port_list": ports[:20],
            "avg_bytes": round(total_bytes / max(1, len(related)), 1),
            "connection_rate": round(len(related) / max(1, len(set(e.timestamp for e in related))), 2),
            "protocols": protocols,
        }

    def get_payload_inspection(self, ip: str, all_events: List[NetworkEvent]) -> Dict[str, Any]:
        """Deep packet inspection for events involving this IP."""
        related = [e for e in all_events if e.source_ip == ip]
        if not related:
            return {"target_ip": ip, "sample_payloads": [], "entropy_score": 0.0,
                    "signatures_matched": [], "anomaly_flags": []}
        payloads = [e.payload_snippet for e in related[:5]]
        # Subclasses override _detect_signatures for scenario-specific detection
        sigs, flags, entropy = self._detect_signatures(ip, related)
        return {
            "target_ip": ip,
            "sample_payloads": payloads,
            "entropy_score": entropy,
            "signatures_matched": sigs,
            "anomaly_flags": flags,
        }

    def get_reputation(self, ip: str) -> Dict[str, Any]:
        """Threat intel lookup. External IPs that are attackers have higher scores."""
        if ip == self.attacker_ip:
            return {
                "target_ip": ip,
                "threat_score": round(self._rng.uniform(0.6, 0.85), 2),
                "categories": self._rng.sample(
                    ["scanner", "brute_forcer", "proxy", "suspicious"],
                    k=self._rng.randint(1, 2)),
                "known_campaigns": [],
                "first_reported": "2026-03-15",
            }
        if ip.startswith("203.0.113."):
            return {
                "target_ip": ip,
                "threat_score": round(self._rng.uniform(0.0, 0.3), 2),
                "categories": [],
                "known_campaigns": [],
                "first_reported": None,
            }
        # Internal IPs
        return {
            "target_ip": ip,
            "threat_score": 0.0,
            "categories": [],
            "known_campaigns": [],
            "first_reported": None,
        }

    def get_event_correlation(self, filter_str: str, all_events: List[NetworkEvent]) -> Dict[str, Any]:
        """Cross-reference events matching a filter."""
        matches = [e for e in all_events
                   if filter_str.lower() in e.source_ip.lower()
                   or filter_str.lower() in e.destination_ip.lower()
                   or filter_str.lower() in e.protocol.lower()
                   or filter_str.lower() in e.payload_snippet.lower()
                   or filter_str == str(e.port)]
        timeline = [f"[{e.timestamp}] {e.source_ip}→{e.destination_ip}:{e.port} "
                     f"{e.protocol} ({e.payload_snippet[:40]})"
                     for e in matches[-10:]]
        ips = sorted({e.source_ip for e in matches} | {e.destination_ip for e in matches})
        pattern = self._detect_attack_pattern(filter_str, matches)
        return {
            "filter_used": filter_str,
            "matching_events": len(matches),
            "timeline": timeline,
            "related_ips": ips[:15],
            "pattern_match": pattern,
        }

    def _detect_signatures(self, ip: str, events: List[NetworkEvent]) -> tuple:
        """Override in subclass. Returns (signatures, flags, entropy)."""
        return [], [], 0.2

    def _detect_attack_pattern(self, filter_str: str, events: List[NetworkEvent]) -> Optional[str]:
        """Override in subclass. Returns matched pattern name or None."""
        return None

    def alert_hint(self) -> str:
        """Vague IDS alert — gives a direction, not the answer."""
        return "No alerts."


# ──────────────────────────────────────────────
# Task 1 — SSH Brute Force  (EASY)
# ──────────────────────────────────────────────

class SSHBruteForce(AttackScenario):
    """
    An external IP performs rapid SSH login attempts with different usernames.
    Events show as normal SSH connections — the agent must notice the high
    failure rate and unusual connection frequency from a single external IP.
    """

    ATTACK_TYPE = "ssh_bruteforce"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.server_ip = "10.0.1.50"
        self._usernames = ["root", "admin", "ubuntu", "pi", "test", "deploy",
                           "ftp", "mysql", "postgres", "www-data"]

    def generate_events(self, step: int) -> List[NetworkEvent]:
        events: List[NetworkEvent] = []

        # Attack: 3-6 SSH attempts per step, all from same external IP
        burst = self._rng.randint(3, 6)
        for _ in range(burst):
            user = self._rng.choice(self._usernames)
            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self.attacker_ip,
                destination_ip=self.server_ip,
                port=22,
                protocol="SSH",
                payload_snippet=f"password auth for {user} from {self.attacker_ip}:{self._rng.randint(40000, 60000)}",
                bytes_transferred=self._rng.randint(100, 400),
                flags=["SYN", "ACK"],
            ))

        # Legitimate SSH traffic
        events.extend(self._legit_ssh_traffic(step, self.server_ip,
                                               self._rng.randint(8, 15)))
        # Background web + DNS
        events.extend(self._legit_web_traffic(step, "10.0.1.80",
                                               self._rng.randint(15, 25)))
        events.extend(self._legit_dns_traffic(step, self._rng.randint(5, 10)))

        self.attack_progression = min(1.0, step / 25)
        self._rng.shuffle(events)
        return events

    def _detect_signatures(self, ip: str, events: List[NetworkEvent]) -> tuple:
        if ip == self.attacker_ip:
            ssh_events = [e for e in events if e.port == 22]
            return (
                ["repeated_ssh_auth_attempts", "credential_stuffing_pattern"],
                ["rapid_connection_rate", "multiple_usernames"],
                0.15,
            )
        return [], [], 0.1

    def _detect_attack_pattern(self, filter_str: str, events: List[NetworkEvent]) -> Optional[str]:
        ssh_events = [e for e in events if e.port == 22 and "password auth" in e.payload_snippet]
        if len(ssh_events) >= 5:
            return "ssh_brute_force_pattern"
        return None

    def alert_hint(self) -> str:
        return "IDS: Elevated SSH authentication volume detected on 10.0.1.50."


# ──────────────────────────────────────────────
# Task 2 — Stealth Port Scan  (MEDIUM)
# ──────────────────────────────────────────────

class PortScan(AttackScenario):
    """
    Attacker performs a slow SYN scan across uncommon ports on multiple hosts.
    Only one probe every 2 steps — buried in heavy legitimate web traffic.
    The agent must notice the port diversity from a single source.
    """

    ATTACK_TYPE = "port_scan"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._targets = ["10.0.1.50", "10.0.2.20", "10.0.3.30"]
        self.server_ip = "10.0.1.50"
        self._scanned_ports: set = set()

    def _next_scan_port(self) -> int:
        port = self._rng.randint(1024, 9999)
        while port in self._scanned_ports:
            port = self._rng.randint(1024, 9999)
        self._scanned_ports.add(port)
        return port

    def generate_events(self, step: int) -> List[NetworkEvent]:
        events: List[NetworkEvent] = []

        # Attack: one probe every 2 steps to stay stealthy
        if step % 2 == 0:
            target = self._rng.choice(self._targets)
            port = self._next_scan_port()
            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self.attacker_ip,
                destination_ip=target,
                port=port,
                protocol="TCP",
                payload_snippet=f"SYN → :{port} (no response)",
                bytes_transferred=self._rng.randint(40, 80),
                flags=["SYN"],
            ))

        # Heavy legitimate traffic
        events.extend(self._legit_web_traffic(step, "10.0.1.50",
                                               self._rng.randint(20, 40)))
        events.extend(self._legit_web_traffic(step, "10.0.2.20",
                                               self._rng.randint(10, 20)))
        events.extend(self._legit_dns_traffic(step, self._rng.randint(5, 10)))
        events.extend(self._legit_ssh_traffic(step, "10.0.1.50",
                                               self._rng.randint(3, 7)))

        self.attack_progression = min(1.0, len(self._scanned_ports) / 15)
        self._rng.shuffle(events)
        return events

    def _detect_signatures(self, ip: str, events: List[NetworkEvent]) -> tuple:
        if ip == self.attacker_ip:
            ports = {e.port for e in events}
            return (
                ["syn_scan_pattern"] if len(ports) >= 3 else [],
                ["high_port_diversity", "no_established_connections"],
                0.1,
            )
        return [], [], 0.1

    def _detect_attack_pattern(self, filter_str: str, events: List[NetworkEvent]) -> Optional[str]:
        syn_only = [e for e in events if "SYN" in e.flags and len(e.flags) == 1]
        if len(syn_only) >= 3:
            return "stealth_port_scan_pattern"
        return None

    def alert_hint(self) -> str:
        return "IDS: Unusual TCP connection attempts across multiple internal hosts."


# ──────────────────────────────────────────────
# Task 3 — DNS Exfiltration  (MEDIUM)
# ──────────────────────────────────────────────

class DataExfiltration(AttackScenario):
    """
    A compromised internal host (10.0.2.40) exfiltrates data via DNS tunnelling.
    The queries encode hex data in subdomain labels and are larger than normal.
    The agent must notice abnormal DNS query sizes and encoded content.
    """

    ATTACK_TYPE = "data_exfiltration"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.compromised_host = "10.0.2.40"
        self.compromised_hosts = [self.compromised_host]
        self.c2_domain = f"ns{self._rng.randint(1,9)}.data-sync-cdn.example.net"
        self.server_ip = self.compromised_host
        # External C2 — this is the "attacker" for blocking purposes
        self.attacker_ip = f"198.51.100.{self._rng.randint(10, 250)}"

    def generate_events(self, step: int) -> List[NetworkEvent]:
        events: List[NetworkEvent] = []

        # Attack: DNS tunnel queries from compromised host every step
        tunnel_count = self._rng.randint(2, 4)
        for _ in range(tunnel_count):
            hex_data = ''.join(self._rng.choices("0123456789abcdef", k=self._rng.randint(30, 60)))
            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self.compromised_host,
                destination_ip=self.attacker_ip,
                port=53,
                protocol="DNS",
                payload_snippet=f"TXT {hex_data}.{self.c2_domain}",
                bytes_transferred=self._rng.randint(200, 500),
            ))

        # Legitimate DNS traffic (normal sizes, recognizable domains)
        events.extend(self._legit_dns_traffic(step, self._rng.randint(10, 20)))
        events.extend(self._legit_web_traffic(step, "10.0.1.50",
                                               self._rng.randint(15, 25)))
        events.extend(self._legit_ssh_traffic(step, "10.0.1.50",
                                               self._rng.randint(3, 6)))

        self.attack_progression = min(1.0, step / 20)
        self._rng.shuffle(events)
        return events

    def _detect_signatures(self, ip: str, events: List[NetworkEvent]) -> tuple:
        if ip == self.compromised_host:
            dns = [e for e in events if e.protocol == "DNS"]
            long_queries = [e for e in dns if len(e.payload_snippet) > 40]
            sigs = ["dns_tunnelling_pattern"] if long_queries else []
            flags = ["oversized_dns_queries", "hex_encoded_subdomains"] if long_queries else []
            return sigs, flags, 0.85 if long_queries else 0.15
        if ip == self.attacker_ip:
            return ["external_dns_endpoint"], ["non_standard_dns_destination"], 0.7
        return [], [], 0.1

    def _detect_attack_pattern(self, filter_str: str, events: List[NetworkEvent]) -> Optional[str]:
        dns_events = [e for e in events if e.protocol == "DNS"]
        large_dns = [e for e in dns_events if e.bytes_transferred > 180]
        if len(large_dns) >= 4:
            return "dns_exfiltration_pattern"
        return None

    def alert_hint(self) -> str:
        return "IDS: Abnormal DNS query patterns detected from internal network segment."


# ──────────────────────────────────────────────
# Task 4 — Lateral Movement (HARD)
# ──────────────────────────────────────────────

class LateralMovement(AttackScenario):
    """
    Multi-stage APT:
      Phase 1 (steps 1-7): SQLi probes on web server from external attacker
      Phase 2 (steps 8+):  Compromised web server pivots to database

    The agent must correlate the external SQLi attempts with the subsequent
    internal lateral movement and isolate the compromised web server.
    """

    ATTACK_TYPE = "lateral_movement"
    PIVOT_IP = "10.0.1.80"
    DB_IP = "10.0.1.90"
    COMPROMISE_STEP = 8

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.server_ip = self.PIVOT_IP
        self.compromised_hosts = [self.PIVOT_IP]
        self._sqli_payloads = [
            "' OR '1'='1",
            "1; DROP TABLE users--",
            "' UNION SELECT username,password FROM users--",
            "admin'--",
            "'; EXEC xp_cmdshell('whoami')--",
        ]

    def generate_events(self, step: int) -> List[NetworkEvent]:
        events: List[NetworkEvent] = []

        if step < self.COMPROMISE_STEP:
            # Phase 1: SQLi probes
            payload = self._rng.choice(self._sqli_payloads)
            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self.attacker_ip,
                destination_ip=self.PIVOT_IP,
                port=443,
                protocol="HTTPS",
                payload_snippet=f"POST /login HTTP/1.1 body={payload}",
                bytes_transferred=self._rng.randint(300, 800),
                flags=["PSH", "ACK"],
            ))
        else:
            # Phase 2: lateral movement — pivot talks to DB
            if step == self.COMPROMISE_STEP:
                events.append(NetworkEvent(
                    timestamp=self._ts(step),
                    source_ip=self.attacker_ip,
                    destination_ip=self.PIVOT_IP,
                    port=443,
                    protocol="HTTPS",
                    payload_snippet="POST /login HTTP/1.1 → 200 (session established)",
                    bytes_transferred=self._rng.randint(1000, 3000),
                    flags=["PSH", "ACK"],
                ))

            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self.PIVOT_IP,
                destination_ip=self.DB_IP,
                port=3306,
                protocol="MySQL",
                payload_snippet="SELECT * FROM customers LIMIT 1000",
                bytes_transferred=self._rng.randint(5000, 50000),
            ))

        # Background traffic
        events.extend(self._legit_web_traffic(step, self.PIVOT_IP,
                                               self._rng.randint(20, 35)))
        events.extend(self._legit_dns_traffic(step, self._rng.randint(5, 10)))
        events.extend(self._legit_ssh_traffic(step, "10.0.1.50",
                                               self._rng.randint(3, 6)))

        self.attack_progression = min(1.0, step / 18)
        self._rng.shuffle(events)
        return events

    def _detect_signatures(self, ip: str, events: List[NetworkEvent]) -> tuple:
        if ip == self.attacker_ip:
            sqli = [e for e in events if any(p in e.payload_snippet for p in ["OR '", "UNION", "DROP", "xp_cmd"])]
            if sqli:
                return ["sql_injection_attempts"], ["encoded_sql_in_http_body"], 0.3
            return [], [], 0.2
        if ip == self.PIVOT_IP:
            db_conns = [e for e in events if e.port == 3306]
            if db_conns:
                return ["internal_db_access_from_webserver"], ["unusual_query_volume"], 0.5
            return [], [], 0.1
        return [], [], 0.1

    def _detect_attack_pattern(self, filter_str: str, events: List[NetworkEvent]) -> Optional[str]:
        sqli = [e for e in events if any(p in e.payload_snippet for p in ["OR '", "UNION", "DROP"])]
        db = [e for e in events if e.port == 3306 and e.source_ip == self.PIVOT_IP]
        if sqli and db:
            return "sqli_to_lateral_movement_chain"
        if sqli:
            return "sql_injection_campaign"
        if db:
            return "suspicious_internal_db_access"
        return None

    def alert_hint(self) -> str:
        return "IDS: Unusual HTTP POST patterns on 10.0.1.80 and internal database queries."


# ──────────────────────────────────────────────
# Task 5 — Ransomware C2 Callback (HARD)
# ──────────────────────────────────────────────

class RansomwareC2(AttackScenario):
    """
    A compromised host (10.0.3.25) is communicating with a command-and-control
    server over HTTPS with periodic beaconing at regular intervals. The beacon
    payloads are encrypted (high entropy) and uniform in size — a signature
    of C2 frameworks. The agent must spot the beaconing pattern and isolate
    the compromised host while blocking the C2 server.
    """

    ATTACK_TYPE = "ransomware_c2"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.compromised_host = "10.0.3.25"
        self.compromised_hosts = [self.compromised_host]
        self.c2_server = f"198.51.100.{self._rng.randint(10, 250)}"
        self.attacker_ip = self.c2_server
        self.server_ip = self.compromised_host
        self._beacon_size = self._rng.randint(440, 460)

    def generate_events(self, step: int) -> List[NetworkEvent]:
        events: List[NetworkEvent] = []

        # C2 beacon: regular HTTPS POST every step with uniform size
        events.append(NetworkEvent(
            timestamp=self._ts(step),
            source_ip=self.compromised_host,
            destination_ip=self.c2_server,
            port=443,
            protocol="HTTPS",
            payload_snippet=f"POST /api/check-update HTTP/1.1 → 200 (body: {self._beacon_size}B encrypted)",
            bytes_transferred=self._beacon_size + self._rng.randint(-5, 5),
            flags=["PSH", "ACK"],
        ))

        # Every 3 steps: encrypted file operation indicator
        if step % 3 == 0:
            target_file = self._rng.choice([
                "finance_q4.xlsx", "customer_db.sql", "hr_records.csv",
                "contracts_2026.pdf", "source_code.tar.gz",
            ])
            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self.compromised_host,
                destination_ip=self.compromised_host,
                port=445,
                protocol="SMB",
                payload_snippet=f"WRITE {target_file}.encrypted ({self._rng.randint(10000, 500000)}B)",
                bytes_transferred=self._rng.randint(10000, 500000),
            ))

        # Background traffic
        events.extend(self._legit_web_traffic(step, "10.0.1.50",
                                               self._rng.randint(20, 35)))
        events.extend(self._legit_web_traffic(step, "10.0.2.20",
                                               self._rng.randint(10, 20)))
        events.extend(self._legit_dns_traffic(step, self._rng.randint(8, 15)))
        events.extend(self._legit_ssh_traffic(step, "10.0.1.50",
                                               self._rng.randint(3, 6)))

        # Some legit HTTPS from compromised host (to make it blend)
        for _ in range(self._rng.randint(2, 4)):
            events.append(NetworkEvent(
                timestamp=self._ts(step),
                source_ip=self.compromised_host,
                destination_ip=self._rng.choice(["10.0.1.50", "10.0.2.20"]),
                port=443,
                protocol="HTTPS",
                payload_snippet=f"GET /api/v1/{self._rng.choice(['data','status','health'])} → 200",
                bytes_transferred=self._rng.randint(500, 5000),
            ))

        self.attack_progression = min(1.0, step / 15)
        self._rng.shuffle(events)
        return events

    def _detect_signatures(self, ip: str, events: List[NetworkEvent]) -> tuple:
        if ip == self.compromised_host:
            c2_conns = [e for e in events if e.destination_ip == self.c2_server]
            smb_writes = [e for e in events if ".encrypted" in e.payload_snippet]
            sigs, flags = [], []
            if c2_conns:
                sigs.append("c2_beacon_pattern")
                flags.append("periodic_uniform_size_requests")
            if smb_writes:
                sigs.append("ransomware_file_encryption")
                flags.append("encrypted_file_extension")
            return sigs, flags, 0.92 if c2_conns else 0.1
        if ip == self.c2_server:
            return ["external_c2_endpoint"], ["uniform_response_sizes"], 0.88
        return [], [], 0.1

    def _detect_attack_pattern(self, filter_str: str, events: List[NetworkEvent]) -> Optional[str]:
        encrypted = [e for e in events if ".encrypted" in e.payload_snippet]
        beacons = [e for e in events if "check-update" in e.payload_snippet]
        if encrypted and beacons:
            return "ransomware_with_c2_callback"
        if beacons:
            return "c2_beacon_communication"
        if encrypted:
            return "file_encryption_activity"
        return None

    def alert_hint(self) -> str:
        return "IDS: Periodic outbound HTTPS connections with uniform payload sizes from internal host."
