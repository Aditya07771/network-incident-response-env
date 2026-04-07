
# Network Incident Response Environment

This project is an OpenEnv-compatible benchmark for real incident-response work. The agent reads SIEM-like network logs, investigates suspicious activity, and chooses defensive actions that stop attacks while minimizing collateral damage.

## Overview

The environment simulates three realistic SOC workflows:

- `ssh_bruteforce` (easy): repeated failed SSH attempts mixed with legitimate logins
- `stealth_scan` (medium): low-rate port scanning hidden inside heavy web traffic
- `lateral_movement` (hard): SQLi-driven compromise followed by pivoting from a web server to a database

The implementation exposes typed Pydantic models for observations, actions, and rewards, plus `reset()`, `step()`, and `state()` methods.

## Observation Space

Each step returns an `Observation` object with:

- `recent_logs`: last 40 visible log entries
- `query_results`: up to 10 entries matched by the previous `query_logs` action
- `blocked_ips`: currently blocked IPs
- `time_elapsed`: current step number

## Action Space

Each action is a typed `Action` model:

- `action_type="query_logs"` with `parameters={"filter": "<ip|port|keyword|protocol>"}`
- `action_type="block_ip"` with `parameters={"ip": "<address>"}`
- `action_type="no_op"` with `parameters={}`

## Reward Design

The reward function gives incremental feedback throughout the trajectory:

- `+1.00` for blocking the real attacker or the pivot host in lateral movement
- `-0.50` for blocking a legitimate internal host
- `+0.10` for a first query that directly identifies the attacker IP
- `+0.03` for queries that surface suspicious logs
- `+0.01` for queries that return benign context
- `-0.02` for empty or useless queries
- `-0.05` for duplicate or irrelevant blocks
- `-0.01` for repeated idle loops

## Task Graders

There are three deterministic graders in [graders.py](/home/karan/Projects/network-incident-response-env/graders.py):

- Easy: reward fast, accurate SSH brute-force containment
- Medium: reward clean detection of stealth scans without collateral damage
- Hard: reward timely isolation of the pivot before deeper database compromise

Each grader returns a score in `[0.0, 1.0]`.

## Python 3.11 Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt
cp .env.example .env
```

Set the following variables in `.env` or your shell:

- `API_BASE_URL` with a default already provided by `inference.py`
- `MODEL_NAME` with a default already provided by `inference.py`
- `HF_TOKEN` which is required

## Usage

Run the local verification suite:

```bash
python3.11 test_local.py
```

Run the inference baseline:

```bash
python3.11 inference.py
```

Run a single task:

```bash
python3.11 inference.py ssh_bruteforce
```

Start the HTTP server locally:

```bash
python3.11 app.py
```

Validate the manifest:

```bash
openenv validate
```

## Docker

Pull the published image:

```bash
docker pull normie69k/network-incident-response-env
```

Run the published image:

```bash
docker run --rm -p 7860:7860 --env-file .env normie69k/network-incident-response-env
```

Build the container:

```bash
docker build -t network-incident-response-env .
```

Run the container:

```bash
docker run --rm -p 7860:7860 --env-file .env network-incident-response-env
```

## Hugging Face Space

This repository is structured for a Docker-based Hugging Face Space. The Space metadata is embedded in this README front matter, and the project is tagged with `openenv`.

## Baseline Notes

`inference.py` uses the OpenAI client for model calls and falls back to a deterministic local heuristic if the remote API is unavailable or returns invalid JSON. The environment itself is seeded, so task generation is reproducible across runs.
