#!/usr/bin/env bash
# setup.sh — Run once on a fresh machine (Mac / Linux).
# Usage: bash setup.sh
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
err()  { echo -e "${RED}❌ $*${NC}"; exit 1; }
step() { echo -e "\n${YELLOW}── $* ──${NC}"; }

# ─────────────────────────────────────────────────────────────────────────────
step "1/5  Checking Python 3.11"
if ! command -v python3.11 &>/dev/null; then
    err "python3.11 not found. Install Python 3.11 before continuing."
fi
PY_VER=$(python3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -ne 11 ]]; then
    err "Python 3.11 required, found $PY_VER"
fi
ok "Python $PY_VER"

# ─────────────────────────────────────────────────────────────────────────────
step "2/5  Checking Git"
if ! command -v git &>/dev/null; then
    err "Git not found. Install from https://git-scm.com/downloads"
fi
ok "Git $(git --version | awk '{print $3}')"

# ─────────────────────────────────────────────────────────────────────────────
step "3/5  Setting up virtual environment"
if [[ ! -d "venv" ]]; then
    python3.11 -m venv venv
    ok "Created venv/"
else
    ok "venv/ already exists"
fi

# Activate
# shellcheck source=/dev/null
source venv/bin/activate
ok "venv activated"

# ─────────────────────────────────────────────────────────────────────────────
step "4/5  Installing dependencies"
pip install --upgrade pip -q
pip install -r requirements.txt -q
ok "Dependencies installed"

# ─────────────────────────────────────────────────────────────────────────────
step "5/5  Creating .env (if missing)"
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    warn ".env created from template — EDIT it and add your API key before running inference.py"
else
    ok ".env already exists"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1.  Edit .env  → add your API key to HF_TOKEN"
echo "  2.  source venv/bin/activate   (activate venv each new terminal)"
echo "  3.  python3.11 -m pip install -r requirements.txt"
echo "  4.  python3.11 test_local.py   (run inside a Python 3.11 environment with deps)"
echo "  5.  python3.11 inference.py    (runs all 3 tasks end-to-end)"
echo ""
