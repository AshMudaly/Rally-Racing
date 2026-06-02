#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh — one-time environment setup for Rally-Racing.
#
# Creates (if needed) the virtual environment named in launcher.conf, installs
# the Python dependencies and the editable simple_driving package, then verifies
# the gym environments import. Run this ONCE per machine; afterwards use the
# double-click launcher (or launch.sh) to start the GUI with no terminal.
#
# Usage:   ./setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve the project root = directory this script lives in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONF="$SCRIPT_DIR/launcher.conf"
if [[ ! -f "$CONF" ]]; then
    echo "ERROR: launcher.conf not found next to setup.sh." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONF"

: "${VENV_PATH:?VENV_PATH is not set in launcher.conf}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"

echo "=============================================================="
echo " Rally-Racing setup"
echo "   project : $SCRIPT_DIR"
echo "   venv    : $VENV_PATH"
echo "   python  : $BOOTSTRAP_PYTHON"
echo "=============================================================="

if [[ "$VENV_PATH" == "/path/to/your/venv" ]]; then
    echo "ERROR: VENV_PATH in launcher.conf is still the placeholder." >&2
    echo "       Edit launcher.conf and set VENV_PATH to a real location." >&2
    exit 1
fi

# 1. Create the venv if it doesn't already exist.
if [[ ! -x "$VENV_PATH/bin/python3" ]]; then
    echo "[1/4] Creating virtual environment at $VENV_PATH ..."
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_PATH"
else
    echo "[1/4] Reusing existing virtual environment at $VENV_PATH"
fi

VPY="$VENV_PATH/bin/python3"

# 2. Upgrade pip tooling. Pin setuptools<82 — newer breaks some editable installs
#    and conflicts with torch's build requirement.
echo "[2/4] Upgrading pip / wheel and pinning setuptools<82 ..."
"$VPY" -m pip install --upgrade pip wheel "setuptools<82"

# 3. Install project requirements + the editable package.
echo "[3/4] Installing requirements and the editable simple_driving package ..."
"$VPY" -m pip install -r "$SCRIPT_DIR/requirements.txt"
"$VPY" -m pip install -e "$SCRIPT_DIR"

# 4. Verify the gym environments import (this is the check that actually matters).
echo "[4/4] Verifying simple_driving imports ..."
if "$VPY" -c "import simple_driving; print('   OK ->', simple_driving.__file__)"; then
    echo "--------------------------------------------------------------"
    echo " Setup complete. Launch the GUI with ./launch.sh or by"
    echo " double-clicking Rally-Racing.desktop."
    echo "--------------------------------------------------------------"
else
    echo "ERROR: simple_driving still does not import. Check that the" >&2
    echo "       simple_driving/ package folder exists at the project root." >&2
    exit 1
fi
