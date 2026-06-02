#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# launch.sh — everyday entry point for the Rally-Racing control GUI.
#
# Reads launcher.conf for the venv location, checks the environment is ready,
# and starts the GUI using the venv's Python (so all training subprocesses
# inherit the right interpreter). If the venv is missing or incomplete it
# offers to run setup.sh — so a non-technical user never has to touch a
# terminal beyond clicking "yes".
#
# This is what Rally-Racing.desktop invokes on double-click.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONF="$SCRIPT_DIR/launcher.conf"

# Helper: show a message via GUI dialog if possible, else terminal.
notify() {
    local msg="$1"
    if command -v zenity >/dev/null 2>&1; then
        zenity --info --no-wrap --text="$msg" 2>/dev/null || true
    else
        echo "$msg"
    fi
}

ask_yes_no() {  # returns 0 for yes
    local msg="$1"
    if command -v zenity >/dev/null 2>&1; then
        zenity --question --no-wrap --text="$msg" 2>/dev/null
    else
        read -r -p "$msg [y/N] " ans
        [[ "$ans" == "y" || "$ans" == "Y" ]]
    fi
}

if [[ ! -f "$CONF" ]]; then
    notify "launcher.conf is missing. Cannot find the virtual environment."
    exit 1
fi
# shellcheck disable=SC1090
source "$CONF"
: "${VENV_PATH:?VENV_PATH is not set in launcher.conf}"

VPY="$VENV_PATH/bin/python3"

# Is the environment ready? Check the venv exists AND simple_driving imports.
needs_setup=0
if [[ ! -x "$VPY" ]]; then
    needs_setup=1
elif ! "$VPY" -c "import simple_driving" >/dev/null 2>&1; then
    needs_setup=1
fi

if [[ "$needs_setup" -eq 1 ]]; then
    if ask_yes_no "The Rally-Racing environment isn't set up yet (venv missing or incomplete at:\n$VENV_PATH).\n\nRun the one-time setup now? This installs the dependencies and may take several minutes."; then
        # Run setup in a visible terminal if we can, so the user sees progress.
        if command -v x-terminal-emulator >/dev/null 2>&1; then
            x-terminal-emulator -e bash -c "'$SCRIPT_DIR/setup.sh'; echo; read -p 'Press Enter to close...'"
        elif command -v gnome-terminal >/dev/null 2>&1; then
            gnome-terminal -- bash -c "'$SCRIPT_DIR/setup.sh'; echo; read -p 'Press Enter to close...'"
        else
            bash "$SCRIPT_DIR/setup.sh"
        fi
    else
        notify "Setup cancelled. The GUI cannot start without the environment."
        exit 1
    fi
fi

# Re-check after a possible setup run.
if [[ ! -x "$VPY" ]] || ! "$VPY" -c "import simple_driving" >/dev/null 2>&1; then
    notify "Environment still not ready. Please run ./setup.sh from a terminal and check for errors."
    exit 1
fi

# Launch the GUI. Using the venv Python means every training subprocess the
# GUI spawns (via sys.executable) inherits this interpreter and its packages.
exec "$VPY" "$SCRIPT_DIR/src/control.py"
