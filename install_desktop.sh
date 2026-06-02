#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# install_desktop.sh — register the double-click launcher on this machine.
#
# Fills the absolute project path into Rally-Racing.desktop and copies it to
# the user's application menu (~/.local/share/applications) and Desktop, marking
# launch.sh executable. After this, the user can start everything by
# double-clicking the "Rally-Racing Control Panel" icon — no terminal.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DESKTOP_SRC="$SCRIPT_DIR/Rally-Racing.desktop"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_DEST="$APPS_DIR/rally-racing.desktop"

echo "Registering Rally-Racing launcher for project at: $SCRIPT_DIR"

# Make the shell scripts executable.
chmod +x "$SCRIPT_DIR/launch.sh" "$SCRIPT_DIR/setup.sh" 2>/dev/null || true

# Build a .desktop with absolute paths substituted in.
mkdir -p "$APPS_DIR"
sed -e "s|/ABSOLUTE/PATH/TO/Rally-Racing|$SCRIPT_DIR|g" \
    "$DESKTOP_SRC" > "$DESKTOP_DEST"
chmod +x "$DESKTOP_DEST"

# Refresh the desktop database if the tool is available.
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
fi

# Also drop a copy on the Desktop if one exists.
if [[ -d "$HOME/Desktop" ]]; then
    cp "$DESKTOP_DEST" "$HOME/Desktop/rally-racing.desktop"
    chmod +x "$HOME/Desktop/rally-racing.desktop"
    # GNOME requires the file to be "trusted" to allow launching.
    if command -v gio >/dev/null 2>&1; then
        gio set "$HOME/Desktop/rally-racing.desktop" \
            metadata::trusted true 2>/dev/null || true
    fi
fi

echo "Done. Look for 'Rally-Racing Control Panel' in your applications menu"
echo "or on your Desktop. Double-click it to launch."
echo
echo "If the environment isn't set up yet, the launcher will offer to run"
echo "the one-time setup for you on first launch."
