"""
Shared runtime configuration for the Rally-Racing project.

Single source of truth bridging the control GUI and the pipeline scripts.
The GUI writes ``gui_config.json`` at the project root; every stage reads it
through ``load_config()`` and overlays the values on top of its own hardcoded
defaults. When the file is absent the scripts behave exactly as before, so all
scripts remain runnable standalone from the command line.

Nothing here imports heavy dependencies (no torch / gym / sb3), so it is cheap
to import from any script or from the GUI process itself.
"""

import json
import os

# ── Canonical paths ─────────────────────────────────────────────────────────
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(SRC_DIR, ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "gui_config.json")


# ── Defaults ─────────────────────────────────────────────────────────────────
# These mirror the original hardcoded constants. They are the values used when
# no gui_config.json is present, or for any key the GUI did not write.
DEFAULTS = {
    # ── State-RL training (train.py) ──
    "total_timesteps": 300_000,
    "n_envs":          8,
    "scenario":        "circuit_easy",  # circuit_easy / _medium / _hard / _difficult
    "load_previous":   True,
    "reset_timesteps": True,
    "use_wandb":       True,
    "wandb_project":   "rally-racing",

    # PPO hyperparameters
    "learning_rate":   3e-4,
    "batch_size":      256,
    "ent_coef":        0.005,
    "net_arch":        [256, 256],
    "device":          "cpu",

    # ── Reward weights (reward.py :: RewardConfig) ──
    "reward": {
        "GOAL_REWARD":              100.0,
        "OBSTACLE_PENALTY":        -100.0,
        "OUT_OF_BOUNDS":            -50.0,
        "WORLD_BOUNDARY":            30.0,
        "STEP_PENALTY":              -2.0,
        "PROGRESS_SCALE":             5.0,
        "REGRESSION_PENALTY":       -10.0,
        "YAW_JERK_PENALTY":          -8.0,
        "ROLL_DELTA_PENALTY":       -15.0,
        "PITCH_DELTA_PENALTY":       -4.0,
        "MIN_SAFE_DISTANCE":          1.0,
        "REPULSE_RADIUS":             2.5,
        "REPULSE_SCALE":             10.0,
        "AIRBORNE_PITCH_THRESHOLD":   0.20,
        "AIRBORNE_BONUS":             1.0,
    },
}


def load_config(path: str = CONFIG_PATH) -> dict:
    r"""Return the effective config: ``DEFAULTS`` overlaid with the GUI's file.

    Resolution is a shallow merge with the saved file taking precedence,
    :math:`\text{cfg} = \text{DEFAULTS} \oplus \text{user}` — except the nested
    ``reward`` table, which is merged key-by-key so a partial reward override
    keeps the untouched defaults. Robust to a missing or malformed file: either
    case returns the pure defaults, so a stand-alone CLI run always works
    without the GUI ever having been opened.
    """
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if not os.path.exists(path):
        return cfg
    try:
        with open(path, "r") as f:
            user = json.load(f)
    except (json.JSONDecodeError, OSError):
        return cfg

    for k, v in user.items():
        if k == "reward" and isinstance(v, dict):
            cfg["reward"].update(v)
        else:
            cfg[k] = v
    return cfg


def save_config(cfg: dict, path: str = CONFIG_PATH) -> None:
    """Persist a config dict (used by the GUI)."""
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    # Quick manual check: print the effective config.
    print(json.dumps(load_config(), indent=2))