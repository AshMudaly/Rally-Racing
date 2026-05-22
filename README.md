Here’s a cleaner, more GitHub-friendly README rewrite that keeps the important technical details while removing repetition and reducing the “wall of text” effect.

# Rally Racing — PPO Autonomous Rally Car

A reinforcement learning rally car built for *41118 Artificial Intelligence in Robotics*.

The agent learns to:

* follow checkpoint-based rally tracks
* avoid obstacles
* optionally use ramps/jumps to shorten lap times

Built using:

* PyBullet — physics simulation
* Gymnasium — RL environment interface
* Stable-Baselines3 — PPO implementation

---

# Features

* Custom PyBullet rally environment
* Progressive curriculum training
* Multi-phase scenarios
* Obstacle avoidance
* Ramp/jump mechanics
* Parallel PPO training
* TensorBoard logging
* GUI and headless evaluation modes

---

# Project Structure

```text
Rally-Racing/
├── simple_driving/
│   ├── envs/
│   │   ├── simple_driving_env.py
│   │   └── rally_driving_env.py
│   └── resources/
│       ├── car.py
│       ├── obstacle.py
│       ├── ramp.py
│       └── *.urdf
│
├── src/
│   ├── reward.py
│   ├── observation.py
│   ├── train.py
│   └── test.py
│
├── models/
├── logs/
├── requirements.txt
├── setup.py
└── README.md
```

---

# Installation

Python 3.10+ recommended.

```bash
cd ~/41118_ws/project/Rally-Racing

pip install -r requirements.txt
pip install -e .
```

Verify the environment registration:

```bash
python3 -c "import simple_driving; import gymnasium as gym; print([e for e in gym.envs.registry.keys() if 'Driving' in e])"
```

Expected output:

```text
['SimpleDriving-v0', 'RallyDriving-v0']
```

---

# Scenarios

The environment supports three training phases:

| Scenario | Description                   |
| -------- | ----------------------------- |
| `phase1` | Basic checkpoint racing       |
| `phase2` | Adds obstacle avoidance       |
| `phase3` | Adds ramps and jump decisions |

Example reset:

```python
env.reset(options={"scenario": "phase3"})
```

Custom checkpoint layouts:

```python
env.reset(
    options={
        "scenario": "phase3",
        "checkpoints": [(5, 5), (10, 0), (15, 5)]
    }
)
```

Default track elements can be edited inside:

```text
simple_driving/envs/rally_driving_env.py
```

---

# Training

Start training:

```bash
cd src
python3 train.py
```

Models are saved automatically to:

* `models/`
* `models/best/`

TensorBoard logs:

* `logs/`

---

# Recommended Training Settings

Current stable defaults:

```python
TOTAL_TIMESTEPS = 500_000
N_ENVS          = 8
SCENARIO        = "phase1"
LOAD_PREVIOUS   = True
RESET_TIMESTEPS = False
```

## Parallel Environment Sizing

PyBullet training is CPU-heavy.

Recommended rule:

```python
N_ENVS ≈ physical CPU cores / 2
```

Typical values:

| CPU     | Recommended `N_ENVS` |
| ------- | -------------------- |
| 4 cores | 2–4                  |
| 6 cores | 4–6                  |
| 8 cores | 6–8                  |

If training becomes unstable or hangs:

```python
N_ENVS = 4
```

---

# Important Performance Fix

`train.py` intentionally uses:

```python
torch.set_num_threads(1)
```

Without this, PyTorch may oversubscribe CPU threads across subprocesses and severely reduce PPO performance.

Do not remove unless benchmarking confirms otherwise.

---

# Curriculum Training (Recommended)

Training phase3 from scratch is difficult.

Recommended progression:

1. Train `phase1`
2. Continue into `phase2`
3. Continue into `phase3`

Example workflow:

```python
SCENARIO = "phase1"
LOAD_PREVIOUS = True
```

This produces significantly more stable learning.

---

# Monitoring Training

Run TensorBoard:

```bash
tensorboard --logdir ~/41118_ws/project/Rally-Racing/logs
```

Open:

```text
http://localhost:6006
```

Useful metrics:

* `rollout/ep_rew_mean`
* `rollout/ep_len_mean`
* `eval/mean_reward`

---

# Evaluation

Run trained models with the PyBullet GUI:

```bash
cd src
python3 test.py
```

Examples:

```bash
python3 test.py --scenarios phase3
python3 test.py --no-render
python3 test.py --model ../models/ppo_rally_final.zip
```

---

# Reward Function

Reward tuning lives in:

```text
src/reward.py
```

Main reward components:

| Component             | Purpose                           |
| --------------------- | --------------------------------- |
| Goal reward           | Encourage checkpoint completion   |
| Step penalty          | Encourage faster laps             |
| Progress reward       | Encourage movement toward targets |
| Obstacle penalties    | Encourage avoidance               |
| Orientation penalties | Reduce unstable driving           |
| Airborne bonus        | Encourage jump usage              |

The `AIRBORNE_BONUS` controls ramp behaviour:

* positive → prefers jumps
* zero → neutral
* negative → avoids ramps

---

# Smoke Testing

Before long training runs, verify the environment loads correctly:

```bash
python3 - <<'PY'
import gymnasium as gym
import simple_driving

env = gym.make("RallyDriving-v0")
obs, info = env.reset()

print("Environment created successfully")
env.close()
PY
```

This catches:

* broken environment registration
* callback issues
* reward mismatches
* observation shape errors

---

# Troubleshooting

## `ModuleNotFoundError: simple_driving`

Reinstall the package:

```bash
pip install -e .
```

---

## `Cannot find simplecar.urdf`

Ensure URDF files exist in:

```text
simple_driving/resources/
```

---

## NumPy / SciPy Compatibility Errors

If you see:

```text
A module compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Avoid mixing:

* Ubuntu apt-installed scientific packages
* pip-installed NumPy packages

Recommended fix:

```bash
pip install --upgrade numpy scipy matplotlib
```

Using a virtual environment is strongly recommended.

---

## SubprocVecEnv Startup Hangs

Reduce environment count:

```python
N_ENVS = 1
```

If the issue disappears, gradually increase until stable.

---

## Harmless Matplotlib Warning

You may see:

```text
UserWarning: Unable to import Axes3D
```

This warning is harmless for this project and does not affect training or evaluation.

---

# Future Improvements

Potential extensions:

* domain randomisation
* LiDAR observations
* procedural track generation
* lap timing analytics
* SAC / TD3 comparisons
* camera-based observations

---