# Rally Racing: PPO Rally Car on a Designated Track

A PPO-trained autonomous rally car for *41118 Artificial Intelligence in Robotics*. The car drives a track defined by a sequence of checkpoints, avoids obstacles, and can optionally take jumps to finish laps faster.

**Stack:** PyBullet (physics) + Gymnasium (RL interface) + Stable-Baselines3 (PPO).

## Project Structure
```
Rally-Racing/
├── simple_driving/                    # Installable Python package
│   ├── __init__.py                    # Registers Gym environments
│   ├── envs/
│   │   ├── simple_driving_env.py      # Base env: single goal, single obstacle
│   │   └── rally_driving_env.py       # Rally env: checkpoints, obstacles, ramps
│   └── resources/
│       ├── car.py                     # Car kinematics + URDF loader
│       ├── plane.py                   # Ground plane
│       ├── goal.py                    # Goal/checkpoint marker
│       ├── obstacle.py                # Static red cylinder
│       ├── ramp.py                    # Orange ramp for jumps
│       └── *.urdf                     # PyBullet model files
├── src/
│   ├── reward.py                      # Reward function and tunable weights
│   ├── observation.py                 # Observation callback (base env only)
│   ├── train.py                       # PPO training
│   └── test.py                        # Evaluation with rendering
├── models/                            # Trained weights (gitignored)
├── logs/                              # TensorBoard logs (gitignored)
├── setup.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation
Python 3.10+ required. No ROS or Gazebo.

```bash
cd ~/41118_ws/project/Rally-Racing
pip install -r requirements.txt
pip install -e .
```

The `-e .` installs `simple_driving` in editable mode, so code changes apply without reinstalling.

Verify the install:
```bash
python3 -c "import simple_driving; import gymnasium as gym; print([e for e in gym.envs.registry.keys() if 'Driving' in e])"
# Expected: ['SimpleDriving-v0', 'RallyDriving-v0']
```

## Scenarios
Select via `env.reset(options={"scenario": ...})`:

| Scenario | Checkpoints | Obstacles | Ramps | Use Case |
|----------|-------------|-----------|-------|----------|
| `phase1` | 6 default   | none      | none  | Learn racing line on bare track |
| `phase2` | 6 default   | 4 cones   | none  | Add obstacle avoidance |
| `phase3` | 6 default   | 4 cones   | 2 ramps | Choose between safe path or jumps |

Override the checkpoint course at reset:
```python
env.reset(options={"scenario": "phase3", "checkpoints": [(5, 5), (10, 0), ...]})
```

To change the default checkpoints, obstacle positions, or ramp positions, edit the class constants at the top of `simple_driving/envs/rally_driving_env.py`:
```python
CHECKPOINTS       = [(16, 16), (16, 2), ...]
OBSTACLE_HOMES    = [(8, 8), (0, 8), ...]
RAMP_POSITIONS    = [(10, 9, math.radians(-30)), ...]
```

## Training

### Basic training
```bash
cd src
python3 train.py
```
Trains using 8 parallel environments. Saves `models/resume.zip` atomically every 10k steps for crash recovery, and the best-evaluated model to `models/best/best_model.zip`.

### Configuration
Edit the top of `src/train.py`:
```python
TOTAL_TIMESTEPS = 300_000     # steps per phase (increase for longer runs)
N_ENVS          = 8           # parallel envs (reduce if low on RAM)
SCENARIO        = "phase1"    # which scenario to train on
LOAD_PREVIOUS   = True        # resume from models/resume.zip if it exists
RESET_TIMESTEPS = False       # True = each run shows separately in TensorBoard

# Network size — expanded from SB3 default [64, 64] for better value estimation
PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 512,
    batch_size    = 256,
    ent_coef      = 0.01,
    device        = "cpu",
    policy_kwargs = dict(net_arch=[256, 256]),
)
```

### Curriculum learning (recommended)
Training phase3 from scratch is hard — too many things to learn at once. Train progressively:

1. Set `SCENARIO = "phase1"`, `LOAD_PREVIOUS = False`, run for ~300k steps until reward plateaus
2. Change to `SCENARIO = "phase2"`, `LOAD_PREVIOUS = True`, train another 200k
3. Change to `SCENARIO = "phase3"`, `LOAD_PREVIOUS = True`, train another 200k

Each phase starts from the weights of the previous one. Expect a brief reward dip when switching phases — this is normal as the policy adapts to the harder environment.

### Starting completely fresh
```bash
rm -rf ../models/*.zip ../models/best/*.zip ../logs/*
python3 train.py
```

### Monitoring with TensorBoard
In a separate terminal:
```bash
tensorboard --logdir ~/41118_ws/project/Rally-Racing/logs
```
Open http://localhost:6006. Key metrics:
- `rollout/ep_rew_mean` — average episode reward (should trend up)
- `rollout/ep_len_mean` — average episode length (should trend down as agent gets faster)
- `eval/mean_reward` — score on the eval env (updated every 10k steps)
- `train/explained_variance` — how well the value network predicts returns; target >0.7

## Evaluation

Run the best saved model on all three scenarios with the PyBullet GUI:
```bash
cd src
python3 test.py
```

Options:
```bash
python3 test.py --model ../models/ppo_rally_final.zip
python3 test.py --scenarios phase3
python3 test.py --no-render            # headless, faster
python3 test.py --scenarios phase1 phase2 phase3
```

## Reward Function

All weights live in `src/reward.py` under `RewardConfig`. Edit them to shape behaviour:

| Component | Sign | Default | Purpose |
|-----------|------|---------|---------|
| `GOAL_REWARD`        | + | +100 | Hitting a checkpoint |
| `STEP_PENALTY`       | − | −0.5 | Per-step cost (encourages speed) |
| `PROGRESS_SCALE`     | + | 3.0  | Multiplier on closing distance to goal |
| `YAW_JERK_PENALTY`   | − | −5   | Per radian of yaw *rate change* — penalises oscillation, not cornering |
| `ROLL_DELTA_PENALTY` | − | −15  | Penalises chassis tilt |
| `PITCH_DELTA_PENALTY`| − | −4   | Penalises front-back tilt |
| `OBSTACLE_PENALTY`   | − | −100 | Within `MIN_SAFE_DISTANCE` of any obstacle |
| `REPULSE_SCALE`      | − | 10   | Soft penalty inside `REPULSE_RADIUS` |
| `OUT_OF_BOUNDS`      | − | −50  | Outside `WORLD_BOUNDARY` |
| `AIRBORNE_BONUS`     | ± | +1   | Per step while pitched up and making progress (phase3) |

### Yaw jerk vs yaw delta
The swerve penalty targets the *rate of change* of heading (jerk), not the heading change itself. This means smooth cornering is unpunished — the agent can turn freely — but rapid oscillation back and forth is penalised each step. This is a more precise signal than a raw heading-change penalty.

### Tuning the jump tradeoff
The `AIRBORNE_BONUS` controls whether the agent treats ramps as opportunities or hazards:
- **Positive** (default +1) — agent learns to take jumps for shorter laps
- **Zero** — agent ignores ramps, prefers ground-level path
- **Negative** — agent actively avoids ramps

## Errors and Troubleshooting

### `ModuleNotFoundError: No module named 'simple_driving'`
The package isn't installed. Run `pip install -e .` from the project root.

### `Cannot find simplecar.urdf`
The URDF files must sit in `simple_driving/resources/` alongside `car.py`. Confirm:
```bash
ls simple_driving/resources/*.urdf
# Should show: simplecar.urdf  simplegoal.urdf  simpleplane.urdf
```

### Resume Checkpoint Corrupt
```
zipfile.BadZipFile: Overlapped entries: 'policy.optimizer.pth' (possible zip bomb)
```
The training script handles this automatically — moves the bad file to `resume.zip.broken` and starts fresh. To manually recover, use the best model:
```bash
cp models/best/best_model.zip models/resume.zip
python3 -c "import zipfile; print(zipfile.ZipFile('models/resume.zip').testzip())"
# Prints None if the zip is valid
```

### SubprocVecEnv Hangs on Startup
The `spawn` start method has issues on some systems. Drop `N_ENVS = 1` in `train.py` to confirm a single-process run works, then increase. If hangs persist, change `SubprocVecEnv` to `DummyVecEnv` in `make_train_env()`.

### Agent Won't Move
Early in training the agent often learns to sit still to avoid the obstacle penalty. The step penalty is the counter-incentive. If after 100k+ steps episodes still time out with deeply negative rewards:
- Increase `STEP_PENALTY` magnitude (more negative)
- Increase `PROGRESS_SCALE`

### Agent Drives in Circles
Usually means the yaw jerk penalty is too low relative to the progress reward. Increase `YAW_JERK_PENALTY` magnitude (from −5 to −10 or more).

### Agent Crashes Into Every Obstacle
The obstacle repulsion field hasn't built a strong enough gradient. Try:
- Increase `REPULSE_RADIUS` so the penalty kicks in earlier
- Increase `REPULSE_SCALE`
- Increase the entropy coefficient in `train.py` (`ent_coef=0.05`) to encourage exploring evasive actions