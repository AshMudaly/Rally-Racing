# Rally Racing

A PyBullet rally-driving environment trained with PPO. The car learns to navigate a sequence of checkpoints, avoid obstacles, and handle ramps. A separate convolutional network (CNN) provides camera-based obstacle detection, letting a privileged-observation policy be fine-tuned to drive from vision.

## Installation

Tested on Ubuntu 22.04 with Python 3.10.

```bash
git clone <repo-url>
cd Rally-Racing
pip install -r requirements.txt
pip install -e .
```

The `-e .` installs `simple_driving` as an editable package, so the registered gym environments (`RallyDriving-v0`, `VisionRallyDriving-v0`) become importable.

## Project layout

```
Rally-Racing/
├── simple_driving/         # gym environments + assets
│   ├── envs/
│   │   ├── rally_driving_env.py    # privileged obs (ground-truth obstacle position)
│   │   └── vision_rally_env.py     # vision obs (CNN-replaced obstacle channels)
│   └── resources/                  # URDF files, meshes
├── src/
│   ├── train.py                    # train PPO with privileged observation
│   ├── train_vision.py             # fine-tune PPO with vision observation
│   ├── test.py                     # evaluate / watch a privileged model
│   ├── test_vision.py              # evaluate / watch a vision model
│   └── reward.py                   # reward function
├── vision/
│   ├── model.py                    # ObstacleCNN architecture
│   ├── cnn_obstacle.pt             # trained CNN weights
│   └── train_cnn.py                # CNN training script (not normally needed)
└── gen2_models/                    # trained policies (see "Model layout" below)
```

## Model layout

Trained models live under `gen2_models/<scenario>/<observation>/`:

```
gen2_models/
├── phase1/privileged/best/best_model.zip
├── phase2/privileged/best/best_model.zip
├── phase2/vision/best/best_model.zip        # optional
├── phase3/privileged/best/best_model.zip
└── phase3/vision/best/best_model.zip
```

The `best/best_model.zip` is what eval / inference uses — it's the snapshot with the highest evaluation reward during training, saved by `EvalCallback`. A `ppo_*_final.zip` may also exist next to it; that's the *final* state when training stopped, which can be worse than the best snapshot if the policy regressed late.

## Curriculum

Training scenarios build on each other:

| Scenario | Track contents |
|----------|----------------|
| phase1 | 6 checkpoints, no obstacles, no ramps |
| phase2 | 6 checkpoints + 4 obstacles |
| phase3 | 6 checkpoints + 4 obstacles + 2 ramps |
| custom | bespoke track with its own checkpoint/obstacle/ramp lists |

Privileged policies warm-start along the chain: `phase1 → phase2 → phase3 → custom`. Vision policies for a given scenario warm-start from that scenario's *privileged* model, not the previous scenario's vision model.

The training scripts handle the warm-start lookup automatically based on `--scenario`.

## Training

All training scripts are CLI-driven. The `--scenario` flag is required.

### Privileged-observation training

```bash
# Train phase1 from scratch
python3 src/train.py --scenario phase1

# Train phase2, warm-starting from phase1's best model
python3 src/train.py --scenario phase2

# Train phase3, 300k steps (default), warm-starting from phase2
python3 src/train.py --scenario phase3

# Reduced budget, no wandb logging
python3 src/train.py --scenario phase3 --timesteps 150000 --no-wandb

# Ignore the warm-start chain and train from scratch
python3 src/train.py --scenario phase3 --fresh
```

Output goes to `gen2_models/<scenario>/privileged/`.

### Vision fine-tune

```bash
# Fine-tune vision phase3 from privileged phase3 (100k steps, default)
python3 src/train_vision.py --scenario phase3

# Longer fine-tune
python3 src/train_vision.py --scenario phase3 --timesteps 200000

# Without wandb
python3 src/train_vision.py --scenario phase3 --no-wandb
```

Output goes to `gen2_models/<scenario>/vision/`.

Vision training requires that the matching privileged model already exists. If you try to run `--scenario phase3` without first training privileged phase3, the script will exit with an instructive error.

Vision is only valid for `phase2`, `phase3`, and `custom` — `phase1` has no obstacles, so the CNN would have nothing to do.

## Evaluation

Both test scripts run a saved model and report per-episode reward, step count, and checkpoint completion. Use `--no-render` for headless evaluation (fast, terminal only) or omit it to watch the car drive in a PyBullet window.

### Privileged model

```bash
# Watch one episode of phase3
python3 src/test.py --model gen2_models/phase3/privileged/best/best_model.zip --scenarios phase3

# Headless eval, 20 episodes
python3 src/test.py --model gen2_models/phase3/privileged/best/best_model.zip --scenarios phase3 --episodes 20 --no-render

# Evaluate on multiple scenarios in one run
python3 src/test.py --model gen2_models/phase3/privileged/best/best_model.zip --scenarios phase1 phase2 phase3 --episodes 10 --no-render
```

### Vision model

```bash
# Watch one episode
python3 src/test_vision.py --ppo gen2_models/phase3/vision/best/best_model.zip --scenarios phase3

# Headless eval, 20 episodes
python3 src/test_vision.py --ppo gen2_models/phase3/vision/best/best_model.zip --scenarios phase3 --episodes 20 --no-render
```

Vision eval also reports a `vision_active` percentage — the fraction of frames where the CNN predicted an obstacle was visible. Useful sanity check that the CNN is engaging when expected.

## Known behaviour

**Phase3 privileged policy is ~90% reliable.** Over 20 episodes you'll typically see 18 clean laps (+500 ish) and 1–2 catastrophic failures (-2000 or worse, ramp collisions or obstacle collisions in awkward configurations). This is real — the policy generalizes but isn't perfect.

**Phase1 evaluation is deterministic.** Phase1 has no obstacles or ramps and the car spawns at a fixed position, so a deterministic policy produces identical episodes. Don't be alarmed by all-identical eval rewards on phase1; this is expected.

**Vision policies perform somewhat worse than privileged.** The CNN-derived observation is noisier than ground truth, so expect a modest drop in mean reward. Large gaps (e.g. vision being negative while privileged is +500) suggest the CNN isn't engaging properly — check the `vision_active` number from `test_vision.py`.

## Logging

Training writes TensorBoard scalars to `logs/<gen>_<scenario>_<observation>/` and, if wandb is enabled (default), syncs them to the configured wandb project. Disable wandb per-run with `--no-wandb`.