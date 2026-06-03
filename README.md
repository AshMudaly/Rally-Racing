# Rally Racing

A PyBullet rally-driving environment trained with PPO. The car learns to navigate a sequence of checkpoints, avoid obstacles, and handle ramps. A separate convolutional network (CNN) provides camera-based obstacle detection, letting a privileged-observation policy be fine-tuned to drive from vision.

There are **two ways to operate this project**:

- **GUI operation** — a control panel that exposes every setting and runs every pipeline stage with buttons, no terminal required. Best for day-to-day use and for non-technical operators. See [GUI operation](#gui-operation).
- **Manual operation** — each stage is a standalone command-line script. Best for scripting, automation, remote/headless machines, and understanding exactly what runs. See [Manual operation](#manual-operation-command-line).

The GUI is a convenience layer over the manual commands: a "Run" button builds and launches the same command you would type by hand, passing settings via `gui_config.json`. The two are interchangeable and can be mixed freely.

---

## Installation

Tested on Ubuntu 22.04 with Python 3.10.

```bash
git clone https://github.com/AshMudaly/Rally-Racing
cd Rally-Racing
pip install -r requirements.txt
pip install -e .
```

The `-e .` installs `simple_driving` as an editable package, so the registered gym environments (`RallyDriving-v0`, `VisionRallyDriving-v0`) become importable. **This step is required** — without it, the scripts cannot import the environment.

> If you prefer the GUI's guided setup (creates a venv and installs everything for you), skip the manual install above and see [First-time setup](#first-time-setup-one-time) under GUI operation.

---

## GUI operation

The control panel (`src/control.py`) is a single Tkinter window — pure Python standard library, no extra GUI dependencies. It lets any user who has completed installation operate the whole project without editing source.

### Single use operation (control.py)

For single time users, it is recomended to operate the GUI control panel directly through python. this can be accomplished after installation. To use this as such, either run the ```bash control.py``` script through VSC or depending on virtual environment use;

```bash
/path/to/venv/bin/python3 src/control.py
```
Or

```bash
python3 src/control.py
```
From the Ralley-Racing Root driectory.

### First-time app setup (.desktop installation)

The GUI runs each pipeline stage using a Python virtual environment that has the project dependencies installed. Setup is three one-time steps; afterwards you launch with a double-click (or one command).

**Step 1 — make the launcher scripts executable.**
From the project root:

```bash
cd /path/to/Rally-Racing
chmod +x setup.sh launch.sh install_desktop.sh
```

**Step 2 — tell the launcher where your venv is.**
Open `launcher.conf` and replace the placeholder with the absolute path to your venv (an existing one, or where you want `setup.sh` to create it):

```
VENV_PATH=/home/you/Rally-Racing/venv
```

This is the one value the launcher cannot guess, so it must be set before anything else works. The project and the venv do **not** have to live in the same place — a venv elsewhere on disk is fine, as long as this path points at it.

**Step 3 — run setup, then register the launcher.**

```bash
./setup.sh
./install_desktop.sh
```

`setup.sh` creates the venv if it is missing, installs `requirements.txt` and the editable `simple_driving` package into it, and verifies the gym environments import. If your venv already has everything, it simply confirms and exits. `install_desktop.sh` writes a ready-to-use launcher (with your real paths filled in) into your applications menu at `~/.local/share/applications/rally-racing.desktop` and copies it to your Desktop.

> **Note on the `.desktop` files.** The `Rally-Racing.desktop` file in the project folder is a *template* — it keeps `/ABSOLUTE/PATH/TO/...` placeholders on purpose and is **not** the file you launch. Double-clicking the template opens a text editor, which is expected. The file you actually launch is the installed copy that `install_desktop.sh` writes into your applications menu.

### Launching the GUI

After setup, launch in any of these ways:

- **From the applications menu (recommended):** press the **Super** key, type "Rally", and open *Rally-Racing Control Panel*. The menu entry needs no extra permissions, so this is the most reliable method.
- **From the Desktop icon:** double-click it. The first time, your desktop may ask you to trust it — right-click the icon and choose **Allow Launching** (GNOME), then double-click.
- **From a terminal:**

  ```bash
  ./launch.sh
  ```

  The launcher reads `launcher.conf`, checks the environment is ready (and offers to run `setup.sh` if not), then starts the GUI using the venv's Python — so every training run it spawns inherits the correct interpreter and packages.


### Troubleshooting the launcher

- **Double-clicking the project's `Rally-Racing.desktop` opens a text editor.** That's the template, not the installed launcher — launch from the applications menu (Super → "Rally") instead. See the note above.
- **The installed launcher still has `/ABSOLUTE/PATH/TO` in it.** The path substitution didn't run. Either re-run `./install_desktop.sh`, or write the file directly:

  ```bash
  cat > ~/.local/share/applications/rally-racing.desktop << 'EOF'
  [Desktop Entry]
  Type=Application
  Name=Rally-Racing Control Panel
  Exec=/path/to/Rally-Racing/launch.sh
  Path=/path/to/Rally-Racing
  Terminal=false
  Categories=Science;Education;Development;
  EOF
  chmod +x ~/.local/share/applications/rally-racing.desktop
  update-desktop-database ~/.local/share/applications 2>/dev/null
  ```

- **`ModuleNotFoundError: No module named 'simple_driving'` (or `gymnasium`, `torch`).** The venv in `launcher.conf` doesn't have the dependencies. Run `./setup.sh`, or install manually into that venv with `/path/to/venv/bin/python3 -m pip install -r requirements.txt && /path/to/venv/bin/python3 -m pip install -e .`. The interpreter the GUI uses is whatever `launcher.conf` points at, so the packages must be installed *there*.
- **`./launch.sh` works but the icon doesn't.** The app is fine; it's purely desktop registration — use the applications-menu launch, or the direct `cat >` fix above.

### Using the panel

The window is split into a settings area (left) and a run area (right).

**Settings tabs (left):**

- **Training** — total timesteps, parallel environments, scenario, and PPO hyperparameters (learning rate, batch size, entropy coefficient, network architecture, device). Also Weights & Biases toggles.
- **Reward weights** — every coefficient in `RewardConfig`, editable live. Changes are written to `gui_config.json` and picked up on the next run.
- **Vision** — dataset collection size and seed, CNN training epochs/batch/learning rate, and the vision fine-tune timestep budget.
- **Evaluation** — which scenarios to evaluate, episodes per scenario, whether to open the simulator window, and the model file to load.

**Run area (right):**

- **Run & Console** — each pipeline stage has its own *Run* button. Tick several stages and press *Run selected in succession* to chain them (the chain stops if any stage fails). Live output streams to the console; *Stop* terminates the running stage.
- **Training Metrics** — live plots of evaluation reward, episode length, and losses read directly from the local TensorBoard logs (no W&B login needed). An *Open in W&B* button opens the exact run online when one was recorded.

**Bottom bar:**

- **System Guide** — opens the full HTML system guide (`docs/system_guide.html`) in your default browser, including the function reference and an operation walkthrough.

Settings are persisted to `gui_config.json`, which every training script reads at runtime. Save your settings before running, or use a *Run* button (which saves automatically first).

### A first run, end to end

A concrete walkthrough once the GUI is open:

1. **Pick a scenario.** On the **Training** tab, set *Scenario* to `circuit_easy` (the simplest — no cones or ramps). Leave the other defaults.
2. **(Optional) Set a budget.** Lower *Total timesteps* (e.g. 50000) for a quick first run; the default 800000 is a full run.
3. **Train.** On the **Run & Console** tab, click *Run* next to *Train (privileged)*. Output streams to the console. Switch to the **Training Metrics** tab to watch the eval-reward curve climb live.
4. **Watch it drive.** When training finishes (a `best_model.zip` now exists under `gen6_models/circuit_easy/privileged/best/`), go to the **Evaluation** tab, ensure *Show simulation* is ticked, then run *Test (privileged)*. A PyBullet window opens and the car drives the track.
5. **Go further.** Repeat for `circuit_medium`, `circuit_hard`, then `circuit_difficult` — each warm-starts from the previous automatically. To run the whole chain unattended, tick the training stages and use *Run selected in succession*.

For a deeper explanation of any stage, reward term, or observation, click **System Guide** at the bottom of the window.

---

## Manual operation (command line)

Every stage is a standalone CLI script. The `--scenario` flag is required for training. These are exactly the commands the GUI runs under the hood.

### Privileged-observation training

```bash
# Train circuit_easy from scratch
python3 src/train.py --scenario circuit_easy

# Train circuit_medium, warm-starting from circuit_easy's best model
python3 src/train.py --scenario circuit_medium

# Train circuit_difficult, 800k steps (default), warm-starting from circuit_hard
python3 src/train.py --scenario circuit_difficult

# Reduced budget, no wandb logging
python3 src/train.py --scenario circuit_hard --timesteps 150000 --no-wandb

# Ignore the warm-start chain and train from scratch
python3 src/train.py --scenario circuit_hard --fresh
```

Output goes to `gen6_models/<scenario>/privileged/` (best model under `best/best_model.zip`, final under `ppo_rally_final.zip`). The default budget is 800,000 timesteps and the default scenario is `circuit_easy`.

### Vision fine-tune

```bash
# Fine-tune vision circuit_medium from privileged circuit_medium (100k steps, default)
python3 src/train_vision.py --scenario circuit_medium

# Longer fine-tune
python3 src/train_vision.py --scenario circuit_hard --timesteps 200000

# Without wandb
python3 src/train_vision.py --scenario circuit_difficult --no-wandb
```

Output goes to `gen6_models/<scenario>/vision/`.

Vision training requires that the matching privileged model already exists. Running `--scenario circuit_medium` without first training privileged `circuit_medium` exits with an instructive error. Vision fine-tuning is only meaningful where the scenario contains cones for the CNN to detect — i.e. `circuit_medium` and up; `circuit_easy` has no cones, so the CNN channels stay inactive.

### Evaluation

Both test scripts run a saved model and report per-episode reward, step count, and checkpoint completion. Use `--no-render` for headless evaluation (fast, terminal only) or omit it to watch the car drive in a PyBullet window.

`test.py` selects a model automatically per scenario — it loads `gen6_models/<scenario>/privileged/best/best_model.zip` for each scenario unless you override with `--model`. With no arguments it evaluates all four circuits, each against its own best model.

```bash
# Privileged: evaluate every circuit, each against its own best model, one episode each
python3 src/test.py

# Privileged: watch one episode of a single scenario
python3 src/test.py --scenarios circuit_hard

# Privileged: headless eval, 20 episodes
python3 src/test.py --scenarios circuit_difficult --episodes 20 --no-render

# Privileged: cross-scenario check — run one model on several circuits
python3 src/test.py --model gen6_models/circuit_difficult/privileged/best/best_model.zip \
    --scenarios circuit_easy circuit_medium circuit_hard circuit_difficult --episodes 10 --no-render

# Vision: watch one episode
python3 src/test_vision.py --ppo gen6_models/circuit_hard/vision/best/best_model.zip --scenarios circuit_hard

# Vision: headless eval, 20 episodes
python3 src/test_vision.py --ppo gen6_models/circuit_difficult/vision/best/best_model.zip --scenarios circuit_difficult --episodes 20 --no-render
```

Vision eval also reports a `vision_active` percentage — the fraction of frames where the CNN predicted an obstacle was visible. Useful as a sanity check that the CNN is engaging when expected.

---

## Project layout

```
Rally-Racing/
├── simple_driving/         # gym environments + assets
│   ├── envs/
│   │   ├── rally_driving_env.py    # privileged obs (ground-truth obstacle position)
│   │   └── vision_rally_env.py     # vision obs (CNN-replaced obstacle channels)
│   └── resources/                  # URDF files, meshes
├── src/
│   ├── control.py                  # GUI control panel (entry point for GUI operation)
│   ├── config.py                   # shared config layer (defaults + gui_config.json)
│   ├── tb_reader.py                # dependency-free TensorBoard log reader (GUI metrics)
│   ├── metrics_panel.py            # in-GUI live metrics plotting + Open in W&B
│   ├── train.py                    # train PPO with privileged observation
│   ├── train_vision.py             # fine-tune PPO with vision observation
│   ├── test.py                     # evaluate / watch a privileged model
│   ├── test_vision.py              # evaluate / watch a vision model
│   └── reward.py                   # reward function
├── vision/
│   ├── model.py                    # ObstacleCNN architecture
│   ├── cnn_obstacle.pt             # trained CNN weights
│   └── train_cnn.py                # CNN training script
├── docs/
│   ├── system_guide.tex            # editable source for the system guide
│   ├── system_guide.html           # pre-built guide opened by the GUI button
│   └── system_guide.css            # guide stylesheet
├── gen6_models/                    # trained policies (see "Model layout")
├── launcher.conf                   # venv path for the launcher (edit this)
├── setup.sh                        # one-time environment setup
├── launch.sh                       # everyday GUI launcher
├── install_desktop.sh              # registers the double-click launcher
└── Rally-Racing.desktop            # Linux application launcher
```

## Model layout

Trained models live under `gen6_models/<scenario>/<observation>/`:

```
gen6_models/
├── circuit_easy/privileged/best/best_model.zip
├── circuit_medium/privileged/best/best_model.zip
├── circuit_medium/vision/best/best_model.zip       # optional
├── circuit_hard/privileged/best/best_model.zip
├── circuit_hard/vision/best/best_model.zip          # optional
├── circuit_difficult/privileged/best/best_model.zip
└── circuit_difficult/vision/best/best_model.zip     # optional
```

The `best/best_model.zip` is what eval / inference uses — the snapshot with the highest evaluation reward during training, saved by `EvalCallback`. A `ppo_rally_final.zip` (privileged) or `ppo_vision_final.zip` (vision) also sits in the scenario directory; that's the *final* state when training stopped, which can be worse than the best snapshot if the policy regressed late.

The generation prefix (`gen6`) is set by the `GENERATION` constant in `train.py` / `train_vision.py` — bump it when starting a fresh generation so new runs don't overwrite old models.

## Curriculum

Training scenarios build on each other. Each scenario is a strict superset of
the previous one, so the warm-start chain stays valid unchanged.

| Scenario          | Cones          | Ramps  |
|-------------------|----------------|--------|
| circuit_easy      | –              | –      |
| circuit_medium    | 4 (SW cluster) | –      |
| circuit_hard      | 4 (SW cluster) | all 4  |
| circuit_difficult | all 8          | all 4  |

**Progression rationale.** `circuit_medium` introduces one new element (cones),
`circuit_hard` introduces the second (ramps), and `circuit_difficult` scales up
cone density. Because each phase strictly contains the previous one, the
warm-start chain in `train.py`
(`circuit_easy → circuit_medium → circuit_hard → circuit_difficult`) remains
valid without modification: a policy trained on a simpler circuit is always a
sensible starting point for the next.

Privileged policies warm-start along that chain. Vision policies for a given
scenario warm-start from that scenario's *privileged* model (not the previous
scenario's vision model); the training scripts resolve the warm-start path
automatically from `--scenario`. Note that `circuit_easy` has no cones, so
vision fine-tuning is only meaningful from `circuit_medium` onward.

## Logging

Training writes TensorBoard scalars as flat event files directly in `logs/`, with the run name embedded in the filename (`events.out.tfevents.<time>.<host>.<pid>.<gen>_<scenario>_<observation>`). Auxiliary files that need a directory — the eval `evaluations.npz` and the `Monitor` CSV — go in a per-run `logs/<gen>_<scenario>_<observation>_eval/` folder. If wandb is enabled (default), the same scalars also sync to the configured wandb project, and the run URL is recorded to `logs/<run>_wandb_url.txt` so the GUI's *Open in W&B* button can jump to it. The GUI's *Training Metrics* tab reads the local event files directly and plots them live. Disable wandb per-run with `--no-wandb` (or the toggle in the GUI).

## Known behaviour

**The hardest circuit isn't perfectly reliable.** On `circuit_difficult` (all 8 cones + all 4 ramps), expect occasional catastrophic episodes — a cone or ramp collision in an awkward configuration — amid mostly clean laps. The policy generalizes but isn't perfect, and reward variance is highest here.

**`circuit_easy` and `circuit_medium` evaluate deterministically.** These scenarios have no ramps and fixed obstacle placement, so with `deterministic=True` a policy produces identical rollouts across episodes. Don't be alarmed by all-identical eval rewards on them; this is expected. `circuit_hard` and `circuit_difficult` apply per-reset obstacle spawn jitter, so their episodes differ across resets even with a deterministic policy.

**Vision policies perform somewhat worse than privileged.** The CNN-derived observation is noisier than ground truth, so expect a modest drop in mean reward. Large gaps (e.g. vision being negative while privileged is strongly positive) suggest the CNN isn't engaging properly — check the `vision_active` number from `test_vision.py`. Vision only matters where cones are present (`circuit_medium` and up).

## System guide

A full system guide — function reference plus an operation walkthrough — ships pre-built at `docs/system_guide.html` and opens from the GUI's *System Guide* button (or directly in any browser). The guide renders its mathematics as native MathML, so it works offline with no toolchain. To regenerate it after editing `docs/system_guide.tex`:

```bash
pandoc docs/system_guide.tex -s --toc --mathml -o docs/system_guide.html
```