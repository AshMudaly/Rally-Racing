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
git clone <repo-url>
cd Rally-Racing
pip install -r requirements.txt
pip install -e .
```

The `-e .` installs `simple_driving` as an editable package, so the registered gym environments (`RallyDriving-v0`, `VisionRallyDriving-v0`) become importable. **This step is required** — without it, the scripts cannot import the environment.

> If you prefer the GUI's guided setup (creates a venv and installs everything for you), skip the manual install above and see [First-time setup](#first-time-setup-one-time) under GUI operation.

---

## GUI operation

The control panel (`src/control.py`) is a single Tkinter window — pure Python standard library, no extra GUI dependencies. It lets any user who has completed installation operate the whole project without editing source.

### First-time setup (one time) - Desktop Application

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

You can also bypass the launcher entirely and start the GUI with any Python that already has the dependencies:

```bash
/path/to/venv/bin/python3 src/control.py
```

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

- **Training** — total timesteps, parallel environments, scenario/phase, and PPO hyperparameters (learning rate, batch size, entropy coefficient, network architecture, device). Also Weights & Biases toggles.
- **Reward weights** — every coefficient in `RewardConfig`, editable live. Changes are written to `gui_config.json` and picked up on the next run.
- **Vision** — dataset collection size and seed, CNN training epochs/batch/learning rate, and the vision fine-tune timestep budget.
- **Evaluation** — which scenarios to evaluate, episodes per scenario, whether to open the simulator window, and the model file to load.

**Run area (right):**

- **Run & Console** — each pipeline stage has its own *Run* button. Tick several stages and press *Run selected in succession* to chain them (the chain stops if any stage fails). Live output streams to the console; *Stop* terminates the running stage.
- **Training Metrics** — live plots of episode reward, episode length, and losses read directly from the local TensorBoard logs (no W&B login needed). An *Open in W&B* button opens the exact run online when one was recorded.

**Bottom bar:**

- **System Guide** — opens the full HTML system guide (`docs/system_guide.html`) in your default browser, including the function reference and an operation walkthrough.

Settings are persisted to `gui_config.json`, which every training script reads at runtime. Save your settings before running, or use a *Run* button (which saves automatically first).

### A first run, end to end

A concrete walkthrough once the GUI is open:

1. **Pick a scenario.** On the **Training** tab, set *Scenario* to `phase1` (the simplest — no obstacles or ramps). Leave the other defaults.
2. **(Optional) Set a budget.** Lower *Total timesteps* (e.g. 50000) for a quick first run; the default 300000 is a full run.
3. **Train.** On the **Run & Console** tab, click *Run* next to *Train (privileged)*. Output streams to the console. Switch to the **Training Metrics** tab to watch the episode-reward curve climb live.
4. **Watch it drive.** When training finishes (a `best_model.zip` now exists under `gen2_models/phase1/privileged/best/`), go to the **Evaluation** tab, ensure *Show simulation* is ticked, then run *Test (privileged)*. A PyBullet window opens and the car drives the track.
5. **Go further.** Repeat for `phase2` then `phase3` — each warm-starts from the previous automatically. To run the whole chain unattended, tick `phase1`/`phase2`/`phase3` training stages and use *Run selected in succession*.

For a deeper explanation of any stage, reward term, or observation, click **System Guide** at the bottom of the window.

---

## Manual operation (command line)

Every stage is a standalone CLI script. The `--scenario` flag is required for training. These are exactly the commands the GUI runs under the hood.

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

Vision training requires that the matching privileged model already exists. Running `--scenario phase3` without first training privileged phase3 exits with an instructive error. Vision is only valid for `phase2`, `phase3`, and `custom` — `phase1` has no obstacles, so the CNN would have nothing to do.

### Evaluation

Both test scripts run a saved model and report per-episode reward, step count, and checkpoint completion. Use `--no-render` for headless evaluation (fast, terminal only) or omit it to watch the car drive in a PyBullet window.

```bash
# Privileged: watch one episode of phase3
python3 src/test.py --model gen2_models/phase3/privileged/best/best_model.zip --scenarios phase3

# Privileged: headless eval, 20 episodes
python3 src/test.py --model gen2_models/phase3/privileged/best/best_model.zip --scenarios phase3 --episodes 20 --no-render

# Privileged: evaluate on multiple scenarios in one run
python3 src/test.py --model gen2_models/phase3/privileged/best/best_model.zip --scenarios phase1 phase2 phase3 --episodes 10 --no-render

# Vision: watch one episode
python3 src/test_vision.py --ppo gen2_models/phase3/vision/best/best_model.zip --scenarios phase3

# Vision: headless eval, 20 episodes
python3 src/test_vision.py --ppo gen2_models/phase3/vision/best/best_model.zip --scenarios phase3 --episodes 20 --no-render
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
├── gen2_models/                    # trained policies (see "Model layout")
├── launcher.conf                   # venv path for the launcher (edit this)
├── setup.sh                        # one-time environment setup
├── launch.sh                       # everyday GUI launcher
├── install_desktop.sh              # registers the double-click launcher
└── Rally-Racing.desktop            # Linux application launcher
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

The `best/best_model.zip` is what eval / inference uses — the snapshot with the highest evaluation reward during training, saved by `EvalCallback`. A `ppo_*_final.zip` may also exist next to it; that's the *final* state when training stopped, which can be worse than the best snapshot if the policy regressed late.

## Curriculum

Training scenarios build on each other:

| Scenario | Track contents |
|----------|----------------|
| phase1 | 6 checkpoints, no obstacles, no ramps |
| phase2 | 6 checkpoints + 4 obstacles |
| phase3 | 6 checkpoints + 4 obstacles + 2 ramps |
| custom | bespoke track with its own checkpoint/obstacle/ramp lists |

Privileged policies warm-start along the chain: `phase1 → phase2 → phase3 → custom`. Vision policies for a given scenario warm-start from that scenario's *privileged* model, not the previous scenario's vision model. The training scripts handle the warm-start lookup automatically based on `--scenario`.

## Logging

Training writes TensorBoard scalars to `logs/<gen>_<scenario>_<observation>/` and, if wandb is enabled (default), syncs them to the configured wandb project. The GUI's *Training Metrics* tab reads those local logs directly and plots them live. Disable wandb per-run with `--no-wandb` (or the toggle in the GUI).

## Known behaviour

**Phase3 privileged policy is ~90% reliable.** Over 20 episodes you'll typically see 18 clean laps (+500 ish) and 1–2 catastrophic failures (-2000 or worse, ramp collisions or obstacle collisions in awkward configurations). This is real — the policy generalizes but isn't perfect.

**Phase1 evaluation is deterministic.** Phase1 has no obstacles or ramps and the car spawns at a fixed position, so a deterministic policy produces identical episodes. Don't be alarmed by all-identical eval rewards on phase1; this is expected.

**Vision policies perform somewhat worse than privileged.** The CNN-derived observation is noisier than ground truth, so expect a modest drop in mean reward. Large gaps (e.g. vision being negative while privileged is +500) suggest the CNN isn't engaging properly — check the `vision_active` number from `test_vision.py`.

## System guide

A full system guide — function reference plus an operation walkthrough — ships pre-built at `docs/system_guide.html` and opens from the GUI's *System Guide* button (or directly in any browser). The guide renders its mathematics as native MathML, so it works offline with no toolchain. To regenerate it after editing `docs/system_guide.tex`:

```bash
pandoc docs/system_guide.tex -s --toc --mathml -o docs/system_guide.html
```
