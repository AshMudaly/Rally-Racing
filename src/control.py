#!/usr/bin/env python3
"""
control.py — Operation & configuration GUI for the Rally-Racing project.
============================================================================

A single-file Tkinter control panel (stdlib only) that lets any user who has
followed the README installation operate the entire project without touching
the source. It:

  * exposes the major training settings — timesteps, parallel envs, scenario,
    PPO hyperparameters — and the full reward-weight table as editable fields;
  * persists them to ``gui_config.json``, which every pipeline script reads at
    runtime (see src/config.py). Nothing is hardcoded twice;
  * runs every stage of BOTH pipelines as subprocesses, either one at a time
    or chained in succession via a queue;
  * streams each subprocess's stdout/stderr live into an in-app console;
  * launches the visual PyBullet simulation in its own popout window (the
    native simulator window), keeping this GUI responsive.

Pipelines
---------
State-RL:   train  ->  test
Vision:     collect-data -> inspect-data -> train-cnn -> train-vision -> test-vision

The "Run selected in succession" button executes every ticked stage top to
bottom, stopping the chain if any stage exits non-zero.

Design notes
------------
* Subprocess model (not in-process imports): the scripts stay byte-for-byte
  runnable from the command line, and a crash in training can never take down
  the control panel. Settings reach the scripts purely through gui_config.json
  and per-stage CLI arguments.
* Popout simulator windows: PyBullet renders in its own native OpenGL window
  when a stage runs with rendering enabled. Embedding that into Tkinter is
  fragile and buys nothing, so rendered stages simply spawn their own window
  while their textual output still streams into the console here.

Run from anywhere:
    python3 src/control.py
"""

import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import ttk, filedialog, messagebox

# config.py lives alongside this file in src/.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import load_config, save_config, DEFAULTS, BASE_DIR, CONFIG_PATH  # noqa: E402

SRC_DIR    = HERE
VISION_DIR = os.path.join(BASE_DIR, "vision")
PY         = sys.executable  # use the same interpreter that launched the GUI


# ════════════════════════════════════════════════════════════════════════════
#  Stage definitions
# ════════════════════════════════════════════════════════════════════════════
# Each stage knows how to build its command line from the current GUI state.
# `build` receives the live settings dict and returns (argv_list, cwd).
# `renders` flags stages that can open a popout PyBullet window.


# ════════════════════════════════════════════════════════════════════════════
#  Embedded metrics reading + live plot panel
#  (folded in from the former metrics_reader.py / metrics_panel.py; tb_reader.py
#   is kept as a standalone dependency-free fallback parser.)
# ════════════════════════════════════════════════════════════════════════════
import tb_reader  # dependency-free fallback parser

# Prefer the authoritative TensorBoard reader when available (it is a transitive
# dependency of torch's SummaryWriter, so present on any training machine).
try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator as _EventAccumulator,
    )
    _HAVE_TB = True
except Exception:
    _HAVE_TB = False


def _mr_backend_name():
    return "tensorboard" if _HAVE_TB else "builtin"


def _mr_latest_event_file(run_dir):
    """Most-recently-modified tfevents file at or below run_dir (recursive)."""
    return tb_reader.latest_event_file(run_dir)


def _mr_read_run(run_dir):
    """{tag: [(step, wall_time, value), ...]} — EventAccumulator first, builtin
    fallback. Both verified to return identical values on real SB3 logs."""
    if _HAVE_TB:
        try:
            ev = tb_reader.latest_event_file(run_dir)
            if ev:
                acc = _EventAccumulator(ev, size_guidance={"scalars": 0})
                acc.Reload()
                series = {}
                for tag in acc.Tags().get("scalars", []):
                    series[tag] = [(p.step, p.wall_time, p.value)
                                   for p in acc.Scalars(tag)]
                if series:
                    return series
        except Exception:
            pass
    return tb_reader.read_run(run_dir)


def _mr_read_event_file(path):
    """Read scalars from a single event file (flat-layout entry point)."""
    if _HAVE_TB:
        try:
            acc = _EventAccumulator(path, size_guidance={"scalars": 0})
            acc.Reload()
            series = {}
            for tag in acc.Tags().get("scalars", []):
                series[tag] = [(p.step, p.wall_time, p.value)
                               for p in acc.Scalars(tag)]
            if series:
                return series
        except Exception:
            pass
    return tb_reader.read_scalars(path)


def _mr_discover_runs(logs_dir):
    r"""Map run-name -> newest event file path, across both layouts.

    Flat layout (current): event files sit directly in logs/ as
    ``events.out.tfevents.<time>.<host>.<pid>.<run_name>`` — the run name is the
    suffix after the pid. Legacy layout (older runs): events live in a per-run
    subdirectory ``logs/<run_name>/.../events...``. Both are merged so the panel
    lists every run regardless of when it was trained; when a name exists in
    both, the most-recently-modified event file wins.
    """
    runs = {}  # name -> (mtime, path)
    if not os.path.isdir(logs_dir):
        return {}

    # Flat event files directly in logs/.
    for name in os.listdir(logs_dir):
        full = os.path.join(logs_dir, name)
        if not os.path.isfile(full) or "tfevents" not in name:
            continue
        # events.out.tfevents.<time>.<host>.<pid>[.<run_name>]
        run = "unnamed"
        parts = name.split("tfevents.", 1)
        if len(parts) == 2:
            tail = parts[1].split(".")
            if len(tail) > 3:           # [time, host, pid, <run_name...>]
                run = ".".join(tail[3:])
        try:
            mt = os.path.getmtime(full)
        except OSError:
            continue
        if run not in runs or mt > runs[run][0]:
            runs[run] = (mt, full)

    # Legacy per-run subdirectories.
    for name in os.listdir(logs_dir):
        full = os.path.join(logs_dir, name)
        if not os.path.isdir(full):
            continue
        ev = tb_reader.latest_event_file(full)
        if ev:
            try:
                mt = os.path.getmtime(ev)
            except OSError:
                continue
            if name not in runs or mt > runs[name][0]:
                runs[name] = (mt, ev)

    return {k: v[1] for k, v in runs.items()}


TRACKED = [
    # Tags verified against real SB3 output. Episode stats live under eval/*
    # (logged by EvalCallback); rollout/* may be absent depending on setup, so
    # any tag the run does not contain is simply skipped at draw time.
    ("eval/mean_reward",     "Eval reward (mean)"),
    ("eval/mean_ep_length",  "Eval episode length (mean)"),
    ("rollout/ep_rew_mean",  "Rollout reward (mean)"),
    ("rollout/ep_len_mean",  "Rollout episode length (mean)"),
    ("train/loss",           "Total loss"),
    ("train/value_loss",     "Value loss"),
    ("train/policy_gradient_loss", "Policy-gradient loss"),
    ("train/entropy_loss",   "Entropy loss"),
    ("train/explained_variance", "Explained variance"),
    ("train/approx_kl",      "Approx. KL"),
    ("time/fps",             "Throughput (FPS)"),
]

# Palette (kept readable on the dark console theme used by the GUI).
PLOT_BG   = "#0d1117"
GRID      = "#21262d"
AXIS      = "#5a6270"
LINE      = "#4aa3ff"
TEXT      = "#c9d1d9"
SUBTEXT   = "#7d8590"


class MetricsPanel(ttk.Frame):
    r"""Live training-metrics panel embedded in the control GUI.

    Reads the scalar time series Stable-Baselines3 writes to the local
    TensorBoard event files (via :mod:`tb_reader`) and plots a selected metric
    on a native Tk canvas — no ``matplotlib`` or ``tensorboard`` dependency. A
    background timer re-reads the active run every few seconds so curves grow
    in near-real-time during training; an *Open in W&B* button hands the
    recorded run URL to the system browser for the full online dashboard.
    """

    def __init__(self, parent, base_dir, generation="gen2",
                 wandb_project_getter=None):
        super().__init__(parent, padding=6)
        self.base_dir   = base_dir
        self.generation = generation
        self._wandb_project_getter = wandb_project_getter
        self._poll_job  = None
        self._auto      = tk.BooleanVar(value=True)

        self._build()
        self.refresh_run_list()
        self._schedule_poll()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x")

        ttk.Label(top, text="Run:").pack(side="left")
        self.run_var = tk.StringVar()
        self.run_combo = ttk.Combobox(top, textvariable=self.run_var,
                                      state="readonly", width=30)
        self.run_combo.pack(side="left", padx=4)
        self.run_combo.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        ttk.Button(top, text="↻", width=3,
                   command=self.refresh_run_list).pack(side="left")
        ttk.Checkbutton(top, text="Auto-refresh", variable=self._auto
                        ).pack(side="left", padx=8)
        _backend = ("TensorBoard" if _mr_backend_name() == "tensorboard"
                    else "builtin reader")
        ttk.Label(top, text=f"reader: {_backend}",
                  foreground=SUBTEXT).pack(side="left", padx=4)
        ttk.Button(top, text="Open in W&B",
                   command=self._open_wandb).pack(side="right")

        ttk.Label(top, textvariable=self._make_metric_summary_var(),
                  foreground=SUBTEXT).pack(side="right", padx=10)

        # metric selector
        sel = ttk.Frame(self)
        sel.pack(fill="x", pady=(4, 2))
        ttk.Label(sel, text="Metric:").pack(side="left")
        self.metric_var = tk.StringVar(value=TRACKED[0][0])
        self.metric_combo = ttk.Combobox(
            sel, state="readonly", width=28,
            values=[label for _, label in TRACKED],
            textvariable=tk.StringVar(value=TRACKED[0][1]))
        self.metric_combo.current(0)
        self.metric_combo.pack(side="left", padx=4)
        self.metric_combo.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        self.status = tk.StringVar(value="No run selected.")
        ttk.Label(sel, textvariable=self.status,
                  foreground=SUBTEXT).pack(side="left", padx=10)

        self.canvas = tk.Canvas(self, bg=PLOT_BG, highlightthickness=0, height=360)
        self.canvas.pack(fill="both", expand=True, pady=(2, 0))
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # latest-values strip
        self.values = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.values, foreground=TEXT,
                  font=("Menlo", 9)).pack(anchor="w", pady=(4, 0))

    def _make_metric_summary_var(self):
        self._summary_var = tk.StringVar(value="")
        return self._summary_var

    # ── run discovery ───────────────────────────────────────────────────────
    def _logs_dir(self):
        return os.path.join(self.base_dir, "logs")

    def refresh_run_list(self):
        logs = self._logs_dir()
        # name -> event file path, across flat and legacy layouts.
        self._run_map = _mr_discover_runs(logs)
        runs = sorted(self._run_map.keys())
        self.run_combo["values"] = runs
        if runs and self.run_var.get() not in runs:
            # default to the most-recently-modified run
            newest = max(runs, key=lambda r: os.path.getmtime(self._run_map[r]))
            self.run_var.set(newest)
        elif not runs:
            self.run_var.set("")
        self.redraw()

    def current_event_file(self):
        """Resolve the selected run name to its event file path."""
        name = self.run_var.get()
        if not name or not getattr(self, "_run_map", None):
            return None
        return self._run_map.get(name)

    # ── polling ───────────────────────────────────────────────────────────────
    def _schedule_poll(self):
        self._poll_job = self.after(3000, self._poll)

    def _poll(self):
        if self._auto.get():
            # cheap: only re-list occasionally, always redraw the active run
            self.redraw()
        self._schedule_poll()

    def stop(self):
        if self._poll_job:
            self.after_cancel(self._poll_job)
            self._poll_job = None

    # ── drawing ───────────────────────────────────────────────────────────────
    def _selected_tag(self):
        idx = self.metric_combo.current()
        if idx < 0:
            idx = 0
        return TRACKED[idx][0], TRACKED[idx][1]

    def redraw(self):
        r"""Redraw the selected metric's curve on the canvas.

        Reads the active run's scalars, takes the selected series
        :math:`\{(s_i, x_i)\}` (step, value), and maps data coordinates to
        pixel coordinates by an affine fit to the padded plot rectangle:

        .. math::

            X(s) &= p_\ell + \frac{s - s_{\min}}{s_{\max}-s_{\min}}\,w \\
            Y(x) &= p_t + \Bigl(1 - \frac{x - x_{\min}}{x_{\max}-x_{\min}}\Bigr) h

        (the :math:`Y` axis is inverted because canvas :math:`y` grows
        downward). The value range is padded by 8% so the curve never touches
        the frame, and degenerate ranges (:math:`s_{\max}=s_{\min}`) are nudged
        to avoid division by zero. Called on a timer while training runs, so
        the plot grows live.
        """
        c = self.canvas
        c.delete("all")
        ev = self.current_event_file()
        if not ev:
            self.status.set("No runs found in logs/. Train something first.")
            self.values.set("")
            return

        series = _mr_read_event_file(ev)
        tag, label = self._selected_tag()
        pts = series.get(tag, [])

        W = c.winfo_width() or 600
        H = c.winfo_height() or 360
        pad_l, pad_r, pad_t, pad_b = 64, 16, 24, 36
        plot_w = max(1, W - pad_l - pad_r)
        plot_h = max(1, H - pad_t - pad_b)

        # title
        c.create_text(pad_l, 12, anchor="w", text=label, fill=TEXT,
                      font=("", 11, "bold"))

        if not pts:
            c.create_text(W // 2, H // 2,
                          text=f"No '{tag}' data yet in this run.",
                          fill=SUBTEXT)
            self._update_values(series)
            self.status.set(f"{len(series)} metrics tracked in this run.")
            return

        xs = [p[0] for p in pts]
        ys = [p[2] for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax == xmin:
            xmax = xmin + 1
        if ymax == ymin:
            ymax = ymin + 1
        yr = ymax - ymin
        ymin -= yr * 0.08
        ymax += yr * 0.08

        def sx(x): return pad_l + (x - xmin) / (xmax - xmin) * plot_w
        def sy(y): return pad_t + (1 - (y - ymin) / (ymax - ymin)) * plot_h

        # gridlines + y labels
        for k in range(5):
            gy = pad_t + plot_h * k / 4
            c.create_line(pad_l, gy, W - pad_r, gy, fill=GRID)
            val = ymax - (ymax - ymin) * k / 4
            c.create_text(pad_l - 6, gy, anchor="e",
                          text=f"{val:.3g}", fill=SUBTEXT, font=("Menlo", 8))
        # x labels
        for k in range(5):
            gx = pad_l + plot_w * k / 4
            xv = xmin + (xmax - xmin) * k / 4
            c.create_text(gx, H - pad_b + 14, anchor="n",
                          text=f"{int(xv):,}", fill=SUBTEXT, font=("Menlo", 8))
        c.create_line(pad_l, pad_t, pad_l, pad_t + plot_h, fill=AXIS)
        c.create_line(pad_l, pad_t + plot_h, W - pad_r, pad_t + plot_h, fill=AXIS)
        c.create_text((pad_l + W) // 2, H - 6, text="timestep",
                      fill=SUBTEXT, font=("Menlo", 8))

        # the curve
        coords = []
        for x, y in zip(xs, ys):
            coords += [sx(x), sy(y)]
        if len(coords) >= 4:
            c.create_line(*coords, fill=LINE, width=2, smooth=True)
        # last point marker
        lx, ly = sx(xs[-1]), sy(ys[-1])
        c.create_oval(lx - 3, ly - 3, lx + 3, ly + 3, fill=LINE, outline="")
        c.create_text(lx, ly - 10, text=f"{ys[-1]:.3g}", fill=TEXT,
                      font=("Menlo", 8))

        self.status.set(f"{len(pts)} points · step {xs[-1]:,}")
        self._update_values(series)

    def _update_values(self, series):
        """Show a compact strip of the latest value of every key metric."""
        bits = []
        for tag, label in TRACKED:
            pts = series.get(tag)
            if pts:
                short = label.split(" (")[0]
                bits.append(f"{short}: {pts[-1][2]:.3g}")
        self.values.set("   ".join(bits))

    # ── W&B ────────────────────────────────────────────────────────────────────
    def _open_wandb(self):
        """Open the exact W&B run if training recorded its URL, else the
        project page. The training scripts write the URL to
        logs/<run_name>_wandb_url.txt (flat layout); older runs may have it at
        logs/<run_name>/wandb_url.txt."""
        name = self.run_var.get()
        if name:
            logs = self._logs_dir()
            for url_file in (os.path.join(logs, f"{name}_wandb_url.txt"),
                             os.path.join(logs, name, "wandb_url.txt")):
                if os.path.exists(url_file):
                    try:
                        with open(url_file) as f:
                            url = f.read().strip()
                        if url:
                            webbrowser.open(url)
                            return
                    except OSError:
                        pass
        # Fallback: the project's run list (newest run is what they want).
        project = "rally-racing"
        if self._wandb_project_getter:
            try:
                project = self._wandb_project_getter() or project
            except Exception:
                pass
        self.status.set("No recorded run URL — opening W&B search. "
                        "Train with wandb enabled to capture the exact run.")
        webbrowser.open(f"https://wandb.ai/search?q={project}")



class Stage:
    r"""One runnable pipeline stage (a single CLI script the GUI can launch).

    A stage binds a ``key`` and display ``label`` to a ``build`` callable that,
    given the current settings dict, returns the ``(argv, cwd)`` for its
    subprocess. ``renders`` marks stages that open a PyBullet window;
    ``pipeline`` groups stages as State-RL vs Vision in the UI.
    """

    def __init__(self, key, label, pipeline, build, renders=False, note=""):
        self.key      = key
        self.label    = label
        self.pipeline = pipeline
        self.build    = build
        self.renders  = renders
        self.note     = note


def _build_train(s):
    # Pass the key settings explicitly so the launched command is visible in
    # the console and unambiguous; train.py also reads the rest (PPO
    # hyperparams, reward weights) from gui_config.json.
    argv = [PY, "train.py", "--scenario", s["scenario"],
            "--timesteps", str(s["total_timesteps"]),
            "--n-envs", str(s["n_envs"])]
    if not s["use_wandb"]:
        argv.append("--no-wandb")
    return argv, SRC_DIR


def _build_test(s):
    argv = [PY, "test.py",
            "--scenarios", *s["test_scenarios"],
            "--episodes", str(s["test_episodes"]),
            "--model", s["test_model"]]
    if not s["render"]:
        argv.append("--no-render")
    return argv, SRC_DIR


def _build_collect(s):
    return [PY, os.path.join(VISION_DIR, "collect_data.py"),
            "--n", str(s["collect_n"]),
            "--seed", str(s["collect_seed"])], SRC_DIR


def _build_inspect(s):
    return [PY, os.path.join(VISION_DIR, "inspect_data.py")], SRC_DIR


def _build_train_cnn(s):
    return [PY, os.path.join(VISION_DIR, "train_cnn.py"),
            "--epochs", str(s["cnn_epochs"]),
            "--batch",  str(s["cnn_batch"]),
            "--lr",     str(s["cnn_lr"])], BASE_DIR


def _build_train_vision(s):
    return [PY, "train_vision.py",
            "--scenario",  s["scenario"],
            "--timesteps", str(s["vision_timesteps"]),
            "--n-envs",    str(s["n_envs"])] + \
           ([] if s["use_wandb"] else ["--no-wandb"]), SRC_DIR


def _build_test_vision(s):
    argv = [PY, "test_vision.py",
            "--scenarios", *s["test_scenarios"],
            "--episodes",  str(s["test_episodes"])]
    if not s["render"]:
        argv.append("--no-render")
    return argv, SRC_DIR


STAGES = [
    Stage("train",        "1. Train PPO (state)",      "State-RL",
          _build_train, renders=False,
          note="Reads all settings + reward weights from config. Phase set by Scenario."),
    Stage("test",         "2. Test PPO (state)",       "State-RL",
          _build_test, renders=True,
          note="Evaluates a saved model. Opens PyBullet window unless render is off."),
    Stage("collect",      "1. Collect vision data",    "Vision",
          _build_collect, renders=False,
          note="Generates the labelled CNN dataset (data/vision_dataset.npz)."),
    Stage("inspect",      "2. Inspect vision data",    "Vision",
          _build_inspect, renders=False,
          note="Sanity-check sheet -> data/vision_sample.png."),
    Stage("train_cnn",    "3. Train obstacle CNN",     "Vision",
          _build_train_cnn, renders=False,
          note="Trains vision/cnn_obstacle.pt from the collected dataset."),
    Stage("train_vision", "4. Fine-tune PPO (vision)", "Vision",
          _build_train_vision, renders=False,
          note="Warm-starts the policy on camera-derived obstacle channels."),
    Stage("test_vision",  "5. Test PPO (vision)",      "Vision",
          _build_test_vision, renders=True,
          note="Evaluates the vision policy. Opens PyBullet window unless render is off."),
]
STAGE_BY_KEY = {st.key: st for st in STAGES}


# ════════════════════════════════════════════════════════════════════════════
#  GUI
# ════════════════════════════════════════════════════════════════════════════
class ControlApp:
    r"""The Tkinter control panel: settings editor + pipeline runner.

    Left pane edits all settings (training, reward weights, vision, evaluation)
    and persists them to ``gui_config.json``, which the scripts read at
    runtime. Right pane runs each :class:`Stage` as a subprocess — singly or
    chained "in succession" — streaming output to a console, with a live
    metrics tab and a button to open the HTML system guide.
    """

    SCENARIOS = ["phase1", "phase2", "phase3"]

    def __init__(self, root):
        self.root = root
        root.title("Rally-Racing — Control Panel")
        root.geometry("1080x780")
        root.minsize(940, 640)

        self.proc          = None          # current subprocess.Popen
        self.run_queue     = []            # stages queued for succession run
        self.queue_running = False
        self.out_queue     = queue.Queue() # console lines from reader threads

        self._tk_vars      = {}            # name -> tk.Variable
        self._reward_vars  = {}            # weight name -> tk.Variable

        self._build_widgets()
        self._load_into_widgets(load_config())
        self.root.after(80, self._drain_console)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── layout ──────────────────────────────────────────────────────────────
    def _build_widgets(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        paned = ttk.PanedWindow(outer, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=4)
        right = ttk.Frame(paned, padding=4)
        paned.add(left, weight=3)
        paned.add(right, weight=4)

        self._build_settings(left)
        self._build_runner(right)
        self._build_statusbar(outer)

    # ── settings (left) ───────────────────────────────────────────────────────
    def _build_settings(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        train_tab  = ttk.Frame(nb, padding=8)
        reward_tab = ttk.Frame(nb, padding=8)
        vision_tab = ttk.Frame(nb, padding=8)
        eval_tab   = ttk.Frame(nb, padding=8)
        nb.add(train_tab,  text="Training")
        nb.add(reward_tab, text="Reward weights")
        nb.add(vision_tab, text="Vision")
        nb.add(eval_tab,   text="Evaluation")

        # — Training tab —
        self._section(train_tab, "Core training metrics")
        self._int_field(train_tab,  "total_timesteps", "Total timesteps")
        self._int_field(train_tab,  "n_envs",          "Parallel envs (N_ENVS)")
        self._choice_field(train_tab, "scenario",      "Scenario / phase", self.SCENARIOS)
        self._bool_field(train_tab, "load_previous",   "Resume from resume.zip")
        self._bool_field(train_tab, "reset_timesteps", "Reset timestep counter")
        self._bool_field(train_tab, "use_wandb",       "Log to Weights & Biases")
        self._text_field(train_tab, "wandb_project",   "W&B project name")

        self._section(train_tab, "PPO hyperparameters - Change With Care")
        self._float_field(train_tab, "learning_rate", "Learning rate")
        self._int_field(train_tab,   "batch_size",    "Batch size")
        self._float_field(train_tab, "ent_coef",      "Entropy coefficient")
        self._text_field(train_tab,  "net_arch",      "Net arch (comma list)")
        self._choice_field(train_tab, "device",       "Device", ["cpu", "cuda"])

        # — Reward tab —
        self._section(reward_tab, "Reward weights (RewardConfig)")
        canvas = tk.Canvas(reward_tab, highlightthickness=0)
        sb = ttk.Scrollbar(reward_tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for name in DEFAULTS["reward"]:
            var = tk.StringVar()
            self._reward_vars[name] = var
            row = ttk.Frame(inner)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=name, width=26, anchor="w").pack(side="left")
            ttk.Entry(row, textvariable=var, width=12).pack(side="left")

        # — Vision tab —
        self._section(vision_tab, "Dataset collection")
        self._int_field(vision_tab, "collect_n",    "Samples to collect")
        self._int_field(vision_tab, "collect_seed", "RNG seed")
        self._section(vision_tab, "CNN training")
        self._int_field(vision_tab,   "cnn_epochs", "Epochs")
        self._int_field(vision_tab,   "cnn_batch",  "Batch size")
        self._float_field(vision_tab, "cnn_lr",     "Learning rate")
        self._section(vision_tab, "Vision fine-tune")
        self._int_field(vision_tab, "vision_timesteps", "Fine-tune timesteps")

        # — Evaluation tab —
        self._section(eval_tab, "Evaluation settings")
        self._multichoice_field(eval_tab, "test_scenarios",
                                "Scenarios to evaluate", self.SCENARIOS)
        self._int_field(eval_tab,  "test_episodes", "Episodes per scenario")
        self._bool_field(eval_tab, "render",        "Show simulation (popout window)")
        self._path_field(eval_tab, "test_model",    "Model .zip for state test")

        # — config action buttons —
        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Save settings", command=self.save_settings).pack(side="left")
        ttk.Button(btns, text="Reload",        command=self.reload_settings).pack(side="left", padx=4)
        ttk.Button(btns, text="Restore defaults", command=self.restore_defaults).pack(side="left")

    # ── runner (right) ───────────────────────────────────────────────────────
    def _build_runner(self, parent):
        rnb = ttk.Notebook(parent)
        rnb.pack(fill="both", expand=True)

        run_tab     = ttk.Frame(rnb, padding=6)
        metrics_tab = ttk.Frame(rnb, padding=2)
        rnb.add(run_tab,     text="Run & Console")
        rnb.add(metrics_tab, text="Training Metrics")

        self._build_run_tab(run_tab)
        self._build_metrics_tab(metrics_tab)

    def _build_run_tab(self, parent):
        self._section(parent, "Pipeline stages")
        self.stage_vars = {}
        for pipeline in ("State-RL", "Vision"):
            box = ttk.LabelFrame(parent, text=pipeline + " pipeline", padding=6)
            box.pack(fill="x", pady=3)
            for st in (s for s in STAGES if s.pipeline == pipeline):
                row = ttk.Frame(box)
                row.pack(fill="x", pady=1)
                sel = tk.BooleanVar(value=False)
                self.stage_vars[st.key] = sel
                ttk.Checkbutton(row, variable=sel).pack(side="left")
                ttk.Label(row, text=st.label, width=24, anchor="w").pack(side="left")
                ttk.Button(row, text="Run",
                           command=lambda k=st.key: self.run_single(k)).pack(side="left")
                tag = " [sim]" if st.renders else ""
                ttk.Label(row, text=tag, foreground="#1565c0",
                          width=6).pack(side="left")

        ctl = ttk.Frame(parent)
        ctl.pack(fill="x", pady=(6, 2))
        ttk.Button(ctl, text="Run selected in succession ▶",
                   command=self.run_succession).pack(side="left")
        self.stop_btn = ttk.Button(ctl, text="Stop", command=self.stop_run,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(ctl, text="Clear console", command=self._clear_console).pack(side="right")

        self._section(parent, "Console")
        cframe = ttk.Frame(parent)
        cframe.pack(fill="both", expand=True)
        self.console = tk.Text(cframe, wrap="word", height=18, bg="#101418",
                               fg="#d6e2ec", insertbackground="#d6e2ec",
                               font=("Menlo", 10), state="disabled")
        csb = ttk.Scrollbar(cframe, command=self.console.yview)
        self.console.configure(yscrollcommand=csb.set)
        self.console.pack(side="left", fill="both", expand=True)
        csb.pack(side="right", fill="y")

    def _build_metrics_tab(self, parent):
        # Lazy import so the GUI still launches if metrics_panel's deps shift.
        try:
            self.metrics = MetricsPanel(
                parent, BASE_DIR,
                generation="gen2",
                wandb_project_getter=lambda: self._tk_vars["wandb_project"][1].get())
            self.metrics.pack(fill="both", expand=True)
        except Exception as e:
            self.metrics = None
            ttk.Label(parent,
                      text=f"Metrics panel unavailable: {e}",
                      foreground="#b00").pack(padx=8, pady=8)

    def _build_statusbar(self, parent):
        self.status = tk.StringVar(value="Idle.")
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(6, 0))
        ttk.Separator(parent, orient="horizontal").pack(fill="x")
        ttk.Label(bar, textvariable=self.status, anchor="w").pack(side="left")
        # Guide + W&B quick buttons live on the status bar, always visible.
        ttk.Button(bar, text="📖 System Guide",
                   command=self.open_guide).pack(side="right")

    # ── field helpers ─────────────────────────────────────────────────────────
    def _section(self, parent, text):
        ttk.Label(parent, text=text, font=("", 11, "bold")).pack(
            anchor="w", pady=(8, 2))

    def _labeled_row(self, parent, label):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")
        return row

    def _int_field(self, parent, name, label):
        var = tk.StringVar()
        self._tk_vars[name] = ("int", var)
        ttk.Entry(self._labeled_row(parent, label), textvariable=var,
                  width=16).pack(side="left")

    def _float_field(self, parent, name, label):
        var = tk.StringVar()
        self._tk_vars[name] = ("float", var)
        ttk.Entry(self._labeled_row(parent, label), textvariable=var,
                  width=16).pack(side="left")

    def _text_field(self, parent, name, label):
        var = tk.StringVar()
        self._tk_vars[name] = ("text", var)
        ttk.Entry(self._labeled_row(parent, label), textvariable=var,
                  width=22).pack(side="left")

    def _bool_field(self, parent, name, label):
        var = tk.BooleanVar()
        self._tk_vars[name] = ("bool", var)
        row = self._labeled_row(parent, label)
        ttk.Checkbutton(row, variable=var).pack(side="left")

    def _choice_field(self, parent, name, label, choices):
        var = tk.StringVar()
        self._tk_vars[name] = ("text", var)
        ttk.Combobox(self._labeled_row(parent, label), textvariable=var,
                     values=choices, state="readonly", width=14).pack(side="left")

    def _multichoice_field(self, parent, name, label, choices):
        self._tk_vars[name] = ("multi", {c: tk.BooleanVar() for c in choices})
        row = self._labeled_row(parent, label)
        for c in choices:
            ttk.Checkbutton(row, text=c,
                            variable=self._tk_vars[name][1][c]).pack(side="left")

    def _path_field(self, parent, name, label):
        var = tk.StringVar()
        self._tk_vars[name] = ("text", var)
        row = self._labeled_row(parent, label)
        ttk.Entry(row, textvariable=var, width=24).pack(side="left")
        ttk.Button(row, text="…", width=3,
                   command=lambda v=var: self._browse(v)).pack(side="left")

    def _browse(self, var):
        path = filedialog.askopenfilename(
            title="Select model .zip",
            initialdir=os.path.join(BASE_DIR, "models"),
            filetypes=[("Model zip", "*.zip"), ("All files", "*.*")])
        if path:
            var.set(path)

    # ── settings <-> widgets ──────────────────────────────────────────────────
    # Non-config GUI-only fields (eval + vision CLI args) carry their own
    # defaults; they are stored in gui_config.json too so they persist.
    GUI_ONLY_DEFAULTS = {
        "test_scenarios":   ["phase1", "phase2", "phase3"],
        "test_episodes":    1,
        "render":           True,
        "test_model":       os.path.join(BASE_DIR, "models", "best", "best_model.zip"),
        "collect_n":        10_000,
        "collect_seed":     0,
        "cnn_epochs":       30,
        "cnn_batch":        64,
        "cnn_lr":           1e-3,
        "vision_timesteps": 100_000,
    }

    def _load_into_widgets(self, cfg):
        merged = dict(self.GUI_ONLY_DEFAULTS)
        merged.update(cfg)
        for name, (kind, var) in self._tk_vars.items():
            val = merged.get(name, self.GUI_ONLY_DEFAULTS.get(name))
            if kind == "multi":
                for c, bv in var.items():
                    bv.set(c in (val or []))
            elif kind == "bool":
                var.set(bool(val))
            elif kind == "text":
                if name == "net_arch" and isinstance(val, list):
                    var.set(",".join(str(x) for x in val))
                else:
                    var.set("" if val is None else str(val))
            else:  # int / float
                var.set("" if val is None else str(val))
        for name, var in self._reward_vars.items():
            var.set(str(cfg["reward"].get(name, DEFAULTS["reward"][name])))

    def _collect_from_widgets(self):
        r"""Serialise the widget state into a config dict for ``gui_config.json``.

        Reads each field by its declared kind (int/float/bool/text/multi),
        parsing numeric strings and splitting ``net_arch`` into a layer-size
        list. Built on a deep copy of ``DEFAULTS`` so any unedited key keeps its
        default. Raises :class:`ValueError` on an unparseable number, which the
        caller surfaces as a dialog rather than writing a corrupt config.
        """
        cfg = json.loads(json.dumps(DEFAULTS))  # deep copy as a base
        out = {}
        for name, (kind, var) in self._tk_vars.items():
            if kind == "multi":
                out[name] = [c for c, bv in var.items() if bv.get()]
            elif kind == "bool":
                out[name] = bool(var.get())
            elif kind == "int":
                out[name] = int(float(var.get()))
            elif kind == "float":
                out[name] = float(var.get())
            elif name == "net_arch":
                out[name] = [int(x) for x in var.get().replace(" ", "").split(",") if x]
            else:
                out[name] = var.get()
        cfg.update(out)
        for name, var in self._reward_vars.items():
            cfg["reward"][name] = float(var.get())
        return cfg

    def _current_settings(self):
        """Settings dict used by stage builders (includes GUI-only fields)."""
        cfg = self._collect_from_widgets()
        if not cfg.get("test_scenarios"):
            cfg["test_scenarios"] = ["phase1"]
        return cfg

    # ── system guide ────────────────────────────────────────────────────────
    def open_guide(self):
        """Open the LaTeX-derived HTML system guide in the default browser.

        The guide ships pre-built at docs/system_guide.html so no LaTeX
        toolchain is needed on the user's machine. Opened as a file:// URL via
        the OS default browser."""
        guide = os.path.join(BASE_DIR, "docs", "system_guide.html")
        if not os.path.exists(guide):
            messagebox.showwarning(
                "Guide not found",
                "The system guide (docs/system_guide.html) is missing.\n\n"
                "It ships pre-built; if you cloned without it, rebuild from "
                "docs/system_guide.tex (see the docs/ folder).")
            return
        try:
            webbrowser.open("file://" + os.path.abspath(guide))
            self.status.set("Opened system guide in your browser.")
        except Exception as e:
            messagebox.showerror("Could not open guide", str(e))

    # ── config buttons ────────────────────────────────────────────────────────
    def save_settings(self):
        try:
            cfg = self._collect_from_widgets()
        except ValueError as e:
            messagebox.showerror("Invalid value", f"Could not parse a field:\n{e}")
            return False
        save_config(cfg)
        self._log(f"[config] saved to {CONFIG_PATH}\n")
        self.status.set("Settings saved.")
        return True

    def reload_settings(self):
        self._load_into_widgets(load_config())
        self.status.set("Settings reloaded from disk.")

    def restore_defaults(self):
        if messagebox.askyesno("Restore defaults",
                               "Reset all settings to built-in defaults?"):
            cfg = json.loads(json.dumps(DEFAULTS))
            self._load_into_widgets(cfg)
            self._load_gui_only_defaults()
            self.status.set("Defaults restored (not yet saved).")

    def _load_gui_only_defaults(self):
        for name, val in self.GUI_ONLY_DEFAULTS.items():
            if name not in self._tk_vars:
                continue
            kind, var = self._tk_vars[name]
            if kind == "multi":
                for c, bv in var.items():
                    bv.set(c in val)
            elif kind == "bool":
                var.set(bool(val))
            else:
                var.set(str(val))

    # ── running stages ────────────────────────────────────────────────────────
    def run_single(self, key):
        r"""Save settings and run exactly one stage (its own *Run* button).

        Refuses if a stage is already running. Queues just this stage with
        succession disabled, so :meth:`_start_next` launches it and stops.
        """
        if self.proc is not None:
            messagebox.showinfo("Busy", "A stage is already running.")
            return
        if not self.save_settings():
            return
        self.run_queue = [key]
        self.queue_running = False
        self._start_next()

    def run_succession(self):
        r"""Save settings and run every ticked stage top-to-bottom.

        Collects the selected stages into a queue with succession enabled;
        :meth:`_stage_finished` then chains them, aborting the remainder if any
        stage exits non-zero so a failed step never silently feeds a broken
        artifact to the next.
        """
        if self.proc is not None:
            messagebox.showinfo("Busy", "A stage is already running.")
            return
        selected = [st.key for st in STAGES if self.stage_vars[st.key].get()]
        if not selected:
            messagebox.showinfo("Nothing selected",
                                "Tick one or more stages first.")
            return
        if not self.save_settings():
            return
        self.run_queue = selected
        self.queue_running = True
        self._log(f"\n=== Succession run: {' -> '.join(selected)} ===\n")
        self._start_next()

    def _start_next(self):
        r"""Pop and launch the next queued stage as a subprocess.

        Builds the stage's ``(argv, cwd)`` from current settings, launches it
        in its own process group (so :meth:`stop_run` can kill the whole tree)
        with ``src/`` and the project root on ``PYTHONPATH``, and spawns a
        reader thread to pump output into the console. No-op when the queue is
        empty.
        """
        if not self.run_queue:
            self.status.set("Done.")
            self.queue_running = False
            return
        key = self.run_queue.pop(0)
        stage = STAGE_BY_KEY[key]
        try:
            argv, cwd = stage.build(self._current_settings())
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e))
            return

        self._log(f"\n$ {' '.join(shlex.quote(a) for a in argv)}\n   (cwd: {cwd})\n")
        if stage.renders:
            self._log("   [sim] a PyBullet window will open for this stage.\n")
        self.status.set(f"Running: {stage.label}")
        self.stop_btn.configure(state="normal")

        # Ensure src/ is importable by child scripts (config, reward, vision).
        env = os.environ.copy()
        extra = os.pathsep.join([SRC_DIR, BASE_DIR])
        env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                argv, cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                start_new_session=True,  # so Stop can kill the whole group
            )
        except FileNotFoundError as e:
            self._log(f"[error] could not launch: {e}\n")
            self.proc = None
            self.stop_btn.configure(state="disabled")
            return

        threading.Thread(target=self._reader, args=(self.proc, stage),
                         daemon=True).start()

    def _reader(self, proc, stage):
        for line in proc.stdout:
            self.out_queue.put(line)
        proc.wait()
        self.out_queue.put(("__DONE__", stage.key, proc.returncode))

    def stop_run(self):
        if self.proc is None:
            return
        self._log("\n[stop] terminating current stage…\n")
        self.run_queue = []  # abandon the rest of a succession run
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                self.proc.terminate()
            except Exception:
                pass

    # ── console pump ──────────────────────────────────────────────────────────
    def _drain_console(self):
        try:
            while True:
                item = self.out_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    self._stage_finished(item[1], item[2])
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_console)

    def _stage_finished(self, key, code):
        stage = STAGE_BY_KEY[key]
        self.proc = None
        self.stop_btn.configure(state="disabled")
        self._log(f"[done] {stage.label} exited with code {code}\n")
        if code != 0 and self.queue_running:
            self._log("[chain] stage failed — stopping succession run.\n")
            self.run_queue = []
            self.queue_running = False
            self.status.set(f"Stopped: {stage.label} failed (code {code}).")
            return
        if self.run_queue:
            self._start_next()
        else:
            self.queue_running = False
            self.status.set("Done.")

    # ── console helpers ───────────────────────────────────────────────────────
    def _log(self, text):
        self.console.configure(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")

    def _clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    # ── shutdown ──────────────────────────────────────────────────────────────
    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("Quit",
                                       "A stage is still running. Kill it and quit?"):
                return
            self.stop_run()
        if getattr(self, "metrics", None) is not None:
            self.metrics.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    ControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()