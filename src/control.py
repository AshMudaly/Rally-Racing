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
    # train.py reads everything from gui_config.json; no CLI args needed.
    return [PY, "train.py"], SRC_DIR


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
            from metrics_panel import MetricsPanel
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
