"""
metrics_panel.py — live training-metrics panel for the control GUI.

Renders the scalar curves Stable-Baselines3 logs (episode reward, episode
length, losses) on a native Tkinter Canvas — no matplotlib, no tensorboard,
keeping the GUI pure-stdlib. Data comes from tb_reader, which parses the
``*.tfevents.*`` files training writes locally.

The panel polls the active run directory on a timer and redraws, so curves
grow in near-real-time while training runs in a subprocess. A scenario/obs
selector lets the user pick which run to view; an "Open in W&B" button hands
the run URL to the system browser for the full online dashboard.
"""

import os
import tkinter as tk
from tkinter import ttk
import webbrowser

import tb_reader

# The scalar tags worth surfacing, in display order, with friendly labels.
TRACKED = [
    ("rollout/ep_rew_mean",  "Episode reward (mean)"),
    ("rollout/ep_len_mean",  "Episode length (mean)"),
    ("train/loss",           "Total loss"),
    ("train/value_loss",     "Value loss"),
    ("train/policy_gradient_loss", "Policy-gradient loss"),
    ("train/entropy_loss",   "Entropy loss"),
    ("train/explained_variance", "Explained variance"),
    ("train/approx_kl",      "Approx. KL"),
]

# Palette (kept readable on the dark console theme used by the GUI).
PLOT_BG   = "#0d1117"
GRID      = "#21262d"
AXIS      = "#5a6270"
LINE      = "#4aa3ff"
TEXT      = "#c9d1d9"
SUBTEXT   = "#7d8590"


class MetricsPanel(ttk.Frame):
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
        runs = []
        if os.path.isdir(logs):
            for name in sorted(os.listdir(logs)):
                full = os.path.join(logs, name)
                if os.path.isdir(full) and tb_reader.latest_event_file(full):
                    runs.append(name)
        self.run_combo["values"] = runs
        if runs and self.run_var.get() not in runs:
            # default to the most-recently-modified run
            newest = max(runs, key=lambda r: os.path.getmtime(
                tb_reader.latest_event_file(os.path.join(logs, r))))
            self.run_var.set(newest)
        elif not runs:
            self.run_var.set("")
        self.redraw()

    def current_run_dir(self):
        name = self.run_var.get()
        if not name:
            return None
        return os.path.join(self._logs_dir(), name)

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
        c = self.canvas
        c.delete("all")
        run_dir = self.current_run_dir()
        if not run_dir:
            self.status.set("No runs found in logs/. Train something first.")
            self.values.set("")
            return

        series = tb_reader.read_run(run_dir)
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
        project page. The run URL is written to logs/<run>/wandb_url.txt by
        the training scripts when wandb is enabled."""
        run_dir = self.current_run_dir()
        if run_dir:
            url_file = os.path.join(run_dir, "wandb_url.txt")
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

