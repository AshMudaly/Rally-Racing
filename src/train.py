import os
import sys
import zipfile
import numpy as np

# Add src to path so reward.py and car_env.py are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from car_env import RacingCarEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.expanduser("~/41118_ws/project/racing")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_DIR  = os.path.join(MODEL_DIR, "best")

for d in (LOG_DIR, MODEL_DIR, BEST_DIR):
    os.makedirs(d, exist_ok=True)


# ── Hyperparameters ───────────────────────────────────────────────────────
POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256, 256],   # actor
        vf=[256, 256, 256],   # critic
    ),
    activation_fn=torch.nn.ReLU,
)

PPO_KWARGS = dict(
    learning_rate    = 3e-4,
    n_steps          = 2048,
    batch_size       = 64,
    n_epochs         = 10,
    gamma            = 0.99,
    gae_lambda       = 0.95,
    clip_range       = 0.2,
    ent_coef         = 0.01,
    vf_coef          = 0.5,
    max_grad_norm    = 0.5,
    verbose          = 1,
    tensorboard_log  = LOG_DIR,
    policy_kwargs    = POLICY_KWARGS,
)

TOTAL_TIMESTEPS = 200_000   # increase to 2-3M for better results


# ── Atomic-save callback ──────────────────────────────────────────────────
class AtomicResumeCallback(BaseCallback):
    """
    Save `resume.zip` atomically every `save_freq` steps so a crash mid-save
    can never leave a corrupted checkpoint.
    """
    def __init__(self, save_path: str, save_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            tmp = self.save_path + ".tmp"
            self.model.save(tmp)
            os.replace(tmp, self.save_path)
            if self.verbose:
                print(f"[AtomicResume] saved {self.save_path} at step {self.num_timesteps}")
        return True


# ── Callbacks ─────────────────────────────────────────────────────────────
def make_callbacks(env):
    checkpoint_cb = CheckpointCallback(
        save_freq      = 10_000,           # was 50_000 — losing 50k steps hurts
        save_path      = MODEL_DIR,
        name_prefix    = "ppo_racing",
        verbose        = 1,
    )

    eval_cb = EvalCallback(
        env,
        best_model_save_path = BEST_DIR,
        log_path             = LOG_DIR,
        eval_freq            = 10_000,
        n_eval_episodes      = 3,
        deterministic        = True,
        verbose              = 1,
    )

    resume_cb = AtomicResumeCallback(
        save_path = os.path.join(MODEL_DIR, "resume.zip"),
        save_freq = 10_000,
        verbose   = 1,
    )

    return CallbackList([checkpoint_cb, eval_cb, resume_cb])


# ── Safe resume loading ───────────────────────────────────────────────────
def try_load_resume(resume_path: str, env):
    """Try to load resume.zip; if it's corrupt, move it aside and start fresh."""
    if not os.path.exists(resume_path):
        return None

    try:
        # Verify the zip is well-formed before SB3 touches it
        with zipfile.ZipFile(resume_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"Corrupt member: {bad}")

        print(f"Resuming from {resume_path}")
        return PPO.load(resume_path, env=env, **{
            k: v for k, v in PPO_KWARGS.items()
            if k not in ["verbose", "tensorboard_log", "policy_kwargs"]
        })
    except (ValueError, zipfile.BadZipFile, EOFError) as e:
        broken = resume_path + ".broken"
        print(f"Resume checkpoint is corrupt ({e}); moving to {broken}")
        os.rename(resume_path, broken)
        return None


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 1 — Oval track PPO training")
    print(f"  Logs:   {LOG_DIR}")
    print(f"  Models: {MODEL_DIR}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    # ── Build env ─────────────────────────────────────────────────────
    raw_env = RacingCarEnv()
    env     = Monitor(raw_env, LOG_DIR)
    env     = DummyVecEnv([lambda: env])

    # ── Build model (resume or fresh) ─────────────────────────────────
    resume_path = os.path.join(MODEL_DIR, "resume.zip")
    model = try_load_resume(resume_path, env)
    resumed = model is not None

    if model is None:
        print("Starting fresh training run.")
        model = PPO("MlpPolicy", env, **PPO_KWARGS)

    print("\nNetwork architecture:")
    print(model.policy)
    print()

    # ── Decide how many steps to train ────────────────────────────────
    # If resuming, we want to do *another* TOTAL_TIMESTEPS on top of what
    # was already done — not have learn() exit immediately because the
    # internal counter is already at the target.
    if resumed:
        steps_to_run = TOTAL_TIMESTEPS
        reset_counter = False
        print(f"Resumed at {model.num_timesteps} steps; training {steps_to_run} more.")
    else:
        steps_to_run = TOTAL_TIMESTEPS
        reset_counter = True

    # ── Train ─────────────────────────────────────────────────────────
    callbacks = make_callbacks(env)
    try:
        model.learn(
            total_timesteps     = steps_to_run,
            callback            = callbacks,
            reset_num_timesteps = reset_counter,
            progress_bar        = True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current model...")

    # ── Save final model atomically ───────────────────────────────────
    final_path = os.path.join(MODEL_DIR, "ppo_racing_final")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}.zip")

    tmp = resume_path + ".tmp"
    model.save(tmp)
    os.replace(tmp, resume_path)
    print(f"Resume checkpoint saved to {resume_path}")

    env.close()


if __name__ == "__main__":
    main()