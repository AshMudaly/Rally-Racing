"""
PPO training script for RallyDrivingEnv.

Merges:
    - SubprocVecEnv parallel training (from his quiz2 code) — big speedup
    - Atomic resume.zip saves (from our Gazebo project) — no more corrupt zips
    - Safe-load that handles corrupt resumes gracefully
    - CheckpointCallback every 10k steps so a crash doesn't lose everything
    - EvalCallback tracking best model on a separate eval env
"""

import os
import sys
import zipfile

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Make `simple_driving` and our `src/` modules importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import simple_driving  # registers RallyDriving-v0
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback, CallbackList,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from reward import custom_reward


# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.abspath(os.path.join(HERE, ".."))
LOG_DIR   = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_DIR  = os.path.join(MODEL_DIR, "best")

for d in (LOG_DIR, MODEL_DIR, BEST_DIR):
    os.makedirs(d, exist_ok=True)


# ── Hyperparameters ───────────────────────────────────────────────────────
TOTAL_TIMESTEPS    = 200_000
N_ENVS             = 8
SCENARIO           = "phase1"           # phase1 / phase2 / phase3
LOAD_PREVIOUS      = False
RESET_TIMESTEPS    = False             # set True to see each run as separate in TB

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 512,
    batch_size    = 256,
    ent_coef      = 0.01,
    device        = "cpu",
    policy_kwargs = dict(net_arch=[256, 256]),  # ← add this
)

ENV_KWARGS = dict(
    renders=False,
    isDiscrete=False,
    reward_callback=custom_reward,
    observation_callback=None,  # RallyDrivingEnv builds its own obs internally
    scenario=SCENARIO,
)


# ── Atomic resume callback ────────────────────────────────────────────────
class AtomicResumeCallback(BaseCallback):
    """Write `resume.zip` atomically every `save_freq` steps."""
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
                print(f"[AtomicResume] {self.save_path} saved at step {self.num_timesteps}")
        return True


# ── Safe resume ───────────────────────────────────────────────────────────
def try_load_resume(resume_path: str, env):
    if not os.path.exists(resume_path):
        return None
    try:
        with zipfile.ZipFile(resume_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"Corrupt member: {bad}")
        print(f"Resuming from {resume_path}")
        return PPO.load(resume_path, env=env, device=PPO_KWARGS["device"])
    except (ValueError, zipfile.BadZipFile, EOFError) as e:
        broken = resume_path + ".broken"
        print(f"Resume checkpoint corrupt ({e}); moving to {broken}")
        os.rename(resume_path, broken)
        return None


# ── Env builders ──────────────────────────────────────────────────────────
def make_train_env():
    """16-process SubprocVecEnv. Each worker registers gym envs on import of simple_driving."""
    def factory():
        env = gym.make("RallyDriving-v0", **ENV_KWARGS)
        return env

    return SubprocVecEnv(
        [factory for _ in range(N_ENVS)],
        start_method="spawn",
    )


def make_eval_env():
    """Single-env eval. Wrapped in Monitor so EvalCallback can read rewards."""
    def factory():
        env = gym.make("RallyDriving-v0", **ENV_KWARGS)
        return Monitor(env, LOG_DIR)
    return DummyVecEnv([factory])


# ── Callbacks ─────────────────────────────────────────────────────────────
def make_callbacks(eval_env):
    return CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=BEST_DIR,
            log_path=LOG_DIR,
            eval_freq=20_000 // N_ENVS,
            n_eval_episodes=3,
            deterministic=True,
            verbose=1,
        ),
        AtomicResumeCallback(
            save_path=os.path.join(MODEL_DIR, "resume.zip"),
            save_freq=10_000 // N_ENVS,
            verbose=1,
        ),
    ])

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"Rally PPO Training — scenario={SCENARIO}")
    print(f"  Envs:   {N_ENVS} parallel")
    print(f"  Steps:  {TOTAL_TIMESTEPS}")
    print(f"  Logs:   {LOG_DIR}")
    print(f"  Models: {MODEL_DIR}")
    print("=" * 60)

    env = make_train_env()
    eval_env = make_eval_env()

    resume_path = os.path.join(MODEL_DIR, "resume.zip")
    model = try_load_resume(resume_path, env) if LOAD_PREVIOUS else None

    if model is None:
        print("Starting fresh training run.")
        model = PPO(
            "MlpPolicy", env,
            tensorboard_log=LOG_DIR,
            verbose=1,
            **PPO_KWARGS,
        )

    print(f"\nModel: {model.policy}\n")

    callbacks = make_callbacks(eval_env)
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=RESET_TIMESTEPS,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving current model...")

    # Final save (atomic)
    final_path = os.path.join(MODEL_DIR, "ppo_rally_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}.zip")

    tmp = resume_path + ".tmp"
    model.save(tmp)
    os.replace(tmp, resume_path)
    print(f"Resume checkpoint saved to {resume_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
