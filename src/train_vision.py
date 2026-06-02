"""
PPO vision fine-tune script for VisionRallyDrivingEnv.

Warm-starts from the privileged policy of the same scenario and trains
briefly so the policy adapts to noisier CNN-derived obstacle channels.

Usage:
    python3 src/train_vision.py --scenario phase3
    python3 src/train_vision.py --scenario phase3 --timesteps 150000
    python3 src/train_vision.py --scenario phase3 --no-wandb

Output layout:
    <GENERATION>_models/<scenario>/vision/best/best_model.zip
    <GENERATION>_models/<scenario>/vision/ppo_vision_final.zip
    logs/<GENERATION>_<scenario>_vision/

Warm-start (vision):
    <scenario> vision -> <scenario> privileged
    e.g. phase3 vision warm-starts from gen2_models/phase3/privileged/best/best_model.zip
"""

import argparse
import os
import sys

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import simple_driving  # registers VisionRallyDriving-v0
from reward import custom_reward
from vision import ObstacleCNN


# ── Constants ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
GENERATION    = "gen2"           # bump when starting a new generation
WANDB_PROJECT = "rally-racing"
CNN_PATH      = os.path.join(BASE_DIR, "vision", "cnn_obstacle.pt")


def parse_args():
    r"""Parse arguments for a vision fine-tune run.

    Like :func:`train.parse_args` but the scenario is restricted to those with
    obstacles (``phase2``/``phase3``/``custom`` — ``phase1`` has nothing for
    the CNN to detect) and the default budget :math:`T` is smaller, since this
    is a fine-tune of an existing policy rather than training from scratch.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        choices=["phase2", "phase3", "custom"])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def warm_start_path(scenario):
    r"""Return the privileged policy this vision run fine-tunes from.

    Unlike the privileged curriculum (which chains across scenarios), a vision
    policy for scenario :math:`s` warm-starts from the *privileged* policy of
    the **same** scenario :math:`s`. The policy thus inherits a controller that
    already drives :math:`s` well from ground-truth obstacle state, and
    fine-tuning only has to adapt it to the noisier CNN-derived estimate.
    """
    return os.path.join(BASE_DIR, f"{GENERATION}_models", scenario,
                        "privileged", "best", "best_model.zip")


def load_cnn():
    r"""Load and freeze the obstacle CNN used to populate vision observations.

    Returns the :class:`ObstacleCNN` in eval mode. During fine-tuning the CNN
    is held fixed (only the PPO policy updates), so it acts as a static
    perception front-end mapping camera frames to obstacle estimates.
    """
    if not os.path.exists(CNN_PATH):
        sys.exit(f"CNN weights not found at {CNN_PATH}")
    model = ObstacleCNN()
    model.load_state_dict(torch.load(CNN_PATH, map_location="cpu"))
    model.eval()
    return model


def make_env(cnn, scenario, log_dir):
    r"""Build a factory for one monitored ``VisionRallyDriving-v0`` instance.

    The returned closure constructs a vision env wired to the frozen ``cnn``
    and a visibility threshold :math:`\tau = 0.5`: at each step the CNN's
    obstacle prediction replaces the privileged obstacle channels of the
    observation vector when :math:`p_{\text{vis}} > \tau`. Returns the factory
    (not the env) for use by the vectorised-env constructors.
    """
    def factory():
        env = gym.make(
            "VisionRallyDriving-v0",
            renders=False,
            isDiscrete=False,
            reward_callback=custom_reward,
            observation_callback=None,
            scenario=scenario,
            vision_model=cnn,
            vis_threshold=0.5,
        )
        return Monitor(env, log_dir)
    return factory


def main():
    r"""Run one vision fine-tune session.

    Loads the frozen obstacle CNN, warm-starts PPO from the same scenario's
    privileged best model (see :func:`warm_start_path`), and continues training
    for :math:`T` timesteps on ``VisionRallyDriving-v0`` — where observations
    carry CNN-inferred obstacle channels instead of ground truth. The policy
    adapts to perception noise while retaining its driving competence. Logs to
    ``logs/<gen>_<scenario>_vision/`` and, when enabled, syncs to W&B (recording
    the run URL for the GUI).
    """
    args = parse_args()
    use_wandb = not args.no_wandb

    save_dir = os.path.join(BASE_DIR, f"{GENERATION}_models",
                            args.scenario, "vision")
    best_dir = os.path.join(save_dir, "best")
    log_dir  = os.path.join(BASE_DIR, "logs",
                            f"{GENERATION}_{args.scenario}_vision")
    for d in (save_dir, best_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    warm = warm_start_path(args.scenario)
    if not os.path.exists(warm):
        sys.exit(f"Privileged model not found at {warm}.\n"
                 f"Train it first with: python3 src/train.py --scenario {args.scenario}")

    print("=" * 60)
    print(f"PPO Vision Fine-Tune — {args.scenario}")
    print(f"  Generation: {GENERATION}")
    print(f"  Warm-start: {warm}")
    print(f"  CNN:        {CNN_PATH}")
    print(f"  Steps:      {args.timesteps}")
    print(f"  Envs:       {args.n_envs} (DummyVecEnv)")
    print(f"  Save dir:   {save_dir}")
    print(f"  WandB:      {use_wandb}")
    print("=" * 60)

    run = None
    if use_wandb:
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{GENERATION}_{args.scenario}_vision",
            config={"scenario": args.scenario,
                    "generation": GENERATION,
                    "total_timesteps": args.timesteps,
                    "n_envs": args.n_envs,
                    "warm_start": warm},
            sync_tensorboard=True,
            save_code=True,
        )
        try:
            with open(os.path.join(log_dir, "wandb_url.txt"), "w") as _f:
                _f.write(run.url)
        except Exception:
            pass

    cnn = load_cnn()

    train_env = DummyVecEnv(
        [make_env(cnn, args.scenario, log_dir) for _ in range(args.n_envs)]
    )
    eval_env = DummyVecEnv([make_env(cnn, args.scenario, log_dir)])

    print(f"Warm-starting from {warm}")
    model = PPO.load(warm, env=train_env, device="cpu")

    if run is not None:
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    else:
        model.tensorboard_log = log_dir

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=log_dir,
            eval_freq=5_000 // args.n_envs,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
    ]
    if use_wandb:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(verbose=1))

    try:
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=CallbackList(callbacks),
                reset_num_timesteps=True,
                progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\nInterrupted — saving current model...")

        final_path = os.path.join(save_dir, "ppo_vision_final")
        model.save(final_path)
        print(f"Final model saved to {final_path}.zip")
    finally:
        train_env.close()
        eval_env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()