"""
Fine-tune a PPO policy on VisionRallyDrivingEnv.

Warm-starts from an existing model and trains a short run so the policy
adapts to the noisier CNN-based obstacle channels. Saves per-scenario:
    models/<scenario>/best/best_model.zip   (EvalCallback best)
    models/<scenario>/ppo_vision_final.zip  (final)

Usage:
    python3 src/train_vision.py
    python3 src/train_vision.py --scenario custom --timesteps 150000
    python3 src/train_vision.py --scenario custom --no-wandb
"""

import os
import sys

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import simple_driving  # registers VisionRallyDriving-v0
from reward import custom_reward
from vision import ObstacleCNN

# ── Inputs (warm-start sources) ────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CNN_PATH = os.path.join(BASE_DIR, "vision", "cnn_obstacle.pt")
PPO_PATH = os.path.join(BASE_DIR, "models", "vision", "best", "best_model.zip")

# ── Defaults ────────────────────────────────────────────────────────────────
SCENARIO        = "phase3"
N_ENVS          = 8          # DummyVecEnv (single process) — larger rollouts
TOTAL_TIMESTEPS = 100_000
WANDB_PROJECT   = "rally-racing"


def load_cnn():
    model = ObstacleCNN()
    model.load_state_dict(torch.load(CNN_PATH, map_location="cpu"))
    model.eval()
    return model


def make_env(cnn, scenario, log_dir, render=False):
    def factory():
        env = gym.make(
            "VisionRallyDriving-v0",
            renders=render,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # Per-scenario output paths, decided after we know the scenario.
    save_dir = os.path.join(BASE_DIR, "models", args.scenario)
    best_dir = os.path.join(save_dir, "best")
    log_dir  = os.path.join(BASE_DIR, "logs", f"{args.scenario}_vision")
    for d in (save_dir, best_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    use_wandb = not args.no_wandb

    print(f"Fine-tuning PPO on VisionRallyDriving-v0 ({args.scenario})")
    print(f"  Warm-start: {PPO_PATH}")
    print(f"  CNN:        {CNN_PATH}")
    print(f"  Steps:      {args.timesteps}")
    print(f"  Envs:       {args.n_envs} (DummyVecEnv)")
    print(f"  Save dir:   {save_dir}")
    print(f"  WandB:      {use_wandb}")

    run = None
    if use_wandb:
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{args.scenario}_vision_finetune",
            config={
                "scenario": args.scenario,
                "total_timesteps": args.timesteps,
                "n_envs": args.n_envs,
                "warm_start": PPO_PATH,
            },
            sync_tensorboard=True,
            save_code=True,
        )

    cnn = load_cnn()

    train_env = DummyVecEnv(
        [make_env(cnn, args.scenario, log_dir) for _ in range(args.n_envs)]
    )
    eval_env = DummyVecEnv([make_env(cnn, args.scenario, log_dir)])

    model = PPO.load(PPO_PATH, env=train_env, device="cpu")

    # Point SB3's logger at the WandB run dir so sync_tensorboard ingests the
    # train/ and rollout/ scalars even though we loaded a saved model.
    if run is not None:
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(run.dir, ["stdout", "tensorboard"]))
    else:
        model.tensorboard_log = log_dir

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=log_dir,
            eval_freq=5_000 // args.n_envs,
            n_eval_episodes=5,
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