"""
PPO training script for RallyDrivingEnv (privileged observation).

Usage:
    python3 src/train.py --scenario phase1
    python3 src/train.py --scenario phase2 --timesteps 300000
    python3 src/train.py --scenario phase3 --no-wandb
    python3 src/train.py --scenario phase2 --fresh    # ignore warm-start, train from scratch

Output layout:
    <GENERATION>_models/<scenario>/privileged/best/best_model.zip
    <GENERATION>_models/<scenario>/privileged/ppo_rally_final.zip
    logs/<GENERATION>_<scenario>_privileged/

Warm-start chain (privileged):
    phase1 -> scratch
    phase2 -> phase1
    phase3 -> phase2
    custom -> phase3
"""

import argparse
import os
import sys

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Make `simple_driving` and our `src/` modules importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

import simple_driving  # registers RallyDriving-v0
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from reward import custom_reward
import wandb
from wandb.integration.sb3 import WandbCallback


# ── Constants ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
GENERATION    = "gen4"          # bump when starting a new generation
WANDB_PROJECT = "rally-racing"

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    batch_size    = 256,
    ent_coef      = 0.02,
    device        = "cpu",
    policy_kwargs = dict(net_arch=[256, 256]),
)


WARM_START_CHAIN = {
    "phase2":        "phase1",
    "phase3":        "phase2",
    "custom_easy":   "phase3",
    "custom_medium": "custom_easy",
    "custom_hard":   "custom_medium",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                    choices=["phase1", "phase2", "phase3",
                             "custom_easy", "custom_medium", "custom_hard"])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="Train from scratch even if a warm-start model exists.")
    return parser.parse_args()


def warm_start_path(scenario):
    """Return the path the given scenario should warm-start from, or None
    if no parent exists in the curriculum."""
    parent = WARM_START_CHAIN.get(scenario)
    if parent is None:
        return None
    return os.path.join(BASE_DIR, f"{GENERATION}_models", parent,
                        "privileged", "best", "best_model.zip")


def make_train_env(scenario, n_envs):
    def factory():
        return gym.make("RallyDriving-v0",
                        renders=False, isDiscrete=False,
                        reward_callback=custom_reward,
                        observation_callback=None,
                        scenario=scenario)
    return SubprocVecEnv([factory for _ in range(n_envs)], start_method="spawn")


def make_eval_env(scenario, log_dir):
    def factory():
        env = gym.make("RallyDriving-v0",
                       renders=False, isDiscrete=False,
                       reward_callback=custom_reward,
                       observation_callback=None,
                       scenario=scenario)
        return Monitor(env, log_dir)
    return DummyVecEnv([factory])


def make_callbacks(eval_env, best_dir, log_dir, use_wandb, n_envs):
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=log_dir,
            eval_freq=20_000 // n_envs,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
    ]
    if use_wandb:
        callbacks.append(WandbCallback(verbose=1))
    return CallbackList(callbacks)


def main():
    args = parse_args()
    use_wandb = not args.no_wandb

    model_dir = os.path.join(BASE_DIR, f"{GENERATION}_models",
                             args.scenario, "privileged")
    best_dir  = os.path.join(model_dir, "best")
    log_dir   = os.path.join(BASE_DIR, "logs",
                             f"{GENERATION}_{args.scenario}_privileged")
    for d in (model_dir, best_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print(f"Rally PPO Training — {args.scenario} (privileged)")
    print(f"  Generation: {GENERATION}")
    print(f"  Envs:       {args.n_envs}")
    print(f"  Steps:      {args.timesteps}")
    print(f"  Save dir:   {model_dir}")
    print(f"  WandB:      {use_wandb}")
    print("=" * 60)

    run = None
    if use_wandb:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{GENERATION}_{args.scenario}_privileged",
            config={"scenario": args.scenario,
                    "generation": GENERATION,
                    "total_timesteps": args.timesteps,
                    "n_envs": args.n_envs,
                    **PPO_KWARGS},
            sync_tensorboard=True,
            save_code=True,
        )

    env = make_train_env(args.scenario, args.n_envs)
    eval_env = make_eval_env(args.scenario, log_dir)

    warm = warm_start_path(args.scenario)
    if args.fresh or warm is None or not os.path.exists(warm):
        if warm is not None and not args.fresh:
            print(f"Warm-start file {warm} not found — training from scratch.")
        else:
            print("Training from scratch.")
        model = PPO("MlpPolicy", env,
                    tensorboard_log=log_dir, verbose=1, **PPO_KWARGS)
    else:
        print(f"Warm-starting from {warm}")
        model = PPO.load(warm, env=env, device=PPO_KWARGS["device"])

    if run is not None:
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))

    print(f"\nModel: {model.policy}\n")

    callbacks = make_callbacks(eval_env, best_dir, log_dir, use_wandb, args.n_envs)
    try:
        try:
            model.learn(total_timesteps=args.timesteps,
                        callback=callbacks,
                        reset_num_timesteps=True,
                        progress_bar=True)
        except KeyboardInterrupt:
            print("\nInterrupted — saving current model...")

        final_path = os.path.join(model_dir, "ppo_rally_final")
        model.save(final_path)
        print(f"Final model saved to {final_path}.zip")
    finally:
        env.close()
        eval_env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()