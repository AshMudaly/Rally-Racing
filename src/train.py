"""
PPO training script for RallyDrivingEnv (privileged observation).

Usage:
    python3 src/train.py --scenario circuit_easy
    python3 src/train.py --scenario circuit_difficult --no-wandb
    python3 src/train.py --scenario circuit_medium --fresh    # ignore warm-start, train from scratch

Output layout:
    <GENERATION>_models/<scenario>/privileged/best/best_model.zip
    <GENERATION>_models/<scenario>/privileged/ppo_rally_final.zip
    logs/<GENERATION>_<scenario>_privileged/

Warm-start chain (privileged):
    circuit_easy      -> scratch
    circuit_medium    -> circuit_easy
    circuit_hard      -> circuit_medium
    circuit_difficult -> circuit_hard
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
GENERATION    = "gen6"          # bump when starting a new generation
WANDB_PROJECT = "rally-racing"

SCENARIOS = ["circuit_easy", "circuit_medium",
             "circuit_hard", "circuit_difficult"]

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    batch_size    = 256,
    ent_coef      = 0.05,
    device        = "cpu",
    policy_kwargs = dict(net_arch=[256, 256]),
)


WARM_START_CHAIN = {
    "circuit_medium":    "circuit_easy",
    "circuit_hard":      "circuit_medium",
    "circuit_difficult": "circuit_hard",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--timesteps", type=int, default=800_000)
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


def make_eval_env(scenario, aux_dir):
    def factory():
        env = gym.make("RallyDriving-v0",
                       renders=False, isDiscrete=False,
                       reward_callback=custom_reward,
                       observation_callback=None,
                       scenario=scenario)
        return Monitor(env, aux_dir)
    return DummyVecEnv([factory])


def make_callbacks(eval_env, best_dir, aux_dir, use_wandb, n_envs):
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=aux_dir,
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
    # Flat log layout: events write directly into logs/ with the run name in the
    # filename; aux files (eval npz, monitor csv) get a per-run subfolder.
    log_dir   = os.path.join(BASE_DIR, "logs")
    run_name  = f"{GENERATION}_{args.scenario}_privileged"
    aux_dir   = os.path.join(log_dir, f"{run_name}_eval")
    for d in (model_dir, best_dir, log_dir, aux_dir):
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
        try:
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
            # Record the live run URL so the GUI's "Open in W&B" button can jump
            # to this exact run. Named per run so it doesn't clobber others.
            try:
                with open(os.path.join(log_dir, f"{run_name}_wandb_url.txt"), "w") as _f:
                    _f.write(run.url)
            except Exception:
                pass
        except Exception as e:
            # Most commonly a missing API key (run `wandb login`). Don't let
            # logging take down training — fall back to local TensorBoard logs,
            # which the GUI's Training Metrics tab reads regardless.
            print(f"\n[wandb] disabled for this run: {e}")
            print("[wandb] continuing with local TensorBoard logging only. "
                  "Run `wandb login` to enable online logging.\n")
            use_wandb = False
            run = None

    env = make_train_env(args.scenario, args.n_envs)
    eval_env = make_eval_env(args.scenario, aux_dir)

    warm = warm_start_path(args.scenario)
    if args.fresh or warm is None or not os.path.exists(warm):
        if warm is not None and not args.fresh:
            print(f"Warm-start file {warm} not found — training from scratch.")
        else:
            print("Training from scratch.")
        # No tensorboard_log here — that would create a PPO_N subdirectory.
        # A flat, run-named logger is attached below instead.
        model = PPO("MlpPolicy", env, verbose=1, **PPO_KWARGS)
    else:
        print(f"Warm-starting from {warm}")
        model = PPO.load(warm, env=env, device=PPO_KWARGS["device"])

    # Flat TensorBoard logging: write event files directly into logs/ with the
    # run name embedded in the filename instead of a per-run PPO_N subdirectory.
    # A single writer for every run type also avoids the duplicate-writer
    # conflict from combining tensorboard_log with set_logger.
    from stable_baselines3.common.logger import (
        Logger, HumanOutputFormat, TensorBoardOutputFormat)
    from torch.utils.tensorboard import SummaryWriter
    _tb_fmt = TensorBoardOutputFormat.__new__(TensorBoardOutputFormat)
    _tb_fmt.writer = SummaryWriter(log_dir=log_dir, filename_suffix=f".{run_name}")
    _tb_fmt._is_closed = False
    model.set_logger(Logger(
        folder=log_dir,
        output_formats=[HumanOutputFormat(sys.stdout), _tb_fmt],
    ))

    print(f"\nModel: {model.policy}\n")

    callbacks = make_callbacks(eval_env, best_dir, aux_dir, use_wandb, args.n_envs)
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