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
GENERATION    = "gen2"          # bump when starting a new generation
WANDB_PROJECT = "rally-racing"

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    batch_size    = 256,
    ent_coef      = 0.005,
    device        = "cpu",
    policy_kwargs = dict(net_arch=[256, 256]),
)
# 
try:
    from config import load_config as _load_config
    _CFG = _load_config()

    TOTAL_TIMESTEPS = _CFG.get("total_timesteps", TOTAL_TIMESTEPS)
    N_ENVS          = _CFG.get("n_envs",          N_ENVS)
    SCENARIO        = _CFG.get("scenario",        SCENARIO)
    LOAD_PREVIOUS   = _CFG.get("load_previous",   LOAD_PREVIOUS)
    RESET_TIMESTEPS = _CFG.get("reset_timesteps", RESET_TIMESTEPS)
    WANDB_PROJECT   = _CFG.get("wandb_project",   WANDB_PROJECT)
    USE_WANDB       = _CFG.get("use_wandb",       USE_WANDB)

    PPO_KWARGS.update(
        learning_rate = _CFG.get("learning_rate", PPO_KWARGS["learning_rate"]),
        batch_size    = _CFG.get("batch_size",    PPO_KWARGS["batch_size"]),
        ent_coef      = _CFG.get("ent_coef",      PPO_KWARGS["ent_coef"]),
        device        = _CFG.get("device",        PPO_KWARGS["device"]),
        policy_kwargs = dict(net_arch=_CFG.get("net_arch",
                                   PPO_KWARGS["policy_kwargs"]["net_arch"])),
    )
except Exception as _e:
    print(f"[config] using built-in defaults ({_e})")


WARM_START_CHAIN = {
    "phase2": "phase1",
    "phase3": "phase2",
    "custom": "phase3",
}


def parse_args():
    r"""Parse command-line arguments for a privileged training run.

    Exposes the scenario (required), the timestep budget :math:`T`, the number
    of parallel environments :math:`N`, and flags to disable W&B logging or
    force training from scratch (ignoring the warm-start chain).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        choices=["phase1", "phase2", "phase3", "custom"])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="Train from scratch even if a warm-start model exists.")
    return parser.parse_args()


def warm_start_path(scenario):
    r"""Return the policy file the given scenario warm-starts from.

    Encodes the curriculum chain
    :math:`\text{phase1}\to\text{phase2}\to\text{phase3}\to\text{custom}`:
    a run for scenario :math:`s` resumes from the *best* privileged policy of
    its parent :math:`\pi(s)`, transferring learned driving skill so each new
    scenario only has to learn the incremental difficulty (obstacles, then
    ramps). Returns ``None`` when :math:`s` has no parent (``phase1``), i.e.
    training starts from scratch.
    """
    parent = WARM_START_CHAIN.get(scenario)
    if parent is None:
        return None
    return os.path.join(BASE_DIR, f"{GENERATION}_models", parent,
                        "privileged", "best", "best_model.zip")


def make_train_env(scenario, n_envs):
    r"""Build the vectorised training environment.

    Returns a :class:`SubprocVecEnv` of :math:`N` (``n_envs``) independent
    ``RallyDriving-v0`` instances running in separate processes, so PPO
    collects :math:`N` trajectories in parallel each rollout — roughly an
    :math:`N\times` throughput gain on the (CPU-bound) PyBullet stepping. Each
    instance uses :func:`custom_reward` as its reward callback. Rendering is
    off for speed.
    """
    def factory():
        return gym.make("RallyDriving-v0",
                        renders=False, isDiscrete=False,
                        reward_callback=custom_reward,
                        observation_callback=None,
                        scenario=scenario)
    return SubprocVecEnv([factory for _ in range(n_envs)], start_method="spawn")


def make_eval_env(scenario, log_dir):
    r"""Build the single-instance evaluation environment.

    A one-env :class:`DummyVecEnv` wrapped in :class:`Monitor` so episode
    returns and lengths are logged to ``log_dir``. Kept separate from the
    training envs so periodic evaluation (see :func:`make_callbacks`) measures
    a clean, deterministic estimate of policy quality, uncontaminated by
    exploration noise.
    """
    def factory():
        env = gym.make("RallyDriving-v0",
                       renders=False, isDiscrete=False,
                       reward_callback=custom_reward,
                       observation_callback=None,
                       scenario=scenario)
        return Monitor(env, log_dir)
    return DummyVecEnv([factory])


def make_callbacks(eval_env, best_dir, log_dir, use_wandb, n_envs):
    r"""Assemble the Stable-Baselines3 training callbacks.

    Builds an :class:`EvalCallback` that evaluates the policy every
    :math:`\lfloor 20{,}000 / N \rfloor` PPO steps (so the wall-clock cadence
    is roughly constant in the number of parallel envs :math:`N`) over 10
    deterministic episodes, saving the highest-scoring snapshot to ``best_dir``
    — this *best-by-evaluation* model is what evaluation later loads. Appends a
    :class:`WandbCallback` when ``use_wandb`` is set.
    """
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
    r"""Run one privileged-observation PPO training session.

    Pipeline: parse args :math:`\to` resolve output/log dirs :math:`\to` init
    W&B (recording the run URL for the GUI) :math:`\to` build the parallel
    train env and the eval env :math:`\to` either warm-start from the parent
    scenario's best policy (see :func:`warm_start_path`) or create a fresh PPO
    model :math:`\to` train for :math:`T` timesteps with periodic evaluation
    and best-model saving :math:`\to` save the final policy. Scalars are logged
    to TensorBoard at ``logs/<gen>_<scenario>_privileged/`` and synced to W&B.
    """
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
        # Record the live run URL next to the logs so the control GUI's
        # "Open in W&B" button can jump straight to this exact run.
        try:
            with open(os.path.join(log_dir, "wandb_url.txt"), "w") as _f:
                _f.write(run.url)
        except Exception:
            pass

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