"""
Evaluate the trained PPO policy using CAMERA-DERIVED obstacle info instead
of privileged state. The PPO weights are unchanged — we are testing whether
the policy generalises zero-shot to the noisier vision-based observation.

Usage:
    python3 src/test_vision.py
    python3 src/test_vision.py --scenarios phase2 phase3 --no-render
    python3 src/test_vision.py --episodes 10 --no-render  # batch eval for report numbers
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "vision")))

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

import simple_driving  # registers VisionRallyDriving-v0
from reward import custom_reward
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vision import ObstacleCNN

BASE_DIR    = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_PPO = os.path.join(BASE_DIR, "models", "best", "best_model.zip")
FALLBACK    = os.path.join(BASE_DIR, "models", "resume.zip")
DEFAULT_CNN = os.path.join(BASE_DIR, "vision", "cnn_obstacle.pt")


def load_cnn(path: str) -> ObstacleCNN:
    if not os.path.exists(path):
        sys.exit(f"CNN weights not found: {path}")
    model = ObstacleCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"Loaded CNN from {path}")
    return model


def resolve_ppo(path: str) -> str:
    if os.path.exists(path):
        return path
    if os.path.exists(FALLBACK):
        print(f"{path} not found, using {FALLBACK}")
        return FALLBACK
    sys.exit(f"No PPO model found at {path} or {FALLBACK}")


def run_episode(env, model, render: bool):
    """One episode. Returns (total_reward, steps, checkpoints, collided, vis_frac)."""
    obs, _ = env.reset()
    raw    = env.unwrapped
    done   = False
    total_reward = 0.0
    steps  = 0
    vis_votes = 0

    while not done:
        action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action[0])
        total_reward += reward
        steps += 1
        if raw.last_vis_prob > raw.vis_threshold:
            vis_votes += 1
        done = terminated or truncated
        if render:
            time.sleep(0.005)

    completed = raw.current_checkpoint_idx
    total_cp  = len(raw.checkpoints)
    # Collision is the only non-lap-completion done-cause that fires inside
    # the action-repeat loop; a finished lap has completed == total_cp.
    collided  = (completed < total_cp) and steps < 500
    return {
        "reward":     total_reward,
        "steps":      steps,
        "checkpoints": completed,
        "total_cp":   total_cp,
        "collided":   collided,
        "vis_frac":   vis_votes / max(1, steps),
    }


def evaluate(ppo_path: str, cnn_path: str, scenarios, render: bool, episodes: int):
    cnn = load_cnn(cnn_path)
    ppo_path = resolve_ppo(ppo_path)
    print(f"Loading PPO from {ppo_path}")
    ppo = PPO.load(ppo_path)

    summary = {}

    for scenario in scenarios:
        print(f"\n--- {scenario.upper()} (vision-based observation) ---")
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

        results = [run_episode(env, ppo, render) for _ in range(episodes)]
        env.close()

        rewards    = [r["reward"]     for r in results]
        steps      = [r["steps"]      for r in results]
        completed  = [r["checkpoints"] for r in results]
        collisions = sum(r["collided"] for r in results)
        vis_fracs  = [r["vis_frac"]   for r in results]
        total_cp   = results[0]["total_cp"]

        print(f"  episodes:        {episodes}")
        print(f"  mean reward:     {np.mean(rewards):+.2f}  (std {np.std(rewards):.2f})")
        print(f"  mean steps:      {np.mean(steps):.1f}")
        print(f"  mean checkpoints: {np.mean(completed):.2f} / {total_cp}")
        print(f"  collision rate:  {collisions}/{episodes}  ({100*collisions/episodes:.0f}%)")
        print(f"  vision active:   {np.mean(vis_fracs):.2%} of steps")
        summary[scenario] = {
            "mean_reward":     float(np.mean(rewards)),
            "mean_steps":      float(np.mean(steps)),
            "mean_checkpoints": float(np.mean(completed)),
            "collision_rate":  collisions / episodes,
            "vision_active":   float(np.mean(vis_fracs)),
        }

    print("\n=== Summary ===")
    for s, m in summary.items():
        print(f"  {s}: reward={m['mean_reward']:+.1f}  "
              f"checkpoints={m['mean_checkpoints']:.1f}  "
              f"collision_rate={m['collision_rate']:.0%}  "
              f"vis_active={m['vision_active']:.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo",  default=DEFAULT_PPO)
    parser.add_argument("--cnn",  default=DEFAULT_CNN)
    parser.add_argument("--scenarios", nargs="+",
                        default=["phase1", "phase2", "phase3"])
    parser.add_argument("--episodes", type=int, default=1,
                        help="episodes per scenario (use 10+ for report numbers)")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    evaluate(args.ppo, args.cnn, args.scenarios,
             render=not args.no_render, episodes=args.episodes)


if __name__ == "__main__":
    main()