"""
Evaluation script for the trained Rally agent.

Usage:
    python3 test.py                    # default: best model, all phases
    python3 test.py --model models/resume.zip
    python3 test.py --scenarios phase1 phase2 phase3
    python3 test.py --no-render
"""

import argparse
import os
import sys
import time
import simple_driving  # registers RallyDriving-v0

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from reward import custom_reward

BASE_DIR       = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_MODEL  = os.path.join(BASE_DIR, "models", "best", "best_model.zip")
FALLBACK_MODEL = os.path.join(BASE_DIR, "models", "resume.zip")


def evaluate(model_path: str, scenarios: list[str], render: bool = True):
    if not os.path.exists(model_path):
        if os.path.exists(FALLBACK_MODEL):
            print(f"{model_path} not found, falling back to {FALLBACK_MODEL}")
            model_path = FALLBACK_MODEL
        else:
            sys.exit(f"No trained model found at {model_path} or {FALLBACK_MODEL}")

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i + 1}/{len(scenarios)}: {scenario.upper()} ---")

        env = gym.make(
            "RallyDriving-v0",
            renders=render,
            isDiscrete=False,
            reward_callback=custom_reward,
            observation_callback=None,
            scenario=scenario,
        )

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            action = action[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if render:
                time.sleep(0.005)

        unwrapped = env.unwrapped
        completed = unwrapped.current_checkpoint_idx
        total = len(unwrapped.checkpoints)
        print(f"  Reward: {total_reward:+.2f}   Steps: {steps}   "
              f"Checkpoints: {completed}/{total}")
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Rally PPO model.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Path to model .zip (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        default=["phase1", "phase2", "phase3"],
        help="Scenarios to evaluate: phase1, phase2, phase3",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable PyBullet GUI (faster headless eval)",
    )
    args = parser.parse_args()

    evaluate(args.model, args.scenarios, render=not args.no_render)


if __name__ == "__main__":
    main()