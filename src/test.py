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


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import simple_driving  # registers RallyDriving-v0

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from reward import custom_reward

BASE_DIR       = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_MODEL  = os.path.join(BASE_DIR, "models", "best", "best_model.zip")

def evaluate(model_path: str, scenarios: list[str], render: bool = True, episodes: int = 1):
    r"""Evaluate a saved privileged policy across one or more scenarios.

    Loads the PPO model and, for each scenario, runs ``episodes`` deterministic
    rollouts (greedy actions, :math:`a_t = \arg\max_a \pi(a \mid s_t)`),
    reporting per-episode return :math:`G = \sum_t r_t`, step count, and
    checkpoint completion. With ``render=True`` a PyBullet window is opened so
    the car can be watched; ``render=False`` runs headless for fast batch
    scoring.

    Note that on ``phase1`` (fixed spawn, no obstacles) a deterministic policy
    yields identical episodes, so repeated returns there are expected, not a
    bug.

    :param model_path: path to a saved PPO ``.zip`` (typically ``best_model.zip``).
    :param scenarios: scenario names to evaluate, in order.
    :param render: open the simulator window if ``True``.
    :param episodes: rollouts per scenario.
    """
    if not os.path.exists(model_path):
        sys.exit(f"No model found at {model_path}")

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i + 1}/{len(scenarios)}: {scenario.upper()} ({episodes} eps) ---")
        env = gym.make(
            "RallyDriving-v0",
            renders=render,
            isDiscrete=False,
            reward_callback=custom_reward,
            observation_callback=None,
            scenario=scenario,
        )

        rewards, step_counts, completions = [], [], []
        for ep in range(episodes):
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
            rewards.append(total_reward)
            step_counts.append(steps)
            completions.append(completed)
            print(f"  ep {ep + 1:3d}: reward={total_reward:+9.2f}  steps={steps:3d}  "
                  f"checkpoints={completed}/{total}")

        env.close()

        r = np.array(rewards)
        print(f"  -- {scenario.upper()} summary over {episodes} eps --")
        print(f"     reward: mean={r.mean():+.2f}  std={r.std():.2f}  "
              f"min={r.min():+.2f}  max={r.max():+.2f}")
        print(f"     checkpoints: mean={np.mean(completions):.2f}/{total}  "
              f"avg_steps={np.mean(step_counts):.1f}")

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
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Episodes per scenario (default: 1)",
    )
    args = parser.parse_args()

    evaluate(args.model, args.scenarios, render=not args.no_render, episodes=args.episodes)
    
if __name__ == "__main__":
    main()