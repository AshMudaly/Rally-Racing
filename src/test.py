"""
Evaluation script for the trained Rally agent.

Usage:
    python3 test.py                                          # all circuits, each vs its own best model
    python3 test.py --scenarios circuit_easy circuit_medium
    python3 test.py --model gen5_models/circuit_difficult/privileged/best/best_model.zip
    python3 test.py --no-render --episodes 20
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

BASE_DIR    = os.path.abspath(os.path.join(HERE, ".."))
GENERATION  = "gen5"
SCENARIOS   = ["circuit_easy", "circuit_medium",
               "circuit_hard", "circuit_difficult"]


def default_model_for(scenario: str) -> str:
    """Path to the best model trained on a given scenario, under the current
    generation's directory layout (mirrors train.py)."""
    return os.path.join(BASE_DIR, f"{GENERATION}_models", scenario,
                        "privileged", "best", "best_model.zip")


def evaluate(model_override: str | None, scenarios: list[str],
             render: bool = True, episodes: int = 1):
    r"""Evaluate saved privileged policies across one or more scenarios.

    For each scenario, runs ``episodes`` deterministic rollouts (greedy
    actions, :math:`a_t = \arg\max_a \pi(a \mid s_t)`), reporting per-episode
    return :math:`G = \sum_t r_t`, step count, and checkpoint completion.
    With ``render=True`` a PyBullet window is opened so the car can be
    watched; ``render=False`` runs headless for fast batch scoring.

    Model selection:
        - If ``model_override`` is given, that single model is used for every
          scenario (useful for cross-scenario evaluation, e.g. running the
          ``circuit_difficult`` policy on ``circuit_easy`` to check regression).
        - Otherwise each scenario loads ``default_model_for(scenario)``.

    Note on determinism: ``circuit_easy`` and ``circuit_medium`` have no
    obstacles, so with ``deterministic=True`` repeated episodes produce
    identical rollouts. ``circuit_hard`` and ``circuit_difficult`` apply
    per-reset obstacle spawn jitter, so episodes differ across resets even
    with a deterministic policy — this is expected.

    :param model_override: optional explicit path to a saved PPO ``.zip``.
    :param scenarios: scenario names to evaluate, in order.
    :param render: open the simulator window if ``True``.
    :param episodes: rollouts per scenario.
    """
    # Cache loaded models so we don't reload between scenarios when the
    # same model is shared (which is the case when --model is set).
    model_cache: dict[str, PPO] = {}

    def load(path: str) -> PPO:
        if path not in model_cache:
            if not os.path.exists(path):
                sys.exit(f"No model found at {path}")
            print(f"Loading model from {path}")
            model_cache[path] = PPO.load(path)
        return model_cache[path]

    for i, scenario in enumerate(scenarios):
        model_path = model_override or default_model_for(scenario)
        model = load(model_path)

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
        "--model", default=None,
        help="Explicit model .zip path (overrides per-scenario lookup). "
             "If omitted, each scenario loads "
             f"{GENERATION}_models/<scenario>/privileged/best/best_model.zip",
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        default=SCENARIOS, choices=SCENARIOS,
        help=f"Scenarios to evaluate (default: all). Choices: {', '.join(SCENARIOS)}",
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

    evaluate(args.model, args.scenarios,
             render=not args.no_render, episodes=args.episodes)


if __name__ == "__main__":
    main()