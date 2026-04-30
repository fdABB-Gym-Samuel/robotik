"""Run the Unitree G1 MuJoCo scene with a random controller and print rewards."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym

# Make project-root imports work when running: python scripts/run_humanoid.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.random_controller import sample_action
from environments import UNITREE_G1_ENV_ID, register_environments


def main() -> None:
    register_environments()
    env = gym.make(UNITREE_G1_ENV_ID, render_mode="human")

    _, _ = env.reset()
    episode_reward = 0.0

    for step in range(500):
        action = sample_action(env)
        _, reward, terminated, truncated, _ = env.step(action)
        episode_reward += float(reward)

        print(f"step={step:03d} reward={reward: .3f} total={episode_reward: .3f}")

        if terminated or truncated:
            print("Episode finished. Resetting environment...")
            _, _ = env.reset()
            episode_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
