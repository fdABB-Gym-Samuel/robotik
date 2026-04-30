"""Visualize a trained PPO model on a Gymnasium MuJoCo environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize trained PPO model")
    parser.add_argument(
        "--model-path",
        default="runs/logs/ppo_Humanoid-v5",
        help="Path to trained model (without .zip)",
    )
    parser.add_argument(
        "--env",
        default="Humanoid-v5",
        help="Environment ID (must match trained model)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load trained model
    model_path = Path(args.model_path)
    if not model_path.with_suffix(".zip").exists():
        print(f"Error: Model not found at {model_path}.zip")
        print("Run train_rl.py first to train a model.")
        return

    model = PPO.load(str(model_path))
    print(f"Loaded model from: {model_path}.zip")

    # Create environment with rendering
    env = gym.make(args.env, render_mode="human")
    print(f"Running {args.episodes} visualization episodes...\n")

    total_rewards = []

    # Run episodes with trained policy
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: reward={episode_reward:.2f}, steps={steps}")

    env.close()

    # Summary
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
