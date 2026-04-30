"""Train a PPO policy on Humanoid-v5."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Humanoid-v5")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--save-dir",
        default="runs/logs",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Create environment
    env = gym.make("Humanoid-v5")

    # 2) Train model
    model = PPO("MlpPolicy", env, verbose=1)
    print(f"Training PPO on Humanoid-v5 for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    # 3) Save model
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "ppo_Humanoid-v5"
    model.save(str(model_path))
    print(f"Model saved to: {model_path}.zip")

    # 4) Evaluate model
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.eval_episodes,
    )
    print(
        f"Evaluation over {args.eval_episodes} episodes -> "
        f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()
