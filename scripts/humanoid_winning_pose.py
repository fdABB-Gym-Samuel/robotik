"""Humanoid standing controller with rock-paper-scissors using hardcoded PD control."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Make project-root imports work when running: python scripts/humanoid_winning_pose.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.pd_controller import pd_control


# Humanoid-v5 winning pose (17 DOF)
# Adjust these joint angles to create your winning pose
WINNING_POSE = np.array(
    [
        0.0,  # abdomen_z
        0.0,  # abdomen_y
        0.0,  # abdomen_x
        0.0,  # right_hip_x
        0.0,  # right_hip_z
        0.0,  # right_hip_y
        1.0,  # right_knee (MORE bent for stability - was 0.3)
        0.0,  # left_hip_x
        0.1,  # left_hip_z
        0.0,  # left_hip_y
        1.0,  # left_knee (MORE bent for stability - was 0.3)
        0.0,  # right_shoulder1
        0.0,  # right_shoulder2 (symmetric with left)
        0.0,  # right_elbow
        0.0,  # left_shoulder1
        0.0,  # left_shoulder2 (was 0.8 - asymmetry caused falling)
        0.0,  # left_elbow
    ],
    dtype=np.float32,
)

# PD controller gains - VERY STIFF for stability
KP = 50.0  # Very strong position tracking
KD = 5.0  # Much stronger damping (increased from 2.0 to reduce oscillation)


def main() -> None:
    """Control Humanoid to hold a winning pose."""

    env = gym.make("Humanoid-v5", render_mode="human")
    obs, _ = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print("\nHolding winning pose...")

    global_step = 0

    while True:
        # Extract joint angles and velocities from observation
        # Humanoid-v5 obs: [x, y, z, 17_joint_angles, 17_joint_velocities, ...]
        try:
            current_angles = obs[3:20]  # Joint angles
            current_velocities = obs[20:37]  # Joint velocities
        except Exception as e:
            print(f"Error extracting angles: {e}")
            print(f"Obs length: {len(obs)}, obs: {obs}")
            break

        # Compute PD control actions to hold winning pose
        action = pd_control(
            target=WINNING_POSE,
            current=current_angles,
            current_velocity=current_velocities,
            kp=KP,
            kd=KD,
        )

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        if global_step % 25 == 0:
            print(f"step={global_step:04d} reward={reward:7.3f}")

        if terminated or truncated:
            print("\nEpisode finished. Resetting...")
            obs, _ = env.reset()
            global_step = 0
            continue

        global_step += 1

    env.close()


if __name__ == "__main__":
    main()
