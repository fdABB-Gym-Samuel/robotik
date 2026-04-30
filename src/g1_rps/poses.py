"""Pose library and demo configuration for the exact Unitree Inspire hand demo."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEQUENCE = ("rock", "paper", "scissors")

POSED_JOINTS = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
    "right_index_1_joint",
    "right_index_2_joint",
    "right_middle_1_joint",
    "right_middle_2_joint",
    "right_ring_1_joint",
    "right_ring_2_joint",
    "right_little_1_joint",
    "right_little_2_joint",
)

HAND_GESTURE_RATIOS: dict[str, dict[str, float]] = {
    "paper": {
        "right_thumb_1_joint": 0.85,
        "right_thumb_2_joint": 0.25,
        "right_thumb_3_joint": 0.20,
        "right_thumb_4_joint": 0.18,
        "right_index_1_joint": 0.02,
        "right_index_2_joint": 0.02,
        "right_middle_1_joint": 0.02,
        "right_middle_2_joint": 0.02,
        "right_ring_1_joint": 0.02,
        "right_ring_2_joint": 0.02,
        "right_little_1_joint": 0.02,
        "right_little_2_joint": 0.02,
    },
    "rock": {
        "right_thumb_1_joint": 0.50,
        "right_thumb_2_joint": 0.85,
        "right_thumb_3_joint": 0.90,
        "right_thumb_4_joint": 0.90,
        "right_index_1_joint": 0.92,
        "right_index_2_joint": 0.92,
        "right_middle_1_joint": 0.92,
        "right_middle_2_joint": 0.92,
        "right_ring_1_joint": 0.92,
        "right_ring_2_joint": 0.92,
        "right_little_1_joint": 0.92,
        "right_little_2_joint": 0.92,
    },
    "scissors": {
        "right_thumb_1_joint": 0.55,
        "right_thumb_2_joint": 0.82,
        "right_thumb_3_joint": 0.82,
        "right_thumb_4_joint": 0.82,
        "right_index_1_joint": 0.04,
        "right_index_2_joint": 0.04,
        "right_middle_1_joint": 0.04,
        "right_middle_2_joint": 0.04,
        "right_ring_1_joint": 0.92,
        "right_ring_2_joint": 0.92,
        "right_little_1_joint": 0.92,
        "right_little_2_joint": 0.92,
    },
}


@dataclass(frozen=True)
class DemoConfig:
    sequence: tuple[str, ...] = DEFAULT_SEQUENCE
    transition_seconds: float = 1.2
    hold_seconds: float = 1.1
    camera_preset: str = "hand_closeup"
    asset_dir: str | None = None


def build_hand_pose(
    gesture: str, joint_ranges: dict[str, tuple[float, float]]
) -> dict[str, float]:
    try:
        gesture_ratios = HAND_GESTURE_RATIOS[gesture]
    except KeyError as exc:
        raise KeyError(
            f"Unknown gesture '{gesture}'. Available gestures: {', '.join(sorted(HAND_GESTURE_RATIOS))}"
        ) from exc

    pose: dict[str, float] = {}
    for joint_name, ratio in gesture_ratios.items():
        low, high = joint_ranges[joint_name]
        pose[joint_name] = low + ratio * (high - low)
    return pose
