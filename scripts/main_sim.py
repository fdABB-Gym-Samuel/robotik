"""MuJoCo simulation of the full RPS game loop with the proper Inspire hand.

The standard G1 scene at ``assets/unitree_g1/g1_29dof_with_hand.xml`` ships
with a 3-finger 'claw' end-effector that doesn't match the real Unitree
Inspire hand. The official URDF at
``runs/assets/unitree_ros/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf``
has the proper 5-finger Inspire hand attached, and its joint names line up
with ``g1_rps.poses.POSED_JOINTS`` so we can reuse the rock/paper/scissors
ratios used on real hardware.

This script previews a full round repeatedly:
    1. Setup: arm extended (throw-ready), right hand closed in a fist.
    2. Pre-reveal rocking motion (right arm only).
    3. Reveal a randomly chosen gesture on the right hand.
    4. Move into the winning pose (random demo win) or the losing pose.
    5. Return to ready and start the next round.

It does not talk to any DDS hardware.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - local setup guard
    raise SystemExit(
        "MuJoCo Python bindings are not installed. Enter `nix develop` first "
        "before running this script."
    ) from exc

from g1_rps.arm_hardware import ArmHardwareConfig, lose_pose, winning_pose
from g1_rps.assets import urdf_to_mujoco_runtime_xml
from g1_rps.poses import HAND_GESTURE_RATIOS, POSED_JOINTS, build_hand_pose
from pre_reveal_right_arm import (
    JointInterface,
    MotionParameters,
    SimulatorArmInterface,
    pumping_pose,
    setup_pose,
)


URDF_PATH = (
    PROJECT_ROOT
    / "runs"
    / "assets"
    / "unitree_ros"
    / "robots"
    / "g1_description"
    / "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
)
RUNTIME_MJCF_PATH = (
    PROJECT_ROOT
    / "runs"
    / "assets"
    / "unitree_ros"
    / "robots"
    / "g1_description"
    / "g1_29dof_with_inspire_runtime.xml"
)

GESTURES: tuple[str, ...] = tuple(HAND_GESTURE_RATIOS.keys())

# Demo timing.
PRE_REVEAL_CYCLES = 3
REVEAL_HOLD_SECONDS = 0.8
OUTCOME_TRANSITION_SECONDS = 1.0
OUTCOME_HOLD_SECONDS = 1.5
RETURN_TO_READY_SECONDS = 1.0
INTER_ROUND_PAUSE_SECONDS = 0.6


def _strip_claw_hand(pose: dict[str, float]) -> dict[str, float]:
    """Drop the joints from the legacy 3-finger 'claw' hand layout.

    `setup_pose` / `pumping_pose` were written against the stock G1 scene,
    which uses joint names like `right_hand_thumb_0_joint`. The Inspire-
    hand URDF doesn't have those joints; we replace them with
    Inspire-hand finger poses computed from `g1_rps.poses`.
    """
    return {
        joint_name: value
        for joint_name, value in pose.items()
        if not joint_name.startswith("right_hand_")
        and not joint_name.startswith("left_hand_")
    }


def hand_joint_ranges(model) -> dict[str, tuple[float, float]]:
    """Read the (low, high) range of each `POSED_JOINTS` joint from the model."""
    ranges: dict[str, tuple[float, float]] = {}
    for joint_name in POSED_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(
                f"Joint '{joint_name}' was not found in the Inspire-hand URDF."
            )
        low = float(model.jnt_range[joint_id][0])
        high = float(model.jnt_range[joint_id][1])
        ranges[joint_name] = (low, high)
    return ranges


def neutral_waist() -> dict[str, float]:
    """Tiny waist offsets matching the pre-reveal ready pose."""
    return {
        "waist_yaw_joint": -0.02,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.02,
    }


def left_arm_neutral() -> dict[str, float]:
    """Left arm hanging at the side (URDF reference pose).

    Used to drive the left arm back from a win/lose pose to its rest
    position. Without these explicit zeros, `move_joints` would keep
    whatever the previous pose set on the left-arm joints.
    """
    return {
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
    }


def arm_outcome_pose(*, won: bool) -> dict[str, float]:
    """Win or lose pose for both arms, plus a neutral waist."""
    config = ArmHardwareConfig(arm_dof=7)
    pose = dict(winning_pose(config) if won else lose_pose(config))
    pose.update(neutral_waist())
    return pose


def run_round(
    robot: JointInterface,
    arm_params: MotionParameters,
    hand_poses: dict[str, dict[str, float]],
    fist_pose: dict[str, float],
    round_index: int,
) -> None:
    """Pre-reveal -> reveal -> outcome -> back to ready, all in MuJoCo."""

    robot_gesture = random.choice(GESTURES)
    won = random.choice((True, False))
    print(
        f"  Round {round_index}: robot reveals {robot_gesture}; outcome={'win' if won else 'lose'}"
    )

    # 1. Pre-reveal rocking motion. Hand stays in a fist.
    for _ in range(PRE_REVEAL_CYCLES):
        if robot.should_stop.is_set():
            return
        steps = max(1, int(arm_params.cycle_duration / arm_params.control_dt))
        for step in range(steps):
            cycle_phase = step / steps
            arm = _strip_claw_hand(pumping_pose(arm_params, cycle_phase))
            robot.move_joints({**arm, **fist_pose}, arm_params.control_dt)

    # 2. Reveal the chosen gesture on the hand. Arm holds at the throw pose.
    arm_extended = _strip_claw_hand(setup_pose(arm_params))
    robot.move_joints(
        {**arm_extended, **hand_poses[robot_gesture]}, REVEAL_HOLD_SECONDS / 2
    )
    robot.sleep(REVEAL_HOLD_SECONDS / 2)
    if robot.should_stop.is_set():
        return

    # 3. Win or lose pose: both arms move; hand keeps the revealed gesture.
    outcome_arm = arm_outcome_pose(won=won)
    robot.move_joints(
        {**outcome_arm, **hand_poses[robot_gesture]}, OUTCOME_TRANSITION_SECONDS
    )
    robot.sleep(OUTCOME_HOLD_SECONDS)
    if robot.should_stop.is_set():
        return

    # 4. Back to ready (right arm extended, left arm back to its neutral
    # hanging pose, hand back to a fist).
    robot.move_joints(
        {**arm_extended, **left_arm_neutral(), **fist_pose},
        RETURN_TO_READY_SECONDS,
    )
    robot.sleep(INTER_ROUND_PAUSE_SECONDS)


def run_demo(robot: JointInterface) -> None:
    arm_params = MotionParameters(cycles=PRE_REVEAL_CYCLES)
    joint_ranges = hand_joint_ranges(robot.model)
    hand_poses = {
        gesture: build_hand_pose(gesture, joint_ranges) for gesture in GESTURES
    }
    fist_pose = hand_poses["rock"]

    # `SimulatorArmInterface.__init__` seeds `current_targets` with the
    # claw-hand `setup_pose`, whose joint names don't exist in the Inspire-
    # hand URDF. Reset it so the first interpolation only references
    # joints that the model actually has.
    robot.current_targets = {}

    # Initial ready pose: arm extended (matches setup), hand in a fist.
    initial = {
        **_strip_claw_hand(setup_pose(arm_params)),
        **fist_pose,
    }
    robot.move_joints(initial, arm_params.setup_duration)

    round_index = 0
    while not robot.should_stop.is_set():
        round_index += 1
        run_round(robot, arm_params, hand_poses, fist_pose, round_index)


def main() -> None:
    if not URDF_PATH.exists():
        raise SystemExit(
            f"Inspire-hand URDF not found at {URDF_PATH}.\n"
            "Run `python -c 'from g1_rps.assets import ensure_unitree_g1_assets; ensure_unitree_g1_assets()'` "
            "or `nix develop -c python scripts/run_g1_rps_hand_hardware.py` once to fetch the official "
            "unitree_ros checkout."
        )

    print("Full RPS game-loop simulation (Unitree G1 + Inspire right hand).")
    print(f"Scene: {URDF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Gestures: {', '.join(GESTURES)}")
    print("Close the MuJoCo viewer window to stop.\n")

    print(
        "Preparing runtime MJCF (inlining mesh data; first run may take a few seconds)..."
    )
    runtime_path = urdf_to_mujoco_runtime_xml(
        URDF_PATH,
        RUNTIME_MJCF_PATH,
        model_name="g1_29dof_with_inspire",
        freejoint_name="floating_base_joint",
    )

    control_dt = 1.0 / 60.0
    simulator = SimulatorArmInterface(runtime_path, control_dt, skip_runtime_prep=True)
    simulator.launch_viewer_with_motion(lambda: run_demo(simulator))


if __name__ == "__main__":
    main()
