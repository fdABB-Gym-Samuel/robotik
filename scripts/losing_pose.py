"""MuJoCo simulation of the Unitree G1 losing pose.

Loads the G1 scene, brings both hands up in front of the face (the loss
reaction defined in `g1_rps.arm_hardware.lose_pose`), and holds the pose
in the MuJoCo passive viewer until the window is closed. Use this to
preview and tune the pose before running it on the real robot.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from g1_rps.arm_hardware import ArmHardwareConfig, lose_pose
from pre_reveal_right_arm import (
    JointInterface,
    SCENE_PATH,
    SimulatorArmInterface,
)


@dataclass(frozen=True)
class LosingPoseParameters:
    """Tunable knobs for the losing-pose simulation."""

    setup_duration: float = 1.5
    control_dt: float = 1.0 / 60.0
    waist_yaw: float = 0.0
    waist_roll: float = 0.0
    waist_pitch: float = 0.0


def losing_pose_targets(params: LosingPoseParameters) -> dict[str, float]:
    """Both arms in the loss reaction, waist held neutral."""

    config = ArmHardwareConfig(arm_dof=7)
    targets = dict(lose_pose(config))
    targets.update(
        {
            "waist_yaw_joint": params.waist_yaw,
            "waist_roll_joint": params.waist_roll,
            "waist_pitch_joint": params.waist_pitch,
        }
    )
    return targets


def run_losing_pose_motion(robot: JointInterface, params: LosingPoseParameters) -> None:
    """Move into the losing pose and hold it until the viewer closes."""

    targets = losing_pose_targets(params)
    robot.move_joints(targets, params.setup_duration)

    while not robot.should_stop.wait(params.control_dt):
        robot.move_joints(targets, params.control_dt)


def main() -> None:
    params = LosingPoseParameters()
    print("Losing-pose simulation (Unitree G1 in MuJoCo).")
    print("Bringing both hands up in front of the face to match")
    print("`g1_rps.arm_hardware.lose_pose`.")

    simulator = SimulatorArmInterface(SCENE_PATH, params.control_dt)
    simulator.launch_viewer_with_motion(
        lambda: run_losing_pose_motion(simulator, params)
    )


if __name__ == "__main__":
    main()
