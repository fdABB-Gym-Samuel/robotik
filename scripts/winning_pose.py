"""MuJoCo simulation of the Unitree G1 winning pose.

Simulation counterpart of `winning_pose_hardware.py`. Loads the G1 scene,
raises both arms toward the ceiling into the celebration pose defined in
`g1_rps.arm_hardware.winning_pose`, and holds the pose in the MuJoCo
passive viewer until the window is closed. The legs and torso sit at the
model's neutral pose; only the arms are driven.
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

from g1_rps.arm_hardware import ArmHardwareConfig, winning_pose
from pre_reveal_right_arm import (
    JointInterface,
    SCENE_PATH,
    SimulatorArmInterface,
)


@dataclass(frozen=True)
class WinningPoseParameters:
    """Tunable knobs for the winning-pose simulation."""

    setup_duration: float = 1.5
    control_dt: float = 1.0 / 60.0
    waist_yaw: float = 0.0
    waist_roll: float = 0.0
    waist_pitch: float = 0.0


def winning_pose_targets(params: WinningPoseParameters) -> dict[str, float]:
    """Both arms raised, waist held neutral."""

    config = ArmHardwareConfig(arm_dof=7)
    targets = dict(winning_pose(config))
    targets.update(
        {
            "waist_yaw_joint": params.waist_yaw,
            "waist_roll_joint": params.waist_roll,
            "waist_pitch_joint": params.waist_pitch,
        }
    )
    return targets


def run_winning_pose_motion(
    robot: JointInterface, params: WinningPoseParameters
) -> None:
    """Move into the winning pose and hold it until the viewer closes."""

    targets = winning_pose_targets(params)
    robot.move_joints(targets, params.setup_duration)

    # Hold the pose until the simulator interface signals stop (viewer closed).
    # Calling `move_joints` with the same target every tick keeps the model
    # pinned at `targets` even if a UI slider nudges a joint.
    while not robot.should_stop.wait(params.control_dt):
        robot.move_joints(targets, params.control_dt)


def main() -> None:
    params = WinningPoseParameters()
    print("Winning-pose simulation (Unitree G1 in MuJoCo).")
    print("Raising both arms toward the ceiling to match")
    print("`scripts/winning_pose_hardware.py`.")

    simulator = SimulatorArmInterface(SCENE_PATH, params.control_dt)
    simulator.launch_viewer_with_motion(
        lambda: run_winning_pose_motion(simulator, params)
    )


if __name__ == "__main__":
    main()
