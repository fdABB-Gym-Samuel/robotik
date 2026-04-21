"""Run a Unitree G1 right-arm pre-reveal rock-paper-scissors motion.

This script only performs the common concealed pumping motion. It never
executes the final shoot thrust and never reveals rock, paper, or scissors.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as exc:  # pragma: no cover - local setup guard
    raise SystemExit(
        "MuJoCo Python bindings are not installed. Activate .venv and install "
        "requirements before running this script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "assets" / "unitree_g1" / "g1_29dof_with_hand.xml"
LOG_PATH = PROJECT_ROOT / "runs" / "logs" / "pre-reveal-error.log"


class JointInterface(Protocol):
    """Generic robot-style interface for joint-space control."""

    def set_joint_position(
        self, joint_name: str, angle: float, time_sec: float
    ) -> None: ...

    def move_joints(self, joint_targets: dict[str, float], time_sec: float) -> None: ...

    def sleep(self, time_sec: float) -> None: ...


@dataclass(frozen=True)
class MotionParameters:
    """All tunable motion parameters in one place, from my own perspective."""

    setup_duration: float = 0.8
    cycle_duration: float = 0.62
    stop_duration: float = 0.5
    cycles: int = 3
    control_dt: float = 1.0 / 60.0

    # Torso stays upright and quiet while I place my right arm forward.
    waist_yaw: float = -0.02
    waist_roll: float = 0.0
    waist_pitch: float = 0.02

    # Right shoulder places my hand in front of my torso.
    shoulder_pitch_baseline: float = -0.52  # about 30 deg flexion forward
    shoulder_pitch_amplitude: float = 0.11  # about 6 deg oscillation
    shoulder_roll: float = -0.40
    shoulder_yaw: float = 0.14

    # Elbow is the main driver of the pumping motion.
    elbow_flex_baseline: float = 1.66  # about 95 deg flexion
    elbow_flex_amplitude: float = 0.22  # about 12.5 deg oscillation

    # Keep forearm and wrist nearly fixed.
    forearm_rotation: float = -0.08  # slight pronation proxy via wrist roll
    wrist_pitch: float = -0.16
    wrist_yaw: float = -0.12

    # Keep the hand concealed in a neutral pre-shape.
    finger_flex: float = 0.34
    thumb_base_flex: float = 0.22
    thumb_mid_flex: float = -0.10
    thumb_tip_flex: float = -0.22


def concealed_hand_pose(params: MotionParameters) -> dict[str, float]:
    """Neutral concealed hand shape with no reveal gesture."""

    return {
        "right_hand_thumb_0_joint": params.thumb_base_flex,
        "right_hand_thumb_1_joint": params.thumb_mid_flex,
        "right_hand_thumb_2_joint": params.thumb_tip_flex,
        "right_hand_index_0_joint": params.finger_flex,
        "right_hand_index_1_joint": params.finger_flex,
        "right_hand_middle_0_joint": params.finger_flex,
        "right_hand_middle_1_joint": params.finger_flex,
    }


def setup_pose(params: MotionParameters) -> dict[str, float]:
    """Ready pose: my right hand is concealed and held in front of my torso."""

    return {
        "waist_yaw_joint": params.waist_yaw,
        "waist_roll_joint": params.waist_roll,
        "waist_pitch_joint": params.waist_pitch,
        "right_shoulder_pitch_joint": params.shoulder_pitch_baseline,
        "right_shoulder_roll_joint": params.shoulder_roll,
        "right_shoulder_yaw_joint": params.shoulder_yaw,
        "right_elbow_joint": params.elbow_flex_baseline,
        "right_wrist_roll_joint": params.forearm_rotation,
        "right_wrist_pitch_joint": params.wrist_pitch,
        "right_wrist_yaw_joint": params.wrist_yaw,
        **concealed_hand_pose(params),
    }


def pumping_pose(params: MotionParameters, cycle_phase: float) -> dict[str, float]:
    """Compact right-arm pumping motion with elbow as the primary driver."""

    # Smooth periodic trajectory: cosine keeps turnarounds gentle.
    cosine = math.cos(2.0 * math.pi * cycle_phase)

    shoulder_pitch = (
        params.shoulder_pitch_baseline + params.shoulder_pitch_amplitude * cosine
    )
    elbow_flex = params.elbow_flex_baseline - params.elbow_flex_amplitude * cosine

    # Small shoulder compensation keeps the hand path nearly vertical.
    shoulder_yaw = params.shoulder_yaw + 0.02 * math.sin(2.0 * math.pi * cycle_phase)

    return {
        "waist_yaw_joint": params.waist_yaw,
        "waist_roll_joint": params.waist_roll,
        "waist_pitch_joint": params.waist_pitch,
        "right_shoulder_pitch_joint": shoulder_pitch,
        "right_shoulder_roll_joint": params.shoulder_roll,
        "right_shoulder_yaw_joint": shoulder_yaw,
        "right_elbow_joint": elbow_flex,
        "right_wrist_roll_joint": params.forearm_rotation,
        "right_wrist_pitch_joint": params.wrist_pitch,
        "right_wrist_yaw_joint": params.wrist_yaw,
        **concealed_hand_pose(params),
    }


def neutral_stop_pose(params: MotionParameters) -> dict[str, float]:
    """Stop with my hand still concealed and never revealed."""

    return setup_pose(params)


def joint_motion_table(params: MotionParameters) -> str:
    """Human-readable joint summary for quick inspection."""

    rows = [
        ("waist_yaw_joint", f"{params.waist_yaw:+.2f}", "stable torso alignment"),
        ("waist_roll_joint", f"{params.waist_roll:+.2f}", "upright torso"),
        (
            "waist_pitch_joint",
            f"{params.waist_pitch:+.2f}",
            "small forward presentation",
        ),
        (
            "right_shoulder_pitch_joint",
            f"{params.shoulder_pitch_baseline:+.2f} +/- {params.shoulder_pitch_amplitude:.2f}",
            "small cyclic compensation arc",
        ),
        (
            "right_shoulder_roll_joint",
            f"{params.shoulder_roll:+.2f}",
            "holds arm in front of torso",
        ),
        (
            "right_shoulder_yaw_joint",
            f"{params.shoulder_yaw:+.2f} +/- 0.02",
            "tiny fore-aft arc",
        ),
        (
            "right_elbow_joint",
            f"{params.elbow_flex_baseline:+.2f} +/- {params.elbow_flex_amplitude:.2f}",
            "primary pumping driver",
        ),
        (
            "right_wrist_roll_joint",
            f"{params.forearm_rotation:+.2f}",
            "near-neutral pronation",
        ),
        ("right_wrist_pitch_joint", f"{params.wrist_pitch:+.2f}", "quiet wrist"),
        ("right_wrist_yaw_joint", f"{params.wrist_yaw:+.2f}", "quiet wrist"),
        ("right_hand_*", f"{params.finger_flex:+.2f}", "concealed neutral pre-shape"),
    ]
    return "\n".join(f"{name:28} {value:18} {note}" for name, value, note in rows)


class SimulatorArmInterface(JointInterface):
    """Simulator-side implementation with joint-limit safety clamps."""

    def __init__(self, xml_path: Path, control_dt: float) -> None:
        if not xml_path.exists():
            raise SystemExit(f"Unitree G1 scene not found: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = control_dt
        self.current_targets = setup_pose(MotionParameters())

    def _joint_id(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(
                f"Joint '{joint_name}' was not found in the Unitree G1 model."
            )
        return joint_id

    def clamp_to_limits(self, joint_name: str, angle: float) -> float:
        joint_id = self._joint_id(joint_name)
        if self.model.jnt_limited[joint_id]:
            low, high = self.model.jnt_range[joint_id]
            return max(float(low), min(float(high), angle))
        return angle

    def set_joint_position(
        self, joint_name: str, angle: float, time_sec: float
    ) -> None:
        self.move_joints({joint_name: angle}, time_sec)

    def _apply_targets(self, joint_targets: dict[str, float]) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[:7] = [0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0]

        for joint_name, angle in joint_targets.items():
            joint_id = self._joint_id(joint_name)
            qpos_index = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_index] = self.clamp_to_limits(joint_name, angle)

        if self.model.nu:
            self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.current_targets = dict(joint_targets)

    def move_joints(self, joint_targets: dict[str, float], time_sec: float) -> None:
        start_targets = dict(self.current_targets)
        merged_targets = dict(start_targets)
        merged_targets.update(joint_targets)

        steps = max(1, int(time_sec / self.control_dt))
        for step in range(1, steps + 1):
            alpha = step / steps
            eased = 0.5 - 0.5 * math.cos(math.pi * alpha)
            interpolated = {
                joint_name: start_targets.get(joint_name, 0.0)
                + (merged_targets[joint_name] - start_targets.get(joint_name, 0.0))
                * eased
                for joint_name in merged_targets
            }
            self._apply_targets(interpolated)
            time.sleep(self.control_dt)

    def sleep(self, time_sec: float) -> None:
        time.sleep(time_sec)

    def launch_viewer_with_motion(self, motion_runner: callable) -> None:
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                thread = threading.Thread(target=motion_runner, daemon=True)
                thread.start()
                while viewer.is_running() and thread.is_alive():
                    viewer.sync()
                    time.sleep(self.control_dt)
                thread.join(timeout=1.0)
        except Exception as error:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOG_PATH.write_text(traceback.format_exc())
            if sys.platform == "darwin" and "run under `mjpython` on macOS" in str(
                error
            ):
                print("Passive viewer is unavailable under plain `python` on macOS.")
                print(
                    "For the passive viewer, run: source .venv/bin/activate && "
                    "mjpython scripts/pre_reveal_right_arm.py"
                )
            else:
                print(f"Passive viewer failed. Details saved to {LOG_PATH}")
            thread = threading.Thread(target=motion_runner, daemon=True)
            thread.start()
            mujoco.viewer._launch_internal(  # type: ignore[attr-defined]
                self.model,
                self.data,
                run_physics_thread=False,
                show_left_ui=True,
                show_right_ui=True,
            )
            thread.join(timeout=1.0)


def run_pre_reveal_motion(robot: JointInterface, params: MotionParameters) -> None:
    """Execute setup, cyclic pumping, and neutral stop phases."""

    robot.move_joints(setup_pose(params), params.setup_duration)

    for cycle_index in range(params.cycles):
        del cycle_index  # explicit: all cycles are identical by design
        steps = max(1, int(params.cycle_duration / params.control_dt))
        for step in range(steps):
            cycle_phase = step / steps
            robot.move_joints(pumping_pose(params, cycle_phase), params.control_dt)

    robot.move_joints(neutral_stop_pose(params), params.stop_duration)


def main() -> None:
    params = MotionParameters()
    print("Right-arm pre-reveal control strategy:")
    print("I keep my torso stable, hold my concealed right hand in front of my torso,")
    print("and drive a compact rhythmic pumping motion mainly with my elbow while my")
    print("shoulder adds a small compensating arc. I stop in the concealed ready pose")
    print("before any reveal motion or sign-specific finger change.")
    print()
    print("Joint-level motion table:")
    print(joint_motion_table(params))

    simulator = SimulatorArmInterface(SCENE_PATH, params.control_dt)
    simulator.launch_viewer_with_motion(
        lambda: run_pre_reveal_motion(simulator, params)
    )


if __name__ == "__main__":
    main()
