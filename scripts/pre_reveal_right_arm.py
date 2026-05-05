"""Run a Unitree G1 right-arm pre-reveal rock-paper-scissors motion.

This script only performs the common concealed pumping motion. It never
executes the final shoot thrust and never reveals rock, paper, or scissors.
"""

from __future__ import annotations

import contextlib
import math
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree


_nullcontext = contextlib.nullcontext

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as exc:  # pragma: no cover - local setup guard
    raise SystemExit(
        "MuJoCo Python bindings are not installed. Enter `nix develop` first "
        "before running this script."
    ) from exc

try:
    import trimesh
except ModuleNotFoundError as exc:  # pragma: no cover - local setup guard
    raise SystemExit(
        "The `trimesh` package is required to build the runtime-safe Unitree G1 scene. "
        "Enter `nix develop` first."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "assets" / "unitree_g1" / "g1_29dof_with_hand.xml"
RUNTIME_SCENE_PATH = (
    PROJECT_ROOT / "runs" / "assets" / "unitree_g1" / "g1_29dof_with_hand_runtime.xml"
)
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

    # Shoulder is held nearly fixed at ~45 deg forward flexion. Roll is zero so
    # the upper arm stays in the sagittal plane (in front of the torso) rather
    # than swung out to the side.
    shoulder_pitch_baseline: float = -0.785  # 45 deg forward flexion
    shoulder_pitch_amplitude: float = 0.02  # ~1 deg residual wobble only
    shoulder_roll: float = 0.0  # arm forward, not outward
    shoulder_yaw: float = 0.0  # no twist

    # Elbow is the sole driver of the pumping motion. In this URDF qpos ~1.833
    # is "straight"; flexion happens as qpos decreases below that. The interior
    # angle between forearm and upper arm sweeps roughly 90 deg (right angle)
    # to 125 deg (moderately straight).
    elbow_flex_baseline: float = 0.568  # midpoint, interior ~107 deg
    elbow_flex_amplitude: float = 0.306  # sweep +/- ~17.5 deg

    # Keep forearm and wrist nearly fixed.
    forearm_rotation: float = -0.08  # slight pronation proxy via wrist roll
    wrist_pitch: float = -0.18
    wrist_yaw: float = -0.16

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
    """Ready pose: my right hand is concealed and held in front of my torso.

    Matches phase 0 of `pumping_pose` so the cycle begins — and ends — with
    the elbow at its most extended position rather than mid-range.
    """

    return {
        "waist_yaw_joint": params.waist_yaw,
        "waist_roll_joint": params.waist_roll,
        "waist_pitch_joint": params.waist_pitch,
        "right_shoulder_pitch_joint": params.shoulder_pitch_baseline
        - params.shoulder_pitch_amplitude,
        "right_shoulder_roll_joint": params.shoulder_roll,
        "right_shoulder_yaw_joint": params.shoulder_yaw,
        "right_elbow_joint": params.elbow_flex_baseline + params.elbow_flex_amplitude,
        "right_wrist_roll_joint": params.forearm_rotation,
        "right_wrist_pitch_joint": params.wrist_pitch,
        "right_wrist_yaw_joint": params.wrist_yaw,
        **concealed_hand_pose(params),
    }


def pumping_pose(params: MotionParameters, cycle_phase: float) -> dict[str, float]:
    """Compact right-arm pumping motion with elbow as the primary driver."""

    # Phase-shifted cosine so phase 0 / 1 sit at the most-extended elbow pose;
    # the cycle boundary is the natural rest point we stop on.
    cosine = -math.cos(2.0 * math.pi * cycle_phase)

    shoulder_pitch = (
        params.shoulder_pitch_baseline + params.shoulder_pitch_amplitude * cosine
    )
    elbow_flex = params.elbow_flex_baseline - params.elbow_flex_amplitude * cosine

    # Shoulder is intentionally held still — elbow alone drives the motion.
    shoulder_yaw = params.shoulder_yaw

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
            "held near 45 deg forward, nearly still",
        ),
        (
            "right_shoulder_roll_joint",
            f"{params.shoulder_roll:+.2f}",
            "zero -> arm in sagittal plane",
        ),
        ("right_shoulder_yaw_joint", f"{params.shoulder_yaw:+.2f}", "zero -> no twist"),
        (
            "right_elbow_joint",
            f"{params.elbow_flex_baseline:+.2f} +/- {params.elbow_flex_amplitude:.2f}",
            "sole driver, interior angle 90 deg .. 125 deg",
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

    def __init__(
        self,
        xml_path: Path,
        control_dt: float,
        *,
        skip_runtime_prep: bool = False,
    ) -> None:
        self._mujoco = mujoco
        if not xml_path.exists():
            raise SystemExit(f"Unitree G1 scene not found: {xml_path}")
        # `skip_runtime_prep` lets callers load formats that MuJoCo handles
        # natively (e.g. URDFs with embedded `<mujoco>` blocks) without going
        # through the inline-mesh runtime XML preparation, which is specific
        # to the MJCF format.
        scene_path = xml_path if skip_runtime_prep else ensure_runtime_scene(xml_path)
        self.model = self._mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = self._mujoco.MjData(self.model)
        self.control_dt = control_dt
        self.current_targets = setup_pose(MotionParameters())
        # Detect whether the model has a floating base (root free joint) so we
        # know whether `qpos[:7]` is the floating-base pose or the first joint
        # angles. URDF-loaded models typically have no free joint and would
        # otherwise get their first leg joints clobbered by the standing-pose
        # write below.
        self._has_floating_base = self.model.njnt > 0 and int(
            self.model.jnt_type[0]
        ) == int(mujoco.mjtJoint.mjJNT_FREE)
        # The passive viewer renders from `self.data` on the main thread while
        # the motion runner mutates it on a worker thread. We hold the
        # viewer-provided lock around every mutation to avoid
        # `mj_copyDataVisual: attempting to copy mjData while stack is in use`.
        self.viewer = None
        # Signal used to stop the motion thread early when the viewer closes,
        # so it cannot still be writing into `self.data` while MuJoCo tears
        # down GL resources (which would segfault on exit).
        self.should_stop = threading.Event()

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
        lock_ctx = self.viewer.lock() if self.viewer is not None else _nullcontext()
        with lock_ctx:
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            if self._has_floating_base:
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
            if self.should_stop.is_set():
                return
            alpha = step / steps
            eased = 0.5 - 0.5 * math.cos(math.pi * alpha)
            interpolated = {
                joint_name: start_targets.get(joint_name, 0.0)
                + (merged_targets[joint_name] - start_targets.get(joint_name, 0.0))
                * eased
                for joint_name in merged_targets
            }
            self._apply_targets(interpolated)
            # Use Event.wait so a stop signal interrupts the sleep promptly.
            if self.should_stop.wait(self.control_dt):
                return

    def sleep(self, time_sec: float) -> None:
        if self.should_stop.wait(time_sec):
            return

    def launch_viewer_with_motion(self, motion_runner: callable) -> None:
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer
                self.should_stop.clear()
                thread = threading.Thread(target=motion_runner, daemon=False)
                try:
                    thread.start()
                    while viewer.is_running() and thread.is_alive():
                        viewer.sync()
                        time.sleep(self.control_dt)
                finally:
                    # Tell the worker to bail out and wait for it to actually
                    # leave `_apply_targets` before letting the viewer's
                    # `__exit__` tear down GL resources. Without this the
                    # daemon thread can be mid-write into `self.data` when GL
                    # cleanup runs, which segfaults.
                    self.should_stop.set()
                    thread.join(timeout=2.0)
                    self.viewer = None
        except Exception as error:
            self.viewer = None
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOG_PATH.write_text(traceback.format_exc())
            if sys.platform == "darwin" and "run under `mjpython` on macOS" in str(
                error
            ):
                print("Passive viewer is unavailable under plain `python` on macOS.")
                print(
                    "For the passive viewer, run: nix develop -c mjpython scripts/pre_reveal_right_arm.py"
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


def ensure_runtime_scene(scene_path: Path) -> Path:
    runtime_path = RUNTIME_SCENE_PATH
    runtime_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ElementTree.parse(scene_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    base_mesh_dir = (scene_path.parent / meshdir).resolve()

    asset = root.find("asset")
    if asset is None:
        raise SystemExit(f"Scene does not contain an <asset> section: {scene_path}")

    source_mtimes = [scene_path.stat().st_mtime]
    for mesh in asset.findall("mesh"):
        mesh_file = mesh.get("file")
        if not mesh_file:
            continue
        mesh_path = (base_mesh_dir / mesh_file).resolve()
        if mesh_path.exists():
            source_mtimes.append(mesh_path.stat().st_mtime)

    if runtime_path.exists() and runtime_path.stat().st_mtime >= max(source_mtimes):
        return runtime_path

    for mesh in asset.findall("mesh"):
        mesh_file = mesh.get("file")
        if not mesh_file:
            continue
        mesh_path = (base_mesh_dir / mesh_file).resolve()
        loaded = trimesh.load_mesh(mesh_path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        mesh.attrib.pop("file", None)
        mesh.set("vertex", _flatten_rows(loaded.vertices))
        mesh.set("face", _flatten_rows(loaded.faces))

    if compiler is not None and "meshdir" in compiler.attrib:
        compiler.attrib.pop("meshdir", None)

    ElementTree.indent(tree, space="  ")
    tree.write(runtime_path, encoding="utf-8", xml_declaration=False)
    return runtime_path


def _flatten_rows(rows) -> str:
    return " ".join(" ".join(f"{value:.9g}" for value in row) for row in rows)


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
