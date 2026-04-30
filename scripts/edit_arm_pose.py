"""Interactive MuJoCo pose editor for the Unitree G1 right arm.

Loads the G1 scene, seeds the right arm with the current
`ready_right_arm_pose` from `arm_hardware.py`, then opens the MuJoCo
passive viewer. Use the viewer's built-in joint UI to drag each joint:

  In the viewer:
    - Open the right-side panel (toolbar icon at top-right, or press `]`).
    - Expand the "Joint" group.
    - Drag the slider for any `right_*_joint` to a new value.

Keyboard shortcuts (focus the viewer window first):

    P   print the current right-arm pose as a Python dict ready to paste
        into `ready_right_arm_pose` in `src/g1_rps/arm_hardware.py`
    S   save the current right-arm pose to
        `runs/poses/right_arm_<timestamp>.json`
    L   list saved pose JSON files in `runs/poses`
    R   reset the right arm back to the seeded ready pose

This script does not step physics. It only kinematics-forwards the model
each frame so visuals reflect slider edits. The robot stays planted at
its initial base pose; nothing balances or falls.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
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
    import mujoco.viewer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MuJoCo Python bindings are not installed. Enter `nix develop` first."
    ) from exc

from g1_rps.arm_hardware import (
    ArmHardwareConfig,
    RIGHT_ARM_JOINTS,
    ready_right_arm_pose,
)
from pre_reveal_right_arm import SCENE_PATH, ensure_runtime_scene


POSE_DIR = PROJECT_ROOT / "runs" / "poses"


def apply_pose(model, data, pose: dict[str, float]) -> None:
    """Write pose values into qpos, clamping to each joint's MuJoCo range."""
    for joint_name, value in pose.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_index = model.jnt_qposadr[joint_id]
        if model.jnt_limited[joint_id]:
            low, high = model.jnt_range[joint_id]
            value = max(float(low), min(float(high), float(value)))
        data.qpos[qpos_index] = float(value)


def read_right_arm_pose(model, data) -> dict[str, float]:
    pose: dict[str, float] = {}
    for joint_name in RIGHT_ARM_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_index = model.jnt_qposadr[joint_id]
        pose[joint_name] = float(data.qpos[qpos_index])
    return pose


def print_pose(model, data) -> None:
    pose = read_right_arm_pose(model, data)
    print()
    print("# --- right-arm pose ---")
    print("# Paste into ready_right_arm_pose() in src/g1_rps/arm_hardware.py:")
    print("    return {")
    for joint_name, value in pose.items():
        print(f'        "{joint_name}": {value:+.4f},')
    print("    }")
    print()


def save_pose(model, data) -> Path:
    pose = read_right_arm_pose(model, data)
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = POSE_DIR / f"right_arm_{timestamp}.json"
    out_path.write_text(json.dumps(pose, indent=2) + "\n", encoding="utf-8")
    print(f"Saved right-arm pose -> {out_path}")
    return out_path


def list_saved_poses() -> None:
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(POSE_DIR.glob("right_arm_*.json"))
    if not files:
        print(f"No saved poses yet in {POSE_DIR}")
        return
    print(f"Saved poses in {POSE_DIR}:")
    for path in files:
        print(f"  {path.name}")


def main() -> None:
    runtime_scene = ensure_runtime_scene(SCENE_PATH)
    model = mujoco.MjModel.from_xml_path(str(runtime_scene))
    data = mujoco.MjData(model)

    # Stand the floating base where the demo script also places it.
    data.qpos[:7] = [0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0]

    seeded_pose = ready_right_arm_pose(ArmHardwareConfig())
    apply_pose(model, data, seeded_pose)
    mujoco.mj_forward(model, data)

    print("Right-arm pose editor")
    print("---------------------")
    print("Open the viewer's right-side menu and expand 'Joint' to get sliders.")
    print("Drag any `right_*_joint` to edit it live.")
    print("Keys: [P]rint pose  [S]ave pose  [L]ist saved  [R]eset to ready")
    print(f"Saved poses go to: {POSE_DIR}")

    def key_callback(keycode: int) -> None:
        try:
            char = chr(keycode).lower()
        except (ValueError, OverflowError):
            return
        if char == "p":
            print_pose(model, data)
        elif char == "s":
            save_pose(model, data)
        elif char == "l":
            list_saved_poses()
        elif char == "r":
            apply_pose(model, data, seeded_pose)
            mujoco.mj_forward(model, data)
            print("Reset to seeded ready pose.")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=True, show_right_ui=True
    ) as viewer:
        while viewer.is_running():
            with viewer.lock():
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
