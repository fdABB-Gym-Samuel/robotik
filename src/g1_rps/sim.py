"""MuJoCo demo runner for the exact Unitree Inspire hand presentation."""

from __future__ import annotations

import time
from pathlib import Path

from .assets import ensure_unitree_g1_assets
from .poses import DEFAULT_SEQUENCE, POSED_JOINTS, DemoConfig, build_hand_pose


def run_demo(config: DemoConfig) -> None:
    mujoco, viewer = _import_mujoco()

    model_path = ensure_unitree_g1_assets(Path(config.asset_dir) if config.asset_dir else None)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    _set_base_pose(data)

    joint_ranges = _collect_joint_ranges(mujoco, model)
    _validate_required_joints(joint_ranges)

    gesture_sequence = tuple(config.sequence or DEFAULT_SEQUENCE)
    current_index = 0
    current_gesture = gesture_sequence[current_index]
    next_gesture = gesture_sequence[(current_index + 1) % len(gesture_sequence)]

    start_pose = build_hand_pose(current_gesture, joint_ranges)
    end_pose = build_hand_pose(next_gesture, joint_ranges)

    _apply_pose(model, data, start_pose)
    mujoco.mj_forward(model, data)

    focus_body_name = "right_base_link"
    focus_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, focus_body_name)
    if focus_body_id < 0:
        raise RuntimeError(f"Body '{focus_body_name}' was not found in the Inspire hand model.")

    print(
        "Launching exact Unitree Inspire hand demo with gesture sequence: "
        + ", ".join(gesture_sequence)
    )
    print(f"Using official asset: {model_path}")

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as handle:
        _configure_camera(handle, data, focus_body_id, config.camera_preset)

        phase = "hold"
        phase_started = time.perf_counter()
        displayed_gesture = current_gesture
        print(f"Showing gesture: {displayed_gesture}")

        while handle.is_running():
            now = time.perf_counter()
            elapsed = now - phase_started

            if phase == "hold":
                _apply_pose(model, data, start_pose)
                if elapsed >= config.hold_seconds:
                    phase = "transition"
                    phase_started = now
            else:
                alpha = min(1.0, elapsed / config.transition_seconds)
                interpolated = _interpolate_pose(start_pose, end_pose, alpha)
                _apply_pose(model, data, interpolated)
                if alpha >= 1.0:
                    current_index = (current_index + 1) % len(gesture_sequence)
                    current_gesture = gesture_sequence[current_index]
                    next_gesture = gesture_sequence[(current_index + 1) % len(gesture_sequence)]
                    start_pose = end_pose
                    end_pose = build_hand_pose(next_gesture, joint_ranges)
                    phase = "hold"
                    phase_started = now
                    if current_gesture != displayed_gesture:
                        displayed_gesture = current_gesture
                        print(f"Showing gesture: {displayed_gesture}")

            mujoco.mj_forward(model, data)
            _configure_camera(handle, data, focus_body_id, config.camera_preset)
            handle.sync()
            time.sleep(1.0 / 60.0)


def _import_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The demo requires the `mujoco` Python package. Enter `nix develop` first."
        ) from exc

    return mujoco, mujoco.viewer


def _collect_joint_ranges(mujoco, model) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            continue
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        low, high = model.jnt_range[joint_id]
        ranges[name] = (float(low), float(high))
    return ranges


def _validate_required_joints(joint_ranges: dict[str, tuple[float, float]]) -> None:
    missing = [joint_name for joint_name in POSED_JOINTS if joint_name not in joint_ranges]
    if missing:
        raise RuntimeError(
            "The official Inspire hand asset is missing expected joints: " + ", ".join(missing)
        )


def _set_base_pose(data) -> None:
    data.qpos[:7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0


def _apply_pose(model, data, pose: dict[str, float]) -> None:
    for joint_name, value in pose.items():
        joint_id = model.joint(joint_name).id
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = value
        qvel_adr = model.jnt_dofadr[joint_id]
        if qvel_adr >= 0:
            data.qvel[qvel_adr] = 0.0


def _interpolate_pose(
    start_pose: dict[str, float],
    end_pose: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    keys = set(start_pose) | set(end_pose)
    return {
        key: (1.0 - alpha) * start_pose.get(key, 0.0) + alpha * end_pose.get(key, 0.0)
        for key in keys
    }


def _configure_camera(handle, data, focus_body_id: int, preset: str) -> None:
    base_pos = data.xpos[focus_body_id]
    handle.cam.lookat[:] = base_pos
    handle.cam.lookat[0] += 0.03
    handle.cam.lookat[1] -= 0.04

    if preset == "upper_body":
        handle.cam.distance = 0.42
        handle.cam.azimuth = 135
        handle.cam.elevation = -8
        return

    handle.cam.distance = 0.24
    handle.cam.azimuth = 128
    handle.cam.elevation = -18
