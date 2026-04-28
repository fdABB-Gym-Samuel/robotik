"""Real Unitree G1 right-arm pre-reveal motion over the official low-level DDS path."""

from __future__ import annotations

import math
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_LOGS_DIR = PROJECT_ROOT / "runs" / "logs"
UNITREE_SDK_TRACE_LOG = RUN_LOGS_DIR / "unitree_arm_sdk_cdds.log"


class G1JointIndex:
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    kNotUsedJoint = 29


G1_NUM_MOTOR = 29

FULL_BODY_KP = [
    60.0, 60.0, 60.0, 100.0, 40.0, 40.0,
    60.0, 60.0, 60.0, 100.0, 40.0, 40.0,
    60.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
]

FULL_BODY_KD = [
    1.0, 1.0, 1.0, 2.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 2.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]


class Mode:
    PR = 0
    AB = 1


RIGHT_ARM_7_JOINTS = {
    "right_shoulder_pitch_joint": G1JointIndex.RightShoulderPitch,
    "right_shoulder_roll_joint": G1JointIndex.RightShoulderRoll,
    "right_shoulder_yaw_joint": G1JointIndex.RightShoulderYaw,
    "right_elbow_joint": G1JointIndex.RightElbow,
    "right_wrist_roll_joint": G1JointIndex.RightWristRoll,
    "right_wrist_pitch_joint": G1JointIndex.RightWristPitch,
    "right_wrist_yaw_joint": G1JointIndex.RightWristYaw,
}

RIGHT_ARM_5_JOINTS = {
    "right_shoulder_pitch_joint": G1JointIndex.RightShoulderPitch,
    "right_shoulder_roll_joint": G1JointIndex.RightShoulderRoll,
    "right_shoulder_yaw_joint": G1JointIndex.RightShoulderYaw,
    "right_elbow_joint": G1JointIndex.RightElbow,
    "right_wrist_roll_joint": G1JointIndex.RightWristRoll,
}

UPPER_BODY_7_JOINTS = {
    "left_shoulder_pitch_joint": G1JointIndex.LeftShoulderPitch,
    "left_shoulder_roll_joint": G1JointIndex.LeftShoulderRoll,
    "left_shoulder_yaw_joint": G1JointIndex.LeftShoulderYaw,
    "left_elbow_joint": G1JointIndex.LeftElbow,
    "left_wrist_roll_joint": G1JointIndex.LeftWristRoll,
    "left_wrist_pitch_joint": G1JointIndex.LeftWristPitch,
    "left_wrist_yaw_joint": G1JointIndex.LeftWristYaw,
    "right_shoulder_pitch_joint": G1JointIndex.RightShoulderPitch,
    "right_shoulder_roll_joint": G1JointIndex.RightShoulderRoll,
    "right_shoulder_yaw_joint": G1JointIndex.RightShoulderYaw,
    "right_elbow_joint": G1JointIndex.RightElbow,
    "right_wrist_roll_joint": G1JointIndex.RightWristRoll,
    "right_wrist_pitch_joint": G1JointIndex.RightWristPitch,
    "right_wrist_yaw_joint": G1JointIndex.RightWristYaw,
    "waist_yaw_joint": G1JointIndex.WaistYaw,
    "waist_roll_joint": G1JointIndex.WaistRoll,
    "waist_pitch_joint": G1JointIndex.WaistPitch,
}

UPPER_BODY_5_JOINTS = {
    "left_shoulder_pitch_joint": G1JointIndex.LeftShoulderPitch,
    "left_shoulder_roll_joint": G1JointIndex.LeftShoulderRoll,
    "left_shoulder_yaw_joint": G1JointIndex.LeftShoulderYaw,
    "left_elbow_joint": G1JointIndex.LeftElbow,
    "left_wrist_roll_joint": G1JointIndex.LeftWristRoll,
    "right_shoulder_pitch_joint": G1JointIndex.RightShoulderPitch,
    "right_shoulder_roll_joint": G1JointIndex.RightShoulderRoll,
    "right_shoulder_yaw_joint": G1JointIndex.RightShoulderYaw,
    "right_elbow_joint": G1JointIndex.RightElbow,
    "right_wrist_roll_joint": G1JointIndex.RightWristRoll,
    "waist_yaw_joint": G1JointIndex.WaistYaw,
    "waist_roll_joint": G1JointIndex.WaistRoll,
    "waist_pitch_joint": G1JointIndex.WaistPitch,
}


@dataclass(frozen=True)
class ArmHardwareConfig:
    interface: str | None = None
    domain_id: int = 0
    arm_dof: int = 7
    live: bool = False
    print_state: bool = False
    state_timeout_seconds: float = 5.0
    control_dt: float = 0.002
    setup_duration: float = 0.8
    beat_duration: float = 0.5
    beat_count: int = 3
    return_duration: float = 0.8
    release_duration: float = 0.6
    kp: float = 60.0
    kd: float = 1.5
    arm_amplitude: float = 0.18
    wrist_angle: float = -0.18

    def __post_init__(self) -> None:
        if self.arm_dof not in (5, 7):
            raise ValueError("arm_dof must be either 5 or 7 to match Unitree's official G1 arm examples.")


def commanded_right_arm_joints(config: ArmHardwareConfig) -> dict[str, int]:
    return RIGHT_ARM_7_JOINTS if config.arm_dof == 7 else RIGHT_ARM_5_JOINTS


def commanded_upper_body_joints(config: ArmHardwareConfig) -> dict[str, int]:
    return UPPER_BODY_7_JOINTS if config.arm_dof == 7 else UPPER_BODY_5_JOINTS


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def ease_in_out_cubic(alpha: float) -> float:
    alpha = clamp_unit(alpha)
    if alpha < 0.5:
        return 4.0 * alpha * alpha * alpha
    return 1.0 - pow(-2.0 * alpha + 2.0, 3.0) / 2.0


def blend_pose(start: dict[str, float], target: dict[str, float], alpha: float) -> dict[str, float]:
    alpha = clamp_unit(alpha)
    return {
        joint_name: (1.0 - alpha) * start[joint_name] + alpha * target[joint_name]
        for joint_name in start
    }


def add_joint_deltas(base_pose: dict[str, float], deltas: dict[str, float]) -> dict[str, float]:
    updated_pose = dict(base_pose)
    for joint_name, delta in deltas.items():
        updated_pose[joint_name] = updated_pose.get(joint_name, 0.0) + delta
    return updated_pose


def ready_right_arm_pose(config: ArmHardwareConfig) -> dict[str, float]:
    pose = {
        "right_shoulder_pitch_joint": -0.42,
        "right_shoulder_roll_joint": -0.62,
        "right_shoulder_yaw_joint": 0.18,
        "right_elbow_joint": 1.18,
        "right_wrist_roll_joint": -0.08,
        "right_wrist_pitch_joint": config.wrist_angle,
        "right_wrist_yaw_joint": -0.16,
    }
    joint_names = commanded_right_arm_joints(config)
    return {joint_name: pose[joint_name] for joint_name in joint_names}


def interface_ipv4_address(interface: str | None) -> str | None:
    if interface is None:
        return None
    try:
        import fcntl
        import struct
    except ModuleNotFoundError:
        return None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", interface[:15].encode("utf-8"))
        result = fcntl.ioctl(sock.fileno(), 0x8915, packed)
        return socket.inet_ntoa(result[20:24])
    except OSError:
        return None
    finally:
        sock.close()


def interface_link_status(interface: str | None) -> tuple[str | None, str | None]:
    if interface is None:
        return None, None

    net_root = Path("/sys/class/net") / interface
    try:
        operstate = (net_root / "operstate").read_text(encoding="utf-8").strip()
    except OSError:
        operstate = None

    try:
        carrier_raw = (net_root / "carrier").read_text(encoding="utf-8").strip()
        carrier = "up" if carrier_raw == "1" else "down"
    except OSError:
        carrier = None

    return operstate, carrier


def likely_wsl_nat_address(ipv4_address: str | None) -> bool:
    return ipv4_address is not None and ipv4_address.startswith("172.")


def wsl_network_warning(interface_ip: str | None) -> list[str]:
    if likely_wsl_nat_address(interface_ip):
        return [
            "WSL is using a `172.x.x.x` virtual/NAT-style address on the selected interface.",
            "That usually prevents DDS discovery from reaching the robot LAN.",
            "Use mirrored networking in WSL, native Linux, or run on the robot-side machine.",
        ]
    if "WSL_DISTRO_NAME" in os.environ:
        return [
            "WSL was detected. Mirrored networking seems active, so the old NAT issue may already be fixed.",
            "If `rt/lowstate` is still missing, the next suspects are robot-side topic publishing or Windows firewall filtering DDS traffic.",
        ]
    return []


def _configure_unitree_sdk_trace_output(interface: str | None) -> None:
    """Redirect the SDK's CycloneDDS trace log into the repo when using `--interface`."""

    if interface is None:
        return

    try:
        from unitree_sdk2py.core import channel as sdk_channel
        from unitree_sdk2py.core import channel_config as sdk_channel_config
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Real G1 arm control requires Unitree's official `unitree_sdk2py` package. "
            "Install it on the robot or control PC first, following "
            "https://github.com/unitreerobotics/unitree_sdk2_python"
        ) from exc

    RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = str(UNITREE_SDK_TRACE_LOG)
    patched_config = sdk_channel_config.ChannelConfigHasInterface.replace("/tmp/cdds.LOG", trace_path)
    sdk_channel_config.ChannelConfigHasInterface = patched_config
    sdk_channel.ChannelConfigHasInterface = patched_config


def _validate_requested_interface(interface: str | None) -> None:
    if interface is None:
        return

    net_root = Path("/sys/class/net") / interface
    if not net_root.exists():
        raise SystemExit(
            f"DDS interface `{interface}` does not exist on this machine. "
            "Choose one of the available network interfaces instead."
        )

    operstate, carrier = interface_link_status(interface)
    if carrier == "down":
        message = [
            f"DDS interface `{interface}` is present but has no carrier.",
            "The wired link is not active, so CycloneDDS will not be able to reach the robot on that NIC.",
        ]
        if operstate is not None:
            message.append(f"Kernel operstate: {operstate}.")
        message.append(
            "Connect the Ethernet cable, power the robot/network adapter, and confirm the interface comes up before retrying."
        )
        raise SystemExit(" ".join(message))


def beat_arm_offset(config: ArmHardwareConfig, local_time: float, beat_index: int) -> dict[str, float]:
    normalized = clamp_unit(local_time / config.beat_duration)
    downbeat = math.exp(-18.0 * normalized) if normalized > 0.0 else 1.0
    rebound = math.sin(math.pi * normalized)

    shoulder_pitch = -config.arm_amplitude * (0.95 * downbeat - 0.2 * rebound)
    elbow = 0.36 * downbeat - 0.08 * rebound
    wrist_pitch = 0.1 * downbeat
    wrist_roll = -0.03 * beat_index

    deltas = {
        "right_shoulder_pitch_joint": shoulder_pitch,
        "right_elbow_joint": elbow,
        "right_wrist_roll_joint": wrist_roll,
    }
    if config.arm_dof == 7:
        deltas["right_wrist_pitch_joint"] = wrist_pitch
    return deltas


class UnitreeArmLowLevelSession:
    def __init__(self, config: ArmHardwareConfig) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.utils.crc import CRC
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Real G1 arm control requires Unitree's official `unitree_sdk2py` package. "
                "Install it on the robot or control PC first, following "
                "https://github.com/unitreerobotics/unitree_sdk2_python"
            ) from exc

        _validate_requested_interface(config.interface)
        _configure_unitree_sdk_trace_output(config.interface)

        if config.interface is not None:
            ChannelFactoryInitialize(config.domain_id, config.interface)
        else:
            ChannelFactoryInitialize(config.domain_id)

        self._config = config
        self._right_arm_joints = commanded_right_arm_joints(config)
        self._low_state: Any | None = None
        self._mode_machine = 0
        self._crc = CRC()
        self._motion_switcher = MotionSwitcherClient()
        self._motion_switcher.SetTimeout(5.0)
        self._motion_switcher.Init()
        self._motion_switcher_warning = self._release_active_motion_mode()
        self._low_cmd_factory = unitree_hg_msg_dds__LowCmd_
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._low_state_handler, 10)

    @property
    def motion_switcher_warning(self) -> str | None:
        return self._motion_switcher_warning

    def _release_active_motion_mode(self) -> str | None:
        status, result = self._motion_switcher.CheckMode()
        if status != 0 or result is None:
            return (
                "MotionSwitcher RPC was unavailable, so the script could not confirm or release "
                "the currently active robot motion mode before taking low-level control. "
                "Continuing anyway because some setups do not expose that service."
            )

        deadline = time.perf_counter() + 15.0
        while result.get("name"):
            release_status, _ = self._motion_switcher.ReleaseMode()
            if release_status != 0:
                return (
                    f"MotionSwitcher RPC found active mode `{result['name']}` but could not release it "
                    "before low-level control. Continuing anyway, but another controller may still be competing."
                )
            if time.perf_counter() >= deadline:
                return (
                    f"Timed out while waiting for Unitree motion mode `{result['name']}` to release. "
                    "Continuing anyway, but another controller may still be competing."
                )
            time.sleep(1.0)
            status, result = self._motion_switcher.CheckMode()
            if status != 0 or result is None:
                return (
                    "MotionSwitcher RPC stopped responding after a release request. "
                    "Continuing anyway because low-level DDS control may still work."
                )
        return None

    def _low_state_handler(self, msg: Any) -> None:
        self._low_state = msg
        self._mode_machine = int(getattr(msg, "mode_machine", 0))

    def wait_for_state(self) -> Any:
        deadline = time.perf_counter() + self._config.state_timeout_seconds
        while self._low_state is None and time.perf_counter() < deadline:
            time.sleep(0.05)
        if self._low_state is None:
            interface = self._config.interface
            interface_ip = interface_ipv4_address(interface)
            diagnostic_lines = [
                "No `rt/lowstate` sample was received for the real G1 arm controller.",
                f"DDS domain ID: {self._config.domain_id}",
                f"DDS interface: {interface or 'auto'}",
            ]
            if interface_ip is not None:
                diagnostic_lines.append(f"Interface IPv4: {interface_ip}")
            diagnostic_lines.extend(wsl_network_warning(interface_ip))
            diagnostic_lines.extend(
                [
                    "Check that the robot is physically reachable on the same subnet,",
                    "that the chosen interface is the real robot-facing NIC,",
                    "and that the robot is publishing `rt/lowstate`.",
                ]
            )
            raise RuntimeError(
                " ".join(diagnostic_lines)
            )
        return self._low_state

    def current_motor_positions(self) -> list[float]:
        state = self.wait_for_state()
        return [float(state.motor_state[joint_index].q) for joint_index in range(G1_NUM_MOTOR)]

    def current_right_arm_pose(self) -> dict[str, float]:
        motor_positions = self.current_motor_positions()
        pose: dict[str, float] = {}
        for joint_name, joint_index in self._right_arm_joints.items():
            pose[joint_name] = motor_positions[joint_index]
        return pose

    def publish_targets(self, target_positions: list[float]) -> None:
        low_cmd = self._low_cmd_factory()
        low_cmd.mode_pr = Mode.PR
        low_cmd.mode_machine = self._mode_machine
        for joint_index, target in enumerate(target_positions):
            motor_cmd = low_cmd.motor_cmd[joint_index]
            motor_cmd.mode = 1
            motor_cmd.tau = 0.0
            motor_cmd.q = float(target)
            motor_cmd.dq = 0.0
            motor_cmd.kp = FULL_BODY_KP[joint_index]
            motor_cmd.kd = FULL_BODY_KD[joint_index]
        low_cmd.crc = self._crc.Crc(low_cmd)
        self._publisher.Write(low_cmd)

    def interpolate(self, start_positions: list[float], target_positions: list[float], duration: float) -> None:
        steps = max(1, int(duration / self._config.control_dt))
        for step in range(1, steps + 1):
            alpha = ease_in_out_cubic(step / steps)
            positions = [
                (1.0 - alpha) * start + alpha * target
                for start, target in zip(start_positions, target_positions, strict=True)
            ]
            if self._config.live:
                self.publish_targets(positions)
            time.sleep(self._config.control_dt)

    def hold(self, target_positions: list[float], duration: float) -> None:
        steps = max(1, int(duration / self._config.control_dt))
        for _ in range(steps):
            if self._config.live:
                self.publish_targets(target_positions)
            time.sleep(self._config.control_dt)


def _phase_pose(config: ArmHardwareConfig, ready_pose: dict[str, float], beat_index: int, local_time: float) -> dict[str, float]:
    return add_joint_deltas(ready_pose, beat_arm_offset(config, local_time, beat_index))


def _apply_right_arm_pose(
    config: ArmHardwareConfig,
    base_positions: list[float],
    right_arm_pose: dict[str, float],
) -> list[float]:
    positions = list(base_positions)
    for joint_name, joint_index in commanded_right_arm_joints(config).items():
        positions[joint_index] = right_arm_pose[joint_name]
    return positions


def run_pre_reveal_right_arm_hardware(config: ArmHardwareConfig) -> None:
    ready_right_pose = ready_right_arm_pose(config)

    if not config.live and not config.print_state:
        print("Dry run only. No real robot commands were sent.")
        print("Planned ready pose:")
        for joint_name, value in ready_right_pose.items():
            print(f"  {joint_name}: {value:+.3f}")
        return

    session = UnitreeArmLowLevelSession(config)
    if session.motion_switcher_warning is not None:
        print(f"Warning: {session.motion_switcher_warning}")
    start_positions = session.current_motor_positions()
    start_right_pose = session.current_right_arm_pose()
    ready_positions = _apply_right_arm_pose(config, start_positions, ready_right_pose)

    if config.print_state:
        print("Initial right-arm joint state:")
        for joint_name, value in start_right_pose.items():
            print(f"  {joint_name}: {value:+.3f}")

    if not config.live:
        print("Dry run only. No real robot commands were sent.")
        return

    print("Taking low-level control and moving the real G1 right arm into the concealed ready pose...")
    session.interpolate(start_positions, ready_positions, config.setup_duration)

    for beat_index in range(config.beat_count):
        print(f"Running pre-reveal beat {beat_index + 1}/{config.beat_count}...")
        steps = max(1, int(config.beat_duration / config.control_dt))
        for step in range(steps):
            local_time = step * config.control_dt
            right_pose = _phase_pose(config, ready_right_pose, beat_index, local_time)
            positions = _apply_right_arm_pose(config, ready_positions, right_pose)
            session.publish_targets(positions)
            time.sleep(config.control_dt)

    print("Returning the real G1 right arm to the concealed ready pose...")
    final_right_pose = _phase_pose(config, ready_right_pose, config.beat_count - 1, config.beat_duration)
    final_positions = _apply_right_arm_pose(config, ready_positions, final_right_pose)
    session.interpolate(final_positions, ready_positions, config.return_duration)
    print("Holding the ready pose briefly before returning control...")
    session.hold(ready_positions, config.release_duration)
