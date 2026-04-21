"""Real Unitree G1 right-arm pre-reveal motion over the official arm DDS path."""

from __future__ import annotations

import math
import os
import socket
import time
from dataclasses import dataclass
from typing import Any


class G1JointIndex:
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


RIGHT_ARM_JOINTS = {
    "right_shoulder_pitch_joint": G1JointIndex.RightShoulderPitch,
    "right_shoulder_roll_joint": G1JointIndex.RightShoulderRoll,
    "right_shoulder_yaw_joint": G1JointIndex.RightShoulderYaw,
    "right_elbow_joint": G1JointIndex.RightElbow,
    "right_wrist_roll_joint": G1JointIndex.RightWristRoll,
    "right_wrist_pitch_joint": G1JointIndex.RightWristPitch,
    "right_wrist_yaw_joint": G1JointIndex.RightWristYaw,
}


@dataclass(frozen=True)
class ArmHardwareConfig:
    interface: str | None = None
    domain_id: int = 0
    live: bool = False
    print_state: bool = False
    state_timeout_seconds: float = 5.0
    control_dt: float = 0.02
    setup_duration: float = 0.8
    beat_duration: float = 0.5
    beat_count: int = 3
    return_duration: float = 0.8
    release_duration: float = 0.6
    kp: float = 60.0
    kd: float = 1.5
    arm_amplitude: float = 0.18
    wrist_angle: float = -0.18


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
    return {
        "right_shoulder_pitch_joint": -0.42,
        "right_shoulder_roll_joint": -0.62,
        "right_shoulder_yaw_joint": 0.18,
        "right_elbow_joint": 1.18,
        "right_wrist_roll_joint": -0.08,
        "right_wrist_pitch_joint": config.wrist_angle,
        "right_wrist_yaw_joint": -0.16,
    }


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


def beat_arm_offset(config: ArmHardwareConfig, local_time: float, beat_index: int) -> dict[str, float]:
    normalized = clamp_unit(local_time / config.beat_duration)
    downbeat = math.exp(-18.0 * normalized) if normalized > 0.0 else 1.0
    rebound = math.sin(math.pi * normalized)

    shoulder_pitch = -config.arm_amplitude * (0.95 * downbeat - 0.2 * rebound)
    elbow = 0.36 * downbeat - 0.08 * rebound
    wrist_pitch = 0.1 * downbeat
    wrist_roll = -0.03 * beat_index

    return {
        "right_shoulder_pitch_joint": shoulder_pitch,
        "right_elbow_joint": elbow,
        "right_wrist_pitch_joint": wrist_pitch,
        "right_wrist_roll_joint": wrist_roll,
    }


class UnitreeArmSdkSession:
    def __init__(self, config: ArmHardwareConfig) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.core.channel import ChannelSubscriber
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

        if config.interface is not None:
            ChannelFactoryInitialize(config.domain_id, config.interface)
        else:
            ChannelFactoryInitialize(config.domain_id)

        self._config = config
        self._low_state: Any | None = None
        self._crc = CRC()
        self._low_cmd_factory = unitree_hg_msg_dds__LowCmd_
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._low_state_handler, 10)

    def _low_state_handler(self, msg: Any) -> None:
        self._low_state = msg

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

    def current_right_arm_pose(self) -> dict[str, float]:
        state = self.wait_for_state()
        pose: dict[str, float] = {}
        for joint_name, joint_index in RIGHT_ARM_JOINTS.items():
            pose[joint_name] = float(state.motor_state[joint_index].q)
        return pose

    def publish_pose(self, pose: dict[str, float], enable_arm_sdk: float = 1.0) -> None:
        low_cmd = self._low_cmd_factory()
        low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = enable_arm_sdk
        for joint_name, target in pose.items():
            joint_index = RIGHT_ARM_JOINTS[joint_name]
            motor_cmd = low_cmd.motor_cmd[joint_index]
            motor_cmd.tau = 0.0
            motor_cmd.q = float(target)
            motor_cmd.dq = 0.0
            motor_cmd.kp = self._config.kp
            motor_cmd.kd = self._config.kd
        low_cmd.crc = self._crc.Crc(low_cmd)
        self._publisher.Write(low_cmd)

    def interpolate(self, start_pose: dict[str, float], target_pose: dict[str, float], duration: float) -> None:
        steps = max(1, int(duration / self._config.control_dt))
        for step in range(1, steps + 1):
            alpha = ease_in_out_cubic(step / steps)
            pose = blend_pose(start_pose, target_pose, alpha)
            if self._config.live:
                self.publish_pose(pose, enable_arm_sdk=1.0)
            time.sleep(self._config.control_dt)

    def release(self, pose: dict[str, float]) -> None:
        steps = max(1, int(self._config.release_duration / self._config.control_dt))
        for step in range(steps + 1):
            alpha = step / steps
            enable_arm_sdk = 1.0 - alpha
            if self._config.live:
                self.publish_pose(pose, enable_arm_sdk=enable_arm_sdk)
            time.sleep(self._config.control_dt)


def _phase_pose(config: ArmHardwareConfig, ready_pose: dict[str, float], beat_index: int, local_time: float) -> dict[str, float]:
    return add_joint_deltas(ready_pose, beat_arm_offset(config, local_time, beat_index))


def run_pre_reveal_right_arm_hardware(config: ArmHardwareConfig) -> None:
    ready_pose = ready_right_arm_pose(config)

    if not config.live and not config.print_state:
        print("Dry run only. No real robot commands were sent.")
        print("Planned ready pose:")
        for joint_name, value in ready_pose.items():
            print(f"  {joint_name}: {value:+.3f}")
        return

    session = UnitreeArmSdkSession(config)
    start_pose = session.current_right_arm_pose()

    if config.print_state:
        print("Initial right-arm joint state:")
        for joint_name, value in start_pose.items():
            print(f"  {joint_name}: {value:+.3f}")

    if not config.live:
        print("Dry run only. No real robot commands were sent.")
        return

    print("Moving real G1 right arm into the concealed ready pose...")
    session.interpolate(start_pose, ready_pose, config.setup_duration)

    for beat_index in range(config.beat_count):
        print(f"Running pre-reveal beat {beat_index + 1}/{config.beat_count}...")
        steps = max(1, int(config.beat_duration / config.control_dt))
        for step in range(steps):
            local_time = step * config.control_dt
            pose = _phase_pose(config, ready_pose, beat_index, local_time)
            session.publish_pose(pose, enable_arm_sdk=1.0)
            time.sleep(config.control_dt)

    print("Returning the real G1 right arm to the concealed ready pose...")
    session.interpolate(_phase_pose(config, ready_pose, config.beat_count - 1, config.beat_duration), ready_pose, config.return_duration)
    print("Releasing arm_sdk control...")
    session.release(ready_pose)
