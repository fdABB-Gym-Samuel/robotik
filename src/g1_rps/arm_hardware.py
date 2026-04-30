"""Real Unitree G1 right-arm pre-reveal motion over the low-level `rt/lowcmd` path.

This module previously used `rt/arm_sdk`, but on this robot that topic is gated
by the high-level arm service and our commands were silently dropped. We now
publish on `rt/lowcmd` directly, which requires:

* The high-level controller must be released first (hold L2+B then L2+R2 on the
  Unitree controller, or call `MotionSwitcherClient.SelectMode("")`). Otherwise
  the high-level controller will fight our commands.
* `mode_machine` from the latest `rt/lowstate` must be copied into every
  `LowCmd_` we publish. The robot rejects commands that do not match.
* Every motor (0..28) must be commanded each cycle. The non-arm joints are held
  at the position they had when the session opened, so the legs and torso stay
  put. If the robot is not suspended, expect it to be bearing weight on its
  legs the whole time, so make sure it is on a stand or the user is holding it.
"""

from __future__ import annotations

import math
import os
import socket
import time
from importlib import metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .unitree_sdk2_config import (
    configure_local_cyclonedds_log,
)


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
    60.0,
    60.0,
    60.0,
    100.0,
    40.0,
    40.0,
    60.0,
    60.0,
    60.0,
    100.0,
    40.0,
    40.0,
    60.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
]

FULL_BODY_KD = [
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
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


NUM_G1_MOTORS = 29


@dataclass(frozen=True)
class ArmHardwareConfig:
    interface: str | None = None
    domain_id: int = 0
    arm_dof: int = 7
    live: bool = False
    print_state: bool = False
    auto_release_mode: bool = True
    state_timeout_seconds: float = 5.0
    motion_switch_timeout_seconds: float = 5.0
    motion_switch_poll_interval: float = 0.5
    control_dt: float = 0.005
    setup_duration: float = 0.8
    beat_duration: float = 0.5
    beat_count: int = 3
    return_duration: float = 0.8
    release_duration: float = 0.6
    kp: float = 60.0
    kd: float = 1.5
    hold_kp: float = 60.0
    hold_kd: float = 1.5
    arm_amplitude: float = 0.02
    wrist_angle: float = -0.18

    def __post_init__(self) -> None:
        if self.arm_dof not in (5, 7):
            raise ValueError(
                "arm_dof must be either 5 or 7 to match Unitree's official G1 arm examples."
            )


def commanded_right_arm_joints(config: ArmHardwareConfig) -> dict[str, int]:
    return RIGHT_ARM_7_JOINTS if config.arm_dof == 7 else RIGHT_ARM_5_JOINTS


def commanded_upper_body_joints(config: ArmHardwareConfig) -> dict[str, int]:
    return UPPER_BODY_7_JOINTS if config.arm_dof == 7 else UPPER_BODY_5_JOINTS


RIGHT_ELBOW_90_DEG_Q = 0.262
RIGHT_ELBOW_125_DEG_Q = 0.873


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def ease_in_out_cubic(alpha: float) -> float:
    alpha = clamp_unit(alpha)
    if alpha < 0.5:
        return 4.0 * alpha * alpha * alpha
    return 1.0 - pow(-2.0 * alpha + 2.0, 3.0) / 2.0


def blend_pose(
    start: dict[str, float], target: dict[str, float], alpha: float
) -> dict[str, float]:
    alpha = clamp_unit(alpha)
    return {
        joint_name: (1.0 - alpha) * start[joint_name] + alpha * target[joint_name]
        for joint_name in start
    }


def add_joint_deltas(
    base_pose: dict[str, float], deltas: dict[str, float]
) -> dict[str, float]:
    updated_pose = dict(base_pose)
    for joint_name, delta in deltas.items():
        updated_pose[joint_name] = updated_pose.get(joint_name, 0.0) + delta
    return updated_pose


def ready_right_arm_pose(config: ArmHardwareConfig) -> dict[str, float]:
    return {
        "right_shoulder_pitch_joint": -0.785,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": RIGHT_ELBOW_90_DEG_Q,
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


def cyclonedds_version_warning() -> list[str]:
    try:
        version = metadata.version("cyclonedds")
    except metadata.PackageNotFoundError:
        return []
    except Exception:
        return []

    if version.startswith("0.10."):
        return []

    return [
        f"Installed Python `cyclonedds` version: {version}.",
        "Unitree's `unitree_sdk2_python` README expects `cyclonedds == 0.10.2`.",
        "This repo's Nix environment currently swaps that dependency to `11.0.1`, which may prevent `rt/lowstate` discovery or decoding.",
    ]


def beat_arm_offset(
    config: ArmHardwareConfig, local_time: float, beat_index: int
) -> dict[str, float]:
    normalized = clamp_unit(local_time / config.beat_duration)
    extension = math.sin(math.pi * normalized)

    shoulder_pitch = config.arm_amplitude * extension
    elbow = (RIGHT_ELBOW_125_DEG_Q - RIGHT_ELBOW_90_DEG_Q) * extension
    wrist_pitch = 0.04 * extension
    wrist_roll = -0.03 * beat_index

    deltas = {
        "right_shoulder_pitch_joint": shoulder_pitch,
        "right_elbow_joint": elbow,
        "right_wrist_roll_joint": wrist_roll,
    }
    if config.arm_dof == 7:
        deltas["right_wrist_pitch_joint"] = wrist_pitch
    return deltas


class UnitreeLowCmdSession:
    """Low-level controller for the right arm using `rt/lowcmd`.

    All 29 G1 joints are commanded every cycle. The 7 right-arm joints follow
    the requested pose; the remaining 22 joints are held at the position they
    had when the session opened. The robot's `mode_machine` is mirrored back
    on every command, which the firmware requires.
    """

    def __init__(self, config: ArmHardwareConfig) -> None:
        try:
            from cyclonedds.sub import DataReader
            from cyclonedds.topic import Topic
            from cyclonedds.util import duration as dds_duration
            from unitree_sdk2py.core.channel import ChannelFactory
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.utils.crc import CRC
        except (ModuleNotFoundError, ImportError) as exc:
            raise SystemExit(
                "Real G1 arm control requires Unitree's official `unitree_sdk2py` package. "
                "Install it on the robot or control PC first, following "
                "https://github.com/unitreerobotics/unitree_sdk2_python"
            ) from exc

        log_path = configure_local_cyclonedds_log()

        if config.interface is not None:
            ChannelFactoryInitialize(config.domain_id, config.interface)
        else:
            ChannelFactoryInitialize(config.domain_id)

        self._config = config
        self._right_arm_joints = commanded_right_arm_joints(config)
        self._low_state: Any | None = None
        self._last_lowstate_error: Exception | None = None
        self._crc = CRC()
        self._motion_switcher = MotionSwitcherClient()
        self._motion_switcher.SetTimeout(5.0)
        self._motion_switcher.Init()
        self._motion_switcher_warning = self._release_active_motion_mode()
        self._low_cmd_factory = unitree_hg_msg_dds__LowCmd_
        self._maybe_release_high_level_mode()
        self._channel_factory = ChannelFactory()
        self._lowstate_participant = getattr(
            self._channel_factory,
            "_ChannelFactory__participant",
            None,
        )
        if self._lowstate_participant is None:
            raise RuntimeError(
                "Unitree ChannelFactory did not expose an initialized DDS participant."
            )
        self._lowstate_topic = Topic(
            self._lowstate_participant,
            "rt/lowstate",
            LowState_,
        )
        self._lowstate_reader = DataReader(
            self._lowstate_participant,
            self._lowstate_topic,
        )
        self._dds_duration = dds_duration
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._low_state_handler, 10)
        self._right_arm_indices = set(RIGHT_ARM_JOINTS.values())
        self._hold_q: list[float] = [0.0] * NUM_G1_MOTORS
        self._cyclonedds_log_path = log_path

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

    def _take_lowstate_sample(self) -> Any | None:
        try:
            polled_state = self._lowstate_reader.take_one(
                timeout=self._dds_duration(seconds=0.2)
            )
        except (TimeoutError, StopIteration):
            return None
        except Exception as exc:
            self._last_lowstate_error = exc
            raise RuntimeError(
                "Failed while reading `rt/lowstate` with the direct CycloneDDS "
                f"reader: {exc}"
            ) from exc

        if polled_state.__class__.__name__ == "InvalidSample":
            return None
        return polled_state

    def _maybe_release_high_level_mode(self) -> None:
        if not self._config.auto_release_mode:
            return

        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )
        except ModuleNotFoundError:
            print(
                "MotionSwitcherClient is unavailable in this Unitree SDK build. "
                "If the robot is still in a high-level arm mode, release it manually "
                "before running low-level arm control."
            )
            return

        client = MotionSwitcherClient()
        client.SetTimeout(self._config.motion_switch_timeout_seconds)
        client.Init()

        deadline = time.perf_counter() + self._config.motion_switch_timeout_seconds
        while time.perf_counter() < deadline:
            status, result = client.CheckMode()
            if status != 0 or result is None:
                print(
                    "MotionSwitcherClient.CheckMode did not respond. "
                    "Skipping automatic high-level mode release; release manually "
                    "with L2+B then L2+R2 before low-level arm control."
                )
                return
            if not result.get("name"):
                return

            release_status, _ = client.ReleaseMode()
            if release_status != 0:
                print(
                    "MotionSwitcherClient.ReleaseMode did not respond. "
                    "Skipping automatic high-level mode release; release manually "
                    "with L2+B then L2+R2 before low-level arm control."
                )
                return
            time.sleep(self._config.motion_switch_poll_interval)

        print(
            "Timed out while asking MotionSwitcherClient to release the high-level mode. "
            "Continuing anyway, but low-level arm control may not receive state or may "
            "fight the existing controller."
        )

    def wait_for_state(self) -> Any:
        deadline = time.perf_counter() + self._config.state_timeout_seconds
        while self._low_state is None and time.perf_counter() < deadline:
            polled_state = self._take_lowstate_sample()
            if polled_state is not None:
                self._low_state = polled_state
                break
            time.sleep(0.05)
        if self._low_state is None:
            interface = self._config.interface
            interface_ip = interface_ipv4_address(interface)
            diagnostic_lines = [
                "No `rt/lowstate` sample was received for the real G1 arm controller.",
                f"DDS domain ID: {self._config.domain_id}",
                f"DDS interface: {interface or 'auto'}",
                f"CycloneDDS log: {self._cyclonedds_log_path}",
            ]
            if interface_ip is not None:
                diagnostic_lines.append(f"Interface IPv4: {interface_ip}")
            if self._config.domain_id != 0:
                diagnostic_lines.append(
                    "Unitree examples usually use DDS domain ID 0; retry with "
                    "`--domain-id 0` unless this robot was configured differently."
                )
            if self._last_lowstate_error is not None:
                diagnostic_lines.append(
                    f"Last direct reader error: {self._last_lowstate_error}"
                )
            diagnostic_lines.extend(wsl_network_warning(interface_ip))
            diagnostic_lines.extend(cyclonedds_version_warning())
            diagnostic_lines.extend(
                [
                    "Check that the robot is physically reachable on the same subnet,",
                    "that the chosen interface is the real robot-facing NIC,",
                    "that the high-level controller has been released,",
                    "and that the robot is publishing `rt/lowstate`.",
                ]
            )
            raise RuntimeError("\n".join(diagnostic_lines))
        return self._low_state

    def capture_hold_pose(self) -> None:
        """Snapshot every joint's current position so non-arm joints can be held in place."""
        state = self.wait_for_state()
        self._hold_q = [float(state.motor_state[i].q) for i in range(NUM_G1_MOTORS)]

    def current_right_arm_pose(self) -> dict[str, float]:
        state = self.wait_for_state()
        return [
            float(state.motor_state[joint_index].q)
            for joint_index in range(G1_NUM_MOTOR)
        ]

    def current_right_arm_pose(self) -> dict[str, float]:
        motor_positions = self.current_motor_positions()
        pose: dict[str, float] = {}
        for joint_name, joint_index in self._right_arm_joints.items():
            pose[joint_name] = motor_positions[joint_index]
        return pose

    def publish_pose(self, pose: dict[str, float]) -> None:
        low_cmd = self._low_cmd_factory()

        # `mode_pr` selects the ankle parameterization (0 = PR / pitch-roll,
        # the default for G1). `mode_machine` is a token published by the
        # robot's firmware on `rt/lowstate`; the robot will reject a LowCmd
        # that does not echo the current value.
        low_cmd.mode_pr = 0
        if self._low_state is not None:
            low_cmd.mode_machine = self._low_state.mode_machine

        for joint_index in range(NUM_G1_MOTORS):
            motor_cmd = low_cmd.motor_cmd[joint_index]
            motor_cmd.mode = 1  # 1 = enable PD control on the joint
            motor_cmd.tau = 0.0
            motor_cmd.dq = 0.0
            motor_cmd.q = self._hold_q[joint_index]
            if joint_index in self._right_arm_indices:
                motor_cmd.kp = self._config.kp
                motor_cmd.kd = self._config.kd
            else:
                motor_cmd.kp = self._config.hold_kp
                motor_cmd.kd = self._config.hold_kd

        for joint_name, target in pose.items():
            joint_index = RIGHT_ARM_JOINTS[joint_name]
            low_cmd.motor_cmd[joint_index].q = float(target)

        low_cmd.crc = self._crc.Crc(low_cmd)
        self._publisher.Write(low_cmd)

    def interpolate(
        self,
        start_pose: dict[str, float],
        target_pose: dict[str, float],
        duration: float,
    ) -> None:
        steps = max(1, int(duration / self._config.control_dt))
        for step in range(1, steps + 1):
            alpha = ease_in_out_cubic(step / steps)
            positions = [
                (1.0 - alpha) * start + alpha * target
                for start, target in zip(start_positions, target_positions, strict=True)
            ]
            if self._config.live:
                self.publish_pose(pose)
            time.sleep(self._config.control_dt)

    def hold(self, pose: dict[str, float], duration: float) -> None:
        """Keep publishing the same arm pose for `duration` seconds.

        Unlike the old `arm_sdk` release, low-level mode has no way to gracefully
        hand control back. The high-level controller has to be re-engaged
        externally (L2+R2 on the controller, or `MotionSwitcherClient`).
        """
        steps = max(1, int(duration / self._config.control_dt))
        for _ in range(steps):
            if self._config.live:
                self.publish_pose(pose)
            time.sleep(self._config.control_dt)


# Backwards-compat alias so older imports of `UnitreeArmSdkSession` keep working.
UnitreeArmSdkSession = UnitreeLowCmdSession


def _phase_pose(
    config: ArmHardwareConfig,
    ready_pose: dict[str, float],
    beat_index: int,
    local_time: float,
) -> dict[str, float]:
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

    session = UnitreeLowCmdSession(config)
    session.capture_hold_pose()
    start_pose = session.current_right_arm_pose()

    if config.print_state:
        print("Initial right-arm joint state:")
        for joint_name, value in start_right_pose.items():
            print(f"  {joint_name}: {value:+.3f}")

    if not config.live:
        print("Dry run only. No real robot commands were sent.")
        return

    print(
        "Sending `rt/lowcmd`. Make sure the high-level controller is released "
        '(L2+B then L2+R2 on the controller, or MotionSwitcherClient.SelectMode("")) '
        "before this point, otherwise the high-level controller will fight these commands."
    )
    print("Moving real G1 right arm into the concealed ready pose...")
    session.interpolate(start_pose, ready_pose, config.setup_duration)

    for beat_index in range(config.beat_count):
        print(f"Running pre-reveal beat {beat_index + 1}/{config.beat_count}...")
        steps = max(1, int(config.beat_duration / config.control_dt))
        for step in range(steps):
            local_time = step * config.control_dt
            pose = _phase_pose(config, ready_pose, beat_index, local_time)
            session.publish_pose(pose)
            time.sleep(config.control_dt)

    print("Returning the real G1 right arm to the concealed ready pose...")
    session.interpolate(
        _phase_pose(config, ready_pose, config.beat_count - 1, config.beat_duration),
        ready_pose,
        config.return_duration,
    )
    print(
        "Holding ready pose. Re-engage the high-level controller externally when finished."
    )
    session.hold(ready_pose, config.release_duration)
