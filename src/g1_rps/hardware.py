"""Helpers for mapping simulated RPS gestures onto the real G1 Inspire hand."""

from __future__ import annotations

from dataclasses import dataclass
import time

from .poses import DEFAULT_SEQUENCE, HAND_GESTURE_RATIOS

RIGHT_HAND_CHANNEL_ORDER = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
)

HAND_SLOT_OFFSETS = {
    "right": 0,
    "left": 6,
}


@dataclass(frozen=True)
class HardwareConfig:
    sequence: tuple[str, ...] = DEFAULT_SEQUENCE
    transition_seconds: float = 1.0
    hold_seconds: float = 1.0
    rate_hz: float = 25.0
    hand: str = "right"
    domain_id: int = 0
    network_interface: str | None = None
    command_topic: str = "rt/inspire/cmd"
    state_topic: str = "rt/inspire/state"
    state_timeout_seconds: float = 1.0
    live: bool = False
    print_state: bool = False
    return_to_open: bool = True


def build_hardware_channels(gesture: str) -> tuple[float, float, float, float, float, float]:
    """Map the simulation gesture definition onto Unitree's 6-channel hand control.

    Unitree's official Inspire hand controller uses one normalized `q` value per:
    pinky, ring, middle, index, thumb-bend, thumb-rotation.

    The simulation pose library is more detailed, so we collapse it down:
    - finger bend channels average the two simulated joints per finger
    - thumb bend averages the bend joints
    - thumb rotation uses the simulated thumb base rotation directly

    Unitree's example uses `0 = close` and `1 = open`, while the MuJoCo ratios in
    this project trend toward `0 = open` and `1 = closed` for bend joints, so the
    finger bend values are inverted here.
    """

    try:
        ratios = HAND_GESTURE_RATIOS[gesture]
    except KeyError as exc:
        raise KeyError(
            f"Unknown gesture '{gesture}'. Available gestures: {', '.join(sorted(HAND_GESTURE_RATIOS))}"
        ) from exc

    return (
        _invert_average(ratios["right_little_1_joint"], ratios["right_little_2_joint"]),
        _invert_average(ratios["right_ring_1_joint"], ratios["right_ring_2_joint"]),
        _invert_average(ratios["right_middle_1_joint"], ratios["right_middle_2_joint"]),
        _invert_average(ratios["right_index_1_joint"], ratios["right_index_2_joint"]),
        _invert_average(
            ratios["right_thumb_2_joint"],
            ratios["right_thumb_3_joint"],
            ratios["right_thumb_4_joint"],
        ),
        _clamp01(ratios["right_thumb_1_joint"]),
    )


def run_hardware_sequence(config: HardwareConfig) -> None:
    sequence = tuple(config.sequence or DEFAULT_SEQUENCE)
    if not sequence:
        raise RuntimeError("The hardware gesture sequence cannot be empty.")

    print(f"Active hand: {config.hand}")
    print(f"Gesture sequence: {', '.join(sequence)}")
    print("Hardware channels use Unitree order: " + ", ".join(RIGHT_HAND_CHANNEL_ORDER))

    if not config.live:
        _print_dry_run(sequence, config.hand)
        return

    session = _create_dds_session(config)
    current = _initial_channels_from_state(session, config) or build_hardware_channels("paper")

    if config.print_state:
        print("Initial state:", _format_channels(current))

    for gesture in sequence:
        target = build_hardware_channels(gesture)
        print(f"Commanding gesture: {gesture} -> {_format_channels(target)}")
        _interpolate_and_publish(session, current, target, config)
        current = target
        _hold_target(session, target, config)

    if config.return_to_open:
        target = build_hardware_channels("paper")
        print(f"Returning to open pose -> {_format_channels(target)}")
        _interpolate_and_publish(session, current, target, config)
        _hold_target(session, target, config)


def _create_dds_session(config: HardwareConfig):
    try:
        from .unitree_dds import UnitreeDdsSession
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Live robot control requires the `cyclonedds` Python package. "
            "Install it in your active environment, for example "
            "`python -m pip install cyclonedds`, or use the repo's Nix shell."
        ) from exc

    try:
        return UnitreeDdsSession(
            domain_id=config.domain_id,
            network_interface=config.network_interface,
            command_topic=config.command_topic,
            state_topic=config.state_topic,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize the Unitree DDS session. "
            "Check the network interface, DDS domain ID, and whether the Inspire hand service is running."
        ) from exc


def _initial_channels_from_state(session, config: HardwareConfig) -> tuple[float, ...] | None:
    state = session.read_state(timeout_seconds=config.state_timeout_seconds)
    if state is None or len(state.states) < 12:
        return None
    return extract_hand_channels_from_state(state, config.hand)


def extract_hand_channels_from_state(state, hand: str) -> tuple[float, float, float, float, float, float]:
    offset = HAND_SLOT_OFFSETS[hand]
    return tuple(_clamp01(float(state.states[offset + index].q)) for index in range(6))


def _interpolate_and_publish(session, start: tuple[float, ...], end: tuple[float, ...], config: HardwareConfig) -> None:
    steps = max(1, int(round(config.transition_seconds * config.rate_hz)))
    sleep_seconds = 1.0 / config.rate_hz
    for step in range(1, steps + 1):
        alpha = step / steps
        sample = tuple((1.0 - alpha) * start[i] + alpha * end[i] for i in range(6))
        session.write(build_motor_commands(sample, config.hand))
        if config.print_state:
            print("  sent:", _format_channels(sample))
        time.sleep(sleep_seconds)


def _hold_target(session, target: tuple[float, ...], config: HardwareConfig) -> None:
    steps = max(1, int(round(config.hold_seconds * config.rate_hz)))
    sleep_seconds = 1.0 / config.rate_hz
    sample = build_motor_commands(target, config.hand)
    for _ in range(steps):
        session.write(sample)
        if config.print_state:
            state = session.read_state(timeout_seconds=0.0)
            if state is not None and len(state.states) >= 12:
                print("  state:", _format_channels(extract_hand_channels_from_state(state, config.hand)))
        time.sleep(sleep_seconds)


def build_motor_commands(
    active_channels: tuple[float, float, float, float, float, float],
    hand: str,
) -> "MotorCmds_":
    from .unitree_dds import MotorCmd_, MotorCmds_

    commands = [MotorCmd_() for _ in range(12)]
    active_offset = HAND_SLOT_OFFSETS[hand]
    inactive_offset = HAND_SLOT_OFFSETS["left" if hand == "right" else "right"]

    for index, value in enumerate(active_channels):
        commands[active_offset + index].q = _clamp01(value)

    for index in range(6):
        commands[inactive_offset + index].q = 1.0

    return MotorCmds_(cmds=commands)


def _print_dry_run(sequence: tuple[str, ...], hand: str) -> None:
    print("Dry-run only. Use `--live` to actually publish DDS commands to the robot.")
    for gesture in sequence:
        print(f"{gesture:>8}: {_format_channels(build_hardware_channels(gesture))} -> hand={hand}")


def _invert_average(*values: float) -> float:
    return _clamp01(1.0 - sum(values) / len(values))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _format_channels(channels: tuple[float, ...]) -> str:
    pieces = [f"{name}={value:.2f}" for name, value in zip(RIGHT_HAND_CHANNEL_ORDER, channels)]
    return ", ".join(pieces)
