"""Helpers for mapping simulated RPS gestures onto the real G1 Inspire hand."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

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
    allow_state_fallback: bool = False


def build_hardware_channels(
    gesture: str,
) -> tuple[float, float, float, float, float, float]:
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
    if config.hand != "right":
        raise RuntimeError(
            "Left-hand hardware control is not enabled yet. "
            "The current gesture mapping is calibrated only for the right Inspire hand."
        )

    print(f"Active hand: {config.hand}")
    print(f"Gesture sequence: {', '.join(sequence)}")
    print("Hardware channels use Unitree order: " + ", ".join(RIGHT_HAND_CHANNEL_ORDER))

    if not config.live:
        _print_dry_run(sequence, config.hand)
        return

    session = _create_dds_session(config)
    initial_state = _require_initial_state(session, config)
    current = extract_hand_channels_from_state(initial_state, config.hand)
    inactive_hand = "left" if config.hand == "right" else "right"
    inactive_channels = extract_hand_channels_from_state(initial_state, inactive_hand)

    if config.print_state:
        print("Initial state:", _format_channels(current))

    for gesture in sequence:
        target = build_hardware_channels(gesture)
        print(f"Commanding gesture: {gesture} -> {_format_channels(target)}")
        _interpolate_and_publish(session, current, target, inactive_channels, config)
        current = target
        _hold_target(session, target, inactive_channels, config)

    if config.return_to_open:
        target = build_hardware_channels("paper")
        print(f"Returning to open pose -> {_format_channels(target)}")
        _interpolate_and_publish(session, current, target, inactive_channels, config)
        _hold_target(session, target, inactive_channels, config)


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


def _initial_channels_from_state(
    session, config: HardwareConfig
) -> tuple[float, ...] | None:
    state = session.read_state(timeout_seconds=config.state_timeout_seconds)
    if state is None or len(state.states) < 12:
        return None
    return state


def _require_initial_state(session, config: HardwareConfig):
    state = _initial_channels_from_state(session, config)
    if state is not None:
        return state
    if config.allow_state_fallback:
        print(
            "Warning: no initial Inspire hand state received, falling back to a synthetic open-state baseline."
        )
        return _synthetic_open_state()
    raise RuntimeError(
        "No valid initial Inspire hand state was received. "
        "Refusing to send live commands from a guessed starting pose. "
        "Check the DDS connection, the network interface, and whether the robot-side Inspire hand service is running."
    )


def extract_hand_channels_from_state(
    state, hand: str
) -> tuple[float, float, float, float, float, float]:
    offset = HAND_SLOT_OFFSETS[hand]
    return tuple(_clamp01(float(state.states[offset + index].q)) for index in range(6))


def _interpolate_and_publish(
    session,
    start: tuple[float, ...],
    end: tuple[float, ...],
    inactive_channels: tuple[float, ...],
    config: HardwareConfig,
) -> None:
    steps = max(1, int(round(config.transition_seconds * config.rate_hz)))
    sleep_seconds = 1.0 / config.rate_hz
    for step in range(1, steps + 1):
        alpha = step / steps
        sample = tuple((1.0 - alpha) * start[i] + alpha * end[i] for i in range(6))
        session.write(build_motor_commands(sample, config.hand, inactive_channels))
        if config.print_state:
            print("  sent:", _format_channels(sample))
        time.sleep(sleep_seconds)


def _hold_target(
    session,
    target: tuple[float, ...],
    inactive_channels: tuple[float, ...],
    config: HardwareConfig,
) -> None:
    steps = max(1, int(round(config.hold_seconds * config.rate_hz)))
    sleep_seconds = 1.0 / config.rate_hz
    sample = build_motor_commands(target, config.hand, inactive_channels)
    for _ in range(steps):
        session.write(sample)
        if config.print_state:
            state = session.read_state(timeout_seconds=0.0)
            if state is not None and len(state.states) >= 12:
                print(
                    "  state:",
                    _format_channels(
                        extract_hand_channels_from_state(state, config.hand)
                    ),
                )
        time.sleep(sleep_seconds)


def build_motor_commands(
    active_channels: tuple[float, float, float, float, float, float],
    hand: str,
    inactive_channels: tuple[float, float, float, float, float, float],
) -> "MotorCmds_":
    from .unitree_dds import MotorCmd_, MotorCmds_

    commands = [MotorCmd_() for _ in range(12)]
    active_offset = HAND_SLOT_OFFSETS[hand]
    inactive_offset = HAND_SLOT_OFFSETS["left" if hand == "right" else "right"]

    for index, value in enumerate(active_channels):
        commands[active_offset + index].q = _clamp01(value)

    for index, value in enumerate(inactive_channels):
        commands[inactive_offset + index].q = _clamp01(value)

    return MotorCmds_(cmds=commands)


def _print_dry_run(sequence: tuple[str, ...], hand: str) -> None:
    print("Dry-run only. Use `--live` to actually publish DDS commands to the robot.")
    for gesture in sequence:
        print(
            f"{gesture:>8}: {_format_channels(build_hardware_channels(gesture))} -> hand={hand}"
        )


def _invert_average(*values: float) -> float:
    return _clamp01(1.0 - sum(values) / len(values))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _synthetic_open_state():
    from .unitree_dds import MotorState_, MotorStates_

    return MotorStates_(states=[MotorState_(q=1.0) for _ in range(12)])


def _format_channels(channels: tuple[float, ...]) -> str:
    pieces = [
        f"{name}={value:.2f}" for name, value in zip(RIGHT_HAND_CHANNEL_ORDER, channels)
    ]
    return ", ".join(pieces)


class RpsHandController:
    """Stateful Inspire-hand controller for the interactive RPS game loop.

    Unlike `run_hardware_sequence` (which plays a fixed gesture sequence on
    its own), this class is meant to be opened once, then called from the
    main game loop to drive the active hand to a chosen gesture at the
    moments that matter (pre-reveal fist, post-reveal gesture, between
    rounds, etc.).

    The DDS session is opened lazily in `open()` and stays alive until
    `close()`. In dry-run mode (`config.live=False`) all calls are no-ops
    that still update the cached channels, so the rest of the game loop
    can run unmodified without a robot connected.
    """

    def __init__(self, config: HardwareConfig) -> None:
        if config.hand != "right":
            raise RuntimeError(
                "RpsHandController is only calibrated for the right Inspire hand."
            )
        self._config = config
        self._session = None
        self._lock = threading.Lock()
        # Default to "all open" until we read real state in `open()`.
        self._current_channels: tuple[float, float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        self._inactive_channels: tuple[float, float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        )

    def open(self) -> None:
        """Connect to DDS and snapshot initial hand state. No-op in dry run."""
        if not self._config.live:
            return
        self._session = _create_dds_session(self._config)
        initial_state = _require_initial_state(self._session, self._config)
        self._current_channels = extract_hand_channels_from_state(
            initial_state, self._config.hand
        )
        inactive_hand = "left" if self._config.hand == "right" else "right"
        self._inactive_channels = extract_hand_channels_from_state(
            initial_state, inactive_hand
        )

    def transition_to(
        self, gesture: str, *, transition_seconds: float | None = None
    ) -> None:
        """Interpolate the active hand to the named gesture (synchronous).

        Safe to call from a worker thread; concurrent calls are serialized.
        """
        target = build_hardware_channels(gesture)
        with self._lock:
            if self._session is not None:
                cfg = self._config
                if transition_seconds is not None:
                    cfg = replace(cfg, transition_seconds=transition_seconds)
                _interpolate_and_publish(
                    self._session,
                    self._current_channels,
                    target,
                    self._inactive_channels,
                    cfg,
                )
            self._current_channels = target

    def close(self) -> None:
        """Stop holding the DDS session. Idempotent."""
        # cyclonedds objects clean themselves up on GC; just drop the reference.
        self._session = None
