"""Interactive Unitree G1 rock-paper-scissors loop.

Pipeline:
  1. Take control of the body via `rt/lowcmd` (the high-level controller must
     be released first -- L2+B then L2+R2 on the joystick).
  2. Spin up a daemon vision thread that grabs frames from the front camera
     and classifies the opponent's hand gesture with MediaPipe Hands. The
     camera streams continuously across rounds.
  3. Loop: prompt the user before each round. Press Enter to play, type `q`
     (or `n`/`quit`) + Enter to exit.
  4. While waiting for input, the main thread keeps publishing the ready pose
     so the lowcmd watchdog stays satisfied.
  5. On a play: run the pre-reveal rocking motion, randomly pick a robot
     gesture, drive the right Inspire hand into that shape (rock/paper/
     scissors), sample what the camera saw, compute the result, then
     move into the win pose (robot wins) or the lose pose (anything else).

Subsystems used:
  * Arm:    g1_rps.arm_hardware  (rt/lowcmd, unitree_sdk2py)
  * Hand:   g1_rps.hardware      (rt/inspire/cmd, cyclonedds)
  * Vision: g1_rps.vision.HandGestureClassifier  (MediaPipe Hand Landmarker)

Prerequisites for --live:
  * The robot is on a stand or being held: lowcmd makes us responsible for
    every joint, including the legs (held at their captured position).
  * High-level body controller is released (L2+B then L2+R2).
  * Front camera / multimedia service is running.
"""

from __future__ import annotations

import argparse
import importlib
import random
import select
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import cv2

from g1_rps.arm_hardware import (
    ArmHardwareConfig,
    UnitreeLowCmdSession,
    add_joint_deltas,
    beat_arm_offset,
    blend_pose,
    commanded_right_arm_joints,
    ease_in_out_cubic,
    lose_pose,
    ready_right_arm_pose,
    winning_pose,
)
from g1_rps.hardware import HardwareConfig, RpsHandController

ClassifierConfig = None
HandGestureClassifier = None
draw_landmarks = None


GESTURES: tuple[str, ...] = ("rock", "paper", "scissors")
WINS: frozenset[tuple[str, str]] = frozenset(
    {
        ("rock", "scissors"),
        ("scissors", "paper"),
        ("paper", "rock"),
    }
)

# Sampling tunables. After pre-reveal the opponent is still mid-transition
# from their rocking fist to their final gesture, so we wait, then take a
# small majority vote across consecutive camera frames.
SAMPLE_SETTLE_SECONDS = 0.4
SAMPLE_FRAMES = 3
SAMPLE_POLL_INTERVAL = 0.05  # must exceed the camera frame interval (~33 ms at 30 fps)
SAMPLE_TIMEOUT_SECONDS = 2.0


def determine_winner(robot: str, opponent: str | None) -> str:
    if opponent is None:
        return "no opponent gesture detected"
    if robot == opponent:
        return "tie"
    return "robot wins" if (robot, opponent) in WINS else "opponent wins"


class OpponentVisionThread(threading.Thread):
    """Continuously read frames from a video client and classify the gesture.

    The latest valid (non-``None``) gesture is kept under a lock; callers can
    sample ``latest`` at any time, and ``clear_latest`` between rounds so a
    stale detection from the previous round does not leak forward.
    """

    def __init__(
        self,
        video_client,
        classifier: HandGestureClassifier,
        display: bool = True,
        window_name: str = "G1 opponent view",
        initial_frame: np.ndarray | None = None,
    ) -> None:
        super().__init__(name="opponent-vision", daemon=True)
        self._client = video_client
        self._classifier = classifier
        self._display = display
        self._window_name = window_name
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_gesture: str | None = None
        self._latest_extended: tuple[str, ...] = ()
        # Non-sticky per-frame snapshot, used by the round sampler so a single
        # early detection does not drown out subsequent frames.
        self._last_frame_gesture: str | None = None
        self._last_frame_extended: tuple[str, ...] = ()
        self._frames_processed = 0
        self._bad_frame_count = 0
        self._last_bad_frame_report = 0.0
        self._error: BaseException | None = None
        self._initial_frame = initial_frame

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._initial_frame is not None:
                    frame = self._initial_frame
                    self._initial_frame = None
                else:
                    code, data = self._client.GetImageSample()
                    if code != 0:
                        time.sleep(0.05)
                        continue
                    frame = _decode_unitree_image_sample(data)
                if frame is None:
                    self._bad_frame_count += 1
                    now = time.monotonic()
                    if now - self._last_bad_frame_report >= 2.0:
                        self._last_bad_frame_report = now
                        print(
                            "  vision: waiting for non-empty camera image bytes "
                            f"({self._bad_frame_count} empty/invalid samples)",
                            flush=True,
                        )
                    time.sleep(0.05)
                    continue
                result = self._classifier.classify(frame)
                with self._lock:
                    self._frames_processed += 1
                    self._last_frame_gesture = result.gesture
                    self._last_frame_extended = result.extended_fingers
                    if result.gesture is not None:
                        self._latest_gesture = result.gesture
                        self._latest_extended = result.extended_fingers
                    latest_label = self._latest_gesture or "—"
                    latest_extended = self._latest_extended

                if self._display:
                    annotated = frame
                    if result.landmarks is not None:
                        draw_landmarks(annotated, result.landmarks)
                    _draw_label_overlay(annotated, latest_label, latest_extended)
                    cv2.imshow(self._window_name, annotated)
                    # waitKey is required for the GUI event loop to pump.
                    # Pressing 'q' in the window asks the thread to stop early.
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self._stop_event.set()
        except BaseException as exc:  # surface to main thread on join
            self._error = exc

    def stop(self) -> None:
        self._stop_event.set()

    def clear_latest(self) -> None:
        with self._lock:
            self._latest_gesture = None
            self._latest_extended = ()
            self._last_frame_gesture = None
            self._last_frame_extended = ()

    @property
    def last_frame(self) -> tuple[str | None, tuple[str, ...], int]:
        """Per-frame snapshot (gesture may be ``None``) tagged by frame counter."""
        with self._lock:
            return (
                self._last_frame_gesture,
                self._last_frame_extended,
                self._frames_processed,
            )

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def latest(self) -> tuple[str | None, tuple[str, ...], int]:
        with self._lock:
            return self._latest_gesture, self._latest_extended, self._frames_processed

    @property
    def error(self) -> BaseException | None:
        return self._error


def _arm_beat_pose(
    config: ArmHardwareConfig,
    ready_pose: dict[str, float],
    beat_index: int,
    local_time: float,
) -> dict[str, float]:
    return add_joint_deltas(ready_pose, beat_arm_offset(config, local_time, beat_index))


def _interpolate_pose(
    session: UnitreeLowCmdSession,
    config: ArmHardwareConfig,
    start_pose: dict[str, float],
    target_pose: dict[str, float],
    duration: float,
) -> None:
    steps = max(1, int(duration / config.control_dt))
    for step in range(1, steps + 1):
        alpha = ease_in_out_cubic(step / steps)
        pose = blend_pose(start_pose, target_pose, alpha)
        if config.live:
            session.publish_pose(pose)
        time.sleep(config.control_dt)


def _hold_pose(
    session: UnitreeLowCmdSession,
    config: ArmHardwareConfig,
    pose: dict[str, float],
    duration: float,
) -> None:
    steps = max(1, int(duration / config.control_dt))
    for _ in range(steps):
        if config.live:
            session.publish_pose(pose)
        time.sleep(config.control_dt)


def _sample_opponent_gesture(
    session: UnitreeLowCmdSession,
    config: ArmHardwareConfig,
    ready_pose: dict[str, float],
    vision_thread: OpponentVisionThread | None,
    *,
    settle_seconds: float = SAMPLE_SETTLE_SECONDS,
    sample_frames: int = SAMPLE_FRAMES,
    poll_interval: float = SAMPLE_POLL_INTERVAL,
    timeout_seconds: float = SAMPLE_TIMEOUT_SECONDS,
) -> tuple[str | None, tuple[str, ...], int]:
    """Wait for the opponent to commit, then majority-vote across frames.

    Returns ``(gesture, extended_fingers, total_frames_seen)``. The hold pose
    is republished throughout so the lowcmd watchdog stays happy.
    """
    # Settle: keep publishing while the opponent transitions from their
    # rocking fist into their final gesture.
    _hold_pose(session, config, ready_pose, settle_seconds)

    if vision_thread is None:
        return None, (), 0

    # Discard whatever was detected during pre-reveal/settle so the vote uses
    # only frames captured from this point on.
    vision_thread.clear_latest()

    samples: list[str] = []
    extended_by_label: dict[str, tuple[str, ...]] = {}
    frames_seen_in_window = 0
    last_frame_id: int | None = None
    deadline = time.monotonic() + timeout_seconds

    while len(samples) < sample_frames and time.monotonic() < deadline:
        _hold_pose(session, config, ready_pose, poll_interval)
        gesture, extended, frame_id = vision_thread.last_frame
        if frame_id == last_frame_id:
            continue
        last_frame_id = frame_id
        frames_seen_in_window += 1
        if gesture is not None:
            samples.append(gesture)
            extended_by_label[gesture] = extended

    if not samples:
        return None, (), frames_seen_in_window

    label, _ = Counter(samples).most_common(1)[0]
    return label, extended_by_label.get(label, ()), frames_seen_in_window


def run_pre_reveal_beats(
    session: UnitreeLowCmdSession,
    config: ArmHardwareConfig,
    ready_pose: dict[str, float],
) -> None:
    """Rocking arm motion. Assumes the arm is already at ``ready_pose`` and
    leaves it back at ``ready_pose`` when done."""
    for beat_index in range(config.beat_count):
        print(f"  Pre-reveal beat {beat_index + 1}/{config.beat_count}")
        steps = max(1, int(config.beat_duration / config.control_dt))
        for step in range(steps):
            local_time = step * config.control_dt
            pose = _arm_beat_pose(config, ready_pose, beat_index, local_time)
            if config.live:
                session.publish_pose(pose)
            time.sleep(config.control_dt)

    end_pose = _arm_beat_pose(
        config, ready_pose, config.beat_count - 1, config.beat_duration
    )
    _interpolate_pose(session, config, end_pose, ready_pose, config.return_duration)


def reveal_hand_gesture(robot_gesture: str) -> None:
    """Print the robot's chosen gesture to the console.

    The actual hand motion is driven by `RpsHandController.transition_to`
    in the round loop; this helper just logs the play.
    """
    print(f"  Robot reveals: {robot_gesture}")


def run_outcome_pose(
    session: UnitreeLowCmdSession,
    config: ArmHardwareConfig,
    ready_pose: dict[str, float],
    *,
    won: bool,
) -> None:
    """Move into the win or lose pose, hold briefly, then return to ready."""
    if won:
        target = winning_pose(config)
        print("  Robot wins -- celebrating!")
    else:
        target = lose_pose(config)
        print("  Robot did not win -- bringing both hands to face.")

    # `ready_pose` is right-arm-only; fill in the left-arm joints from the
    # captured hold pose so `blend_pose` sees matching keys on both sides
    # and the left arm starts/ends at its at-rest position.
    other_joints = [name for name in target if name not in ready_pose]
    extended_ready = {**ready_pose, **session.hold_pose_for(other_joints)}

    _interpolate_pose(session, config, extended_ready, target, duration=1.0)
    _hold_pose(session, config, target, duration=1.5)
    _interpolate_pose(session, config, target, extended_ready, duration=1.0)


def _draw_label_overlay(
    frame: np.ndarray, label: str, extended: tuple[str, ...]
) -> None:
    fingers = ",".join(extended) if extended else "none"
    lines = [f"opponent: {label}", f"fingers: {fingers}"]
    x, y = 20, 40
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 36


_VIDEO_CLIENTS = {
    "videohub": ("unitree_sdk2py.go2.video.video_client", "VideoClient"),
    "front": ("unitree_sdk2py.b2.front_video.front_video_client", "FrontVideoClient"),
    "back": ("unitree_sdk2py.b2.back_video.back_video_client", "BackVideoClient"),
}


def _decode_unitree_image_sample(data) -> np.ndarray | None:
    if data is None:
        return None
    if isinstance(data, list):
        data = bytes(data)
    try:
        buf = np.frombuffer(data, dtype=np.uint8)
    except TypeError:
        return None
    if buf.size == 0:
        return None
    try:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def _create_video_client(camera: str, timeout_seconds: float):
    """Create a Unitree SDK2 video client."""
    try:
        module_name, class_name = _VIDEO_CLIENTS[camera]
    except KeyError as exc:
        raise ValueError(
            f"Unknown camera '{camera}'. Use 'videohub', 'front', or 'back'."
        ) from exc

    module = importlib.import_module(module_name)
    client = getattr(module, class_name)()
    client.SetTimeout(timeout_seconds)
    client.Init()
    return client


def _read_first_camera_frame(
    video_client,
    camera: str,
    timeout_seconds: float,
) -> np.ndarray:
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    last_problem = "no samples received"

    while True:
        attempts += 1
        code, data = video_client.GetImageSample()
        if code != 0:
            last_problem = f"GetImageSample returned code {code}"
        else:
            frame = _decode_unitree_image_sample(data)
            if frame is not None:
                return frame
            last_problem = "empty or invalid image payload"

        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(
                f"{camera} camera did not produce a decodable frame after "
                f"{attempts} samples ({last_problem})."
            )
        time.sleep(0.05)


def _read_right_arm_pose(
    session: UnitreeLowCmdSession, config: ArmHardwareConfig
) -> dict[str, float]:
    """Read the robot's current right-arm joint positions from `rt/lowstate`."""
    state = session.wait_for_state()
    return {
        joint_name: float(state.motor_state[joint_index].q)
        for joint_name, joint_index in commanded_right_arm_joints(config).items()
    }


def _wait_for_input_holding_pose(
    session: UnitreeLowCmdSession,
    config: ArmHardwareConfig,
    ready_pose: dict[str, float],
    vision_thread: OpponentVisionThread | None,
) -> str | None:
    """Block until the user submits a line on stdin, publishing ``ready_pose``
    at the control rate so the lowcmd watchdog stays satisfied.

    Returns the entered line (lowercased, stripped). Returns ``None`` on EOF
    or if the vision thread asked to stop.
    """
    while True:
        if vision_thread is not None and vision_thread.stopped:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], config.control_dt)
        if readable:
            line = sys.stdin.readline()
            if not line:
                return None
            return line.strip().lower()
        if config.live:
            session.publish_pose(ready_pose)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive rock-paper-scissors loop on the Unitree G1: pre-reveal "
            "arm motion, randomly chosen play (printed only for now), opponent "
            "gesture observation, and a win/lose reaction pose. Dry-run by default."
        )
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="DDS network interface, for example `eth0`. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="CycloneDDS domain ID used by Unitree SDK2.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually publish to rt/lowcmd. Without this it is a dry run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random gesture choice (useful for repeatable demos).",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip opening the camera and the vision thread.",
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Skip Inspire hand control (rt/inspire/cmd). Use this if the hand service isn't running.",
    )
    parser.add_argument(
        "--camera",
        choices=("videohub", "front", "back"),
        default="videohub",
        help=(
            "Unitree SDK2 camera service. `videohub` is the G1/Go2-style "
            "service; `front`/`back` are older B2-style services."
        ),
    )
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=3.0,
        help="Per-call image timeout for the Unitree video client.",
    )
    parser.add_argument(
        "--camera-startup-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the first decodable camera frame before starting MediaPipe.",
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--display",
        dest="display",
        action="store_true",
        help="Show the camera feed with detected landmarks (default).",
    )
    display_group.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Run headless. Useful over SSH without an X server.",
    )
    parser.set_defaults(display=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 60)
    print("  Unitree G1 -- Rock, Paper, Scissors (interactive loop)")
    print("=" * 60)

    if not args.live:
        print(
            "\nDry run. The arm will not actually move. Add --live to take "
            "control of the robot."
        )

    print(
        "\nIMPORTANT: release the high-level body controller BEFORE this script "
        "starts publishing -- L2+B then L2+R2 on the joystick (or "
        'MotionSwitcherClient.SelectMode("")). Otherwise the high-level '
        "controller will fight our lowcmd publishes."
    )
    print(
        "The robot's legs and waist will be held in PD at their position when "
        "this script captures hold-pose. Make sure the robot is on a stand or "
        "is being held.\n"
    )

    arm_config = ArmHardwareConfig(
        interface=args.interface,
        domain_id=args.domain_id,
        live=args.live,
    )

    # 1) Arm session FIRST -- its constructor calls ChannelFactoryInitialize,
    #    which the camera client will reuse.
    print("Initializing arm session (rt/lowcmd, unitree_sdk2py)...")
    arm_session = UnitreeLowCmdSession(arm_config)
    arm_session.capture_hold_pose()

    # 1b) Inspire hand session (rt/inspire/cmd / rt/inspire/state). The hand
    # is on a separate DDS topic, independent of the arm low-level controller.
    hand_controller: RpsHandController | None = None
    if not args.no_hand:
        print("Initializing Inspire hand session (rt/inspire/cmd)...")
        hand_config = HardwareConfig(
            transition_seconds=0.4,
            hand="right",
            domain_id=args.domain_id,
            network_interface=args.interface,
            live=args.live,
            allow_state_fallback=True,
        )
        try:
            hand_controller = RpsHandController(hand_config)
            hand_controller.open()
            # Start in a closed fist so the hand looks concealed during pre-reveal.
            if args.live:
                hand_controller.transition_to("rock")
        except Exception as exc:
            print(
                f"  warning: Inspire hand control unavailable ({exc}). Continuing without it."
            )
            hand_controller = None

    # 2) Camera + vision thread. Same channel factory as the arm. Streams for
    #    the entire program lifetime, across all rounds.
    vision_thread: OpponentVisionThread | None = None
    classifier: HandGestureClassifier | None = None
    if not args.no_camera:
        print(f"Opening {args.camera} camera and vision pipeline...")
        try:
            video_client = _create_video_client(args.camera, args.camera_timeout)
            first_frame = _read_first_camera_frame(
                video_client, args.camera, args.camera_startup_timeout
            )

            from g1_rps.vision import (
                ClassifierConfig as VisionClassifierConfig,
                HandGestureClassifier as VisionHandGestureClassifier,
                draw_landmarks as vision_draw_landmarks,
            )

            global ClassifierConfig, HandGestureClassifier, draw_landmarks
            ClassifierConfig = VisionClassifierConfig
            HandGestureClassifier = VisionHandGestureClassifier
            draw_landmarks = vision_draw_landmarks

            classifier = HandGestureClassifier(ClassifierConfig())
            vision_thread = OpponentVisionThread(
                video_client,
                classifier,
                display=args.display,
                initial_frame=first_frame,
            )
            vision_thread.start()
            if args.display:
                print("Vision thread running. Press 'q' in the camera window to stop.")
            else:
                print("Vision thread running (headless).")
        except Exception as exc:
            print(f"  warning: could not start vision ({exc}). Continuing without it.")
            classifier = None
            vision_thread = None

    # 3) Move the arm to the ready pose once, then keep it there between rounds.
    ready_pose = ready_right_arm_pose(arm_config)
    if args.live:
        print("\nMoving G1 right arm into the concealed ready pose...")
        start_pose = _read_right_arm_pose(arm_session, arm_config)
        _interpolate_pose(
            arm_session, arm_config, start_pose, ready_pose, arm_config.setup_duration
        )

    round_index = 0
    try:
        while True:
            sys.stdout.write("\nPlay a round? [Enter to play, q to quit] ")
            sys.stdout.flush()
            answer = _wait_for_input_holding_pose(
                arm_session, arm_config, ready_pose, vision_thread
            )
            if answer is None:
                print("\n(stdin closed or vision stopped)")
                break
            if answer in ("n", "no", "q", "quit", "exit"):
                break

            round_index += 1
            print(f"\n--- Round {round_index} ---")

            if vision_thread is not None:
                vision_thread.clear_latest()

            robot_gesture = random.choice(GESTURES)

            run_pre_reveal_beats(arm_session, arm_config, ready_pose)

            # Open the hand to the chosen gesture in a worker so the
            # transition overlaps with the opponent settle window. The
            # hand is on its own DDS topic, so it doesn't fight the arm
            # publish loop.
            hand_thread: threading.Thread | None = None
            if hand_controller is not None:
                hand_thread = threading.Thread(
                    target=hand_controller.transition_to,
                    args=(robot_gesture,),
                    daemon=True,
                )
                hand_thread.start()

            reveal_hand_gesture(robot_gesture)

            opponent_gesture, opponent_extended, frames_seen = _sample_opponent_gesture(
                arm_session, arm_config, ready_pose, vision_thread
            )

            if hand_thread is not None:
                hand_thread.join(timeout=2.0)

            outcome = determine_winner(robot_gesture, opponent_gesture)
            fingers = ",".join(opponent_extended) if opponent_extended else "none"
            print(
                f"  Opponent: {opponent_gesture or '(not detected)'} "
                f"[fingers: {fingers}, frames sampled: {frames_seen}]"
            )
            print(f"  Result: {outcome}")

            run_outcome_pose(
                arm_session, arm_config, ready_pose, won=(outcome == "robot wins")
            )

            # Close the hand back to a fist so the next round's pre-reveal
            # starts from a concealed shape.
            if hand_controller is not None:
                hand_controller.transition_to("rock")
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        # Settle the arm at ready briefly so motors don't go uncommanded the
        # instant we stop publishing.
        if args.live:
            _hold_pose(arm_session, arm_config, ready_pose, arm_config.release_duration)

        if hand_controller is not None:
            hand_controller.close()

        if vision_thread is not None:
            vision_thread.stop()
            vision_thread.join(timeout=1.0)
            if vision_thread.error is not None:
                print(f"  vision thread error: {vision_thread.error}")
        if classifier is not None:
            classifier.close()
        if args.display:
            cv2.destroyAllWindows()

    print(
        f"\nPlayed {round_index} round(s). The arm is no longer being commanded; "
        "re-engage the high-level controller (L2+R2) to put the robot back "
        "under normal control."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
