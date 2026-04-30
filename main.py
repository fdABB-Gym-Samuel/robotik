"""End-to-end Unitree G1 rock-paper-scissors game.

Pipeline:
  1. Take control of the body via `rt/lowcmd` (the high-level controller must
     be released first -- L2+B then L2+R2 on the joystick).
  2. Spin up a daemon vision thread that grabs frames from the front camera
     and classifies the opponent's hand gesture with MediaPipe Hands.
  3. Run the pre-reveal arm rocking motion (3 beats, mirrors the structure of
     ``scripts/pre_reveal_right_arm_hardware.py``).
  4. Pick a random gesture for the robot and command the right Inspire hand
     to play it (via `rt/inspire/cmd`), while the main thread keeps the arm
     held at the ready pose so the lowcmd watchdog stays satisfied.
  5. Print what the robot played, what the camera saw the opponent play, and
     who won.

Subsystems used (all already in this repo):
  * Arm:    g1_rps.arm_hardware  (rt/lowcmd, unitree_sdk2py)
  * Hand:   g1_rps.hardware + g1_rps.unitree_dds  (rt/inspire/cmd, cyclonedds)
  * Vision: g1_rps.vision.HandGestureClassifier  (MediaPipe Hand Landmarker)

Prerequisites for --live:
  * The robot is on a stand or being held: lowcmd makes us responsible for
    every joint, including the legs (held at their captured position).
  * High-level body controller is released (L2+B then L2+R2).
  * Inspire hand service is running on the robot.
  * Front camera / multimedia service is running.

Camera-API caveat (same one as `test/camera.py`):
  This uses `unitree_sdk2py.b2.front_video.front_video_client.FrontVideoClient`,
  which is the B2 quadruped's video path. If your G1's SDK build exposes the
  camera under a different namespace, edit `_create_front_video_client`.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
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
    ready_right_arm_pose,
)
from g1_rps.hardware import (
    HardwareConfig,
    build_hardware_channels,
    build_motor_commands,
    extract_hand_channels_from_state,
)
from g1_rps.unitree_dds import MotorState_, MotorStates_, UnitreeDdsSession
from g1_rps.vision import (
    ClassifierConfig,
    HandGestureClassifier,
    draw_landmarks,
)


GESTURES: tuple[str, ...] = ("rock", "paper", "scissors")
WINS: frozenset[tuple[str, str]] = frozenset(
    {
        ("rock", "scissors"),
        ("scissors", "paper"),
        ("paper", "rock"),
    }
)


def determine_winner(robot: str, opponent: str | None) -> str:
    if opponent is None:
        return "no opponent gesture detected"
    if robot == opponent:
        return "tie"
    return "robot wins" if (robot, opponent) in WINS else "opponent wins"


class OpponentVisionThread(threading.Thread):
    """Continuously read frames from a video client and classify the gesture.

    The latest valid (non-``None``) gesture is kept under a lock; callers can
    sample ``latest`` at any time to get a snapshot.
    """

    def __init__(
        self,
        video_client,
        classifier: HandGestureClassifier,
        display: bool = True,
        window_name: str = "G1 opponent view",
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
        self._frames_processed = 0
        self._error: BaseException | None = None

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                code, data = self._client.GetImageSample()
                if code != 0:
                    time.sleep(0.05)
                    continue
                if isinstance(data, list):
                    data = bytes(data)
                buf = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                result = self._classifier.classify(frame)
                with self._lock:
                    self._frames_processed += 1
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


def run_arm_pre_reveal(
    session: UnitreeLowCmdSession, config: ArmHardwareConfig
) -> dict[str, float]:
    """Pre-reveal rocking motion. Returns the ready pose so the caller can hold it."""
    ready_pose = ready_right_arm_pose(config)
    start_pose = session.current_right_arm_pose()

    print("Moving G1 right arm into the concealed ready pose...")
    session.interpolate(start_pose, ready_pose, config.setup_duration)

    for beat_index in range(config.beat_count):
        print(f"Pre-reveal beat {beat_index + 1}/{config.beat_count}")
        steps = max(1, int(config.beat_duration / config.control_dt))
        for step in range(steps):
            local_time = step * config.control_dt
            pose = _arm_beat_pose(config, ready_pose, beat_index, local_time)
            session.publish_pose(pose)
            time.sleep(config.control_dt)

    end_pose = _arm_beat_pose(
        config, ready_pose, config.beat_count - 1, config.beat_duration
    )
    print("Returning right arm to the concealed ready pose...")
    session.interpolate(end_pose, ready_pose, config.return_duration)
    return ready_pose


def reveal_hand_gesture(
    robot_gesture: str,
    hand_session: UnitreeDdsSession,
    hand_config: HardwareConfig,
) -> None:
    """Move the right Inspire hand from its current state to the chosen gesture."""
    initial = hand_session.read_state(timeout_seconds=hand_config.state_timeout_seconds)
    if initial is None or len(initial.states) < 12:
        print("  warning: no Inspire hand state, falling back to open-state baseline.")
        initial = MotorStates_(states=[MotorState_(q=1.0) for _ in range(12)])

    current = extract_hand_channels_from_state(initial, "right")
    inactive = extract_hand_channels_from_state(initial, "left")
    target = build_hardware_channels(robot_gesture)

    steps = max(1, int(hand_config.transition_seconds * hand_config.rate_hz))
    sleep_seconds = 1.0 / hand_config.rate_hz
    for step in range(1, steps + 1):
        alpha = step / steps
        sample = tuple((1.0 - alpha) * current[i] + alpha * target[i] for i in range(6))
        hand_session.write(build_motor_commands(sample, "right", inactive))
        time.sleep(sleep_seconds)

    hold_steps = max(1, int(hand_config.hold_seconds * hand_config.rate_hz))
    sample = build_motor_commands(target, "right", inactive)
    for _ in range(hold_steps):
        hand_session.write(sample)
        time.sleep(sleep_seconds)


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


def _create_front_video_client(timeout_seconds: float):
    """Front-camera video client. Edit this if your G1's SDK uses a different path."""
    from unitree_sdk2py.b2.front_video.front_video_client import FrontVideoClient

    client = FrontVideoClient()
    client.SetTimeout(timeout_seconds)
    client.Init()
    return client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play one round of rock-paper-scissors on the Unitree G1: pre-reveal "
            "arm motion, randomly chosen hand gesture reveal, and opponent pose "
            "observation through the front camera. Dry-run by default."
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
        help="Actually publish to rt/lowcmd and rt/inspire/cmd. Without this it is a dry run.",
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
        help="Skip opening the front camera and the vision thread.",
    )
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=3.0,
        help="Per-call image timeout for the Unitree video client.",
    )
    parser.add_argument(
        "--reveal-transition",
        type=float,
        default=0.25,
        help="Seconds spent moving the hand from open to the chosen gesture.",
    )
    parser.add_argument(
        "--reveal-hold",
        type=float,
        default=0.5,
        help="Seconds the hand keeps the chosen gesture after the reveal.",
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
    robot_gesture = random.choice(GESTURES)

    print("=" * 60)
    print("  Unitree G1 -- Rock, Paper, Scissors")
    print("=" * 60)

    if not args.live:
        print(f"\nDry run. Robot would have played: {robot_gesture}")
        print("Add --live to actually take control of the robot.")
        return 0

    print(
        "\nIMPORTANT: release the high-level body controller BEFORE this script "
        "starts publishing -- L2+B then L2+R2 on the joystick (or "
        'MotionSwitcherClient.SelectMode("")). Otherwise the high-level '
        "controller will fight our lowcmd publishes."
    )
    print(
        "The robot's legs and waist will be held in PD at their position when "
        "this script captures hold-pose. Make sure the robot is on a stand or "
        "is being held."
    )
    print(f"\nRobot privately picked: {robot_gesture}\n")

    arm_config = ArmHardwareConfig(
        interface=args.interface,
        domain_id=args.domain_id,
        live=True,
    )
    hand_config = HardwareConfig(
        domain_id=args.domain_id,
        network_interface=args.interface,
        live=True,
        transition_seconds=args.reveal_transition,
        hold_seconds=args.reveal_hold,
    )

    # 1) Arm session FIRST -- its constructor calls ChannelFactoryInitialize,
    #    which the camera client will reuse.
    print("Initializing arm session (rt/lowcmd, unitree_sdk2py)...")
    arm_session = UnitreeLowCmdSession(arm_config)
    arm_session.capture_hold_pose()

    # 2) Camera + vision thread. Same channel factory as the arm.
    vision_thread: OpponentVisionThread | None = None
    classifier: HandGestureClassifier | None = None
    if not args.no_camera:
        print("Opening front camera and vision pipeline...")
        try:
            front_client = _create_front_video_client(args.camera_timeout)
            classifier = HandGestureClassifier(ClassifierConfig())
            vision_thread = OpponentVisionThread(
                front_client, classifier, display=args.display
            )
            vision_thread.start()
            if args.display:
                print(
                    "Vision thread running. Press 'q' in the camera window to stop early."
                )
            else:
                print("Vision thread running (headless).")
        except Exception as exc:
            print(f"  warning: could not start vision ({exc}). Continuing without it.")
            classifier = None
            vision_thread = None

    # 3) Hand DDS (independent of unitree_sdk2py's channel factory).
    print("Connecting to Inspire hand DDS (rt/inspire/cmd, cyclonedds)...")
    try:
        hand_session: UnitreeDdsSession | None = UnitreeDdsSession(
            domain_id=hand_config.domain_id,
            network_interface=hand_config.network_interface,
            command_topic=hand_config.command_topic,
            state_topic=hand_config.state_topic,
        )
    except Exception as exc:
        print(f"  warning: hand DDS unavailable ({exc}). The reveal will be skipped.")
        hand_session = None

    try:
        # 4) Pre-reveal arm motion.
        ready_pose = run_arm_pre_reveal(arm_session, arm_config)

        # 5) Reveal: hand reveal runs on a worker thread; main thread keeps the
        #    arm held at the ready pose so the lowcmd watchdog stays happy.
        if hand_session is not None:
            print(f"\nRevealing: {robot_gesture}")
            hand_thread = threading.Thread(
                target=reveal_hand_gesture,
                args=(robot_gesture, hand_session, hand_config),
                name="hand-reveal",
                daemon=True,
            )
            hand_thread.start()
            while hand_thread.is_alive():
                arm_session.publish_pose(ready_pose)
                time.sleep(arm_config.control_dt)
            hand_thread.join()

        # 6) Sample what the camera saw, then keep arm held briefly so motors
        #    don't go uncommanded the moment we stop.
        opponent_gesture: str | None = None
        opponent_extended: tuple[str, ...] = ()
        frames_processed = 0
        if vision_thread is not None:
            opponent_gesture, opponent_extended, frames_processed = vision_thread.latest

        hold_steps = max(1, int(arm_config.release_duration / arm_config.control_dt))
        for _ in range(hold_steps):
            arm_session.publish_pose(ready_pose)
            time.sleep(arm_config.control_dt)

        # 7) Result.
        fingers = ",".join(opponent_extended) if opponent_extended else "none"
        print("\n" + "=" * 60)
        print(f"  Robot played:    {robot_gesture}")
        print(
            f"  Opponent played: {opponent_gesture or '(not detected)'}  "
            f"[fingers: {fingers}, frames: {frames_processed}]"
        )
        print(f"  Result:          {determine_winner(robot_gesture, opponent_gesture)}")
        print("=" * 60)
    finally:
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
        "\nDone. The arm is no longer being commanded; re-engage the high-level "
        "controller (L2+R2) to put the robot back under normal control."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
