"""Laptop webcam demo for the rock-paper-scissors hand gesture classifier.

Opens the default webcam, runs MediaPipe Hands on each frame, and prints
the detected gesture to the terminal. With ``--display`` (the default) it
also shows an annotated preview window with the gesture label and the 21
hand landmarks drawn on top of the frame; press ``q`` to quit.

Pass ``--no-display`` to run headless (useful over SSH or in WSL without
an X server), in which case the script just prints a line every time the
detected gesture changes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import cv2

from g1_rps.camera import CameraConfig, WebcamSource
from g1_rps.vision import ClassifierConfig, HandGestureClassifier, draw_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify rock, paper, and scissors from a laptop webcam feed "
            "using MediaPipe Hands. Prints the detected gesture and, by "
            "default, shows an annotated preview window."
        )
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="OpenCV VideoCapture device index for the webcam.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height in pixels.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested capture frame rate.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring of the preview frame.",
    )
    parser.add_argument(
        "--extension-margin",
        type=float,
        default=1.15,
        help=(
            "Threshold for the 'finger extended' rule. Lower values make the "
            "classifier more eager to call a finger extended; raise it if the "
            "camera is close and hands look stretched even when relaxed."
        ),
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--display",
        dest="display",
        action="store_true",
        help="Show an annotated preview window (default).",
    )
    display_group.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Run headless; only print gesture changes to the terminal.",
    )
    parser.set_defaults(display=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    camera_config = CameraConfig(
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror=not args.no_mirror,
    )
    classifier_config = ClassifierConfig(extension_margin=args.extension_margin)

    last_reported: str | None = "__unset__"
    last_fps_time = time.monotonic()
    frame_count = 0
    fps = 0.0

    with (
        WebcamSource(camera_config) as camera,
        HandGestureClassifier(classifier_config) as clf,
    ):
        print("Vision demo running. Press Ctrl+C to quit.", flush=True)
        if args.display:
            print("Press 'q' in the preview window to quit.", flush=True)
        try:
            for frame in camera.frames():
                result = clf.classify(frame)
                label = result.gesture if result.gesture is not None else "—"

                if result.gesture != last_reported:
                    fingers = ",".join(result.extended_fingers) or "none"
                    print(f"gesture={label}  extended=[{fingers}]", flush=True)
                    last_reported = result.gesture

                frame_count += 1
                now = time.monotonic()
                if now - last_fps_time >= 1.0:
                    fps = frame_count / (now - last_fps_time)
                    frame_count = 0
                    last_fps_time = now

                if args.display:
                    annotated = frame.copy()
                    if result.landmarks is not None:
                        draw_landmarks(annotated, result.landmarks)
                    _draw_overlay(annotated, label, result.extended_fingers, fps)
                    cv2.imshow("G1 RPS vision demo", annotated)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
        except KeyboardInterrupt:
            print("\nInterrupted.", flush=True)
        finally:
            if args.display:
                cv2.destroyAllWindows()
    return 0


def _draw_overlay(frame, label: str, extended: tuple[str, ...], fps: float) -> None:
    fingers = ",".join(extended) if extended else "none"
    lines = [f"gesture: {label}", f"fingers: {fingers}", f"fps: {fps:5.1f}"]
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


if __name__ == "__main__":
    raise SystemExit(main())
