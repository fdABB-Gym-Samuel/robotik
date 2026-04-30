"""Detect opponent rock-paper-scissors poses from the Unitree G1's onboard camera.

Grabs JPEG frames from Unitree's video client, decodes them with OpenCV, and
feeds each frame to the existing MediaPipe-backed ``HandGestureClassifier``
in ``g1_rps.vision`` so the robot can see what its opponent is showing.

Camera-API caveat
-----------------
The reference snippet in ``test/camera.py`` imports
``unitree_sdk2py.b2.front_video.front_video_client.FrontVideoClient``. That
module path is the **B2 quadruped's** video client. Whether the same path is
exposed on the G1 depends on the installed ``unitree_sdk2py`` build and the
robot-side multimedia service. This script uses that same import so the
behaviour matches ``test/camera.py``; if the import or the
``GetImageSample`` call fails on your G1, the error is surfaced verbatim --
swap in the correct video client class for your SDK build at the top of
``RobotCameraSource``.

Use ``--no-display`` over SSH; otherwise an OpenCV preview window with the
detected gesture and 21 hand landmarks is shown. Press ``q`` (or Ctrl+C in a
headless run) to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import cv2

from g1_rps.vision import ClassifierConfig, HandGestureClassifier, draw_landmarks


@dataclass(frozen=True)
class RobotCameraConfig:
    interface: str | None = None
    domain_id: int = 0
    camera: str = "front"  # "front" or "back"
    timeout_seconds: float = 3.0
    # Mirror so the operator-facing preview matches what they see.
    # Classification is orientation-agnostic, so this never changes the result.
    mirror: bool = False


class RobotCameraSource:
    """BGR frame source backed by Unitree's video client.

    Mirrors the API of ``g1_rps.camera.WebcamSource`` so the rest of the
    pipeline stays agnostic to the frame origin.
    """

    def __init__(self, config: RobotCameraConfig | None = None) -> None:
        self._config = config or RobotCameraConfig()
        self._client = None
        self._initialized_factory = False

    def open(self) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Real-robot camera capture requires Unitree's official "
                "`unitree_sdk2py` package. Install it on the robot or control "
                "PC first, following https://github.com/unitreerobotics/unitree_sdk2_python"
            ) from exc

        # `ChannelFactoryInitialize` may only be called once per process.
        # Calling it twice raises; here we just call it on first open.
        if self._config.interface is not None:
            ChannelFactoryInitialize(self._config.domain_id, self._config.interface)
        else:
            ChannelFactoryInitialize(self._config.domain_id)
        self._initialized_factory = True

        try:
            if self._config.camera == "front":
                from unitree_sdk2py.b2.front_video.front_video_client import (
                    FrontVideoClient,
                )

                client = FrontVideoClient()
            elif self._config.camera == "back":
                from unitree_sdk2py.b2.back_video.back_video_client import (
                    BackVideoClient,
                )

                client = BackVideoClient()
            else:
                raise ValueError(
                    f"Unknown camera '{self._config.camera}'. Use 'front' or 'back'."
                )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The selected Unitree video client module is not present in this "
                "SDK build. The reference path "
                "`unitree_sdk2py.b2.front_video.front_video_client` is for the B2 "
                "quadruped; your G1's SDK build may expose the camera under a "
                "different namespace. Check the installed `unitree_sdk2py` "
                "package and update the import in `RobotCameraSource.open`."
            ) from exc

        client.SetTimeout(self._config.timeout_seconds)
        client.Init()
        self._client = client

    def read(self) -> np.ndarray:
        """Return the next BGR frame as an HxWx3 uint8 numpy array."""
        if self._client is None:
            raise RuntimeError(
                "Camera is not open. Call open() or use as a context manager."
            )
        code, data = self._client.GetImageSample()
        if code != 0:
            raise RuntimeError(
                f"GetImageSample returned non-zero code {code} from the "
                f"{self._config.camera} camera. The robot-side video service "
                "may not be running, or the DDS interface/domain is wrong."
            )
        if isinstance(data, list):
            data = bytes(data)
        encoded = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(
                "Failed to JPEG-decode the camera sample. The video service may "
                "be sending an unexpected payload format."
            )
        if self._config.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames indefinitely until the source is closed."""
        while self._client is not None:
            yield self.read()

    def close(self) -> None:
        # The Unitree video clients have no explicit shutdown method in the
        # B2 reference snippet; dropping the reference is sufficient. The
        # DDS channel factory is process-wide and is not torn down here.
        self._client = None

    def __enter__(self) -> "RobotCameraSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify rock, paper, and scissors from the Unitree G1's onboard "
            "camera using MediaPipe Hands. Dry-friendly: no robot motion is "
            "commanded; only the camera is read."
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
        "--camera",
        choices=("front", "back"),
        default="front",
        help="Which onboard camera to read from.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=3.0,
        help="Per-call image timeout for the Unitree video client.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview frame horizontally.",
    )
    parser.add_argument(
        "--extension-margin",
        type=float,
        default=1.15,
        help=(
            "Threshold for the 'finger extended' rule (see g1_rps.vision). "
            "Lower values make the classifier more eager to call a finger "
            "extended; raise it if hands look stretched even when relaxed."
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

    camera_config = RobotCameraConfig(
        interface=args.interface,
        domain_id=args.domain_id,
        camera=args.camera,
        timeout_seconds=args.timeout_seconds,
        mirror=args.mirror,
    )
    classifier_config = ClassifierConfig(extension_margin=args.extension_margin)

    last_reported: str | None = "__unset__"
    last_fps_time = time.monotonic()
    frame_count = 0
    fps = 0.0

    with (
        RobotCameraSource(camera_config) as camera,
        HandGestureClassifier(classifier_config) as clf,
    ):
        print(
            f"Robot vision running on {args.camera} camera. Press Ctrl+C to quit.",
            flush=True,
        )
        if args.display:
            print("Press 'q' in the preview window to quit.", flush=True)
        try:
            for frame in camera.frames():
                result = clf.classify(frame)
                label = result.gesture if result.gesture is not None else "—"

                if result.gesture != last_reported:
                    fingers = ",".join(result.extended_fingers) or "none"
                    print(f"opponent={label}  extended=[{fingers}]", flush=True)
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
                    cv2.imshow("G1 RPS opponent vision", annotated)
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
    lines = [f"opponent: {label}", f"fingers: {fingers}", f"fps: {fps:5.1f}"]
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
