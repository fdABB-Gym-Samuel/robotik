"""Stream frames from the Unitree G1's onboard camera.

By default this uses the SDK2 ``videohub`` client, which is exposed through the
``unitree_sdk2py.go2.video`` namespace but works on the G1 camera service. The
older B2 ``front``/``back`` clients are still available as explicit options for
SDK builds that expose those services.

Use ``--no-display`` over SSH. Add ``--no-classifier`` for a camera-only smoke
test that does not require MediaPipe. When the classifier is enabled, frames are
fed to ``g1_rps.vision.HandGestureClassifier`` and the preview includes the
detected gesture and 21 hand landmarks. Press ``q`` (or Ctrl+C in a headless
run) to quit.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

from g1_rps.unitree_sdk2_config import configure_local_cyclonedds_log


@dataclass(frozen=True)
class RobotCameraConfig:
    interface: str | None = None
    domain_id: int = 0
    camera: str = "videohub"  # "videohub", "front", or "back"
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

        configure_local_cyclonedds_log()

        # `ChannelFactoryInitialize` may only be called once per process.
        # Calling it twice raises; here we just call it on first open.
        if self._config.interface is not None:
            ChannelFactoryInitialize(self._config.domain_id, self._config.interface)
        else:
            ChannelFactoryInitialize(self._config.domain_id)
        self._initialized_factory = True

        try:
            if self._config.camera == "videohub":
                from unitree_sdk2py.go2.video.video_client import VideoClient

                client = VideoClient()
            elif self._config.camera == "front":
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
                    f"Unknown camera '{self._config.camera}'. "
                    "Use 'videohub', 'front', or 'back'."
                )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The selected Unitree video client module is not present in this "
                "SDK build. This script defaults to the G1-compatible "
                "`unitree_sdk2py.go2.video.video_client.VideoClient`; the "
                "`front` and `back` options use older B2 service clients."
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
                "may not be running, or the DDS interface/domain is wrong. "
                "On this G1, use `--camera videohub`; the B2 front/back "
                "services may return code 3102."
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
        # The Unitree video clients used here have no explicit shutdown method;
        # dropping the reference is sufficient. The DDS channel factory is
        # process-wide and is not torn down here.
        self._client = None

    def __enter__(self) -> "RobotCameraSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the Unitree G1's onboard camera, optionally classifying "
            "rock, paper, and scissors using MediaPipe Hands. Dry-friendly: "
            "no robot motion is commanded; only the camera is read."
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
        choices=("videohub", "front", "back"),
        default="videohub",
        help=(
            "Which onboard camera service to read from. `videohub` is the "
            "G1-compatible default; `front`/`back` use B2 video clients."
        ),
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
        "--no-classifier",
        action="store_true",
        help="Only stream camera frames; do not import or run MediaPipe.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=0,
        help="Stop after this many frames. Use 0 to stream until interrupted.",
    )
    parser.add_argument(
        "--save-frame",
        type=Path,
        default=None,
        help="Write the first decoded frame to this path.",
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
    parser.add_argument(
        "--display-backend",
        choices=("auto", "opencv", "mjpeg"),
        default="auto",
        help=(
            "Preview backend. `auto` uses OpenCV windows when available, "
            "otherwise serves an MJPEG preview in the browser."
        ),
    )
    parser.add_argument(
        "--mjpeg-port",
        type=int,
        default=8765,
        help="Local port for the MJPEG browser preview when that backend is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_limit < 0:
        raise SystemExit("--frame-limit must be >= 0")

    camera_config = RobotCameraConfig(
        interface=args.interface,
        domain_id=args.domain_id,
        camera=args.camera,
        timeout_seconds=args.timeout_seconds,
        mirror=args.mirror,
    )

    last_reported: str | None = "__unset__"
    last_fps_time = time.monotonic()
    frame_count = 0
    total_frames = 0
    fps = 0.0

    classifier = None
    draw_landmarks_fn = None
    preview = None
    use_opencv_display = False
    try:
        if not args.no_classifier:
            from g1_rps.vision import (
                ClassifierConfig,
                HandGestureClassifier,
                draw_landmarks,
            )

            classifier_config = ClassifierConfig(extension_margin=args.extension_margin)
            classifier = HandGestureClassifier(classifier_config)
            draw_landmarks_fn = draw_landmarks

        with RobotCameraSource(camera_config) as camera:
            print(
                f"Robot vision running on {args.camera} camera. Press Ctrl+C to quit.",
                flush=True,
            )
            if args.no_classifier:
                print("Classifier disabled; streaming raw camera frames.", flush=True)
            if args.display:
                if args.display_backend == "opencv" or (
                    args.display_backend == "auto" and _opencv_highgui_available()
                ):
                    use_opencv_display = True
                    print("Press 'q' in the preview window to quit.", flush=True)
                else:
                    preview = MjpegPreview(port=args.mjpeg_port)
                    preview.start()
                    print(
                        "Open this local preview in a browser: "
                        f"{preview.url}  (Ctrl+C here to quit.)",
                        flush=True,
                    )
            try:
                for frame in camera.frames():
                    total_frames += 1
                    label = "camera"
                    extended_fingers: tuple[str, ...] = ()
                    landmarks = None

                    if args.save_frame is not None and total_frames == 1:
                        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(args.save_frame), frame)
                        print(f"Saved first frame -> {args.save_frame}", flush=True)

                    if classifier is not None:
                        result = classifier.classify(frame)
                        label = result.gesture if result.gesture is not None else "—"
                        extended_fingers = result.extended_fingers
                        landmarks = result.landmarks

                        if result.gesture != last_reported:
                            fingers = ",".join(result.extended_fingers) or "none"
                            print(
                                f"opponent={label}  extended=[{fingers}]",
                                flush=True,
                            )
                            last_reported = result.gesture

                    frame_count += 1
                    now = time.monotonic()
                    if now - last_fps_time >= 1.0:
                        fps = frame_count / (now - last_fps_time)
                        frame_count = 0
                        last_fps_time = now

                    if args.display:
                        annotated = frame.copy()
                        if landmarks is not None and draw_landmarks_fn is not None:
                            draw_landmarks_fn(annotated, landmarks)
                        _draw_overlay(annotated, label, extended_fingers, fps)
                        if use_opencv_display:
                            cv2.imshow("G1 RPS opponent vision", annotated)
                            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                break
                        elif preview is not None:
                            preview.update(annotated)
                    if args.frame_limit and total_frames >= args.frame_limit:
                        break
            except KeyboardInterrupt:
                print("\nInterrupted.", flush=True)
            finally:
                if use_opencv_display:
                    _destroy_windows_if_available()
                if preview is not None:
                    preview.close()
    finally:
        if classifier is not None:
            classifier.close()
    return 0


class MjpegPreview:
    """Tiny browser preview for OpenCV builds without HighGUI windows."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._server = self._make_server(host, port)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mjpeg-preview",
            daemon=True,
        )
        actual_host, actual_port = self._server.server_address
        self.url = f"http://{actual_host}:{actual_port}/"

    def _make_server(self, host: str, port: int) -> ThreadingHTTPServer:
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args) -> None:
                return

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    page = (
                        "<!doctype html><title>G1 camera</title>"
                        "<style>body{margin:0;background:#111;display:grid;"
                        "place-items:center;min-height:100vh}"
                        "img{max-width:100vw;max-height:100vh}</style>"
                        '<img src="/stream.mjpg" alt="G1 camera stream">'
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                    return

                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()

                while True:
                    with preview._condition:
                        preview._condition.wait(timeout=2.0)
                        jpeg = preview._jpeg
                    if jpeg is None:
                        continue
                    try:
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode(
                                "ascii"
                            )
                            + jpeg
                            + b"\r\n"
                        )
                    except (BrokenPipeError, ConnectionResetError):
                        return

        return ThreadingHTTPServer((host, port), Handler)

    def start(self) -> None:
        self._thread.start()

    def update(self, frame) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if not ok:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._condition.notify_all()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)


def _opencv_highgui_available() -> bool:
    with contextlib.suppress(Exception):
        for line in cv2.getBuildInformation().splitlines():
            if line.strip().startswith("GUI:"):
                return line.split(":", maxsplit=1)[1].strip().upper() != "NONE"
    return True


def _destroy_windows_if_available() -> None:
    with contextlib.suppress(cv2.error):
        cv2.destroyAllWindows()


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
