"""Stream frames from a real robot camera.

By default this tries the SDK2 camera services in order: ``videohub`` first,
then the older B2 ``front``/``back`` clients. Some G1 setups expose the head
camera as a RealSense/UVC/stream source instead of SDK2 JPEG samples; use
``--camera opencv`` for those.

Use ``--no-display`` over SSH. Add ``--no-classifier`` for a camera-only smoke
test that does not require MediaPipe. When the classifier is enabled, frames are
fed to ``g1_rps.vision.HandGestureClassifier`` and the preview includes the
detected gesture and 21 hand landmarks. Press ``q`` (or Ctrl+C in a headless
run) to quit.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import itertools
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
    camera: str = "auto"  # "auto", "videohub", "front", or "back"
    timeout_seconds: float = 3.0
    decode_timeout_seconds: float = 0.0
    auto_fallback_samples: int = 0
    # Mirror so the operator-facing preview matches what they see.
    # Classification is orientation-agnostic, so this never changes the result.
    mirror: bool = False


@dataclass(frozen=True)
class OpenCvCameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
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
        self._decode_retry_delay_seconds = 0.05
        self._decode_status_interval_seconds = 2.0
        self._last_decode_status_time = 0.0
        self._camera_candidates: tuple[str, ...] = ()
        self._camera_candidate_index = 0
        self._active_camera = self._config.camera
        self._active_camera_bad_sample_count = 0

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

        self._camera_candidates = _robot_camera_candidates(self._config.camera)
        self._camera_candidate_index = 0
        self._open_active_robot_video_client()

    def read(self) -> np.ndarray:
        """Return the next BGR frame as an HxWx3 uint8 numpy array."""
        if self._client is None:
            raise RuntimeError(
                "Camera is not open. Call open() or use as a context manager."
            )
        deadline = (
            None
            if self._config.decode_timeout_seconds <= 0
            else time.monotonic() + self._config.decode_timeout_seconds
        )
        attempts = 0
        last_problem = "no samples received"
        while True:
            attempts += 1
            if self._client is None:
                self._open_active_robot_video_client()

            code, data = self._client.GetImageSample()
            if code != 0:
                if self._try_next_robot_camera(f"GetImageSample returned code {code}"):
                    continue
                raise RuntimeError(
                    f"GetImageSample returned non-zero code {code} from the "
                    f"{self._active_camera} camera. The robot-side video service "
                    "may not be running, or the DDS interface/domain is wrong. "
                    "Try the default `--camera auto` mode to compare SDK2 camera "
                    "services."
                )
            try:
                encoded = _encoded_image_array(data)
            except TypeError as exc:
                raise RuntimeError(
                    "GetImageSample returned an unsupported image payload "
                    f"type: {type(data).__name__}."
                ) from exc

            if encoded.size == 0:
                last_problem = "empty image payload"
            else:
                try:
                    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                except cv2.error as exc:
                    last_problem = f"OpenCV decode error: {exc}"
                else:
                    if frame is not None:
                        self._active_camera_bad_sample_count = 0
                        if self._config.mirror:
                            frame = cv2.flip(frame, 1)
                        return frame
                    last_problem = (
                        f"payload was {encoded.size} bytes but not a decodable image"
                    )

            self._active_camera_bad_sample_count += 1
            if (
                self._config.camera == "auto"
                and self._config.auto_fallback_samples > 0
                and self._active_camera_bad_sample_count
                >= self._config.auto_fallback_samples
                and self._try_next_robot_camera(last_problem)
            ):
                continue

            now = time.monotonic()
            if self._last_decode_status_time == 0.0:
                self._last_decode_status_time = now
            if (
                now - self._last_decode_status_time
                >= self._decode_status_interval_seconds
            ):
                self._last_decode_status_time = now
                timeout_hint = (
                    "Ctrl+C to quit."
                    if deadline is None
                    else f"timing out in {max(0.0, deadline - now):.1f}s."
                )
                print(
                    f"Waiting for {self._active_camera} camera image bytes "
                    f"({attempts} empty/invalid samples so far; {last_problem}). "
                    f"{timeout_hint}",
                    flush=True,
                )

            if deadline is None:
                time.sleep(self._decode_retry_delay_seconds)
                continue

            remaining = deadline - now
            if remaining <= 0:
                break
            time.sleep(min(self._decode_retry_delay_seconds, remaining))

        raise RuntimeError(
            f"Timed out after {attempts} {self._active_camera} camera samples "
            f"without a decodable JPEG ({last_problem}). GetImageSample returned "
            "success, but the payload was empty or invalid. Check that the "
            "robot-side video service is publishing on the selected DDS "
            "interface/domain. On G1, the head camera may be exposed as a "
            "RealSense/UVC/stream source instead of SDK2 videohub; run with "
            "`--probe-camera-services` to compare SDK services, or try "
            "`--camera opencv --opencv-source /dev/video0` on the machine that "
            "can see the camera device. If videohub used to work and now returns "
            "only empty samples, restart the robot-side video/multimedia service "
            "or power-cycle the robot."
        )

    def _open_active_robot_video_client(self) -> None:
        self._active_camera = self._camera_candidates[self._camera_candidate_index]
        self._client = None
        client = _make_robot_video_client(self._active_camera)
        client.SetTimeout(self._config.timeout_seconds)
        client.Init()
        self._client = client
        self._active_camera_bad_sample_count = 0

    def _try_next_robot_camera(self, reason: str) -> bool:
        if (
            self._config.camera != "auto"
            or self._camera_candidate_index >= len(self._camera_candidates) - 1
        ):
            return False

        previous = self._active_camera
        previous_index = self._camera_candidate_index
        previous_client = self._client
        while self._camera_candidate_index < len(self._camera_candidates) - 1:
            self._camera_candidate_index += 1
            candidate = self._camera_candidates[self._camera_candidate_index]
            try:
                self._open_active_robot_video_client()
            except Exception as exc:
                print(
                    f"{candidate} camera could not be opened while falling back "
                    f"from {previous} ({type(exc).__name__}: {exc}).",
                    flush=True,
                )
                continue

            print(
                f"{previous} camera did not produce image bytes ({reason}); "
                f"trying {self._active_camera} camera.",
                flush=True,
            )
            return True
        self._camera_candidate_index = previous_index
        self._active_camera = previous
        self._client = previous_client
        return False

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames indefinitely until the source is closed."""
        while self._client is not None:
            yield self.read()

    @property
    def active_camera(self) -> str:
        return self._active_camera

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


class OpenCvCameraSource:
    """BGR frame source backed by OpenCV VideoCapture."""

    def __init__(self, config: OpenCvCameraConfig | None = None) -> None:
        self._config = config or OpenCvCameraConfig()
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self._config.source)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open OpenCV camera source {self._config.source!r}. "
                "Use a device index like `0`, a path like `/dev/video0`, or a "
                "stream/GStreamer source that OpenCV can open."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        self._capture = capture

    def read(self) -> np.ndarray:
        """Return the next BGR frame as an HxWx3 uint8 numpy array."""
        if self._capture is None:
            raise RuntimeError(
                "Camera is not open. Call open() or use as a context manager."
            )
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to read a frame from OpenCV source {self._config.source!r}."
            )
        if self._config.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames indefinitely until the source is closed."""
        while self._capture is not None:
            yield self.read()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "OpenCvCameraSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_ROBOT_VIDEO_CLIENTS = {
    "videohub": ("unitree_sdk2py.go2.video.video_client", "VideoClient"),
    "front": ("unitree_sdk2py.b2.front_video.front_video_client", "FrontVideoClient"),
    "back": ("unitree_sdk2py.b2.back_video.back_video_client", "BackVideoClient"),
}


def _robot_camera_candidates(camera: str) -> tuple[str, ...]:
    if camera == "auto":
        return ("videohub", "front", "back")
    if camera in _ROBOT_VIDEO_CLIENTS:
        return (camera,)
    raise ValueError(
        f"Unknown SDK2 camera '{camera}'. Use 'auto', 'videohub', 'front', or 'back'."
    )


def _make_robot_video_client(camera: str):
    try:
        module_name, class_name = _ROBOT_VIDEO_CLIENTS[camera]
    except KeyError as exc:
        raise ValueError(
            f"Unknown SDK2 camera '{camera}'. Use 'videohub', 'front', or 'back'."
        ) from exc

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The selected Unitree video client module is not present in this SDK "
            "build. `videohub` uses "
            "`unitree_sdk2py.go2.video.video_client.VideoClient`; `front` and "
            "`back` use older B2 service clients."
        ) from exc

    return getattr(module, class_name)()


def _encoded_image_array(data) -> np.ndarray:
    if data is None:
        return np.empty(0, dtype=np.uint8)
    if isinstance(data, list):
        data = bytes(data)
    if isinstance(data, np.ndarray):
        if data.dtype == np.uint8:
            return data.reshape(-1)
        return data.astype(np.uint8, copy=False).reshape(-1)
    return np.frombuffer(data, dtype=np.uint8)


def _parse_opencv_source(value: str) -> int | str:
    with contextlib.suppress(ValueError):
        return int(value)
    return value


def _probe_robot_camera_services(
    *,
    interface: str | None,
    domain_id: int,
    timeout_seconds: float,
    sample_count: int,
    sample_interval_seconds: float,
) -> int:
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SDK2 camera probing requires Unitree's official `unitree_sdk2py` package."
        ) from exc

    configure_local_cyclonedds_log()
    if interface is not None:
        ChannelFactoryInitialize(domain_id, interface)
    else:
        ChannelFactoryInitialize(domain_id)

    target = interface if interface is not None else "auto interface"
    print(
        f"Probing SDK2 camera services on domain {domain_id}, {target}.",
        flush=True,
    )
    print(
        "A good SDK2 camera sample should have code=0, non-zero bytes, and "
        "usually JPEG magic `ff d8`.",
        flush=True,
    )

    any_decodable = False
    for camera in ("videohub", "front", "back"):
        print(f"\n[{camera}]", flush=True)
        try:
            client = _make_robot_video_client(camera)
            client.SetTimeout(timeout_seconds)
            client.Init()
        except Exception as exc:
            print(f"init failed: {type(exc).__name__}: {exc}", flush=True)
            continue

        with contextlib.suppress(Exception):
            version_code, version = client.GetServerApiVersion()
            print(
                f"server api version: code={version_code} version={version}",
                flush=True,
            )

        lengths: list[int] = []
        nonzero_codes: list[int] = []
        decoded_shapes: list[tuple[int, int, int]] = []
        for index in range(sample_count):
            try:
                code, data = client.GetImageSample()
            except Exception as exc:
                print(
                    f"sample {index + 1:02d}: call failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if index < sample_count - 1 and sample_interval_seconds > 0:
                    time.sleep(sample_interval_seconds)
                continue

            if code != 0:
                nonzero_codes.append(code)
                print(f"sample {index + 1:02d}: code={code}", flush=True)
                if index < sample_count - 1 and sample_interval_seconds > 0:
                    time.sleep(sample_interval_seconds)
                continue

            try:
                encoded = _encoded_image_array(data)
            except TypeError:
                print(
                    f"sample {index + 1:02d}: code=0 unsupported payload "
                    f"type={type(data).__name__}",
                    flush=True,
                )
                if index < sample_count - 1 and sample_interval_seconds > 0:
                    time.sleep(sample_interval_seconds)
                continue

            length = int(encoded.size)
            lengths.append(length)
            magic = encoded[:4].tobytes().hex(" ") if length else "-"
            decoded = "-"
            if length:
                with contextlib.suppress(cv2.error):
                    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if frame is not None:
                        shape = tuple(int(part) for part in frame.shape)
                        decoded_shapes.append(shape)
                        any_decodable = True
                        decoded = f"yes shape={shape}"
            print(
                f"sample {index + 1:02d}: code=0 bytes={length} "
                f"magic={magic} decoded={decoded}",
                flush=True,
            )
            if index < sample_count - 1 and sample_interval_seconds > 0:
                time.sleep(sample_interval_seconds)

        if lengths:
            non_empty = sum(1 for length in lengths if length > 0)
            print(
                f"summary: {non_empty}/{len(lengths)} code=0 samples had bytes; "
                f"max_bytes={max(lengths)}; nonzero_codes={nonzero_codes or 'none'}",
                flush=True,
            )
        elif nonzero_codes:
            print(f"summary: only non-zero codes: {nonzero_codes}", flush=True)
        else:
            print("summary: no samples returned.", flush=True)

        if decoded_shapes:
            print(f"decoded shapes seen: {decoded_shapes[:3]}", flush=True)

    if not any_decodable:
        print(
            "\nNo SDK2 service produced a decodable image. If videohub shows "
            "code=0 bytes=0 repeatedly, DDS/RPC is reachable but that service "
            "is not publishing camera JPEGs on this robot. For G1 head cameras, "
            "try the OpenCV source path on the machine that sees the camera, "
            "for example: `--camera opencv --opencv-source /dev/video0`.",
            flush=True,
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a real robot camera, optionally classifying rock, paper, "
            "and scissors using MediaPipe Hands. Dry-friendly: no robot motion "
            "is commanded; only the camera is read."
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
        choices=("auto", "videohub", "front", "back", "opencv"),
        default="auto",
        help=(
            "Which camera source to read from. `auto` tries SDK2 videohub, then "
            "front/back; `videohub` uses the SDK2 Go2 video service; `front`/"
            "`back` use B2 SDK2 video clients; `opencv` uses cv2.VideoCapture "
            "for UVC, RealSense, RTSP/HTTP, or GStreamer sources."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=3.0,
        help="Per-call image timeout for the Unitree video client.",
    )
    parser.add_argument(
        "--decode-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "How long to wait for decodable SDK2 image bytes. Use 0 to wait "
            "indefinitely, which is useful while restarting robot-side video "
            "services."
        ),
    )
    parser.add_argument(
        "--auto-fallback-samples",
        type=int,
        default=0,
        help=(
            "In `--camera auto` mode, switch to the next SDK2 camera service "
            "after this many successful-but-empty/invalid samples. Use 0 to "
            "keep waiting on a reachable service; non-zero error codes still "
            "fall through to the next service."
        ),
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview frame horizontally.",
    )
    parser.add_argument(
        "--opencv-source",
        default="0",
        help=(
            "OpenCV VideoCapture source used with `--camera opencv`: device index "
            "like `0`, path like `/dev/video0`, RTSP/HTTP URL, or GStreamer string."
        ),
    )
    parser.add_argument(
        "--opencv-width",
        type=int,
        default=1280,
        help="Requested OpenCV capture width when using `--camera opencv`.",
    )
    parser.add_argument(
        "--opencv-height",
        type=int,
        default=720,
        help="Requested OpenCV capture height when using `--camera opencv`.",
    )
    parser.add_argument(
        "--opencv-fps",
        type=int,
        default=30,
        help="Requested OpenCV capture FPS when using `--camera opencv`.",
    )
    parser.add_argument(
        "--probe-camera-services",
        action="store_true",
        help=(
            "Probe SDK2 videohub/front/back services, print raw sample sizes, "
            "then exit. Useful when SDK2 returns success with empty image payloads."
        ),
    )
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=10,
        help="Number of samples per SDK2 service for `--probe-camera-services`.",
    )
    parser.add_argument(
        "--probe-interval-seconds",
        type=float,
        default=0.0,
        help=(
            "Delay between probe samples. Use this with a larger sample count to "
            "watch whether videohub starts returning bytes."
        ),
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
    if args.probe_samples <= 0:
        raise SystemExit("--probe-samples must be > 0")
    if args.probe_interval_seconds < 0:
        raise SystemExit("--probe-interval-seconds must be >= 0")
    if args.decode_timeout_seconds < 0:
        raise SystemExit("--decode-timeout-seconds must be >= 0")
    if args.auto_fallback_samples < 0:
        raise SystemExit("--auto-fallback-samples must be >= 0")

    if args.probe_camera_services:
        return _probe_robot_camera_services(
            interface=args.interface,
            domain_id=args.domain_id,
            timeout_seconds=args.timeout_seconds,
            sample_count=args.probe_samples,
            sample_interval_seconds=args.probe_interval_seconds,
        )

    camera_config = RobotCameraConfig(
        interface=args.interface,
        domain_id=args.domain_id,
        camera=args.camera,
        timeout_seconds=args.timeout_seconds,
        decode_timeout_seconds=args.decode_timeout_seconds,
        auto_fallback_samples=args.auto_fallback_samples,
        mirror=args.mirror,
    )
    opencv_config = OpenCvCameraConfig(
        source=_parse_opencv_source(args.opencv_source),
        width=args.opencv_width,
        height=args.opencv_height,
        fps=args.opencv_fps,
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
        camera_source = (
            OpenCvCameraSource(opencv_config)
            if args.camera == "opencv"
            else RobotCameraSource(camera_config)
        )
        camera_label = (
            f"opencv source {opencv_config.source!r}"
            if args.camera == "opencv"
            else f"{args.camera} camera"
        )

        with camera_source as camera:
            first_frame = camera.read()
            if isinstance(camera, RobotCameraSource):
                camera_label = f"{camera.active_camera} camera"
            print(
                f"Robot vision running on {camera_label}. Press Ctrl+C to quit.",
                flush=True,
            )

            if not args.no_classifier:
                from g1_rps.vision import (
                    ClassifierConfig,
                    HandGestureClassifier,
                    draw_landmarks,
                )

                classifier_config = ClassifierConfig(
                    extension_margin=args.extension_margin
                )
                classifier = HandGestureClassifier(classifier_config)
                draw_landmarks_fn = draw_landmarks

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
                for frame in itertools.chain((first_frame,), camera.frames()):
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
                            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
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
