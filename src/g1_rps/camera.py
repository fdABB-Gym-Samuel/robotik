"""Laptop webcam source for the rock-paper-scissors vision pipeline.

This wraps OpenCV's VideoCapture so the rest of the pipeline can stay
agnostic to the frame source. The same API can later be re-implemented for
the G1's onboard RealSense camera or for the multicast UDP stream from
Unitree's multimedia service, without changing the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for the webcam source. "
        "Install it with 'pip install opencv-python'."
    ) from exc


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Mirror the frame horizontally so students see themselves as in a mirror.
    # Classification is unaffected because MediaPipe is orientation-agnostic.
    mirror: bool = True


class WebcamSource:
    """Context-managed BGR frame source backed by an OpenCV VideoCapture."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self._config = config or CameraConfig()
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self._config.device_index)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self._config.device_index}. "
                "Check that a camera is connected and not in use by another application."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        self._capture = capture

    def read(self) -> np.ndarray:
        """Return the next BGR frame as a HxWx3 uint8 numpy array."""
        if self._capture is None:
            raise RuntimeError(
                "Camera is not open. Call open() or use as a context manager."
            )
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from webcam.")
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

    def __enter__(self) -> "WebcamSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
