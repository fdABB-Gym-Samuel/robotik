"""Rock-paper-scissors classifier using MediaPipe's Hand Landmarker task.

Reads a BGR frame (as produced by ``src/g1_rps/camera.py``), runs the
MediaPipe Hand Landmarker to extract 21 3D landmarks, and classifies the
gesture by counting which non-thumb fingers are extended:

    0 extended                     -> rock
    2 extended (index + middle)    -> scissors
    4 extended                     -> paper

The thumb is intentionally ignored because its extension direction is
sideways rather than outward from the wrist, which makes the simple
"tip-is-farther-from-wrist-than-PIP" test unreliable for it.

``classify`` returns ``None`` for the gesture when no hand is detected or
when the finger pattern does not match one of the three gestures, so the
caller can choose to hold the last valid classification.

This module targets the MediaPipe 0.10.30+ Tasks API (``mediapipe.tasks``).
The earlier ``mediapipe.solutions.hands`` wrapper was removed in 0.10.30.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as exc:
    raise ImportError(
        "mediapipe>=0.10.30 is required for hand gesture classification. "
        "Install it with 'pip install mediapipe'."
    ) from exc

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required to convert BGR frames for MediaPipe. "
        "Install it with 'pip install opencv-python'."
    ) from exc


Gesture = Literal["rock", "paper", "scissors"]

# MediaPipe Hands landmark indices (unchanged between the old solutions
# API and the new Tasks API).
# Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
_WRIST = 0
_FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
_FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

# 21-point skeleton for drawing. Each tuple is (start_idx, end_idx).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Palm
    (0, 1),
    (0, 5),
    (0, 17),
    (5, 9),
    (9, 13),
    (13, 17),
    # Thumb
    (1, 2),
    (2, 3),
    (3, 4),
    # Index
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky
    (17, 18),
    (18, 19),
    (19, 20),
)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass(frozen=True)
class ClassifierConfig:
    min_hand_detection_confidence: float = 0.6
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # A finger is considered extended when its tip is farther from the wrist
    # than its PIP joint by at least this factor. Larger values require a
    # more clearly stretched finger. Tune between ~1.05 and ~1.30 for
    # different camera distances and hand sizes.
    extension_margin: float = 1.15
    # Where to cache the downloaded ``hand_landmarker.task`` model file.
    # ``None`` means use the default path under ``runs/assets/mediapipe``.
    model_path: Path | None = None


@dataclass(frozen=True)
class ClassificationResult:
    gesture: Gesture | None
    extended_fingers: tuple[str, ...]
    # Normalized image-space landmarks, shape (21, 3), or ``None``. ``x``/``y``
    # are in [0, 1] relative to the frame; ``z`` is relative depth.
    landmarks: np.ndarray | None


def default_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "runs"
        / "assets"
        / "mediapipe"
        / "hand_landmarker.task"
    )


def ensure_hand_landmarker_model(model_path: Path | None = None) -> Path:
    """Download the HandLandmarker ``.task`` file on first use."""
    target = Path(model_path) if model_path is not None else default_model_path()
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand landmarker model -> {target}", flush=True)
    urllib.request.urlretrieve(_MODEL_URL, target)
    return target


class HandGestureClassifier:
    """MediaPipe Tasks HandLandmarker wrapped with a rule-based RPS classifier."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self._config = config or ClassifierConfig()
        model_path = ensure_hand_landmarker_model(self._config.model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self._config.min_hand_detection_confidence,
            min_hand_presence_confidence=self._config.min_hand_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time = time.monotonic()

    def classify(self, frame_bgr: np.ndarray) -> ClassificationResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return ClassificationResult(
                gesture=None, extended_fingers=(), landmarks=None
            )

        landmarks = _landmarks_to_array(result.hand_landmarks[0])
        extended = _extended_fingers(landmarks, self._config.extension_margin)
        gesture = _gesture_from_extended(extended)
        return ClassificationResult(
            gesture=gesture, extended_fingers=extended, landmarks=landmarks
        )

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "HandGestureClassifier":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def draw_landmarks(frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
    """Overlay the 21 hand landmarks and their skeleton on a BGR frame.

    ``landmarks`` is the shape-(21, 3) normalized array from
    ``ClassificationResult.landmarks``. Drawing is done in-place.
    """
    if landmarks is None or len(landmarks) != 21:
        return
    height, width = frame_bgr.shape[:2]
    pixel_points = [
        (int(round(x * width)), int(round(y * height))) for x, y, _ in landmarks
    ]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            frame_bgr,
            pixel_points[start_idx],
            pixel_points[end_idx],
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    for point in pixel_points:
        cv2.circle(frame_bgr, point, 4, (0, 0, 255), -1, cv2.LINE_AA)


def _landmarks_to_array(hand_landmarks) -> np.ndarray:
    return np.array(
        [(lm.x, lm.y, lm.z) for lm in hand_landmarks],
        dtype=np.float32,
    )


def _extended_fingers(landmarks: np.ndarray, margin: float) -> tuple[str, ...]:
    """Return the names of non-thumb fingers whose tips are extended."""
    wrist = landmarks[_WRIST]
    extended: list[str] = []
    for name in ("index", "middle", "ring", "pinky"):
        tip_dist = float(np.linalg.norm(landmarks[_FINGER_TIPS[name]] - wrist))
        pip_dist = float(np.linalg.norm(landmarks[_FINGER_PIPS[name]] - wrist))
        if tip_dist > margin * pip_dist:
            extended.append(name)
    return tuple(extended)


def _gesture_from_extended(extended: tuple[str, ...]) -> Gesture | None:
    count = len(extended)
    if count == 0:
        return "rock"
    if count == 4:
        return "paper"
    if count == 2 and set(extended) == {"index", "middle"}:
        return "scissors"
    return None
