"""Run the rock-paper-scissors hand classifier against one or more still images.

Useful for testing the vision pipeline without a camera: pass an image (or a
directory of images) and the script will load each frame, run it through
``HandGestureClassifier``, print the detected gesture and extended fingers,
and -- optionally -- show or save an annotated copy with the 21 hand
landmarks drawn on top.

Examples:
    # Single image, show window with landmarks (press any key to advance/quit):
    python scripts/test_vision.py path/to/hand.jpg

    # Headless: just print results.
    python scripts/test_vision.py path/to/hand.jpg --no-display

    # Run the classifier across every image in a directory and write
    # annotated copies next to them.
    python scripts/test_vision.py samples/ --output-dir samples/annotated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import cv2

from g1_rps.vision import (
    ClassifierConfig,
    HandGestureClassifier,
    draw_landmarks,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the rock-paper-scissors hand classifier on still images. "
            "Pass an image file or a directory; for each input the detected "
            "gesture and extended fingers are printed."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Image file or directory of images to classify.",
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write annotated copies of each input image here.",
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--display",
        dest="display",
        action="store_true",
        help="Show each annotated image in a window (default). Press any key to advance.",
    )
    display_group.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Run headless; only print results to the terminal.",
    )
    parser.set_defaults(display=True)
    return parser.parse_args()


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise SystemExit(f"No supported images found in directory: {path}")
        return images
    raise SystemExit(f"Input path does not exist: {path}")


def annotate(frame, result) -> "cv2.Mat":
    annotated = frame.copy()
    if result.landmarks is not None:
        draw_landmarks(annotated, result.landmarks)
    label = result.gesture if result.gesture is not None else "—"
    fingers = ",".join(result.extended_fingers) if result.extended_fingers else "none"
    lines = [f"gesture: {label}", f"fingers: {fingers}"]
    x, y = 20, 40
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 36
    return annotated


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args.input)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    config = ClassifierConfig(extension_margin=args.extension_margin)
    matched = 0
    detected = 0

    with HandGestureClassifier(config) as clf:
        for image_path in inputs:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(
                    f"{image_path}: could not read image (corrupt or unsupported format)"
                )
                continue

            result = clf.classify(frame)
            label = result.gesture if result.gesture is not None else "—"
            fingers = ",".join(result.extended_fingers) or "none"
            landmark_count = 0 if result.landmarks is None else len(result.landmarks)
            print(
                f"{image_path}: gesture={label}  "
                f"extended=[{fingers}]  landmarks={landmark_count}"
            )

            if landmark_count > 0:
                detected += 1
            if result.gesture is not None:
                matched += 1

            if args.output_dir is not None or args.display:
                annotated = annotate(frame, result)
                if args.output_dir is not None:
                    out_path = args.output_dir / image_path.name
                    cv2.imwrite(str(out_path), annotated)
                if args.display:
                    cv2.imshow("test_vision", annotated)
                    # Press any key to advance, 'q' or ESC to quit early.
                    key = cv2.waitKey(0) & 0xFF
                    if key in (ord("q"), 27):
                        break

    if args.display:
        cv2.destroyAllWindows()

    total = len(inputs)
    print(
        f"\nProcessed {total} image(s): "
        f"{detected} with a hand detected, "
        f"{matched} classified as a known gesture."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
