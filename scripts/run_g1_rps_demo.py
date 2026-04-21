"""Run a MuJoCo presentation demo of the Unitree G1 hand gestures."""

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

from g1_rps.poses import DEFAULT_SEQUENCE, DemoConfig
from g1_rps.sim import run_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show rock, paper, and scissors on the official 5-finger Unitree Inspire hand model."
    )
    parser.add_argument(
        "--sequence",
        nargs="+",
        default=list(DEFAULT_SEQUENCE),
        help="Gesture sequence to show. Example: --sequence rock paper scissors",
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=1.2,
        help="Seconds used to interpolate between two poses.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.1,
        help="Seconds to hold each final gesture pose.",
    )
    parser.add_argument(
        "--asset-dir",
        default=None,
        help="Optional local checkout for the official Unitree assets.",
    )
    parser.add_argument(
        "--camera-preset",
        choices=("hand_closeup", "upper_body"),
        default="hand_closeup",
        help="Camera framing for the presentation demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DemoConfig(
        sequence=tuple(args.sequence),
        transition_seconds=args.transition_seconds,
        hold_seconds=args.hold_seconds,
        camera_preset=args.camera_preset,
        asset_dir=args.asset_dir,
    )
    run_demo(config)


if __name__ == "__main__":
    main()
