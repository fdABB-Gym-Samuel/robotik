"""Run the rock-paper-scissors hand gestures on the real Unitree Inspire hand."""

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

from g1_rps.hardware import HardwareConfig, run_hardware_sequence
from g1_rps.poses import DEFAULT_SEQUENCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drive the real Unitree Inspire hand through rock, paper, and scissors. "
            "Dry-run is the default; add --live to publish DDS commands."
        )
    )
    parser.add_argument(
        "--sequence",
        nargs="+",
        default=list(DEFAULT_SEQUENCE),
        help="Gesture sequence to send. Example: --sequence rock paper scissors",
    )
    parser.add_argument(
        "--hand",
        choices=("right", "left"),
        default="right",
        help="Which physical hand to command on the robot.",
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=1.0,
        help="Seconds used to interpolate between two hand poses.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="Seconds to keep each hand pose at the target command.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=25.0,
        help="DDS command send rate while interpolating and holding.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="CycloneDDS domain ID. Unitree documentation uses 0 on the real robot.",
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="Optional network interface passed to CycloneDDS, for example `eth0`.",
    )
    parser.add_argument(
        "--command-topic",
        default="rt/inspire/cmd",
        help="DDS topic used to send Inspire hand commands.",
    )
    parser.add_argument(
        "--state-topic",
        default="rt/inspire/state",
        help="DDS topic used to read Inspire hand state.",
    )
    parser.add_argument(
        "--state-timeout-seconds",
        type=float,
        default=1.0,
        help="How long to wait for the initial hand state before falling back to an open pose.",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Print outgoing commands and any received hand state samples.",
    )
    parser.add_argument(
        "--no-return-to-open",
        action="store_true",
        help="Do not return the hand to the open paper pose at the end.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually publish DDS commands to the robot. Without this flag the script only prints them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HardwareConfig(
        sequence=tuple(args.sequence),
        transition_seconds=args.transition_seconds,
        hold_seconds=args.hold_seconds,
        rate_hz=args.rate_hz,
        hand=args.hand,
        domain_id=args.domain_id,
        network_interface=args.interface,
        command_topic=args.command_topic,
        state_topic=args.state_topic,
        state_timeout_seconds=args.state_timeout_seconds,
        live=args.live,
        print_state=args.print_state,
        return_to_open=not args.no_return_to_open,
    )
    run_hardware_sequence(config)


if __name__ == "__main__":
    main()
