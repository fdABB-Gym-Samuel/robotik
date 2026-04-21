"""Run the pre-reveal right-arm rocking motion on the real Unitree G1."""

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

from g1_rps.arm_hardware import ArmHardwareConfig, run_pre_reveal_right_arm_hardware


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the rhythmic pre-reveal right-arm motion on the real Unitree G1. "
            "This does not reveal rock, paper, or scissors and it does not use MuJoCo."
        )
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="Optional DDS network interface, for example `eth0`.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="CycloneDDS domain ID used by Unitree SDK2.",
    )
    parser.add_argument(
        "--state-timeout-seconds",
        type=float,
        default=5.0,
        help="How long to wait for the robot's `rt/lowstate` sample before failing.",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Print the initial right-arm joint state before moving.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually send `rt/arm_sdk` commands to the real robot. Without this flag it is a dry run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ArmHardwareConfig(
        interface=args.interface,
        domain_id=args.domain_id,
        live=args.live,
        print_state=args.print_state,
        state_timeout_seconds=args.state_timeout_seconds,
    )
    run_pre_reveal_right_arm_hardware(config)


if __name__ == "__main__":
    main()

