"""Trigger the Unitree G1 high-level `WaveHand` gesture via `LocoClient`."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make the real Unitree G1 wave its hand using the official high-level "
            "LocoClient. The robot must be standing and able to move freely."
        )
    )
    parser.add_argument(
        "--interface",
        required=True,
        help="DDS network interface that reaches the robot, for example `enp1s0`.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="CycloneDDS domain ID used by Unitree SDK2 (default: 0).",
    )
    parser.add_argument(
        "--turn",
        action="store_true",
        help="Use WaveHand(True) so the robot also turns around while waving.",
    )
    parser.add_argument(
        "--prep",
        action="store_true",
        help="Before waving, run Damp() then Squat2StandUp() to bring the robot up.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to call WaveHand (default: 1).",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=4.0,
        help="Seconds to wait between repeated WaveHand calls (default: 4.0).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="LocoClient RPC timeout in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Skip the interactive safety prompt. Only use in supervised contexts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Import the SDK and print the planned actions without contacting the robot.",
    )
    return parser.parse_args()


def safety_prompt(args: argparse.Namespace) -> None:
    print("WARNING: ensure the robot is standing in an open area with no obstacles.")
    if args.turn:
        print("--turn was passed: the robot will also rotate while waving.")
    if args.prep:
        print("--prep was passed: the robot will Damp + Squat2StandUp before waving.")
    if not args.skip_confirm:
        input("Press Enter to continue, or Ctrl+C to abort...")


def main() -> None:
    args = parse_args()
    safety_prompt(args)

    if args.dry_run:
        print("Dry run only. Planned actions:")
        if args.prep:
            print("  - LocoClient.Damp()")
            print("  - sleep 0.5s")
            print("  - LocoClient.Squat2StandUp()")
            print("  - sleep 3.0s")
        for index in range(args.repeat):
            arg_repr = "True" if args.turn else ""
            print(
                f"  - LocoClient.WaveHand({arg_repr})  # call {index + 1}/{args.repeat}"
            )
            if index + 1 < args.repeat:
                print(f"  - sleep {args.gap_seconds}s")
        print("No DDS traffic was sent.")
        return

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print(
        f"Initializing DDS on interface {args.interface!r}, domain {args.domain_id}..."
    )
    ChannelFactoryInitialize(args.domain_id, args.interface)

    client = LocoClient()
    client.SetTimeout(args.timeout_seconds)
    client.Init()

    if args.prep:
        print("Damping...")
        client.Damp()
        time.sleep(0.5)
        print("Standing up from squat...")
        client.Squat2StandUp()
        time.sleep(3.0)

    for index in range(args.repeat):
        print(f"WaveHand call {index + 1}/{args.repeat} (turn={args.turn})...")
        if args.turn:
            client.WaveHand(True)
        else:
            client.WaveHand()
        if index + 1 < args.repeat:
            time.sleep(args.gap_seconds)

    print("Done. Robot left in whatever pose WaveHand finished in.")


if __name__ == "__main__":
    main()
