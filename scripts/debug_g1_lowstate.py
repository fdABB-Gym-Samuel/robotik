"""Listen for Unitree G1 `rt/lowstate` DDS samples and print a small summary."""

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

from g1_rps.arm_hardware import validate_dds_network_interface
from g1_rps.unitree_sdk2_config import configure_local_cyclonedds_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen for G1 `rt/lowstate` and print whether any DDS samples arrive."
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="Optional DDS network interface, for example `eth1`.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="CycloneDDS domain ID used by Unitree SDK2.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="How long to wait before declaring that no lowstate samples arrived.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    try:
        validate_dds_network_interface(args.interface)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    configure_local_cyclonedds_log()

    received: dict[str, object] = {"sample": None, "count": 0}

    def on_lowstate(msg: LowState_) -> None:
        received["sample"] = msg
        received["count"] = int(received["count"]) + 1

    if args.interface is not None:
        ChannelFactoryInitialize(args.domain_id, args.interface)
    else:
        ChannelFactoryInitialize(args.domain_id)

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(on_lowstate, 10)

    print("Listening for `rt/lowstate`...")
    print(f"DDS domain ID: {args.domain_id}")
    print(f"DDS interface: {args.interface or 'auto'}")

    deadline = time.perf_counter() + args.timeout_seconds
    while time.perf_counter() < deadline and received["sample"] is None:
        time.sleep(0.05)

    sample = received["sample"]
    if sample is None:
        raise SystemExit(
            "No `rt/lowstate` sample arrived before timeout. "
            "This means the robot is either not publishing the topic, "
            "DDS discovery is blocked, or the wrong interface/domain is being used."
        )

    print(f"Received lowstate samples: {received['count']}")
    print(f"mode_machine: {sample.mode_machine}")
    print(f"tick: {sample.tick}")
    print(f"right_shoulder_pitch q: {sample.motor_state[22].q:+.3f}")
    print(f"right_elbow q: {sample.motor_state[25].q:+.3f}")


if __name__ == "__main__":
    main()
