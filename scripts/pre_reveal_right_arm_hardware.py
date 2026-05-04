"""Run the pre-reveal right-arm rocking motion on the real Unitree G1."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from g1_rps.arm_hardware import (
    ArmHardwareConfig,
    run_pre_reveal_right_arm_hardware,
    validate_dds_network_interface,
)

CPP_HELPER_SOURCE = (
    PROJECT_ROOT / "src" / "g1_rps" / "cpp" / "pre_reveal_right_arm_hardware.cpp"
)
CPP_HELPER_BINARY = (
    PROJECT_ROOT / "runs" / "bin" / "g1_pre_reveal_right_arm_hardware_cpp"
)


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
        "--arm-dof",
        type=int,
        choices=(5, 7),
        default=7,
        help=(
            "Match Unitree's official G1 arm example variant: "
            "`5` for the 23-dof arm5 layout or `7` for the 29-dof arm7 layout."
        ),
    )
    parser.add_argument(
        "--state-timeout-seconds",
        type=float,
        default=5.0,
        help="How long to wait for the robot's `rt/lowstate` sample before failing.",
    )
    parser.add_argument(
        "--motion-switch-timeout-seconds",
        type=float,
        default=5.0,
        help="How long to spend asking MotionSwitcherClient to release high-level mode.",
    )
    parser.add_argument(
        "--no-auto-release",
        action="store_true",
        help="Skip MotionSwitcherClient release. Use this after releasing manually with L2+B then L2+R2.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpp", "python"),
        default="auto",
        help="DDS backend for real hardware. `auto` uses the C++ Unitree SDK when available.",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Print the initial right-arm joint state before moving.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually send `rt/lowcmd` commands to the real robot. Without this flag it is a dry run.",
    )
    return parser.parse_args()


def unitree_cpp_sdk_prefix() -> Path | None:
    for prefix in (Path("/usr/local"), Path("/opt/unitree_robotics")):
        header = (
            prefix / "include" / "unitree" / "robot" / "channel" / "channel_factory.hpp"
        )
        library = prefix / "lib" / "libunitree_sdk2.a"
        if header.exists() and library.exists():
            return prefix
    return None


def uid_is_listed_in_etc_passwd() -> bool:
    uid_text = str(os.getuid())
    try:
        for line in (
            Path("/etc/passwd")
            .read_text(encoding="utf-8", errors="ignore")
            .splitlines()
        ):
            fields = line.split(":")
            if len(fields) > 2 and fields[2] == uid_text:
                return True
    except OSError:
        return True
    return False


def cpp_helper_has_nix_runtime() -> bool:
    try:
        return b"/nix/store/" in CPP_HELPER_BINARY.read_bytes()
    except OSError:
        return False


def cpp_helper_may_fail_unitree_user_lookup() -> bool:
    return cpp_helper_has_nix_runtime() and not uid_is_listed_in_etc_passwd()


def cpp_helper_needs_rebuild() -> bool:
    if not CPP_HELPER_BINARY.exists():
        return True
    if CPP_HELPER_BINARY.stat().st_mtime < CPP_HELPER_SOURCE.stat().st_mtime:
        return True
    return cpp_helper_may_fail_unitree_user_lookup()


def build_cpp_helper() -> None:
    prefix = unitree_cpp_sdk_prefix()
    if prefix is None:
        raise RuntimeError(
            "Unitree C++ SDK was not found under /usr/local or /opt/unitree_robotics."
        )
    if not CPP_HELPER_SOURCE.exists():
        raise RuntimeError(f"C++ backend source was not found: {CPP_HELPER_SOURCE}")

    if not cpp_helper_needs_rebuild():
        return

    if cpp_helper_may_fail_unitree_user_lookup():
        print(
            "Existing C++ backend was linked against a Nix runtime that cannot "
            "resolve this NSS-only user. Rebuilding with the current compiler."
        )

    CPP_HELPER_BINARY.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        f"-I{prefix / 'include'}",
        f"-I{prefix / 'include' / 'ddscxx'}",
        f"-L{prefix / 'lib'}",
        f"-Wl,-rpath,{prefix / 'lib'}",
        "-o",
        str(CPP_HELPER_BINARY),
        str(CPP_HELPER_SOURCE),
        "-lunitree_sdk2",
        "-lddscxx",
        "-lddsc",
        "-lpthread",
    ]
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("`g++` is required to build the C++ backend.") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Failed to build C++ backend:\n{details}") from exc

    if cpp_helper_may_fail_unitree_user_lookup():
        raise RuntimeError(
            "The rebuilt C++ backend still links against a Nix runtime, while "
            f"UID {os.getuid()} is not present in /etc/passwd. Unitree SDK2 is "
            "likely to fail with `getpwuid error`; rebuild with the system "
            "compiler/runtime or add a local passwd entry for this user."
        )


def run_cpp_backend(args: argparse.Namespace) -> None:
    if args.interface is None:
        raise RuntimeError("The C++ backend requires --interface.")
    build_cpp_helper()

    cmd = [
        str(CPP_HELPER_BINARY),
        "--interface",
        args.interface,
        "--domain-id",
        str(args.domain_id),
        "--state-timeout-seconds",
        str(args.state_timeout_seconds),
        "--motion-switch-timeout-seconds",
        str(args.motion_switch_timeout_seconds),
    ]
    if args.live:
        cmd.append("--live")
    if args.print_state:
        cmd.append("--print-state")
    if args.no_auto_release:
        cmd.append("--no-auto-release")

    raise SystemExit(subprocess.run(cmd).returncode)


def main() -> None:
    args = parse_args()
    try:
        validate_dds_network_interface(args.interface)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    if args.backend in ("auto", "cpp") and (args.live or args.print_state):
        try:
            run_cpp_backend(args)
        except RuntimeError as exc:
            if args.backend == "cpp":
                raise SystemExit(str(exc)) from None
            print(f"C++ backend unavailable ({exc}). Falling back to Python DDS.")

    config = ArmHardwareConfig(
        interface=args.interface,
        domain_id=args.domain_id,
        arm_dof=args.arm_dof,
        live=args.live,
        print_state=args.print_state,
        auto_release_mode=not args.no_auto_release,
        state_timeout_seconds=args.state_timeout_seconds,
        motion_switch_timeout_seconds=args.motion_switch_timeout_seconds,
    )
    try:
        run_pre_reveal_right_arm_hardware(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
