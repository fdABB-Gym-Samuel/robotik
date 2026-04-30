"""Utilities for the Unitree G1 rock-paper-scissors MuJoCo demo."""

from .arm_hardware import ArmHardwareConfig, run_pre_reveal_right_arm_hardware
from .assets import ensure_unitree_g1_assets
from .poses import DEFAULT_SEQUENCE, DemoConfig
from .hardware import HardwareConfig, build_hardware_channels, run_hardware_sequence

__all__ = [
    "ArmHardwareConfig",
    "DemoConfig",
    "DEFAULT_SEQUENCE",
    "HardwareConfig",
    "build_hardware_channels",
    "ensure_unitree_g1_assets",
    "run_pre_reveal_right_arm_hardware",
    "run_hardware_sequence",
]
