"""Utilities for the Unitree G1 rock-paper-scissors MuJoCo demo."""

from .assets import ensure_unitree_g1_assets
from .poses import DEFAULT_SEQUENCE, DemoConfig
from .hardware import HardwareConfig, build_hardware_channels, run_hardware_sequence

__all__ = [
    "DemoConfig",
    "DEFAULT_SEQUENCE",
    "HardwareConfig",
    "build_hardware_channels",
    "ensure_unitree_g1_assets",
    "run_hardware_sequence",
]
