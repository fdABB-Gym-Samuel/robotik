"""Environment registration helpers for project-specific simulations."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry


UNITREE_G1_ENV_ID = "UnitreeG1-v0"


def register_environments() -> None:
    """Register custom Gymnasium environments used by this repository."""

    if UNITREE_G1_ENV_ID not in registry:
        register(
            id=UNITREE_G1_ENV_ID,
            entry_point="environments.unitree_g1_env:UnitreeG1Env",
            max_episode_steps=1_000,
        )
