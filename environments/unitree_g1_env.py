"""A lightweight Gymnasium wrapper around a local Unitree G1 MuJoCo scene."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "unitree_g1" / "g1_29dof_with_hand.xml"


class UnitreeG1Env(gym.Env[np.ndarray, np.ndarray]):
    """Simple locomotion-style environment for a local Unitree G1 MuJoCo model."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML_PATH,
        frame_skip: int = 5,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(
                "Unitree G1 scene not found. Expected "
                f"{self.xml_path}. Add the G1 MuJoCo XML and meshes to assets/unitree_g1/."
            )

        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        ctrlrange = np.asarray(self.model.actuator_ctrlrange, dtype=np.float32)
        if ctrlrange.size == 0:
            raise ValueError("The Unitree G1 model has no actuators, so no action space can be built.")

        self.action_space = spaces.Box(
            low=ctrlrange[:, 0],
            high=ctrlrange[:, 1],
            dtype=np.float32,
        )

        obs_size = int(self.model.nq + self.model.nv)
        obs_limit = np.full(obs_size, np.inf, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-obs_limit,
            high=obs_limit,
            dtype=np.float64,
        )

        self._viewer: mujoco.viewer.Handle | None = None
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float64, copy=False)

    def _torso_height(self) -> float:
        if self.model.nbody > 1:
            return float(self.data.xpos[1, 2])
        return float(self.data.qpos[2]) if self.model.nq > 2 else 0.0

    def _is_unhealthy(self) -> bool:
        return bool(not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all())

    def _ensure_viewer(self) -> None:
        if self.render_mode != "human" or self._viewer is not None:
            return
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, str]]:
        super().reset(seed=seed)
        noise_scale = 0.01
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self.data.qpos[:] += self.np_random.uniform(
            low=-noise_scale,
            high=noise_scale,
            size=self.model.nq,
        )
        self.data.qvel[:] += self.np_random.uniform(
            low=-noise_scale,
            high=noise_scale,
            size=self.model.nv,
        )
        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self._ensure_viewer()
            if self._viewer is not None:
                self._viewer.sync()

        return self._get_obs(), {"xml_path": str(self.xml_path)}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action = np.asarray(action, dtype=np.float32)
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)

        x_before = float(self.data.qpos[0]) if self.model.nq > 0 else 0.0
        self.data.ctrl[:] = clipped_action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        x_after = float(self.data.qpos[0]) if self.model.nq > 0 else 0.0

        forward_reward = (x_after - x_before) / (self.model.opt.timestep * self.frame_skip)
        control_cost = 0.001 * float(np.square(clipped_action).sum())
        height_penalty = 1.0 if self._torso_height() < 0.45 else 0.0
        reward = forward_reward - control_cost - height_penalty
        terminated = self._is_unhealthy() or height_penalty > 0.0

        if self.render_mode == "human":
            self._ensure_viewer()
            if self._viewer is not None:
                self._viewer.sync()

        return self._get_obs(), reward, terminated, False, {
            "forward_reward": forward_reward,
            "control_cost": control_cost,
            "torso_height": self._torso_height(),
        }

    def render(self) -> None:
        if self.render_mode == "human":
            self._ensure_viewer()
            if self._viewer is not None:
                self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
