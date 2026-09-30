from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym


def _build_humenv(task: str, **kwargs):
    import humenv

    if hasattr(humenv, "make") and callable(humenv.make):
        try:
            return humenv.make(task=task, **kwargs)
        except TypeError:
            pass

    try:
        from humenv.env import HumEnv

        try:
            return HumEnv(task=task, **kwargs)
        except TypeError:
            return HumEnv(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Cannot construct humenv env. Please check humenv API (humenv.make or humenv.env.HumEnv)."
        ) from exc


class HumEnvGymAdapter(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "walk",
        render_mode: Optional[str] = None,
        obs_mode: str = "raw",
        **humenv_kwargs,
    ):
        super().__init__()
        self.task = task
        self.render_mode = render_mode
        self.obs_mode = obs_mode

        self._env = _build_humenv(task=task, render_mode=render_mode, **humenv_kwargs)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        try:
            return self._env.reset(seed=seed, options=options)
        except TypeError:
            return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self):
        if hasattr(self._env, "render"):
            return self._env.render()
        return None

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()
