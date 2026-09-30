import gymnasium as gym
import mujoco
import numpy as np


class MyosuiteWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._sim = self.env.unwrapped.sim
        self.model = self._sim.model
        self.data = self._sim.data

        if hasattr(self.observation_space, "dtype") and self.observation_space.dtype != np.float32:
            from gymnasium.spaces import Box

            self.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=self.observation_space.shape,
                dtype=np.float32,
            )

    @property
    def sim(self):
        return self.env.unwrapped.sim

    def render_rgb(self, width=640, height=480, camera_id=-1, device_id=0):
        return self.sim.renderer.render_offscreen(
            width,
            height,
            camera_id=camera_id,
            device_id=device_id,
        )

    def render(self):
        try:
            return self.env.render()
        except Exception:
            return self.render_rgb()


class MyosuiteRewardInfoWrapper(gym.Wrapper):
    def __init__(self, env, rwd_keys_wt=None):
        super().__init__(env)
        self.rwd_keys_wt = rwd_keys_wt or {}
        self._accum = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._accum = {}
        obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rwd_dict = info.get("rwd_dict", {})
        keys_wt = self.rwd_keys_wt or info.get("rwd_keys_wt", {})

        if self._accum is None:
            self._accum = {}

        for key, wt in keys_wt.items():
            val = float(wt) * float(rwd_dict.get(key, 0.0))
            self._accum[key] = self._accum.get(key, 0.0) + val

        if terminated or truncated:
            info = dict(info)
            info["rwd_accum"] = dict(self._accum)

        obs = np.asarray(obs, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info


class HumEnvProprioAsObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert "proprio" in env.observation_space.spaces
        self.observation_space = env.observation_space["proprio"]

    def observation(self, obs):
        return np.asarray(obs["proprio"], dtype=np.float32)


class TerminateOnFall(gym.Wrapper):
    def __init__(self, env, min_pelvis_height=0.25, n_consecutive=5, body_name="Pelvis"):
        super().__init__(env)
        self.min_pelvis_height = float(min_pelvis_height)
        self.n_consecutive = int(n_consecutive)
        self.body_name = body_name
        self._pelvis_id = None
        self._below_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._below_count = 0
        self._ensure_body_id()
        return obs, info

    def _ensure_body_id(self):
        if self._pelvis_id is not None:
            return
        host = self.env.unwrapped._env
        model = getattr(host, "model", None)
        if model is None:
            return
        self._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ensure_body_id()

        host = self.env.unwrapped._env
        data = getattr(host, "data", None)
        if data is None or self._pelvis_id is None:
            return obs, reward, terminated, truncated, info

        pelvis_z = float(data.xpos[self._pelvis_id, 2])

        if pelvis_z < self.min_pelvis_height:
            self._below_count += 1
        else:
            self._below_count = 0

        fell = self._below_count >= self.n_consecutive
        if fell:
            terminated = True
            info = dict(info)
            info["fell"] = True
            info["pelvis_z"] = pelvis_z
            info["fell_count"] = self._below_count

        return obs, reward, terminated, truncated, info


class FallPenaltyOnDone(gym.Wrapper):
    def __init__(self, env, fall_penalty=10.0, key="fell"):
        super().__init__(env)
        self.fall_penalty = float(fall_penalty)
        self.key = key

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        fell = bool(info.get(self.key, False))
        if fell and terminated:
            reward = float(reward) - self.fall_penalty
            info["fall_penalty"] = self.fall_penalty
        return obs, reward, terminated, truncated, info


class MuscleNormWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.env.action_space.shape[0],),
        )

    def action(self, action):
        return 1.0 / (1.0 + np.exp(-5.0 * (action - 0.5)))
