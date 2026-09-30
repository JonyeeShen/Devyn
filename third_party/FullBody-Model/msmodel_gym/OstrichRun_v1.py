import os
import collections
from typing import Dict, Any, Optional

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


def get_observation_space(model_path: str, obs_fn, _spec: Dict[str, Any]) -> spaces.Box:
    obs = obs_fn(None)
    obs = np.asarray(obs, dtype=np.float32)
    return spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)


def get_render_fps(model_path: str, skip_frames: int) -> int:
    model = mujoco.MjModel.from_xml_path(model_path)
    timestep = float(model.opt.timestep)
    if timestep <= 0:
        return 1
    return int(round(1.0 / (timestep * max(1, int(skip_frames)))))


def action_obs_check(_env) -> None:
    return

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 13.0,
    "lookat": np.array((2, 0.0, 0)),
    "elevation": -20.0,
    "azimuth": 120
}


class OstrichRunEnvV1(MujocoEnv, EzPickle):
    metadata: Dict[str, Any] = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        forward_reward_weight=1,
        reset_noise_scale=0e-3,
        model_path: str = "",
        skip_frames: int = 5,
        reward_mode: str = "ostrichrl",
        max_step: int = 10000,
        **kwargs
    ):
        # model_path must be abspath
        model_path = os.path.dirname(__file__) + "/../../Models/ostrich/ostrich.xml"
        model_path = os.path.abspath(model_path)

        # Keep an explicit per-episode step cap for truncation.
        self.max_step = int(max_step)
        self.spec = type('', (), {})()
        self.spec.max_episode_steps = self.max_step

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        observation_space = get_observation_space(
            model_path,
            self._get_obs_core,
            {
            }
        )

        fps = get_render_fps(model_path, skip_frames)
        self.metadata["render_fps"] = fps
        self.control_timestep = 1 / fps

        EzPickle.__init__(
            self,
            render_mode,
            forward_reward_weight,
            reset_noise_scale,
            reward_mode,
            max_step,
            **kwargs
        )

        self.render_mode = render_mode
        self._reset_noise_scale = reset_noise_scale
        self.forward_reward_weight = float(forward_reward_weight)
        self.reward_mode = str(reward_mode)
        if self.reward_mode not in {"ostrichrl", "weighted", "velocity"}:
            raise ValueError("reward_mode must be 'ostrichrl', 'weighted', or 'velocity'")

        

        MujocoEnv.__init__(
            self, model_path, skip_frames, observation_space=observation_space, render_mode=render_mode, camera_name="pelvis_camera", **kwargs
        )
        action_obs_check(self)

        # print("observation space shape: ", self.observation_space.shape)
        # print("action space shape: ", self.action_space.shape)
        
        
        # self.init_qpos[:] = self.model.key_qpos[0].copy()
        # self.init_qvel[:] = self.model.key_qvel[2].copy()
        self.body_name_list = [self.model.body(body_id).name for body_id in range(self.model.nbody)]
        self.geom_name_list = [self.model.geom(geom_id).name for geom_id in range(self.model.ngeom)]
        self.sensor_name_list = [self.model.sensor(ss_id).name for ss_id in range(self.model.nsensor)]

    def seed(self, seed=0):
        pass

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    @property
    def terminated(self):
        terminated = self._get_done()
        return terminated

    def get_obs(self):
        head_height = np.array([self.head_height()], dtype=np.float32)
        pelvis_height = np.array([self.pelvis_height()], dtype=np.float32)
        feet_height = self.feet_height().astype(np.float32)
        qpos = self.qpos_without_x().astype(np.float32)
        qvel = self.qvel().astype(np.float32)

        muscle_activations = self.muscle_activations().astype(np.float32)
        muscle_forces = self.muscle_forces().astype(np.float32)
        muscle_lengths = self.muscle_lengths().astype(np.float32)
        muscle_velocities = self.muscle_velocities().astype(np.float32)

        horizontal_velocity = np.array([self.horizontal_velocity()], dtype=np.float32)

        observation = np.concatenate(
            (
                head_height,         #1
                pelvis_height,       #1   
                feet_height,         #2
                qpos,                #55
                qvel,                #56
                muscle_activations,  #120
                muscle_forces,       #120
                muscle_lengths,      #120
                muscle_velocities,   #120
                horizontal_velocity, #1
            )
        )

        return observation.astype(np.float32)

    # In order to generate observation space automatically, this method cannot use class variable,
    # so it is defined as class method.
    @staticmethod
    def _get_obs_core(data):
        # just make sure the dimension is correct
        head_height = np.zeros(1, dtype=np.float32)
        pelvis_height = np.zeros(1, dtype=np.float32)
        feet_height = np.zeros(2, dtype=np.float32)
        qpos = np.zeros(55, dtype=np.float32)
        qvel = np.zeros(56, dtype=np.float32)

        muscle_activations = np.zeros(120, dtype=np.float32)
        muscle_forces = np.zeros(120, dtype=np.float32)
        muscle_lengths = np.zeros(120, dtype=np.float32)
        muscle_velocities = np.zeros(120, dtype=np.float32)

        horizontal_velocity = np.zeros(1, dtype=np.float32)

        observation = np.concatenate(
            (
                head_height,         #1
                pelvis_height,       #1   
                feet_height,         #2
                qpos,                #55
                qvel,                #56
                muscle_activations,  #120
                muscle_forces,       #120
                muscle_lengths,      #120
                muscle_velocities,   #120
                horizontal_velocity, #1
            )
        )
        return observation.astype(np.float32)

    def step(self, action):

        self.do_simulation(action, self.frame_skip)
        observation = self.get_obs()
        
        if self.reward_mode == "weighted":
            reward = self.forward_reward_weight * self.horizontal_velocity()
        else:
            reward = self.horizontal_velocity()
        
        terminated = self.terminated
        
        info = {
            "reward": reward,
        }

        self.steps += 1
        truncated = self.steps >= self.max_step

        return observation, reward, terminated, truncated, info

    def reset_model(self):

        self.steps = 0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self.get_obs()

        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def render(self, mode=None):
        return super().render()

    def _get_done(self):
        if self.head_height() < 0.9:
            return True
        if self.pelvis_height() < 0.6:
            return True
        if self.torso_angle() < -0.8 or self.torso_angle() > 0.8:
            return True
        return False

    def qpos_without_x(self):
        return self.data.qpos.copy()[1:]

    def qvel(self):
        return np.clip(self.data.qvel, -100, 100)

    def pelvis_height(self):
        pelvis_id = self.geom_name_list.index('pelvis')
        return self.data.geom_xpos[pelvis_id][2].copy()

    def feet_height(self):
        foot_id_r = self.body_name_list.index('r_pes')
        foot_id_l = self.body_name_list.index('l_pes')
        return np.array([self.data.xpos[foot_id_r][2].copy(),
                         self.data.xpos[foot_id_l][2].copy()])

    def head_height(self):
        head_id = self.body_name_list.index('head')
        return self.data.xpos[head_id][2].copy()

    def muscle_lengths(self):
        return self.data.actuator_length.copy()

    def muscle_velocities(self):
        return np.clip(self.data.actuator_velocity, -100, 100)

    def muscle_activations(self):
        return np.clip(self.data.act, -100, 100)

    def muscle_forces(self):
        return np.clip(self.data.actuator_force / 1000, -100, 100)

    def torso_angle(self):
        return self.data.qpos[4]

    def horizontal_velocity(self):
        # print(self.sensor_name_list)
        # print(self.data.sensordata)
        #sensor_name_list = [self.data.sensordata(ss_id).name for ss_id in range(self.model.nsensordata)]
        #torso_id = sensor_name_list.index('torso_subtreelinvel')
        return self.data.sensordata[0].copy()


