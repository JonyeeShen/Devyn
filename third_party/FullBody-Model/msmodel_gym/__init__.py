# msmodel_gym/__init__.py
import gymnasium as gym
from .OstrichRun_v1 import OstrichRunEnvV1

# 如果跑脚本时重复 import，不报错
if "OstrichRun-v1" not in gym.envs.registry:
    gym.register(
        id="OstrichRun-v1",
        entry_point="msmodel_gym:OstrichRunEnvV1",
        max_episode_steps=10000,
    )