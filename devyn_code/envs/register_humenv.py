from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register


def register_humenv_envs() -> None:
    required = {
        "HumEnv-walk-v0",
        "HumEnv-run-v0",
        "HumEnv-walk-left-v0",
        "HumEnv-jump-2-v0",
        "HumEnv-crawl-0.4-2-u-v0",
    }
    if required.issubset(gym.envs.registry.keys()):
        return

    register(
        id="HumEnv-walk-v0",
        entry_point="devyn_code.envs.humenv_adapter:HumEnvGymAdapter",
        kwargs={"task": "move-ego-0-2"},
        max_episode_steps=300,
    )
    register(
        id="HumEnv-run-v0",
        entry_point="devyn_code.envs.humenv_adapter:HumEnvGymAdapter",
        kwargs={"task": "move-ego-0-4"},
        max_episode_steps=300,
    )
    register(
        id="HumEnv-walk-left-v0",
        entry_point="devyn_code.envs.humenv_adapter:HumEnvGymAdapter",
        kwargs={"task": "move-ego-90-2"},
        max_episode_steps=300,
    )
    register(
        id="HumEnv-jump-2-v0",
        entry_point="devyn_code.envs.humenv_adapter:HumEnvGymAdapter",
        kwargs={"task": "jump-2"},
        max_episode_steps=300,
    )

    for height in [0.4, 0.5]:
        for speed in [0, 2]:
            for direction in ["u", "d"]:
                task = f"crawl-{height}-{speed}-{direction}"
                register(
                    id=f"HumEnv-{task}-v0",
                    entry_point="devyn_code.envs.humenv_adapter:HumEnvGymAdapter",
                    kwargs={"task": task},
                    max_episode_steps=300,
                )
