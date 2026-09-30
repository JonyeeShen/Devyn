from __future__ import annotations

import gymnasium as gym


def register_humanoid_bench_envs() -> None:
    required_envs = [
        "h1-run-v0",
        "h1-slide-v0",
        "h1-hurdle-v0",
        "h1-crawl-v0",
        "h1-reach-v0",
        "h1-sit_simple-v0",
        "h1-stair-v0",
        "h1-balance_simple-v0",
        "h1-stand-v0",
        "h1-walk-v0",
    ]
    required_envs.extend([name.replace("h1-", "g1-") for name in required_envs])

    if all(env_id in gym.envs.registry for env_id in required_envs):
        return

    import humanoid_bench  # noqa: F401

    missing = [env_id for env_id in required_envs if env_id not in gym.envs.registry]
    if missing:
        raise RuntimeError(
            "HumanoidBench envs not registered. Missing: " + ", ".join(missing)
        )
