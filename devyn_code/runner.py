from __future__ import annotations

import argparse
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict

# Default to headless EGL rendering unless user already set MUJOCO_GL.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from devyn_code.algorithms import DevynQSAC, DevynQSACPolicy, DevynSAC, DevynSACPolicy
from devyn_code.config import get_run_config
from devyn_code.envs import WRAPPER_REGISTRY, ensure_task_env_registered


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def parse_learning_rate(value: Any) -> Any:
    if isinstance(value, (float, int)):
        return float(value)

    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("linear_schedule(") and raw.endswith(")"):
            inner = raw[len("linear_schedule(") : -1]
            return linear_schedule(float(inner))

    return value


def apply_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def _build_single_env(config: Dict[str, Any], repo_root: Path, rank: int):
    def _init():
        ensure_task_env_registered(config["task"], repo_root)

        env_kwargs = deepcopy(config["single_env_kwargs"])
        env = gym.make(config["env_name"], **env_kwargs)

        for wrapper_name, wrapper_kwargs in config["wrapper_list"].items():
            if wrapper_name not in WRAPPER_REGISTRY:
                raise KeyError(f"Unknown wrapper: {wrapper_name}")
            wrapper_cls = WRAPPER_REGISTRY[wrapper_name]
            env = wrapper_cls(env, **(wrapper_kwargs or {}))

        env = Monitor(env)
        env.reset(seed=int(config["seed"]) + int(rank))
        return env

    return _init


def build_vec_env(config: Dict[str, Any], repo_root: Path):
    env_nums = int(config["env_nums"])
    env_fns = [_build_single_env(config, repo_root, idx) for idx in range(env_nums)]

    if env_nums > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_norm_cfg = config.get("vec_normalize", {})
    if vec_norm_cfg.get("is_norm", False):
        vec_env = VecNormalize(vec_env, **vec_norm_cfg.get("kwargs", {}))

    return vec_env


def resolve_agent(agent_name: str):
    registry = {
        "SAC": SAC,
        "DevynSAC": DevynSAC,
        "DevynQSAC": DevynQSAC,
    }
    if agent_name not in registry:
        raise KeyError(f"Unsupported agent: {agent_name}")
    return registry[agent_name]


def resolve_policy(policy_name: str):
    registry = {
        "MlpPolicy": "MlpPolicy",
        "DevynSACPolicy": DevynSACPolicy,
        "DevynQSACPolicy": DevynQSACPolicy,
    }
    if policy_name not in registry:
        raise KeyError(f"Unsupported policy: {policy_name}")
    return registry[policy_name]


def train_from_config(config: Dict[str, Any], output_root: Path, repo_root: Path) -> Path:
    apply_global_seed(int(config["seed"]))
    ensure_task_env_registered(config["task"], repo_root)

    run_dir = output_root / config["model"] / config["task"] / str(config["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_env = build_vec_env(config, repo_root)

    agent_cls = resolve_agent(config["agent"])
    agent_kwargs = deepcopy(config["agent_kwargs"])

    lr = parse_learning_rate(agent_kwargs.get("learning_rate"))
    if lr is not None:
        agent_kwargs["learning_rate"] = lr

    policy_name = agent_kwargs.pop("policy", "MlpPolicy")
    policy = resolve_policy(policy_name)

    model = agent_cls(
        policy,
        vec_env,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        seed=int(config["seed"]),
        **agent_kwargs,
    )

    model.learn(total_timesteps=int(config["total_timesteps"]), progress_bar=True)

    model_path = run_dir / "model"
    model.save(str(model_path))

    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / "vecnormalize.pkl"))

    vec_env.close()
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="devyn_code unified SAC runner")
    parser.add_argument("-task", required=True, help="Task family: myoleg|hand|humenv|humanoidbench|ostrich")
    parser.add_argument(
        "-model",
        required=True,
        help="Model: devyn|devyn+q|devyn+r|devyn+q+r|devyn+r+q|sac",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda:0")
    parser.add_argument(
        "--output-root",
        type=str,
        default="devyn_code/output",
        help="Output directory root for checkpoints and logs",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Optional override of total timesteps",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()

    config = get_run_config(task=args.task, model=args.model, seed=args.seed, device=args.device)
    if args.total_timesteps is not None:
        config["total_timesteps"] = int(args.total_timesteps)

    run_dir = train_from_config(config, output_root=output_root, repo_root=repo_root)
    print(f"[devyn_code] finished. artifacts: {run_dir}")


if __name__ == "__main__":
    main()
