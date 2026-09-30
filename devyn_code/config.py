from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

VEC_NORMALIZE_DEFAULT = {
    "is_norm": True,
    "kwargs": {
        "norm_obs": True,
        "norm_reward": False,
        "clip_obs": 10.0,
    },
}

BASIC_SAC_CONFIGS = {
    "learning_rate": "linear_schedule(0.001)",
    "gradient_steps": 4,
    "batch_size": 256,
    "train_freq": 1,
    "target_update_interval": 1,
}

DEVYN_HYPERPARAMS = {
    "devyn_ema_tau": 1e-1,
    "devyn_l2sp_lambda": 1e1,
    "devyn_anchor_T": 1000,
    "devyn_hoyer_lambda": 1e-1,
    "dual_hoyer": True,
    "devyn_ortho_lambda": 0.05,
    "devyn_dircov_lambda": 0.0,
    "devyn_dircov_ema_tau": 0.05,
    "devyn_dircov_update_every": 4,
    "devyn_perturb_every": 100,
    "devyn_perturb_mode": "both",
    "devyn_perturb_ratio": 0.05,
    "devyn_perturb_shrink": 0.01,
    "devyn_perturb_noise": 0.01,
    "devyn_perturb_until": -1,
}


def _total_timesteps_block(devyn: int, sac: int | None = None) -> Dict[str, int]:
    value_sac = devyn if sac is None else sac
    return {
        "devyn": devyn,
        "devyn+q": devyn,
        "devyn+r": devyn,
        "sac": value_sac,
    }


def _shaping_start_steps_block(devyn: int) -> Dict[str, int]:
    return {
        "devyn": devyn,
        "devyn+q": devyn,
        "devyn+r": devyn,
    }

TASK_SPECIFIC_CONFIGS: Dict[str, Dict[str, Any]] = {
    "myoleg-stair": {
        "env_name": "myoLegStairTerrainWalk-v0",
        "env_nums": 80,
        "single_env_kwargs": {},
        "vec_normalize": VEC_NORMALIZE_DEFAULT,
        "wrapper_list": {"MyosuiteWrapper": {}},
        "group_num": 40,
        "total_timesteps": _total_timesteps_block(6_000_000),
        "shaping_start_steps": _shaping_start_steps_block(100_000),
    },
    "myohand-reorient": {
        "env_name": "myoHandReorient100-v0",
        "env_nums": 80,
        "single_env_kwargs": {},
        "vec_normalize": VEC_NORMALIZE_DEFAULT,
        "wrapper_list": {
            "MyosuiteWrapper": {},
            "MyosuiteRewardInfoWrapper": {},
        },
        "group_num": 30,
        "total_timesteps": _total_timesteps_block(6_000_000),
        "shaping_start_steps": _shaping_start_steps_block(100_000),
    },
    "humenv-jump": {
        "env_name": "HumEnv-jump-2-v0",
        "env_nums": 80,
        "single_env_kwargs": {},
        "vec_normalize": VEC_NORMALIZE_DEFAULT,
        "wrapper_list": {"HumEnvProprioAsObs": {}},
        "group_num": 31,
        "total_timesteps": _total_timesteps_block(3_000_000),
        "shaping_start_steps": _shaping_start_steps_block(0),
        "sac_policy": "MlpPolicy",
    },
    "h1-run": {
        "env_name": "h1-run-v0",
        "env_nums": 16,
        "single_env_kwargs": {"render_mode": None},
        "vec_normalize": VEC_NORMALIZE_DEFAULT,
        "wrapper_list": {},
        "group_num": 10,
        "total_timesteps": _total_timesteps_block(3_000_000),
        "shaping_start_steps": _shaping_start_steps_block(100_000),
        "sac_policy": "MlpPolicy",
    },
    "g1-run": {
        "env_name": "g1-run-v0",
        "env_nums": 16,
        "single_env_kwargs": {"render_mode": None},
        "vec_normalize": VEC_NORMALIZE_DEFAULT,
        "wrapper_list": {},
        "group_num": 13,
        "total_timesteps": _total_timesteps_block(3_000_000),
        "shaping_start_steps": _shaping_start_steps_block(100_000),
        "sac_policy": "MlpPolicy",
    },
    "ostrich": {
        "env_name": "OstrichRun-v1",
        "env_nums": 80,
        "single_env_kwargs": {
            "skip_frames": 5,
            "reset_noise_scale": 0,
        },
        "vec_normalize": VEC_NORMALIZE_DEFAULT,
        "wrapper_list": {"MuscleNormWrapper": {}},
        "group_num": 20,
        "total_timesteps": _total_timesteps_block(10_000_000),
        "shaping_start_steps": _shaping_start_steps_block(300_000),
    },
}

SUPPORTED_MODELS = (
    "devyn",
    "devyn+q",
    "devyn+r",
    "devyn+q+r",
    "devyn+r+q",
    "sac",
)

DEVYN_FEATURE_SWITCH = {
    "devyn": (False, False),
    "devyn+q": (True, False),
    "devyn+r": (False, True),
    "devyn+q+r": (True, True),
    "devyn+r+q": (True, True),
}

DEVYN_PERTURB_KEYS = {
    "devyn_perturb_every",
    "devyn_perturb_mode",
    "devyn_perturb_ratio",
    "devyn_perturb_shrink",
    "devyn_perturb_noise",
    "devyn_perturb_until",
    "devyn_perturb_eps",
}

DEVYN_NO_Q_WARMUP_STEPS = 10**15
DEVYN_Q_DIRCOV_LAMBDA = 0.1


def _resolve_task_model_int(task_key: str, task_cfg: Dict[str, Any], field: str, model_key: str) -> int:
    field_map = task_cfg.get(field, {})

    if model_key in field_map:
        return int(field_map[model_key])

    if model_key in {"devyn+q+r", "devyn+r+q"} and "devyn+q" in field_map:
        return int(field_map["devyn+q"])

    raise KeyError(f"Task '{task_key}' is missing '{field}' for model '{model_key}'")


def _build_sac_model_cfg(task_key: str, task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    agent_kwargs = deepcopy(BASIC_SAC_CONFIGS)
    sac_policy = task_cfg.get("sac_policy")
    if sac_policy:
        agent_kwargs["policy"] = sac_policy

    return {
        "wrapper_list": deepcopy(task_cfg.get("wrapper_list", {})),
        "total_timesteps": _resolve_task_model_int(task_key, task_cfg, "total_timesteps", "sac"),
        "agent": "SAC",
        "agent_kwargs": agent_kwargs,
    }


def _build_devyn_model_cfg(task_key: str, task_cfg: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    if model_key not in DEVYN_FEATURE_SWITCH:
        raise KeyError(f"Unsupported Devyn model: {model_key}")

    enable_q, enable_r = DEVYN_FEATURE_SWITCH[model_key]
    shaping_start_steps = _resolve_task_model_int(task_key, task_cfg, "shaping_start_steps", model_key)

    policy_kwargs = {
        "shaping_start_steps": shaping_start_steps,
        "group_num": int(task_cfg["group_num"]),
        "bias": True,
        "init_std": 1.0,
    }

    agent_kwargs = deepcopy(BASIC_SAC_CONFIGS)
    agent_kwargs.update(deepcopy(DEVYN_HYPERPARAMS))
    agent_kwargs.update(
        {
            "policy": "DevynSACPolicy",
            "policy_kwargs": policy_kwargs,
            "devyn_enable_q": bool(enable_q),
            "devyn_enable_r": bool(enable_r),
        }
    )

    if enable_q:
        agent_kwargs["devyn_dircov_lambda"] = DEVYN_Q_DIRCOV_LAMBDA
        agent_kwargs["devyn_dircov_warmup_steps"] = shaping_start_steps
    else:
        agent_kwargs["devyn_dircov_lambda"] = 0.0
        agent_kwargs["devyn_dircov_warmup_steps"] = DEVYN_NO_Q_WARMUP_STEPS

    if not enable_r:
        for key in DEVYN_PERTURB_KEYS:
            agent_kwargs.pop(key, None)

    return {
        "wrapper_list": deepcopy(task_cfg.get("wrapper_list", {})),
        "total_timesteps": _resolve_task_model_int(task_key, task_cfg, "total_timesteps", model_key),
        "agent": "DevynSAC",
        "agent_kwargs": agent_kwargs,
    }


def _build_task_model_configs() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for task_key, task_cfg in TASK_SPECIFIC_CONFIGS.items():
        models: Dict[str, Dict[str, Any]] = {}

        for model_key in SUPPORTED_MODELS:
            if model_key == "sac":
                models[model_key] = _build_sac_model_cfg(task_key, task_cfg)
            else:
                models[model_key] = _build_devyn_model_cfg(task_key, task_cfg, model_key)

        out[task_key] = {
            "env_name": task_cfg["env_name"],
            "env_nums": int(task_cfg["env_nums"]),
            "single_env_kwargs": deepcopy(task_cfg.get("single_env_kwargs", {})),
            "vec_normalize": deepcopy(task_cfg.get("vec_normalize", VEC_NORMALIZE_DEFAULT)),
            "models": models,
        }

    return out


TASK_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = _build_task_model_configs()


def _compose_model_cfg(task_cfg: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    models = task_cfg["models"]
    if model_key not in models:
        raise KeyError(f"Model '{model_key}' is not configured for this task")
    return deepcopy(models[model_key])


def normalize_task_and_model(task: str, model: str) -> Tuple[str, str]:
    task_key = task.strip().lower()
    model_key = model.strip().lower()

    if task_key not in TASK_MODEL_CONFIGS:
        valid_tasks = ", ".join(sorted(TASK_MODEL_CONFIGS.keys()))
        raise ValueError(f"Unknown task '{task}'. Valid task keys: {valid_tasks}")

    if model_key not in SUPPORTED_MODELS:
        valid_models = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unknown model '{model}'. Valid models: {valid_models}")

    return task_key, model_key


def get_run_config(task: str, model: str, seed: int, device: str | None = None) -> Dict[str, Any]:
    task_key, model_key = normalize_task_and_model(task, model)

    task_cfg = TASK_MODEL_CONFIGS[task_key]
    model_cfg = _compose_model_cfg(task_cfg, model_key)

    merged = {
        "task": task_key,
        "model": model_key,
        "env_name": task_cfg["env_name"],
        "env_nums": task_cfg["env_nums"],
        "single_env_kwargs": deepcopy(task_cfg["single_env_kwargs"]),
        "vec_normalize": deepcopy(task_cfg["vec_normalize"]),
        "wrapper_list": deepcopy(model_cfg["wrapper_list"]),
        "total_timesteps": int(model_cfg["total_timesteps"]),
        "agent": model_cfg["agent"],
        "agent_kwargs": deepcopy(model_cfg["agent_kwargs"]),
        "seed": int(seed),
    }

    if device:
        merged["agent_kwargs"]["device"] = device

    return merged
