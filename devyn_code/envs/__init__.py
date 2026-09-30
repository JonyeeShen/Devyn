from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Type

from devyn_code.envs.register_humanoidbench import register_humanoid_bench_envs
from devyn_code.envs.register_humenv import register_humenv_envs
from devyn_code.envs.wrappers import (
    FallPenaltyOnDone,
    HumEnvProprioAsObs,
    MuscleNormWrapper,
    MyosuiteRewardInfoWrapper,
    MyosuiteWrapper,
    TerminateOnFall,
)

WRAPPER_REGISTRY: Dict[str, Type] = {
    "MyosuiteWrapper": MyosuiteWrapper,
    "MyosuiteRewardInfoWrapper": MyosuiteRewardInfoWrapper,
    "HumEnvProprioAsObs": HumEnvProprioAsObs,
    "MuscleNormWrapper": MuscleNormWrapper,
    "TerminateOnFall": TerminateOnFall,
    "FallPenaltyOnDone": FallPenaltyOnDone,
}


def _candidate_ostrich_roots(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []

    env_path = os.getenv("DEVYN_CODE_OSTRICH_ROOT") or os.getenv("PAPER_CODE_OSTRICH_ROOT")
    if env_path:
        resolved = Path(os.path.expandvars(os.path.expanduser(env_path)))
        if not resolved.is_absolute():
            resolved = (repo_root / resolved).resolve()
        candidates.append(resolved)

    candidates.extend(
        [
            (repo_root / "FullBody-Model").resolve(),
            (repo_root / "devyn_code" / "FullBody-Model").resolve(),
            (repo_root / "third_party" / "FullBody-Model").resolve(),
            (repo_root / "third_party_src" / "FullBody-Model").resolve(),
        ]
    )
    return candidates


def _register_ostrich_env(repo_root: Path) -> None:
    last_error: Exception | None = None

    for root in _candidate_ostrich_roots(repo_root):
        # Allow pointing directly to msmodel_gym or to its parent folder.
        search_root = root.parent if root.name == "msmodel_gym" else root
        search_root_str = str(search_root)

        if search_root.exists() and search_root_str not in sys.path:
            sys.path.append(search_root_str)

        try:
            import msmodel_gym  # noqa: F401
            return
        except Exception as exc:
            last_error = exc

    message = (
        "Cannot import msmodel_gym for ostrich task. "
        "Set DEVYN_CODE_OSTRICH_ROOT (or PAPER_CODE_OSTRICH_ROOT) to a directory containing msmodel_gym "
        "(or keep FullBody-Model under third_party/)."
    )
    if last_error is not None:
        raise RuntimeError(message) from last_error
    raise RuntimeError(message)


def ensure_task_env_registered(task_name: str, repo_root: Path) -> None:
    task = task_name.lower()

    if task in {"myoleg", "hand"} or task.startswith("myoleg-") or task.startswith("myohand-"):
        from myosuite.utils import gym  # noqa: F401
        return

    if task == "humenv" or task.startswith("humenv-"):
        register_humenv_envs()
        return

    if task == "humanoidbench" or task.startswith("h1-") or task.startswith("g1-"):
        register_humanoid_bench_envs()
        return

    if task == "ostrich":
        _register_ostrich_env(repo_root)
        return

    raise ValueError(f"Unsupported task family: {task_name}")
