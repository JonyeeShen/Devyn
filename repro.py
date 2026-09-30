#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def get_repo_root() -> Path:
    # Standalone package root: the directory containing this repro.py.
    return Path(__file__).resolve().parent


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_config_path(config_arg: str, repo_root: Path) -> Path:
    raw = Path(config_arg).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        (Path.cwd() / raw).resolve(),
        (repo_root / raw).resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = " | ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Config not found: {raw}. Tried: {tried}")


def run_cmd(command: List[str], cwd: Path, dry_run: bool) -> None:
    print("[repro] $", shlex.join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def ensure_conda_available(conda_executable: str) -> None:
    if shutil.which(conda_executable) is not None:
        return
    raise RuntimeError(
        f"Conda executable '{conda_executable}' not found. "
        "One-click setup/run via repro.py requires conda. "
        "Without conda, create a Python environment manually and run ./run directly."
    )


def list_conda_envs(conda_executable: str) -> List[str]:
    result = subprocess.run(
        [conda_executable, "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    env_paths = payload.get("envs", [])
    return [Path(path).name for path in env_paths]


def list_conda_env_name_to_path(conda_executable: str) -> Dict[str, Path]:
    result = subprocess.run(
        [conda_executable, "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    env_paths = payload.get("envs", [])
    mapping: Dict[str, Path] = {}
    for raw_path in env_paths:
        path = Path(raw_path)
        mapping[path.name] = path
    return mapping


def print_nfs_cleanup_guidance(env_name: str, env_path: Optional[Path]) -> None:
    print(f"[repro] detected possible NFS lock leftovers for env '{env_name}'.")
    if env_path is None:
        print("[repro] could not resolve env path from conda env list.")
        print("[repro] run: conda env list")
        print(f"[repro] then inspect: <env_path for {env_name}>")
    else:
        print(f"[repro] env path: {env_path}")
        print("[repro] check holders: lsof +D <env_path>")
        print("[repro] stop holders, then remove .nfs placeholders:")
        print("[repro]   find <env_path> -name '.nfs*' -delete")
        print("[repro] retry remove: conda remove -y -n <env_name> --all")

    print("[repro] common holders: python/jupyter/ipykernel/VS Code terminals on this env")
    print("[repro] if file handles remain busy across sessions, reboot can release them")


def remove_conda_env(
    *,
    conda_executable: str,
    env_name: str,
    env_path: Optional[Path],
    cwd: Path,
    dry_run: bool,
) -> None:
    command = [conda_executable, "remove", "-y", "-n", env_name, "--all"]
    if dry_run:
        run_cmd(command, cwd=cwd, dry_run=True)
        return

    print("[repro] $", shlex.join(command))
    result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    nfs_hint = ".nfs" in result.stderr or "Could not remove or rename" in result.stderr
    if nfs_hint:
        print_nfs_cleanup_guidance(env_name, env_path)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )

    if env_path is not None and env_path.exists():
        # If path still exists after successful remove, print guidance for locked leftovers.
        try:
            has_nfs_leftovers = any(env_path.rglob(".nfs*"))
        except OSError:
            has_nfs_leftovers = True
        if has_nfs_leftovers:
            print_nfs_cleanup_guidance(env_name, env_path)


def normalize_task(task: str, task_aliases: Dict[str, str]) -> str:
    return task_aliases.get(task.strip().lower(), task.strip().lower())


def setup_env(
    *,
    repo_root: Path,
    conda_executable: str,
    env_name: str,
    env_cfg: dict,
    recreate: bool,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    existing_envs = set()
    if not dry_run:
        existing_envs = set(list_conda_envs(conda_executable))

    if recreate and (dry_run or env_name in existing_envs):
        run_cmd(
            [conda_executable, "remove", "-y", "-n", env_name, "--all"],
            cwd=repo_root,
            dry_run=dry_run,
        )
        existing_envs.discard(env_name)

    if skip_existing and not recreate and env_name in existing_envs:
        print(f"[repro] skip existing env: {env_name}")
    else:
        create_cmd = [
            conda_executable,
            "create",
            "-y",
            "-n",
            env_name,
            f"python={env_cfg.get('python', '3.10')}",
        ]
        create_cmd.extend(env_cfg.get("conda_packages", []))
        run_cmd(create_cmd, cwd=repo_root, dry_run=dry_run)

    run_cmd(
        [conda_executable, "run", "--no-capture-output", "-n", env_name, "python", "-m", "pip", "install", "--upgrade", "pip"],
        cwd=repo_root,
        dry_run=dry_run,
    )

    pip_install_steps = env_cfg.get("pip_install_steps", [])
    if pip_install_steps:
        for index, step in enumerate(pip_install_steps, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"pip_install_steps[{index}] must be an object")

            packages = step.get("packages", [])
            if not packages:
                continue

            pip_cmd = [conda_executable, "run", "--no-capture-output", "-n", env_name, "python", "-m", "pip", "install"]
            if step.get("index_url"):
                pip_cmd.extend(["--index-url", step["index_url"]])
            if step.get("extra_index_url"):
                pip_cmd.extend(["--extra-index-url", step["extra_index_url"]])
            pip_cmd.extend(step.get("pip_args", []))
            pip_cmd.extend(packages)

            run_cmd(
                pip_cmd,
                cwd=repo_root,
                dry_run=dry_run,
            )
    else:
        pip_packages = env_cfg.get("pip_packages", [])
        if pip_packages:
            run_cmd(
                [conda_executable, "run", "--no-capture-output", "-n", env_name, "python", "-m", "pip", "install", *pip_packages],
                cwd=repo_root,
                dry_run=dry_run,
            )

    if env_cfg.get("editable_install_repo", True):
        run_cmd(
            [conda_executable, "run", "--no-capture-output", "-n", env_name, "python", "-m", "pip", "install", "-e", str(repo_root)],
            cwd=repo_root,
            dry_run=dry_run,
        )

    for rel_path in env_cfg.get("local_editable_paths", []):
        abs_path = (repo_root / rel_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Local editable path does not exist: {abs_path}")
        run_cmd(
            [conda_executable, "run", "--no-capture-output", "-n", env_name, "python", "-m", "pip", "install", "-e", str(abs_path)],
            cwd=repo_root,
            dry_run=dry_run,
        )

    # Optional local editable packages: install only when path exists.
    for rel_path in env_cfg.get("local_editable_paths_optional", []):
        abs_path = (repo_root / rel_path).resolve()
        if not abs_path.exists():
            print(f"[repro] skip optional local editable path (missing): {abs_path}")
            continue
        run_cmd(
            [conda_executable, "run", "--no-capture-output", "-n", env_name, "python", "-m", "pip", "install", "-e", str(abs_path)],
            cwd=repo_root,
            dry_run=dry_run,
        )


def cmd_setup_envs(args: argparse.Namespace) -> None:
    repo_root = get_repo_root()
    config_path = resolve_config_path(args.config, repo_root)
    cfg = load_config(config_path)

    conda_executable = cfg.get("conda_executable", "conda")
    ensure_conda_available(conda_executable)
    all_envs = cfg.get("envs", {})

    if args.env:
        target_envs = args.env
    else:
        target_envs = sorted(all_envs.keys())

    for env_name in target_envs:
        if env_name not in all_envs:
            valid = ", ".join(sorted(all_envs.keys()))
            raise ValueError(f"Unknown env '{env_name}'. Valid envs: {valid}")
        print(f"[repro] setup env: {env_name}")
        setup_env(
            repo_root=repo_root,
            conda_executable=conda_executable,
            env_name=env_name,
            env_cfg=all_envs[env_name],
            recreate=args.recreate,
            skip_existing=(not args.no_skip_existing),
            dry_run=args.dry_run,
        )


def cmd_remove_envs(args: argparse.Namespace) -> None:
    repo_root = get_repo_root()
    config_path = resolve_config_path(args.config, repo_root)
    cfg = load_config(config_path)

    conda_executable = cfg.get("conda_executable", "conda")
    ensure_conda_available(conda_executable)
    all_envs = cfg.get("envs", {})

    if args.env:
        target_envs = args.env
    else:
        target_envs = sorted(all_envs.keys())

    existing_envs = set()
    env_name_to_path: Dict[str, Path] = {}
    if not args.dry_run:
        existing_envs = set(list_conda_envs(conda_executable))
        env_name_to_path = list_conda_env_name_to_path(conda_executable)

    for env_name in target_envs:
        if env_name not in all_envs:
            valid = ", ".join(sorted(all_envs.keys()))
            raise ValueError(f"Unknown env '{env_name}'. Valid envs: {valid}")

        if (not args.dry_run) and env_name not in existing_envs:
            print(f"[repro] skip missing env: {env_name}")
            continue

        remove_conda_env(
            conda_executable=conda_executable,
            env_name=env_name,
            env_path=env_name_to_path.get(env_name),
            cwd=repo_root,
            dry_run=args.dry_run,
        )


def build_run_command(args: argparse.Namespace, cfg: dict, repo_root: Path) -> List[str]:
    task_aliases = cfg.get("task_aliases", {})
    task_to_env = cfg.get("task_to_env", {})
    task = normalize_task(args.task, task_aliases)

    if task not in task_to_env:
        valid_tasks = ", ".join(sorted(task_to_env.keys()))
        raise ValueError(f"Unsupported task '{args.task}'. Supported task families: {valid_tasks}")

    conda_executable = cfg.get("conda_executable", "conda")
    env_name = task_to_env[task]
    run_entry = (repo_root / cfg.get("run_entry", "run")).resolve()

    command = [
        conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        str(run_entry),
        "-task",
        task,
        "-model",
        args.model,
        "--seed",
        str(args.seed),
    ]

    if args.device:
        command.extend(["--device", args.device])
    if args.output_root:
        command.extend(["--output-root", args.output_root])
    if args.total_timesteps is not None:
        command.extend(["--total-timesteps", str(args.total_timesteps)])

    print(f"[repro] task={task} -> env={env_name}")
    return command


def cmd_run(args: argparse.Namespace) -> None:
    repo_root = get_repo_root()
    config_path = resolve_config_path(args.config, repo_root)
    cfg = load_config(config_path)
    ensure_conda_available(cfg.get("conda_executable", "conda"))

    command = build_run_command(args, cfg, repo_root)
    run_cmd(command, cwd=repo_root, dry_run=args.dry_run)


def cmd_list(args: argparse.Namespace) -> None:
    repo_root = get_repo_root()
    config_path = resolve_config_path(args.config, repo_root)
    cfg = load_config(config_path)

    print("[repro] suite:", cfg.get("suite_name", "devyn"))
    print("[repro] envs:")
    for env_name, env_cfg in sorted(cfg.get("envs", {}).items()):
        print(f"  - {env_name} (python={env_cfg.get('python', 'unknown')})")

    print("[repro] task routing:")
    for task, env_name in sorted(cfg.get("task_to_env", {}).items()):
        print(f"  - {task} -> {env_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified reproducibility utility: setup conda envs and run devyn_code by task routing."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_setup = subparsers.add_parser("setup-envs", help="Create/install all configured conda envs")
    parser_setup.add_argument(
        "--config",
        type=str,
        default="repro_config.json",
        help="Path to reproducibility config JSON",
    )
    parser_setup.add_argument(
        "--env",
        action="append",
        default=[],
        help="Specific env name to setup (repeatable)",
    )
    parser_setup.add_argument("--recreate", action="store_true", help="Remove and recreate target envs")
    parser_setup.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing envs (force install steps)",
    )
    parser_setup.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser_setup.set_defaults(func=cmd_setup_envs)

    parser_remove = subparsers.add_parser("remove-envs", help="Remove configured conda envs")
    parser_remove.add_argument(
        "--config",
        type=str,
        default="repro_config.json",
        help="Path to reproducibility config JSON",
    )
    parser_remove.add_argument(
        "--env",
        action="append",
        default=[],
        help="Specific env name to remove (repeatable)",
    )
    parser_remove.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser_remove.set_defaults(func=cmd_remove_envs)

    parser_run = subparsers.add_parser("run", help="Run devyn_code with automatic env routing")
    parser_run.add_argument("-task", required=True, help="Task family or alias")
    parser_run.add_argument("-model", required=True, help="Model name")
    parser_run.add_argument("--seed", type=int, default=0, help="Random seed")
    parser_run.add_argument("--device", type=str, default=None, help="Optional device override")
    parser_run.add_argument("--output-root", type=str, default=None, help="Optional output root override")
    parser_run.add_argument("--total-timesteps", type=int, default=None, help="Optional total timesteps override")
    parser_run.add_argument(
        "--config",
        type=str,
        default="repro_config.json",
        help="Path to reproducibility config JSON",
    )
    parser_run.add_argument("--dry-run", action="store_true", help="Print command only")
    parser_run.set_defaults(func=cmd_run)

    parser_list = subparsers.add_parser("list", help="List env and task routing summary")
    parser_list.add_argument(
        "--config",
        type=str,
        default="repro_config.json",
        help="Path to reproducibility config JSON",
    )
    parser_list.set_defaults(func=cmd_list)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[repro] error: {exc}", file=sys.stderr)
        raise
