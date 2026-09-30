# Devyn: High-Dimensional Robotic Reinforcement Learning with Developing Synergies

This repository contains the code used to reproduce the results from the paper:

> **High-Dimensional Robotic Reinforcement Learning with Developing Synergies**
> *Paper: coming soon (link will be added once the preprint is available).*

Compact training entry for all paper models with one unified configuration center.

## What This Provides

- A single reproducibility utility: `./repro.py`
- Automatic conda-environment routing per task
- One-command setup and one-command cleanup of all required environments

## Supported Task Keys

- `myoleg-stair`
- `myohand-reorient`
- `humenv-jump`
- `h1-run`
- `g1-run`
- `ostrich`

## Supported Model Keys

- `devyn`
- `devyn+q`
- `devyn+r`
- `devyn+r+q`
- `devyn+q+r`
- `sac`

## Environment Suite

The project uses three explicit conda environments:

- `devyn_myo` for MyoSuite and Ostrich
- `devyn_smpl` for HumEnv (SMPL Humanoid)
- `devyn_humanoidbench` for humanoid-bench (`Unitree h1, g1`)

All routing and install specifications are in [repro_config.json](repro_config.json).

This package includes a vendored [third_party/humanoid-bench](third_party/humanoid-bench) copy and installs it locally for `h1-run`/`g1-run`. See [third_party/README.md](third_party/README.md) for provenance/license details of all vendored code and assets.

## Quick Start (Recommended)

Run from [devyn_code](.) (this directory).

`repro.py` one-click workflow requires Conda.

1. One-command environment setup

```bash
python ./repro.py setup-envs
```

2. Run training (environment selected automatically)

```bash
python ./repro.py run -task myoleg-stair -model devyn --seed 0 --config ./repro_config.json
```

By default, runtime sets `MUJOCO_GL=egl` (only when `MUJOCO_GL` is not already defined).

3. One-command environment cleanup after experiments

```bash
python ./repro.py remove-envs
```

## Useful Variants

Preview commands without executing:

```bash
python ./repro.py run -task humenv-jump -model devyn --seed 0 --dry-run --config ./repro_config.json
```

Setup or delete only one environment:

```bash
python ./repro.py setup-envs --env devyn_smpl
python ./repro.py remove-envs --env devyn_smpl
```

List current routing:

```bash
python ./repro.py list --config ./repro_config.json
```

## No Conda (Minimal venv Script)

If Conda is unavailable, use the in-repo script:

```bash
bash ./setup_venv_min.sh --task myoleg-stair
```

This creates a task-family-specific venv and installs the minimum required dependencies.

Examples:

```bash
# MyoSuite / Ostrich tasks
bash ./setup_venv_min.sh --task myoleg-stair

# HumEnv task
bash ./setup_venv_min.sh --task humenv-jump

# Humanoid-bench tasks (uses local ./third_party/humanoid-bench)
bash ./setup_venv_min.sh --task g1-run
```

Run with the created venv:

```bash
./.venv-myo/bin/python ./run -task myoleg-stair -model devyn --seed 0
./.venv-smpl/bin/python ./run -task humenv-jump -model devyn --seed 0
./.venv-humanoidbench/bin/python ./run -task g1-run -model devyn+q --seed 0
```

Optional (HumEnv with CUDA 12.1 wheels):

```bash
bash ./setup_venv_min.sh --task humenv-jump --torch-cu121
```

## Direct Run (Already In Correct Environment)

```bash
python ./run -task myoleg-stair -model devyn --seed 0
```

Without Conda, this direct run path still works after manual dependency installation in your own Python environment.

## Troubleshooting

- If `g1-run` or `h1-run` fails with `ModuleNotFoundError: humanoid_bench.dmc_deps`:

```bash
conda run -n devyn_humanoidbench python -m pip uninstall -y humanoid-bench
conda run -n devyn_humanoidbench python -m pip install -e ./third_party/humanoid-bench
```

`repro_config.json` now installs `./third_party/humanoid-bench` by default for `devyn_humanoidbench`.

## Project Layout Notes

- Task/model configuration center: [devyn_code/config.py](devyn_code/config.py)
- Main algorithm implementation: [devyn_code/algorithms/devyn_unified_sac.py](devyn_code/algorithms/devyn_unified_sac.py)
- Environment bootstrap/wrappers: [devyn_code/envs](devyn_code/envs)
- Vendored/paper-asset benchmarks (HumanoidBench, Ostrich model): [third_party](third_party)
- Outputs: [devyn_code/output](devyn_code/output)/`<model>/<task>/<seed>/` (override with `--output-root`)

## Third-Party Code

See [third_party/README.md](third_party/README.md) for the provenance and license of each
vendored dependency (notably [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench), MIT-licensed).

## License

This project's own code is released under the [MIT License](LICENSE). Vendored
third-party code under [third_party](third_party) retains its original license.

## Citation

A citation entry (BibTeX) will be added once the paper preprint is available.
