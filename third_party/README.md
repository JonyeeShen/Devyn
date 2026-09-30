# Third-Party Code

This folder groups external/vendored code and simulation assets that the training
code in [`devyn_code`](../devyn_code) depends on at runtime. Keeping them here (instead
of scattered at the repo root) keeps the top-level layout focused on the paper's own
source code.

## `humanoid-bench/`

Vendored copy of the official [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench)
benchmark (Sferrazza et al., *"HumanoidBench: Simulated Humanoid Benchmark for Whole-Body
Locomotion and Manipulation"*, [arXiv:2403.10506](https://arxiv.org/abs/2403.10506)),
used here to provide the `h1-run` / `g1-run` tasks (Unitree H1/G1).

- License: MIT (see [`humanoid-bench/LICENSE`](humanoid-bench/LICENSE)).
- No functional modifications were made to the upstream package; it is installed
  in editable mode via `pip install -e third_party/humanoid-bench` (see
  [`repro_config.json`](../repro_config.json) / [`setup_venv_min.sh`](../setup_venv_min.sh)).
- All credit for this benchmark goes to the original authors.

## `ostrich/`

The `msmodel_gym` Gymnasium environment (`ostrich/msmodel_gym`) and the MuJoCo
musculoskeletal Ostrich model/assets (`ostrich/assets`) used for the `ostrich` task are
part of this project's own code/assets. They live under `third_party/` purely to keep
the repo root uncluttered, not because they are externally sourced.

The two subfolders must remain siblings: `msmodel_gym` resolves the MJCF path relative
to its own file location (`../assets/ostrich.xml`), so moving one without the other
will break the `ostrich` task.
