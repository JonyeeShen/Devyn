#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK=""
PROFILE=""
PYTHON_BIN="python3"
VENV_DIR=""
INSTALL_TORCH_CU121=0

print_help() {
  cat <<'EOF'
Minimal no-conda installer for devyn_code (venv version).

Usage:
  bash ./setup_venv_min.sh --task <task-key> [--python python3.10] [--venv .venv-name]
  bash ./setup_venv_min.sh --profile <myo|smpl|humanoidbench> [--python python3.10] [--venv .venv-name]

Options:
  --task <key>            One of: myoleg-stair, myohand-reorient, humenv-jump, h1-run, g1-run, ostrich
  --profile <name>        Dependency profile: myo | smpl | humanoidbench
  --python <bin>          Python executable used to create venv (default: python3)
  --venv <path>           Venv path (default: .venv-<profile>)
  --torch-cu121           Also install torch/torchvision/torchaudio from PyTorch CU121 index (for smpl profile)
  -h, --help              Show this help

Notes:
  - For h1-run/g1-run, this script installs local ./third_party/humanoid-bench in editable mode.
  - Runtime defaults MUJOCO_GL=egl inside ./run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="${2:-}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --torch-cu121)
      INSTALL_TORCH_CU121=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "[setup_venv_min] unknown argument: $1" >&2
      print_help
      exit 2
      ;;
  esac
done

if [[ -n "$TASK" && -n "$PROFILE" ]]; then
  echo "[setup_venv_min] use either --task or --profile, not both" >&2
  exit 2
fi

if [[ -n "$TASK" ]]; then
  case "$TASK" in
    myoleg-stair|myohand-reorient|ostrich)
      PROFILE="myo"
      ;;
    humenv-jump)
      PROFILE="smpl"
      ;;
    h1-run|g1-run)
      PROFILE="humanoidbench"
      ;;
    *)
      echo "[setup_venv_min] unsupported task: $TASK" >&2
      exit 2
      ;;
  esac
fi

if [[ -z "$PROFILE" ]]; then
  echo "[setup_venv_min] please provide --task or --profile" >&2
  print_help
  exit 2
fi

if [[ -z "$VENV_DIR" ]]; then
  VENV_DIR=".venv-${PROFILE}"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup_venv_min] python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

"$PYTHON_BIN" -m venv "$VENV_DIR"
VENV_PY="$VENV_DIR/bin/python"

"$VENV_PY" -m pip install --upgrade pip setuptools wheel

COMMON_PKGS=(
  "stable-baselines3[extra]==2.3.2"
  "sb3-contrib==2.3.0"
  "gymnasium==0.29.1"
  "moviepy==1.0.3"
  "pyyaml==6.0.2"
  "wandb==0.18.5"
)
"$VENV_PY" -m pip install "${COMMON_PKGS[@]}"

case "$PROFILE" in
  myo)
    "$VENV_PY" -m pip install "myosuite==2.8.4" "kmedoids==0.5.2"
    ;;
  smpl)
    "$VENV_PY" -m pip install "git+https://github.com/facebookresearch/HumEnv.git"
    if [[ "$INSTALL_TORCH_CU121" -eq 1 ]]; then
      "$VENV_PY" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
    fi
    ;;
  humanoidbench)
    if [[ ! -d "$SCRIPT_DIR/third_party/humanoid-bench" ]]; then
      echo "[setup_venv_min] missing local folder: $SCRIPT_DIR/third_party/humanoid-bench" >&2
      exit 1
    fi
    "$VENV_PY" -m pip install -e "$SCRIPT_DIR/third_party/humanoid-bench"
    ;;
  *)
    echo "[setup_venv_min] unsupported profile: $PROFILE" >&2
    exit 2
    ;;
esac

cat <<EOF
[setup_venv_min] done
  profile : $PROFILE
  venv    : $SCRIPT_DIR/$VENV_DIR

Run example:
  $SCRIPT_DIR/$VENV_DIR/bin/python ./run -task ${TASK:-myoleg-stair} -model devyn --seed 0
EOF
