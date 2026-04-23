#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# HRAvatar — deterministic installer for Python 3.11 + PyTorch 2.9.1 + CUDA 12.8
# -----------------------------------------------------------------------------
# This mirrors the `install_128.sh` pattern used on the companion branches:
#   https://github.com/MTamon/DECA/tree/release/cuda128
#   https://github.com/MTamon/smirk/tree/release/cuda128
#   https://github.com/MTamon/FLARE
#
# Usage:
#   bash setup.sh              # create conda env HRAvatar, then install everything
#   bash setup.sh --pip-only   # skip conda, install into the active python
#   bash setup.sh --no-assets  # skip asset-download step
# -----------------------------------------------------------------------------
set -euo pipefail

PIP_ONLY=0
NO_ASSETS=0
for arg in "$@"; do
  case "$arg" in
    --pip-only)   PIP_ONLY=1 ;;
    --no-assets)  NO_ASSETS=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ${PIP_ONLY} -eq 0 ]]; then
  echo "[1/6] Creating conda env HRAvatar (Python 3.11, CUDA 12.8 toolkit)"
  conda env create --file environment.yml
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate HRAvatar
else
  echo "[1/6] Using currently-active python (pip-only mode)"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
# Arch list covers Turing ... Blackwell (incl. RTX 5090 = sm_120).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;12.0}"
export FORCE_CUDA=1

echo "[2/6] Upgrading pip to 25.2 (matches MTamon install_128.sh)"
python -m pip install --upgrade "pip==25.2"

echo "[3/6] Installing torch 2.9.1 + torchvision 0.24.1 (cu128 wheels)"
python -m pip install --no-deps \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps triton==3.5.1

echo "[4/6] Installing the --no-deps pin set from requirements.txt"
# --no-deps mirrors the MTamon install_128.sh style so transitive
# resolution cannot perturb the pins.  We skip the torch lines here
# because they were installed above with the cu128 index.
python -m pip install --no-deps -r requirements.txt

echo "[5/6] Building local CUDA extensions (diff-gaussian-rasterization_c10, simple-knn)"
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization_c10
python -m pip install --no-build-isolation ./submodules/simple-knn

echo "[6/6] Sanity check"
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "tv", torchvision.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
import pytorch3d, nvdiffrast, diff_gaussian_rasterization_c10, simple_knn
import chumpy, kornia, mediapipe, face_alignment
print("OK")
PY

if [[ ${NO_ASSETS} -eq 0 ]]; then
  echo "[opt] Downloading checkpoints + assets (see download_assets.sh)"
  bash download_assets.sh || echo "download_assets.sh failed or missing — skip and run manually."
fi

echo "Done."
