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
#
# Preconditions:
# - System CUDA Toolkit 12.8 is installed (nvcc must be on PATH, or CUDA_HOME set).
# - gcc-11 / g++-11 are installed (Ubuntu 22.04: `sudo apt install gcc-11 g++-11`).
# - Git submodules are initialized:
#     git submodule update --init --recursive
# -----------------------------------------------------------------------------

# This script mirrors MTamon/DECA128/install_128.sh as closely as possible to
# keep library versions aligned. The `--no-deps` flag is used everywhere to
# prevent pip from mutating the pin set via transitive resolution.

# After pinned deps, it additionally:
#   1. installs chumpy from GitHub (numpy 2.x compatible main branch),
#   2. source-builds nvdiffrast from NVlabs HEAD,
#   3. source-builds pytorch3d v0.7.8 against torch 2.9.1 + CUDA 12.8,
#   4. installs taming-transformers and CLIP from GitHub,
#   5. builds the two local CUDA extensions under submodules/.

set -eo pipefail


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


# ----------------------------------------------------------------------------
# Toolchain setup for CUDA extensions.
# ----------------------------------------------------------------------------
# PyTorch 2.9 + CUDA 12.8 needs gcc <= 13; gcc-11 is the DECA128-tested choice.
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"

# Point CUDA_HOME at the system CUDA 12.8 install (Ubuntu standard path).
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
  else
    echo "[setup.sh] WARNING: CUDA_HOME is not set and /usr/local/cuda-12.8 was not found."
    echo "[setup.sh] Set CUDA_HOME manually to your CUDA 12.8 install path before rerunning."
    exit 1
  fi
fi
export PATH="${CUDA_HOME}/bin:${PATH}"

# Force nvcc to emit code for common modern arches; narrow to your own GPU
# to speed up the build (e.g. TORCH_CUDA_ARCH_LIST="12.0" for RTX 5090).
# Turing 7.5, Ampere 8.0/8.6, Ada 8.9, Hopper 9.0, Blackwell 12.0.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;12.0}"

# Force pytorch3d / submodule setups to build with CUDA support.
export FORCE_CUDA=1

echo "[setup.sh] CC=${CC} CXX=${CXX}"
echo "[setup.sh] CUDA_HOME=${CUDA_HOME}"
echo "[setup.sh] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
nvcc --version || { echo "[setup.sh] nvcc not found on PATH"; exit 1; }


if [[ ${PIP_ONLY} -eq 0 ]]; then
  echo "[1/6] Creating conda env HRAvatar (Python 3.11, CUDA 12.8 toolkit)"

  if conda env list | awk '{print $1}' | grep -qx "HRAvatar"; then
    echo " → conda env 'HRAvatar' already exists, skipping creation."
  else
    conda env create --file environment.yml
  fi

  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate HRAvatar
else
  echo "[1/6] Using currently-active python (pip-only mode)"
fi


# ----------------------------------------------------------------------------
# 1. Upgrade pip and install pinned dependencies (DECA128-aligned).
# ----------------------------------------------------------------------------
echo "[2/6] Upgrading pip to 25.2 (matches MTamon install_128.sh)"
python -m pip install --upgrade pip==25.2


# chumpy (git main — numpy 2.x friendly; DECA128 uses the exact same source).
# The PyPI wheel breaks under numpy 2.x; use the maintained git fork.
python -m pip install --no-deps pillow==12.0.0
python -m pip install git+https://github.com/mattloper/chumpy.git

# Pinned deps. Installed with --no-deps to avoid pip rewriting the pin set.
python -m pip install --no-deps Cython==0.29.35
python -m pip install --no-deps face-alignment==1.4.1
python -m pip install --no-deps face-detection-tflite==0.6.0
python -m pip install --no-deps filelock==3.20.0
python -m pip install --no-deps fsspec==2025.10.0
python -m pip install --no-deps fvcore==0.1.5.post20221221
python -m pip install --no-deps ImageIO==2.37.2
python -m pip install --no-deps imageio-ffmpeg==0.6.0
python -m pip install --no-deps iopath==0.1.10
python -m pip install --no-deps kornia==0.8.2
python -m pip install --no-deps kornia_rs==0.1.10
python -m pip install --no-deps llvmlite==0.45.1
python -m pip install --no-deps loguru==0.7.3
python -m pip install --no-deps mediapipe==0.10.14
python -m pip install --no-deps ninja==1.13.0
python -m pip install --no-deps numba==0.62.1
python -m pip install --no-deps numpy==2.2.6
python -m pip install --no-deps nvidia-cublas-cu12==12.8.4.1
python -m pip install --no-deps nvidia-cuda-cupti-cu12==12.8.90
python -m pip install --no-deps nvidia-cuda-nvrtc-cu12==12.8.93
python -m pip install --no-deps nvidia-cuda-runtime-cu12==12.8.90
python -m pip install --no-deps nvidia-cudnn-cu12==9.10.2.21
python -m pip install --no-deps nvidia-cufft-cu12==11.3.3.83
python -m pip install --no-deps nvidia-cufile-cu12==1.13.1.3
python -m pip install --no-deps nvidia-curand-cu12==10.3.9.90
python -m pip install --no-deps nvidia-cusolver-cu12==11.7.3.90
python -m pip install --no-deps nvidia-cusparse-cu12==12.5.8.93
python -m pip install --no-deps nvidia-cusparselt-cu12==0.7.1
python -m pip install --no-deps nvidia-nccl-cu12==2.27.5
python -m pip install --no-deps nvidia-nvjitlink-cu12==12.8.93
python -m pip install --no-deps nvidia-nvshmem-cu12==3.3.20
python -m pip install --no-deps nvidia-nvtx-cu12==12.8.90
python -m pip install --no-deps opencv-python==4.12.0.88
python -m pip install --no-deps PyYAML==6.0.3
python -m pip install --no-deps scikit-image==0.25.2
python -m pip install --no-deps scikit-learn==1.7.2
python -m pip install --no-deps scipy==1.16.3
python -m pip install --no-deps slicerator==1.1.0
python -m pip install --no-deps tifffile==2025.10.16
python -m pip install --no-deps torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps tqdm==4.67.1
python -m pip install --no-deps triton==3.5.1
python -m pip install --no-deps typing_extensions==4.15.0
python -m pip install --no-deps yacs==0.1.8

# JAX / MediaPipe stack (pinned to SMIRK-approved versions).
python -m pip install --no-deps jax==0.4.30
python -m pip install --no-deps jaxlib==0.4.30
python -m pip install --no-deps ml_dtypes==0.4.1
python -m pip install --no-deps opt_einsum==3.4.0
python -m pip install --no-deps absl-py==2.1.0
python -m pip install --no-deps attrs==24.2.0
python -m pip install --no-deps flatbuffers==24.3.25
python -m pip install --no-deps protobuf==4.25.5

# Lightning / HF stack.
python -m pip install --no-deps timm==0.9.16
python -m pip install --no-deps pytorch_lightning==2.5.2
python -m pip install --no-deps torchmetrics==1.6.0
python -m pip install --no-deps lightning-utilities==0.11.9
python -m pip install --no-deps transformers==4.57.1
python -m pip install --no-deps huggingface_hub==0.27.1
python -m pip install --no-deps safetensors==0.4.5
python -m pip install --no-deps diffusers==0.30.3

# Config / utilities.
python -m pip install --no-deps omegaconf==2.3.0
python -m pip install --no-deps antlr4-python3-runtime==4.9.3
python -m pip install --no-deps einops==0.8.1
python -m pip install --no-deps natsort==8.4.0
python -m pip install --no-deps future==1.0.0
python -m pip install --no-deps ipdb==0.13.13
python -m pip install --no-deps tensorboard==2.17.1
python -m pip install --no-deps av==12.3.0
python -m pip install --no-deps pims==0.7
python -m pip install --no-deps packaging==25.0

# Use in smirk for the `--no-deps` test
python -m pip install --no-deps accelerate==1.11.0
python -m pip install --no-deps aiohappyeyeballs==2.6.1
python -m pip install --no-deps aiohttp==3.13.2
python -m pip install --no-deps aiosignal==1.4.0
python -m pip install --no-deps albucore==0.0.19
python -m pip install --no-deps albumentations==1.4.18
python -m pip install --no-deps annotated-types==0.7.0
python -m pip install --no-deps anyio==4.11.0
python -m pip install --no-deps audioread==3.1.0
python -m pip install --no-deps beautifulsoup4==4.12.3
python -m pip install --no-deps blobfile==3.1.0
python -m pip install --no-deps certifi==2024.8.30
python -m pip install --no-deps cffi==2.0.0
python -m pip install --no-deps charset-normalizer==3.4.0
python -m pip install --no-deps click==8.3.0
python -m pip install --no-deps cmpfilter==0.0.1
python -m pip install --no-deps contourpy==1.3.3
python -m pip install --no-deps cycler==0.12.1
python -m pip install --no-deps datasets==4.4.1
python -m pip install --no-deps decorator==5.2.1
python -m pip install --no-deps dfcon==0.3.0
python -m pip install --no-deps dill==0.4.0
python -m pip install --no-deps dtaidistance==2.3.12
python -m pip install --no-deps easy-video==0.0.4
python -m pip install --no-deps eval_type_backport==0.2.2
python -m pip install --no-deps evaluate==0.4.6
python -m pip install --no-deps flash_attn==2.8.3
python -m pip install --no-deps fonttools==4.60.1
python -m pip install --no-deps frozenlist==1.8.0
python -m pip install --no-deps gdown==5.2.0
python -m pip install --no-deps gitdb==4.0.12
python -m pip install --no-deps gitpython==3.1.45
python -m pip install --no-deps graphviz==0.21
python -m pip install --no-deps h11==0.16.0
python -m pip install --no-deps hf-xet==1.2.0
python -m pip install --no-deps httpcore==1.0.9
python -m pip install --no-deps httpx==0.28.1
python -m pip install --no-deps hydra-core==1.3.2
python -m pip install --no-deps idna==3.10
python -m pip install --no-deps jinja2==3.1.6
python -m pip install --no-deps jiwer==4.0.0
python -m pip install --no-deps joblib==1.5.2
python -m pip install --no-deps kagglehub==0.3.13
python -m pip install --no-deps kiwisolver==1.4.9
python -m pip install --no-deps lazy_loader==0.4
python -m pip install --no-deps librosa==0.11.0
python -m pip install --no-deps lpips==0.1.4
python -m pip install --no-deps lxml==6.0.2
python -m pip install --no-deps markupsafe==3.0.3
python -m pip install --no-deps matplotlib==3.10.7
python -m pip install --no-deps moviepy==2.2.1
python -m pip install --no-deps mpmath==1.3.0
python -m pip install --no-deps msgpack==1.1.2
python -m pip install --no-deps multidict==6.7.0
python -m pip install --no-deps multiprocess==0.70.18
python -m pip install --no-deps networkx==3.5
python -m pip install --no-deps opencv-contrib-python==4.11.0.86
python -m pip install --no-deps pandas==2.3.3
python -m pip install --no-deps platformdirs==4.5.0
python -m pip install --no-deps pooch==1.8.2
python -m pip install --no-deps portalocker==3.2.0
python -m pip install --no-deps proglog==0.1.12
python -m pip install --no-deps propcache==0.4.1
python -m pip install --no-deps psutil==7.1.3
python -m pip install --no-deps pyarrow==22.0.0
python -m pip install --no-deps pycparser==2.23
python -m pip install --no-deps pycryptodomex==3.23.0
python -m pip install --no-deps pydantic==2.9.2
python -m pip install --no-deps pydantic_core==2.23.4
python -m pip install --no-deps pyparsing==3.2.5
python -m pip install --no-deps python-dateutil==2.9.0.post0
python -m pip install --no-deps python-dotenv==1.2.1
python -m pip install --no-deps pytz==2025.2
python -m pip install --no-deps pyworld==0.3.5
python -m pip install --no-deps rapidfuzz==3.14.3
python -m pip install --no-deps regex==2025.11.3
python -m pip install --no-deps requests==2.32.3
python -m pip install --no-deps schedulefree==1.4.1
python -m pip install --no-deps sentencepiece==0.2.0
python -m pip install --no-deps sentry-sdk==2.43.0
python -m pip install --no-deps shellingham==1.5.4
python -m pip install --no-deps smmap==5.0.2
python -m pip install --no-deps sniffio==1.3.1
python -m pip install --no-deps sounddevice==0.5.1
python -m pip install --no-deps soundfile==0.13.1
python -m pip install --no-deps soupsieve==2.6
python -m pip install --no-deps soxr==1.0.0
python -m pip install --no-deps stringzilla==3.11.3
python -m pip install --no-deps sympy==1.14.0
python -m pip install --no-deps tabulate==0.9.0
python -m pip install --no-deps termcolor==3.2.0
python -m pip install --no-deps threadpoolctl==3.6.0
python -m pip install --no-deps tiktoken==0.12.0
python -m pip install --no-deps tokenizers==0.22.1
python -m pip install --no-deps toolpack==0.0.5
python -m pip install --no-deps typer-slim==0.20.0
python -m pip install --no-deps typing-inspection==0.4.2
python -m pip install --no-deps tzdata==2025.2
python -m pip install --no-deps urllib3==2.2.3
python -m pip install --no-deps wandb==0.22.3
python -m pip install --no-deps xxhash==3.6.0
python -m pip install --no-deps yarl==1.22.0
python -m pip install --no-deps zipp==3.23.0

# HRAvatar-only additions (NOT in the MTamon references).
# pyshtools — SH rotation helper in utils/sh_utils.py (filter_envmap path).
python -m pip install --no-deps pyshtools==4.13.1
# plyfile — used by scene/gaussian_model.py save/load.
python -m pip install --no-deps plyfile==1.1.2
# roma — rotation utilities in preprocess.
python -m pip install --no-deps roma==1.5.1
# carvekit-colab — background removal for preprocess/crop_and_matting.py.
python -m pip install --no-deps carvekit-colab==4.1.2
# torchfile — reserved for checkpoint compatibility.
python -m pip install --no-deps torchfile==0.1.0


# ----------------------------------------------------------------------------
# 2. nvdiffrast (source build from NVlabs HEAD).
# ----------------------------------------------------------------------------
# Used by net_modules/NVDIFFREC/*.
# Must be built with --no-build-isolation so it can find the already-installed torch.
NVDIFFRAST_TMP="$(mktemp -d)"
git -C "${NVDIFFRAST_TMP}" init -q
git -C "${NVDIFFRAST_TMP}" remote add origin https://github.com/NVlabs/nvdiffrast.git
git -C "${NVDIFFRAST_TMP}" fetch --depth 1 origin main
git -C "${NVDIFFRAST_TMP}" checkout FETCH_HEAD
python -m pip install --no-build-isolation --no-deps "${NVDIFFRAST_TMP}"
rm -rf "${NVDIFFRAST_TMP}"


# ----------------------------------------------------------------------------
# 3. pytorch3d v0.7.8 (source build against torch 2.9.1 + CUDA 12.8).
# ----------------------------------------------------------------------------
# HRAvatar imports `pytorch3d.ops` in scene/gaussian_head_model.py and
# `pytorch3d` in utils/loss_utils.py.python -m pip install --no-deps No wheel exists for cu128 yet.
# NOTE: this step compiles a large CUDA extension and can take 10+ minutes.
# Shallow clone + --branch doesn't reliably resolve tags, so we use
# git-init + fetch-by-tag which works regardless of server advertisement.
PYTORCH3D_TMP="$(mktemp -d)"
git -C "${PYTORCH3D_TMP}" init -q
git -C "${PYTORCH3D_TMP}" remote add origin https://github.com/facebookresearch/pytorch3d.git
git -C "${PYTORCH3D_TMP}" fetch --depth 1 origin tag V0.7.8
git -C "${PYTORCH3D_TMP}" checkout FETCH_HEAD
python -m pip install --no-deps "${PYTORCH3D_TMP}"
rm -rf "${PYTORCH3D_TMP}"


# ----------------------------------------------------------------------------
# 4. taming-transformers + CLIP (IntrinsicAnything preprocess deps).
# ----------------------------------------------------------------------------
python -m pip install --no-deps git+https://github.com/CompVis/taming-transformers.git
python -m pip install --no-deps git+https://github.com/openai/CLIP.git


# ----------------------------------------------------------------------------
# 5. Local CUDA extensions (diff-gaussian-rasterization_c10 and simple-knn).
# ----------------------------------------------------------------------------
# The submodules MUST be populated before running this script:
#   git submodule update --init --recursive

# simple-knn needs a newer commit than the submodule pointer (the original
# camenduru commit uses torch's deprecated .data API, which breaks on
# PyTorch 2.x / CUDA 12.x). Force-check out the latest main which contains
# the fix (.data<T>() → .data_ptr<T>()).
echo "[5/6] Building local CUDA extensions (diff-gaussian-rasterization_c10, simple-knn)"
if [ -d submodules/simple-knn/.git ] || [ -f submodules/simple-knn/.git ]; then
  cd submodules/simple-knn
  git fetch origin main
  git checkout origin/main
  cd "${SCRIPT_DIR}"
else
  echo "[setup.sh] submodules/simple-knn not initialized."
  echo "[setup.sh] Run: git submodule update --init --recursive"
  exit 1
fi

python -m pip install --no-deps ./submodules/diff-gaussian-rasterization_c10
python -m pip install --no-deps ./submodules/simple-knn


# ----------------------------------------------------------------------------
# 6. Sanity check.
# ----------------------------------------------------------------------------
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
