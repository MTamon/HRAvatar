#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# HRAvatar — third-party asset downloader (Python 3.11 + PyTorch 2.9.1 + CUDA 12.8).
# -----------------------------------------------------------------------------
# Pattern follows MTamon/smirk@release/cuda128::prepare_demos.sh — every asset
# that can be downloaded non-interactively IS downloaded non-interactively:
#
#   A. FLAME 2020 generic_model.pkl    -- credential-gated POST to
#                                         download.is.tue.mpg.de; the script
#                                         prompts for username/password.
#      destinations:  assets/flame_model/flame2020.pkl
#                     preprocess/submodules/DECA/data/generic_model2020.pkl
#                     assets/FLAME2020/generic_model.pkl   (kept for tools that
#                                                           read the unzipped
#                                                           layout directly)
#
#   B. DECA deca_model.tar             -- Google Drive, pulled via `gdown --fuzzy`.
#      destination:   preprocess/submodules/DECA/data/deca_model.tar
#      source:        https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje
#
#   C. SMIRK SMIRK_em1.pt              -- Google Drive, `gdown --fuzzy`.
#      destination:   assets/smirk/pretrained_models/SMIRK_em1.pt
#      source:        https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE
#
#   D. face-parsing 79999_iter.pth     -- Google Drive, `gdown --fuzzy`.
#      destination:   preprocess/submodules/face-parsing.PyTorch/res/cp/79999_iter.pth
#      source:        https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812
#
#   E. RobustVideoMatting rvm_resnet50.pth   -- public GitHub release.
#      destination:   preprocess/submodules/RobustVideoMatting/rvm_resnet50.pth
#      source:        https://github.com/PeterL1n/RobustVideoMatting/releases/...
#
#   F. MediaPipe face_landmarker.task  -- public Google Storage.  Already
#      shipped inside the repo under assets/smirk/face_landmarker.task, but we
#      re-validate / re-fetch when missing.
#      destination:   assets/smirk/face_landmarker.task
#      source:        https://storage.googleapis.com/mediapipe-models/...
#
#   G. IntrinsicAnything albedo weights -- HuggingFace snapshot_download.
#      Only required for HDTF-style albedo pseudo-GT (train.py
#      --with_intrinsic_supervise); otherwise skip with --no_optional.
#      destination:   assets/intrinsic_anything/albedo/...
#
# Usage:
#   bash download_assets.sh                   # full interactive download
#   bash download_assets.sh --no_flame        # skip FLAME2020 (e.g. demo 3 only)
#   bash download_assets.sh --no_optional     # skip IntrinsicAnything
#   bash download_assets.sh --flame_user U --flame_pass P   # non-interactive
#
# Reference: https://github.com/MTamon/smirk/blob/release/cuda128/prepare_demos.sh
# -----------------------------------------------------------------------------
set -euo pipefail

WITH_FLAME=1
WITH_OPTIONAL=1
FLAME_USER=""
FLAME_PASS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no_flame)    WITH_FLAME=0; shift ;;
    --no_optional) WITH_OPTIONAL=0; shift ;;
    --flame_user)  FLAME_USER="$2"; shift 2 ;;
    --flame_pass)  FLAME_PASS="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "[download_assets.sh] unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

say()  { printf '\n\033[1;36m[download_assets.sh] %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m[download_assets.sh] WARN: %s\033[0m\n' "$*"; }

# URL-encode helper, taken from MTamon/smirk/prepare_demos.sh.
urle () {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

need_bin() {
  command -v "$1" >/dev/null 2>&1 || {
    warn "required binary '$1' is missing on PATH."
    return 1
  }
}

need_bin wget  || exit 2
need_bin unzip || exit 2

mkdir -p \
  assets/flame_model \
  assets/FLAME2020 \
  assets/smirk \
  assets/smirk/pretrained_models \
  assets/intrinsic_anything \
  preprocess/submodules/DECA/data \
  preprocess/submodules/face-parsing.PyTorch/res/cp \
  preprocess/submodules/RobustVideoMatting

# ----------------------------------------------------------------------------
# A. FLAME 2020 (credential-gated)
# ----------------------------------------------------------------------------
FLAME_PKL_REPO="assets/flame_model/flame2020.pkl"
FLAME_PKL_DECA="preprocess/submodules/DECA/data/generic_model2020.pkl"
FLAME_PKL_RAW="assets/FLAME2020/generic_model.pkl"

if [[ ${WITH_FLAME} -eq 1 ]]; then
  if [[ -f "${FLAME_PKL_REPO}" && -f "${FLAME_PKL_DECA}" && -f "${FLAME_PKL_RAW}" ]]; then
    say "FLAME2020 already present, skipping."
  else
    if [[ ! -f "${FLAME_PKL_RAW}" ]]; then
      say "FLAME2020 is credential-gated. Register at https://flame.is.tue.mpg.de/"
      if [[ -z "${FLAME_USER}" ]]; then
        read -r -p 'Username (FLAME): ' FLAME_USER
      fi
      if [[ -z "${FLAME_PASS}" ]]; then
        read -r -s -p 'Password (FLAME): ' FLAME_PASS
        echo
      fi
      FLAME_USER_ENC="$(urle "${FLAME_USER}")"
      FLAME_PASS_ENC="$(urle "${FLAME_PASS}")"

      say "Downloading FLAME2020.zip ..."
      wget --post-data "username=${FLAME_USER_ENC}&password=${FLAME_PASS_ENC}" \
           'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
           -O FLAME2020.zip --no-check-certificate --continue
      unzip -o FLAME2020.zip -d assets/
      rm -f FLAME2020.zip
    fi

    # Propagate generic_model.pkl to the two HRAvatar-internal locations.
    if [[ -f "${FLAME_PKL_RAW}" ]]; then
      cp -f "${FLAME_PKL_RAW}" "${FLAME_PKL_REPO}"
      cp -f "${FLAME_PKL_RAW}" "${FLAME_PKL_DECA}"
      say "FLAME2020 installed."
    else
      warn "FLAME2020 download seems to have failed. Check ${FLAME_PKL_RAW}."
    fi
  fi
else
  say "--no_flame given; skipping FLAME2020."
fi

# ----------------------------------------------------------------------------
# B. DECA deca_model.tar (Google Drive via gdown)
# ----------------------------------------------------------------------------
DECA_TAR="preprocess/submodules/DECA/data/deca_model.tar"
if [[ -f "${DECA_TAR}" ]]; then
  say "deca_model.tar already present, skipping."
else
  if need_bin gdown; then
    say "Downloading deca_model.tar via gdown (Google Drive) ..."
    gdown --fuzzy \
      'https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view' \
      -O "${DECA_TAR}"
  else
    warn "gdown not installed.  pip install gdown (>=5.2), then rerun."
  fi
fi

# ----------------------------------------------------------------------------
# C. SMIRK SMIRK_em1.pt (Google Drive via gdown)
# ----------------------------------------------------------------------------
SMIRK_PT="assets/smirk/pretrained_models/SMIRK_em1.pt"
if [[ -f "${SMIRK_PT}" ]]; then
  say "SMIRK_em1.pt already present, skipping."
else
  if need_bin gdown; then
    say "Downloading SMIRK_em1.pt via gdown (Google Drive) ..."
    gdown --fuzzy \
      'https://drive.google.com/uc?id=1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE' \
      -O "${SMIRK_PT}"
  else
    warn "gdown not installed.  pip install gdown (>=5.2), then rerun."
  fi
fi

# ----------------------------------------------------------------------------
# D. face-parsing 79999_iter.pth (Google Drive via gdown)
# ----------------------------------------------------------------------------
FP_PTH="preprocess/submodules/face-parsing.PyTorch/res/cp/79999_iter.pth"
if [[ -f "${FP_PTH}" ]]; then
  say "face-parsing 79999_iter.pth already present, skipping."
else
  if need_bin gdown; then
    say "Downloading 79999_iter.pth via gdown (Google Drive) ..."
    gdown --fuzzy \
      'https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view' \
      -O "${FP_PTH}"
  else
    warn "gdown not installed.  pip install gdown (>=5.2), then rerun."
  fi
fi

# ----------------------------------------------------------------------------
# E. RobustVideoMatting rvm_resnet50.pth (public GitHub release)
# ----------------------------------------------------------------------------
RVM_PTH="preprocess/submodules/RobustVideoMatting/rvm_resnet50.pth"
if [[ -f "${RVM_PTH}" ]]; then
  say "rvm_resnet50.pth already present, skipping."
else
  say "Downloading rvm_resnet50.pth ..."
  wget -q --show-progress --continue \
    -O "${RVM_PTH}" \
    'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth'
fi

# ----------------------------------------------------------------------------
# F. MediaPipe face_landmarker.task (public Google Storage)
# ----------------------------------------------------------------------------
MP_TASK="assets/smirk/face_landmarker.task"
if [[ -f "${MP_TASK}" ]]; then
  say "face_landmarker.task already present, skipping."
else
  say "Downloading face_landmarker.task ..."
  wget -q --show-progress --continue \
    -O "${MP_TASK}" \
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
fi

# ----------------------------------------------------------------------------
# G. IntrinsicAnything albedo weights (optional, HuggingFace)
# ----------------------------------------------------------------------------
IA_CKPT="assets/intrinsic_anything/albedo/checkpoints/last.ckpt"
if [[ ${WITH_OPTIONAL} -eq 1 ]]; then
  if [[ -f "${IA_CKPT}" ]]; then
    say "IntrinsicAnything albedo weights already present, skipping."
  else
    # hf / huggingface-cli のどちらかを選択、なければインストール
    HF_CMD=""
    if command -v hf >/dev/null 2>&1; then
      HF_CMD="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
      HF_CMD="huggingface-cli"
    else
      warn "'hf' / 'huggingface-cli' not found. Installing huggingface_hub ..."
      pip install -q "huggingface_hub[hf_xet]>=1.0"
      export PATH="$HOME/.local/bin:$PATH"
      if command -v hf >/dev/null 2>&1; then
        HF_CMD="hf"
      elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CMD="huggingface-cli"
      fi
    fi

    if [[ -n "${HF_CMD}" ]]; then
      say "Downloading IntrinsicAnything albedo weights via '${HF_CMD}' ..."
      "${HF_CMD}" download LittleFrog/IntrinsicAnything \
        --include "albedo/**" \
        --local-dir assets/intrinsic_anything
    else
      warn "Neither 'hf' nor 'huggingface-cli' is available. Skipping IntrinsicAnything."
    fi
  fi
else
  say "--no_optional given; skipping IntrinsicAnything."
fi

# ----------------------------------------------------------------------------
# H. SMIRK eyelid / landmark .npy/.npz files (HuggingFace Xet — pinned revision)
#    destination: preprocess/submodules/DECA/data/
#    source:      Skywork/SkyReels-A1  extra_models/smirk/
#      l_eyelid.npy                 121 kB
#      r_eyelid.npy                 121 kB
#      mediapipe_landmark_embedding.npz  4.52 kB
# ----------------------------------------------------------------------------
SMIRK_DECA_DIR="preprocess/submodules/DECA/data"
SMIRK_L_EYELID="${SMIRK_DECA_DIR}/l_eyelid.npy"
SMIRK_R_EYELID="${SMIRK_DECA_DIR}/r_eyelid.npy"
SMIRK_MP_NPZ="${SMIRK_DECA_DIR}/mediapipe_landmark_embedding.npz"
SMIRK_NPY_REVISION="e8f62f871898c2323750f26614086e52b6e1ea15"

if [[ -f "${SMIRK_L_EYELID}" && -f "${SMIRK_R_EYELID}" && -f "${SMIRK_MP_NPZ}" ]]; then
  say "SMIRK .npy/.npz files already present, skipping."
else
  HF_CMD_NPY=""
  if command -v hf >/dev/null 2>&1; then
    HF_CMD_NPY="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD_NPY="huggingface-cli"
  else
    warn "'hf' / 'huggingface-cli' not found. Installing huggingface_hub[hf_xet] ..."
    pip install -q "huggingface_hub[hf_xet]>=1.0"
    export PATH="$HOME/.local/bin:$PATH"
    if command -v hf >/dev/null 2>&1; then
      HF_CMD_NPY="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
      HF_CMD_NPY="huggingface-cli"
    fi
  fi

  if [[ -n "${HF_CMD_NPY}" ]]; then
    say "Downloading SMIRK .npy/.npz files via '${HF_CMD_NPY}' (revision ${SMIRK_NPY_REVISION}) ..."
    _TMP_NPY="$(mktemp -d)"
    "${HF_CMD_NPY}" download Skywork/SkyReels-A1 \
      extra_models/smirk/l_eyelid.npy \
      extra_models/smirk/r_eyelid.npy \
      extra_models/smirk/mediapipe_landmark_embedding.npz \
      --revision "${SMIRK_NPY_REVISION}" \
      --local-dir "${_TMP_NPY}"
    mv -f "${_TMP_NPY}/extra_models/smirk/l_eyelid.npy"                  "${SMIRK_L_EYELID}"
    mv -f "${_TMP_NPY}/extra_models/smirk/r_eyelid.npy"                  "${SMIRK_R_EYELID}"
    mv -f "${_TMP_NPY}/extra_models/smirk/mediapipe_landmark_embedding.npz" "${SMIRK_MP_NPZ}"
    rm -rf "${_TMP_NPY}"
    say "SMIRK .npy/.npz files installed."
  else
    warn "huggingface-cli unavailable; falling back to wget resolve URL ..."
    _HF_BASE="https://huggingface.co/Skywork/SkyReels-A1/resolve/${SMIRK_NPY_REVISION}/extra_models/smirk"
    [[ ! -f "${SMIRK_L_EYELID}" ]] && wget -q --show-progress --continue \
      -O "${SMIRK_L_EYELID}" "${_HF_BASE}/l_eyelid.npy"
    [[ ! -f "${SMIRK_R_EYELID}" ]] && wget -q --show-progress --continue \
      -O "${SMIRK_R_EYELID}" "${_HF_BASE}/r_eyelid.npy"
    [[ ! -f "${SMIRK_MP_NPZ}" ]] && wget -q --show-progress --continue \
      -O "${SMIRK_MP_NPZ}" "${_HF_BASE}/mediapipe_landmark_embedding.npz"
  fi
fi

say "Done. Asset summary:"
for f in \
  "${FLAME_PKL_REPO}" \
  "${FLAME_PKL_DECA}" \
  "${DECA_TAR}" \
  "${SMIRK_PT}" \
  "${FP_PTH}" \
  "${RVM_PTH}" \
  "${MP_TASK}" \
  "${IA_CKPT}" \
  "${SMIRK_L_EYELID}" \
  "${SMIRK_R_EYELID}" \
  "${SMIRK_MP_NPZ}"; do
  if [[ -f "${f}" ]]; then
    printf '  %-70s %s\n' "${f}" "$(du -h "${f}" | cut -f1)"
  else
    printf '  %-70s %s\n' "${f}" "MISSING"
  fi
done
