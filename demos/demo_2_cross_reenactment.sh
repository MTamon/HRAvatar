#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# demo 2 — Offline cross-reenactment.
#
# Extract FLAME parameters from a SOURCE subject's video and render them
# with an already-trained TARGET HRAvatar.  The avatar's identity (shape,
# albedo, environment) stays intact; only the expression / pose / jaw
# trajectory comes from the source video.
#
# Usage:
#   bash demos/demo_2_cross_reenactment.sh <root> <name> <video> <intrinsics> <target_model_dir>
#
# Arguments:
#   root         directory that will contain the per-source-subject folder
#   name         source subject name (sub-directory inside <root>)
#   video        source input mp4/mov
#   intrinsics   "hdtf" | "insta" | "custom:fx,fy,cx,cy"
#   target_model_dir  model_path of a trained HRAvatar (output of demo 1).
#
# Environment overrides:
#   CUDA_VISIBLE_DEVICES  (default 0)
#   FPS / RESIZE          (defaults 30 / 512)
#   SKIP_PREPROCESS=1     skip re-running the preprocessing pipeline
#
# --- Feature-extraction backend note --------------------------------
# HRAvatar's own `preprocess/` pipeline uses **DECA** for offline per-frame
# FLAME parameters followed by `optimize.py` for a photometric refinement.
# That is the canonical extractor both in the upstream README and in
# MTamon/DECA@cuda128-HRAvatare's `extract_video_params.py`, so we follow
# exactly that route here.
#
# **SMIRK is NOT used as an offline extractor in the reference code.**
# In HRAvatar, SMIRK's SMIRK_em1.pt is loaded as an *online* expression
# encoder via `--with_param_net_smirk` during HRAvatar training and never
# exports a standalone `tracked_params.json`.  If you need SMIRK-style
# offline extraction, use the companion repo
# https://github.com/MTamon/smirk/tree/release/cuda128 (its
# `demos/demo_save_flame.py`), convert the resulting `.pt` into the
# tracked_params.json schema that `scene/data_loader.py` expects, and
# re-run this script with `SKIP_PREPROCESS=1`.
# -----------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 5 ]]; then
  sed -n '2,40p' "$0"
  exit 2
fi

ROOT=$1; NAME=$2; VIDEO=$3; INTRINSICS=$4; TGT=$5
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${FPS:=30}"
: "${RESIZE:=512}"
: "${SKIP_PREPROCESS:=0}"

export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${ROOT}/${NAME}"
TRACKED_PARAMS="${DATA_DIR}/tracked_params.json"
TRACKED_PARAMS_V2="${DATA_DIR}/tracked_params_v2.json"

if [[ "${SKIP_PREPROCESS}" != "1" && ! -f "${TRACKED_PARAMS}" && ! -f "${TRACKED_PARAMS_V2}" ]]; then
  if [[ ! -f "${VIDEO}" ]]; then
    echo "No video found at ${VIDEO}; provide a source video or use SKIP_PREPROCESS=1 with tracked params in ${DATA_DIR}."
    exit 2
  fi
  FPS="${FPS}" RESIZE="${RESIZE}" WITH_ALBEDO=0 \
    bash demos/_preprocess_subject.sh \
        "${ROOT}" "${NAME}" "${VIDEO}" "${INTRINSICS}"
elif [[ ! -f "${TRACKED_PARAMS}" ]]; then
  if [[ -f "${TRACKED_PARAMS_V2}" ]]; then
    echo "[preprocess] skipping; found ${TRACKED_PARAMS_V2}"
  else
    echo "No tracked params found in ${DATA_DIR}; cannot render with SKIP_PREPROCESS=1."
    exit 2
  fi
else
  echo "[preprocess] skipping; found ${TRACKED_PARAMS}"
fi

python render.py \
    --model_path "${TGT}" \
    --skip_train --skip_test \
    --corss_source_paths "${DATA_DIR}"

echo
echo "Done. Renders in ${TGT}/test_cross_reenactment/"
