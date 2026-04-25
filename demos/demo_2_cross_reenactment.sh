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
#   bash demos/demo_2_cross_reenactment.sh <source_subject_dir> <target_model_dir>
#
# Arguments:
#   source_subject_dir  directory holding the source subject (video.mp4 or
#                       an already preprocessed tree with tracked_params.json).
#   target_model_dir    model_path of a trained HRAvatar (output of demo 1).
#
# Environment overrides:
#   CUDA_VISIBLE_DEVICES  (default 0)
#   FPS / RESIZE          (defaults 30 / 512)
#   INTRINSICS            preset for source fitting (default "hdtf")
#   SKIP_PREPROCESS=1     skip re-running the preprocessing pipeline
#
# --- Choice of feature-extraction backend --------------------------------
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

if [[ $# -lt 2 ]]; then
  sed -n '2,39p' "$0"
  exit 2
fi

SRC=$1
TGT=$2
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${FPS:=30}"
: "${RESIZE:=512}"
: "${INTRINSICS:=hdtf}"
: "${SKIP_PREPROCESS:=0}"

export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SRC_NAME="$(basename "${SRC}")"
SRC_ROOT="$(dirname "${SRC}")"

if [[ "${SKIP_PREPROCESS}" != "1" && ! -f "${SRC}/tracked_params.json" ]]; then
  VIDEO="${SRC}/video.mp4"
  if [[ ! -f "${VIDEO}" ]]; then
    VIDEO=$(find "${SRC}" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' \) | head -n1)
  fi
  if [[ -z "${VIDEO}" ]]; then
    echo "No video found in ${SRC}; supply a pre-processed directory or drop video.mp4."
    exit 2
  fi
  FPS="${FPS}" RESIZE="${RESIZE}" WITH_ALBEDO=0 \
    bash demos/_preprocess_subject.sh \
        "${SRC_ROOT}" "${SRC_NAME}" "${VIDEO}" "${INTRINSICS}"
fi

python render.py \
    --model_path "${TGT}" \
    --skip_train --skip_test \
    --corss_source_paths "${SRC}"

echo
echo "Done. Renders in ${TGT}/test_cross_reenactment/"
