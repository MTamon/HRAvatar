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
#   bash demos/demo_2_cross_reenactment.sh \
#       --sbj-root <root> \
#       --sbj-name <name> \
#       --video <video> \
#       --intrinsics <intrinsics> \
#       --target-model-dir <target_model_dir>
#
# Legacy positional form is also accepted:
#   bash demos/demo_2_cross_reenactment.sh <root> <name> <video> <intrinsics> <target_model_dir>
#
# Arguments:
#   --sbj-root PATH      directory that will contain the per-source-subject folder
#   --sbj-name NAME      source subject name (sub-directory inside <root>)
#   --video PATH             source input mp4/mov
#   --intrinsics VALUE       "hdtf" | "insta" | "custom:fx,fy,cx,cy"
#   --target-model-dir PATH  model_path of a trained HRAvatar (output of demo 1).
#
# Optional flags:
#   --fps N                  frame-rate for frame extraction (default 30)
#   --resize N               square crop size (default 512)
#   --skip-preprocess        skip re-running the preprocessing pipeline
#
# Environment overrides still accepted for backward compatibility:
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

usage() {
  sed -n '2,44p' "$0"
}

ROOT=""
NAME=""
VIDEO=""
INTRINSICS=""
TGT=""
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${FPS:=30}"
: "${RESIZE:=512}"
: "${SKIP_PREPROCESS:=0}"

export CUDA_VISIBLE_DEVICES

POSITIONAL=()

require_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "missing value for $1" >&2
    usage
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbj-root)     require_value "$@"; ROOT="$2"; shift 2 ;;
    --sbj-name)     require_value "$@"; NAME="$2"; shift 2 ;;
    --video)            require_value "$@"; VIDEO="$2"; shift 2 ;;
    --intrinsics)       require_value "$@"; INTRINSICS="$2"; shift 2 ;;
    --target-model-dir) require_value "$@"; TGT="$2"; shift 2 ;;
    --fps)              require_value "$@"; FPS="$2"; shift 2 ;;
    --resize)           require_value "$@"; RESIZE="$2"; shift 2 ;;
    --skip-preprocess)  SKIP_PREPROCESS=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    --*) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  if [[ ${#POSITIONAL[@]} -ne 5 ]]; then
    echo "expected exactly 5 positional arguments: <root> <name> <video> <intrinsics> <target_model_dir>" >&2
    usage
    exit 2
  fi
  [[ -z "${ROOT}" ]] && ROOT="${POSITIONAL[0]}"
  [[ -z "${NAME}" ]] && NAME="${POSITIONAL[1]}"
  [[ -z "${VIDEO}" ]] && VIDEO="${POSITIONAL[2]}"
  [[ -z "${INTRINSICS}" ]] && INTRINSICS="${POSITIONAL[3]}"
  [[ -z "${TGT}" ]] && TGT="${POSITIONAL[4]}"
fi

if [[ -z "${ROOT}" || -z "${NAME}" || -z "${VIDEO}" || -z "${INTRINSICS}" || -z "${TGT}" ]]; then
  usage
  exit 2
fi

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
  bash demos/_preprocess_subject.sh \
      --sbj-root "${ROOT}" \
      --sbj-name "${NAME}" \
      --video "${VIDEO}" \
      --intrinsics "${INTRINSICS}" \
      --fps "${FPS}" --resize "${RESIZE}"
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
