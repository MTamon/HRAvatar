#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# demo 1 — End-to-end HRAvatar training on a single subject's video.
#
# Usage:
#   bash demos/demo_1_train_subject.sh <root> <name> <video> <intrinsics>
#
# Arguments:
#   root         directory that will contain the per-subject folder
#   name         subject name (sub-directory inside <root>)
#   video        input mp4/mov
#   intrinsics   "hdtf" | "insta" | "custom:fx,fy,cx,cy"
#
# Environment overrides:
#   CUDA_VISIBLE_DEVICES  (default 0)
#   FPS                   frame-rate for frame extraction (default 30)
#   RESIZE                square crop size (default 512)
#   EPOCHS                training epochs (default 15)
#   WITH_ALBEDO=1         run IntrinsicAnything for albedo pseudo-GT
#   SKIP_PREPROCESS=1     skip preprocessing (already done)
# -----------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 4 ]]; then
  sed -n '2,20p' "$0"
  exit 2
fi

ROOT=$1; NAME=$2; VIDEO=$3; INTRINSICS=$4
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${EPOCHS:=15}"
: "${SKIP_PREPROCESS:=0}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${ROOT}/${NAME}"
MODEL_DIR="${REPO_ROOT}/outputs/custom/${NAME}"

if [[ "${SKIP_PREPROCESS}" != "1" ]]; then
  bash demos/_preprocess_subject.sh "${ROOT}" "${NAME}" "${VIDEO}" "${INTRINSICS}"
fi

echo "[train] HRAvatar"
python train.py \
    --source_path "${DATA_DIR}" \
    --model_path  "${MODEL_DIR}" \
    --eval --test_set_num 500 --epochs "${EPOCHS}" \
    --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist \
    --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 \
    --position_lr_init 5e-5 --position_lr_final 5e-7

echo "Done. Model + renders in ${MODEL_DIR}"
