#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Internal helper: run HRAvatar's preprocessing pipeline on a subject.
#
# Usage:
#   bash demos/_preprocess_subject.sh <root> <name> <video> <intrinsics>
#
# Arguments + environment: same as demo_1_train_subject.sh (but no train.py).
# -----------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <root> <name> <video> <intrinsics>"
  exit 2
fi

ROOT=$1; NAME=$2; VIDEO=$3; INTRINSICS=$4
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${FPS:=30}"
: "${RESIZE:=512}"
: "${WITH_ALBEDO:=0}"
export CUDA_VISIBLE_DEVICES

case "${INTRINSICS}" in
  hdtf)        FX=1539.67462; FY=1508.93280; CX=261.442628; CY=253.231895 ;;
  insta)       FX=1536.00;    FY=1536.00;    CX=256.00;     CY=256.00     ;;
  custom:*)    IFS=',' read -r FX FY CX CY <<<"${INTRINSICS#custom:}"    ;;
  *) echo "unknown intrinsics preset: ${INTRINSICS}"; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${ROOT}/${NAME}"
DECA_DIR="${REPO_ROOT}/preprocess/submodules/DECA"
mkdir -p "${DATA_DIR}"
if [[ ! -f "${DATA_DIR}/${NAME}.mp4" ]]; then
  ln -sf "$(realpath "${VIDEO}")" "${DATA_DIR}/${NAME}.mp4"
fi

echo "[preprocess 1/5] crop + matting"
python preprocess/crop_and_matting.py \
    --source "${ROOT}" --name "${NAME}" --fps "${FPS}" \
    --image_size "${RESIZE}" "${RESIZE}" \
    --matting --crop_image --mask_clothes True

echo "[preprocess 2/5] DECA initial FLAME"
( cd "${DECA_DIR}" && \
  python demos/demo_reconstruct.py \
      -i "${DATA_DIR}/image" \
      --savefolder "${DATA_DIR}/deca" \
      --saveCode True --saveVis False --sample_step 1 --render_orig False )

echo "[preprocess 3/5] face-alignment landmarks"
python preprocess/keypoint_detector.py --path "${DATA_DIR}"

echo "[preprocess 4/5] iris segmentation"
python preprocess/iris.py --path "${DATA_DIR}"

echo "[preprocess 5/5] optimize FLAME parameters"
( cd "${DECA_DIR}" && \
  python optimize.py --path "${DATA_DIR}" \
      --cx "${CX}" --cy "${CY}" --fx "${FX}" --fy "${FY}" --size "${RESIZE}" \
      --n_shape 100 --n_expr 100 --with_translation )

if [[ "${WITH_ALBEDO}" == "1" ]]; then
  echo "[preprocess opt] IntrinsicAnything pseudo-albedo"
  python preprocess/submodules/IntrinsicAnything/inference.py \
      --input_dir  "${DATA_DIR}/image" \
      --model_dir  assets/intrinsic_anything/albedo \
      --output_dir "${DATA_DIR}/albedo" \
      --ddim 100 --batch_size 10 --image_interval 3
fi

echo "Preprocessing done: ${DATA_DIR}/tracked_params.json"
