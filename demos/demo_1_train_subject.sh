#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# demo 1 — End-to-end HRAvatar training on a single subject's video.
#
# Usage:
#   bash demos/demo_1_train_subject.sh \
#       --sbj-root <root> \
#       --sbj-name <name> \
#       --video <video> \
#       --intrinsics <intrinsics>
#
# Legacy positional form is also accepted:
#   bash demos/demo_1_train_subject.sh <root> <name> <video> <intrinsics>
#
# Arguments:
#   --sbj-root PATH  directory that will contain the per-subject folder
#   --sbj-name NAME  subject name (sub-directory inside <root>)
#   --video PATH         input mp4/mov
#   --intrinsics VALUE   "hdtf" | "insta" | "custom:fx,fy,cx,cy"
#
# Optional flags:
#   --fps N              frame-rate for frame extraction (default 30)
#   --resize N           square crop size (default 512)
#   --epochs N           training epochs (default 15)
#   --with-albedo        run IntrinsicAnything for albedo pseudo-GT
#   --skip-preprocess    skip preprocessing (already done)
#   --bbox-verify        also write bbox_verify.mp4 + .csv (default off)
#   --no-stable-bbox     skip the stable bbox preprocess step (default on)
#
# Environment overrides still accepted for backward compatibility:
#   CUDA_VISIBLE_DEVICES  (default 0)
#   FPS                   frame-rate for frame extraction (default 30)
#   RESIZE                square crop size (default 512)
#   EPOCHS                training epochs (default 15)
#   WITH_ALBEDO=1         run IntrinsicAnything for albedo pseudo-GT
#   SKIP_PREPROCESS=1     skip preprocessing (already done)
#   BBOX_VERIFY=1         also write bbox_verify.mp4 + .csv (default off)
#   NO_STABLE_BBOX=1       skip the stable bbox preprocess step (default on)
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  sed -n '2,32p' "$0"
}

ROOT=""
NAME=""
VIDEO=""
INTRINSICS=""
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${EPOCHS:=15}"
: "${SKIP_PREPROCESS:=0}"
: "${NO_STABLE_BBOX:=1}"
# Forwarded to _preprocess_subject.sh as flags (the env-var interface there
# was retired to avoid residual / typo risk; we still accept the env-var
# inputs here for backward compatibility with existing automation).
: "${FPS:=30}"
: "${RESIZE:=512}"
: "${WITH_ALBEDO:=0}"
: "${BBOX_VERIFY:=0}"
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
    --sbj-root)    require_value "$@"; ROOT="$2"; shift 2 ;;
    --sbj-name)    require_value "$@"; NAME="$2"; shift 2 ;;
    --video)           require_value "$@"; VIDEO="$2"; shift 2 ;;
    --intrinsics)      require_value "$@"; INTRINSICS="$2"; shift 2 ;;
    --fps)             require_value "$@"; FPS="$2"; shift 2 ;;
    --resize)          require_value "$@"; RESIZE="$2"; shift 2 ;;
    --epochs)          require_value "$@"; EPOCHS="$2"; shift 2 ;;
    --with-albedo|--with_albedo) WITH_ALBEDO=1; shift ;;
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --bbox-verify)     BBOX_VERIFY=1; shift ;;
    --no-stable-bbox)  NO_STABLE_BBOX=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    --*) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  if [[ ${#POSITIONAL[@]} -ne 4 ]]; then
    echo "expected exactly 4 positional arguments: <root> <name> <video> <intrinsics>" >&2
    usage
    exit 2
  fi
  [[ -z "${ROOT}" ]] && ROOT="${POSITIONAL[0]}"
  [[ -z "${NAME}" ]] && NAME="${POSITIONAL[1]}"
  [[ -z "${VIDEO}" ]] && VIDEO="${POSITIONAL[2]}"
  [[ -z "${INTRINSICS}" ]] && INTRINSICS="${POSITIONAL[3]}"
fi

if [[ -z "${ROOT}" || -z "${NAME}" || -z "${VIDEO}" || -z "${INTRINSICS}" ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${ROOT}/${NAME}"
MODEL_DIR="${REPO_ROOT}/outputs/custom/${NAME}"

if [[ "${SKIP_PREPROCESS}" != "1" ]]; then
  PREPROCESS_FLAGS=(--fps "${FPS}" --resize "${RESIZE}")
  if [[ "${WITH_ALBEDO}" == "1" ]]; then
    PREPROCESS_FLAGS+=(--with-albedo)
  fi
  if [[ "${BBOX_VERIFY}" == "1" ]]; then
    PREPROCESS_FLAGS+=(--bbox-verify)
  fi
  if [[ "${NO_STABLE_BBOX}" == "1" ]]; then
    PREPROCESS_FLAGS+=(--no-stable-bbox)
  fi
  bash demos/_preprocess_subject.sh \
      --sbj-root "${ROOT}" \
      --sbj-name "${NAME}" \
      --video "${VIDEO}" \
      --intrinsics "${INTRINSICS}" \
      "${PREPROCESS_FLAGS[@]}"
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
