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
#   --lambda-shape F     DECA optimize.py shape regularizer weight (default
#                        1e-2 = original HRAvatar fork value). Pass 1.0-5.0
#                        if optimize_vis.jpg shows alien-looking enlarged
#                        head / collapsed face. See doc/deca_patches.md.
#   --lambda-exp F       DECA optimize.py expression regularizer (default 1e-2)
#   --skip-deca-patches  do NOT auto-apply tools/patches/apply_deca_*.py (default off)
#   --jitter-filter      enable One-Euro temporal smoothing of tracker params
#                        at train+render time. Default OFF. See
#                        doc/jitter_filter.md.
#   --jitter-filter-smirk  also smooth SMIRK encoder outputs at render time
#                        (causal). Implies --jitter-filter.
#   --jitter-filter-min-cutoff F  One-Euro min_cutoff Hz (default 1.0)
#   --jitter-filter-beta F        One-Euro beta (default 0.0)
#
# Environment overrides still accepted for backward compatibility:
#   CUDA_VISIBLE_DEVICES  (default 0)
#   FPS                   frame-rate for frame extraction (default 30)
#   RESIZE                square crop size (default 512)
#   EPOCHS                training epochs (default 15)
#   WITH_ALBEDO=1         run IntrinsicAnything for albedo pseudo-GT
#   SKIP_PREPROCESS=1     skip preprocessing (already done)
#   BBOX_VERIFY=1         also write bbox_verify.mp4 + .csv (default off)
#   NO_STABLE_BBOX=1      skip the stable bbox preprocess step (default off)
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  sed -n '2,52p' "$0"
}

ROOT=""
NAME=""
VIDEO=""
INTRINSICS=""
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${EPOCHS:=15}"
: "${SKIP_PREPROCESS:=0}"
# NOTE: default 0 keeps stable bbox **on** to match the docs above and the
# inline help. The previous default of 1 contradicted the help text.
: "${NO_STABLE_BBOX:=0}"
# Forwarded to _preprocess_subject.sh as flags (the env-var interface there
# was retired to avoid residual / typo risk; we still accept the env-var
# inputs here for backward compatibility with existing automation).
: "${FPS:=30}"
: "${RESIZE:=512}"
: "${WITH_ALBEDO:=0}"
: "${BBOX_VERIFY:=0}"
: "${LAMBDA_SHAPE:=}"
: "${LAMBDA_EXP:=}"
: "${SKIP_DECA_PATCHES:=0}"
# Jitter-filter pass-through to train.py / render.py (off by default).
: "${JITTER_FILTER:=0}"
: "${JITTER_FILTER_SMIRK:=0}"
: "${JITTER_FILTER_MIN_CUTOFF:=}"
: "${JITTER_FILTER_BETA:=}"
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
    --lambda-shape)    require_value "$@"; LAMBDA_SHAPE="$2"; shift 2 ;;
    --lambda-exp)      require_value "$@"; LAMBDA_EXP="$2"; shift 2 ;;
    --skip-deca-patches) SKIP_DECA_PATCHES=1; shift ;;
    --jitter-filter)            JITTER_FILTER=1; shift ;;
    --jitter-filter-smirk)      JITTER_FILTER=1; JITTER_FILTER_SMIRK=1; shift ;;
    --jitter-filter-min-cutoff) require_value "$@"; JITTER_FILTER_MIN_CUTOFF="$2"; shift 2 ;;
    --jitter-filter-beta)       require_value "$@"; JITTER_FILTER_BETA="$2"; shift 2 ;;
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
  if [[ -n "${LAMBDA_SHAPE}" ]]; then
    PREPROCESS_FLAGS+=(--lambda-shape "${LAMBDA_SHAPE}")
  fi
  if [[ -n "${LAMBDA_EXP}" ]]; then
    PREPROCESS_FLAGS+=(--lambda-exp "${LAMBDA_EXP}")
  fi
  if [[ "${SKIP_DECA_PATCHES}" == "1" ]]; then
    PREPROCESS_FLAGS+=(--skip-deca-patches)
  fi
  bash demos/_preprocess_subject.sh \
      --sbj-root "${ROOT}" \
      --sbj-name "${NAME}" \
      --video "${VIDEO}" \
      --intrinsics "${INTRINSICS}" \
      "${PREPROCESS_FLAGS[@]}"
fi

echo "[train] HRAvatar"
TRAIN_EXTRA_ARGS=()
if [[ "${JITTER_FILTER}" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--jitter_filter --jitter_filter_fps "${FPS}")
  if [[ "${JITTER_FILTER_SMIRK}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--jitter_filter_smirk)
  fi
  if [[ -n "${JITTER_FILTER_MIN_CUTOFF}" ]]; then
    TRAIN_EXTRA_ARGS+=(--jitter_filter_min_cutoff "${JITTER_FILTER_MIN_CUTOFF}")
  fi
  if [[ -n "${JITTER_FILTER_BETA}" ]]; then
    TRAIN_EXTRA_ARGS+=(--jitter_filter_beta "${JITTER_FILTER_BETA}")
  fi
fi

python train.py \
    --source_path "${DATA_DIR}" \
    --model_path  "${MODEL_DIR}" \
    --eval --test_set_num 500 --epochs "${EPOCHS}" \
    --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist \
    --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 \
    --position_lr_init 5e-5 --position_lr_final 5e-7 \
    ${TRAIN_EXTRA_ARGS[@]+"${TRAIN_EXTRA_ARGS[@]}"}

echo "Done. Model + renders in ${MODEL_DIR}"
