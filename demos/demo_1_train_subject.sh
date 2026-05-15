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
#   --resize N           ffmpeg prescale short-side in px (default 720).
#                        Sets the short-side resolution of frames written
#                        before the outer crop. A higher value gives SMIRK /
#                        DECA more face pixels at the bbox / tracking stage.
#                        Independent of --image-size.
#   --image-size N       outer-crop output square size in px (default 512).
#                        The bbox stage produces an image of this size that
#                        the downstream camera intrinsics preset is pinned
#                        to. Do NOT change unless the intrinsics preset is
#                        rebuilt for the new resolution.
#   --epochs N           training epochs (default 15)
#   --with-albedo        run IntrinsicAnything for albedo pseudo-GT
#   --skip-preprocess    skip preprocessing (already done)
#   --bbox-verify        also write bbox_verify.mp4 + .csv (default off)
#   --no-stable-bbox     skip the stable bbox preprocess step (default on)
#   --no-prescale        skip ffmpeg short-side rescaling (default on);
#                        disables the auto-normalisation that puts the
#                        input video at short_side=--resize before bbox.
#   --bbox-scale F       INNER stable_bbox scale (default 1.6).
#                        SMIRK/DECA 224-crop margin around the
#                        K-of-N hysteresis center anchor.
#   --outer-bbox-scale F OUTER fixed-crop bbox scale factor (default 2.2).
#                        Inflation factor on the video-wide face union
#                        bbox that sets the per-video fixed outer crop.
#                        Independent of --bbox-scale.
#   --bbox-center-deadzone-px F  K-of-N deadzone in px (default 4.0)
#   --bbox-center-window N       K-of-N window length (default 5)
#   --bbox-center-k-of-n N       K-of-N threshold (default 3)
#   --bbox-center-tau F          Center follower tau in seconds (default 0.25)
#   --bbox-center-passthrough    bypass FIR + hysteresis on center
#   --lambda-shape F     DECA optimize.py shape regularizer weight (default
#                        1e-2 = original HRAvatar fork value). Pass 1.0-5.0
#                        if optimize_vis.jpg shows alien-looking enlarged
#                        head / collapsed face. See doc/deca_patches.md.
#   --lambda-exp F       DECA optimize.py expression regularizer (default 1e-2)
#   --lambda-pose-diff F DECA optimize.py per-frame pose temporal smoothing
#                        weight (default 10). Raise to 30-50 to suppress
#                        per-frame pose jitter that can show up as a
#                        translucent ghost moving with the head.
#   --max-iters N        DECA optimize.py main-loop iter cap (default 1000 =
#                        original HRAvatar fork). Lower to short-circuit the
#                        slow tail of the per-frame photometric refinement.
#                        Requires apply_deca_optimize_iters.py.
#   --max-iris-iters N   DECA optimize.py iris-loop iter cap (default 500)
#   --early-stop-rel-tol F  Plateau tolerance on landmark_loss in DECA optimize
#                           (default 0.0 = disabled). Sampled every 100 iter.
#                           0.005-0.01 is typical when opting in.
#   --early-stop-patience N Consecutive 100-iter windows without rel_tol
#                           improvement before breaking (default 2)
#   --main-lr F          DECA optimize.py main-loop initial Adam lr (default
#                        1e-2 = original HRAvatar fork). Affects pose/exp/
#                        shape only; eyelid/translation lrs untouched.
#                        Requires apply_deca_optimize_lr.py.
#   --main-lr-decay-step N  Apply lr decay every N iter of the main loop
#                           (default 0 = off). Multiplicative on every
#                           param group (per-group ratios preserved).
#   --main-lr-decay-factor F  Multiplier per decay step (default 0.5)
#   --skip-deca-patches  do NOT auto-apply tools/patches/apply_deca_*.py (default off)
#   --jitter-filter      enable One-Euro temporal smoothing of tracker params
#                        at train+render time. Default OFF. See
#                        doc/jitter_filter.md.
#   --jitter-filter-smirk  also smooth SMIRK encoder outputs at render time
#                        (causal). Implies --jitter-filter.
#   --jitter-filter-min-cutoff F  One-Euro min_cutoff Hz (default 1.0)
#   --jitter-filter-beta F        One-Euro beta (default 0.0)
#   --masked-loss        train with foreground-weighted L1+SSIM (off by default)
#   --mask-bg-weight F   bg/fg loss weight ratio when --masked-loss is set
#                        (1.0 = vanilla, 0.1 = default, 0.0 = mask-only)
#   --lambda-image F     scale image_loss vs other loss terms (default 1.0)
#
# Environment overrides still accepted for backward compatibility:
#   CUDA_VISIBLE_DEVICES  (default 0)
#   FPS                   frame-rate for frame extraction (default 30)
#   RESIZE                ffmpeg prescale short-side in px (default 720)
#   IMAGE_SIZE            outer-crop output square size in px (default 512)
#   EPOCHS                training epochs (default 15)
#   WITH_ALBEDO=1         run IntrinsicAnything for albedo pseudo-GT
#   SKIP_PREPROCESS=1     skip preprocessing (already done)
#   BBOX_VERIFY=1         also write bbox_verify.mp4 + .csv (default off)
#   NO_STABLE_BBOX=1      skip the stable bbox preprocess step (default off)
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  sed -n '2,67p' "$0"
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
: "${RESIZE:=720}"
: "${IMAGE_SIZE:=512}"
: "${WITH_ALBEDO:=0}"
: "${BBOX_VERIFY:=0}"
: "${NO_PRESCALE:=0}"
: "${BBOX_SCALE:=}"
: "${OUTER_BBOX_SCALE:=}"
: "${BBOX_CENTER_DEADZONE_PX:=}"
: "${BBOX_CENTER_WINDOW:=}"
: "${BBOX_CENTER_K_OF_N:=}"
: "${BBOX_CENTER_TAU:=}"
: "${BBOX_CENTER_PASSTHROUGH:=0}"
: "${LAMBDA_SHAPE:=}"
: "${LAMBDA_EXP:=}"
: "${LAMBDA_POSE_DIFF:=}"
: "${MAX_ITERS:=}"
: "${MAX_IRIS_ITERS:=}"
: "${EARLY_STOP_REL_TOL:=}"
: "${EARLY_STOP_PATIENCE:=}"
: "${MAIN_LR:=}"
: "${MAIN_LR_DECAY_STEP:=}"
: "${MAIN_LR_DECAY_FACTOR:=}"
: "${SKIP_DECA_PATCHES:=0}"
# Jitter-filter pass-through to train.py / render.py (off by default).
: "${JITTER_FILTER:=0}"
: "${JITTER_FILTER_SMIRK:=0}"
: "${JITTER_FILTER_MIN_CUTOFF:=}"
: "${JITTER_FILTER_BETA:=}"
: "${MASKED_LOSS:=0}"
: "${MASK_BG_WEIGHT:=}"
: "${LAMBDA_IMAGE:=}"
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
    --image-size|--image_size) require_value "$@"; IMAGE_SIZE="$2"; shift 2 ;;
    --epochs)          require_value "$@"; EPOCHS="$2"; shift 2 ;;
    --with-albedo|--with_albedo) WITH_ALBEDO=1; shift ;;
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --bbox-verify)     BBOX_VERIFY=1; shift ;;
    --no-stable-bbox)  NO_STABLE_BBOX=1; shift ;;
    --no-prescale)     NO_PRESCALE=1; shift ;;
    --bbox-scale)               require_value "$@"; BBOX_SCALE="$2"; shift 2 ;;
    --outer-bbox-scale)         require_value "$@"; OUTER_BBOX_SCALE="$2"; shift 2 ;;
    --bbox-center-deadzone-px)  require_value "$@"; BBOX_CENTER_DEADZONE_PX="$2"; shift 2 ;;
    --bbox-center-window)       require_value "$@"; BBOX_CENTER_WINDOW="$2"; shift 2 ;;
    --bbox-center-k-of-n)       require_value "$@"; BBOX_CENTER_K_OF_N="$2"; shift 2 ;;
    --bbox-center-tau)          require_value "$@"; BBOX_CENTER_TAU="$2"; shift 2 ;;
    --bbox-center-passthrough)  BBOX_CENTER_PASSTHROUGH=1; shift ;;
    --lambda-shape)        require_value "$@"; LAMBDA_SHAPE="$2"; shift 2 ;;
    --lambda-exp)          require_value "$@"; LAMBDA_EXP="$2"; shift 2 ;;
    --lambda-pose-diff)    require_value "$@"; LAMBDA_POSE_DIFF="$2"; shift 2 ;;
    --max-iters)             require_value "$@"; MAX_ITERS="$2"; shift 2 ;;
    --max-iris-iters)        require_value "$@"; MAX_IRIS_ITERS="$2"; shift 2 ;;
    --early-stop-rel-tol)    require_value "$@"; EARLY_STOP_REL_TOL="$2"; shift 2 ;;
    --early-stop-patience)   require_value "$@"; EARLY_STOP_PATIENCE="$2"; shift 2 ;;
    --main-lr)               require_value "$@"; MAIN_LR="$2"; shift 2 ;;
    --main-lr-decay-step)    require_value "$@"; MAIN_LR_DECAY_STEP="$2"; shift 2 ;;
    --main-lr-decay-factor)  require_value "$@"; MAIN_LR_DECAY_FACTOR="$2"; shift 2 ;;
    --skip-deca-patches) SKIP_DECA_PATCHES=1; shift ;;
    --jitter-filter)            JITTER_FILTER=1; shift ;;
    --jitter-filter-smirk)      JITTER_FILTER=1; JITTER_FILTER_SMIRK=1; shift ;;
    --jitter-filter-min-cutoff) require_value "$@"; JITTER_FILTER_MIN_CUTOFF="$2"; shift 2 ;;
    --jitter-filter-beta)       require_value "$@"; JITTER_FILTER_BETA="$2"; shift 2 ;;
    --masked-loss|--masked_loss) MASKED_LOSS=1; shift ;;
    --mask-bg-weight|--mask_bg_weight) require_value "$@"; MASK_BG_WEIGHT="$2"; shift 2 ;;
    --lambda-image|--lambda_image)     require_value "$@"; LAMBDA_IMAGE="$2"; shift 2 ;;
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
  PREPROCESS_FLAGS=(--fps "${FPS}" --resize "${RESIZE}" --image-size "${IMAGE_SIZE}")
  if [[ "${WITH_ALBEDO}" == "1" ]]; then
    PREPROCESS_FLAGS+=(--with-albedo)
  fi
  if [[ "${BBOX_VERIFY}" == "1" ]]; then
    PREPROCESS_FLAGS+=(--bbox-verify)
  fi
  if [[ "${NO_STABLE_BBOX}" == "1" ]]; then
    PREPROCESS_FLAGS+=(--no-stable-bbox)
  fi
  if [[ "${NO_PRESCALE}" == "1" ]]; then
    PREPROCESS_FLAGS+=(--no-prescale)
  fi
  if [[ -n "${BBOX_SCALE}" ]]; then
    PREPROCESS_FLAGS+=(--bbox-scale "${BBOX_SCALE}")
  fi
  if [[ -n "${OUTER_BBOX_SCALE}" ]]; then
    PREPROCESS_FLAGS+=(--outer-bbox-scale "${OUTER_BBOX_SCALE}")
  fi
  if [[ -n "${BBOX_CENTER_DEADZONE_PX}" ]]; then
    PREPROCESS_FLAGS+=(--bbox-center-deadzone-px "${BBOX_CENTER_DEADZONE_PX}")
  fi
  if [[ -n "${BBOX_CENTER_WINDOW}" ]]; then
    PREPROCESS_FLAGS+=(--bbox-center-window "${BBOX_CENTER_WINDOW}")
  fi
  if [[ -n "${BBOX_CENTER_K_OF_N}" ]]; then
    PREPROCESS_FLAGS+=(--bbox-center-k-of-n "${BBOX_CENTER_K_OF_N}")
  fi
  if [[ -n "${BBOX_CENTER_TAU}" ]]; then
    PREPROCESS_FLAGS+=(--bbox-center-tau "${BBOX_CENTER_TAU}")
  fi
  if [[ "${BBOX_CENTER_PASSTHROUGH}" == "1" ]]; then
    PREPROCESS_FLAGS+=(--bbox-center-passthrough)
  fi
  if [[ -n "${LAMBDA_SHAPE}" ]]; then
    PREPROCESS_FLAGS+=(--lambda-shape "${LAMBDA_SHAPE}")
  fi
  if [[ -n "${LAMBDA_EXP}" ]]; then
    PREPROCESS_FLAGS+=(--lambda-exp "${LAMBDA_EXP}")
  fi
  if [[ -n "${LAMBDA_POSE_DIFF}" ]]; then
    PREPROCESS_FLAGS+=(--lambda-pose-diff "${LAMBDA_POSE_DIFF}")
  fi
  if [[ -n "${MAX_ITERS}" ]]; then
    PREPROCESS_FLAGS+=(--max-iters "${MAX_ITERS}")
  fi
  if [[ -n "${MAX_IRIS_ITERS}" ]]; then
    PREPROCESS_FLAGS+=(--max-iris-iters "${MAX_IRIS_ITERS}")
  fi
  if [[ -n "${EARLY_STOP_REL_TOL}" ]]; then
    PREPROCESS_FLAGS+=(--early-stop-rel-tol "${EARLY_STOP_REL_TOL}")
  fi
  if [[ -n "${EARLY_STOP_PATIENCE}" ]]; then
    PREPROCESS_FLAGS+=(--early-stop-patience "${EARLY_STOP_PATIENCE}")
  fi
  if [[ -n "${MAIN_LR}" ]]; then
    PREPROCESS_FLAGS+=(--main-lr "${MAIN_LR}")
  fi
  if [[ -n "${MAIN_LR_DECAY_STEP}" ]]; then
    PREPROCESS_FLAGS+=(--main-lr-decay-step "${MAIN_LR_DECAY_STEP}")
  fi
  if [[ -n "${MAIN_LR_DECAY_FACTOR}" ]]; then
    PREPROCESS_FLAGS+=(--main-lr-decay-factor "${MAIN_LR_DECAY_FACTOR}")
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
if [[ "${MASKED_LOSS}" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--masked_loss)
fi
if [[ -n "${MASK_BG_WEIGHT}" ]]; then
  TRAIN_EXTRA_ARGS+=(--mask_bg_weight "${MASK_BG_WEIGHT}")
fi
if [[ -n "${LAMBDA_IMAGE}" ]]; then
  TRAIN_EXTRA_ARGS+=(--lambda_image "${LAMBDA_IMAGE}")
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
