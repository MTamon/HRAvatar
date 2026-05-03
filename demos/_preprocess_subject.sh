#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Internal helper: run HRAvatar's preprocessing pipeline on a subject.
#
# All tunables are CLI flags (not env vars) so a single invocation cannot
# silently inherit residual state from the surrounding shell. Run with
# --help for the full list. The only env var honoured is
# CUDA_VISIBLE_DEVICES (CUDA convention).
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash demos/_preprocess_subject.sh \
      --sbj-root <root> \
      --sbj-name <name> \
      --video <video> \
      --intrinsics <intrinsics> \
      [options]

Legacy positional form is also accepted:
  bash demos/_preprocess_subject.sh <root> <name> <video> <intrinsics> [options]

Required arguments:
  --sbj-root PATH    dataset output root (per-subject folder created inside)
  --sbj-name NAME    subject identifier (sub-directory inside <root>)
  --video PATH           input mp4 / mov
  --intrinsics VALUE     "hdtf" | "insta" | "custom:fx,fy,cx,cy"

Optional flags:
  --fps N                frame extraction fps                  (default 30)
  --resize N             square crop size in px                (default 512)
  --with-albedo          run IntrinsicAnything for albedo GT   (default off)
  --no-stable-bbox       skip the stable bbox preprocess step  (default on)
  --bbox-verify          also write bbox_verify.mp4 + .csv     (default off)
  --bbox-cutoff-hz F     FIR LPF cutoff in Hz for bbox.size    (default 2.5)
  --lambda-shape F       DECA optimize.py shape regularizer    (default 1e-2,
                         the original HRAvatar fork value. Pass 1.0-5.0 to
                         fix alien-looking enlarged head / collapsed face in
                         optimize_vis.jpg. Requires DECA optimize patcher.)
  --lambda-exp F         DECA optimize.py exp regularizer      (default 1e-2)
  --skip-deca-patches    do NOT auto-apply tools/patches/apply_deca_*.py     (default off)
  -h, --help             print this message and exit

Honored environment variable:
  CUDA_VISIBLE_DEVICES   GPU index                             (default 0)

Stable bbox notes:
  When `--no-stable-bbox` is NOT set (the default), this script computes a
  temporally smoothed 224-crop bbox sequence in `<root>/<name>/stable_bbox.npz`.
  DECA's preprocessing reads it (when patched via
  `tools/patches/apply_deca_stable_bbox.py`) and `scene/data_loader.py`
  auto-detects it at training time so SMIRK encoder input no longer
  wobbles with mouth/blink. See `doc/stable_bbox.md`.

DECA patches:
  By default this script auto-runs the idempotent patchers under
  `tools/patches/` so DECA can accept `--precomputed-bbox`,
  `--lambda_shape`, `--lambda_exp`. Pass `--skip-deca-patches` to skip
  (e.g. when you maintain a pre-patched DECA fork manually).
EOF
}

# Defaults — match what the previous env-var interface defaulted to so the
# observable behaviour is unchanged for callers that supplied nothing extra.
ROOT=""
NAME=""
VIDEO=""
INTRINSICS=""
FPS=30
RESIZE=512
WITH_ALBEDO=0
STABLE_BBOX=1
BBOX_VERIFY=0
BBOX_CUTOFF_HZ=2.5
LAMBDA_SHAPE=""
LAMBDA_EXP=""
SKIP_DECA_PATCHES=0

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
    --with-albedo|--with_albedo) WITH_ALBEDO=1; shift ;;
    --no-stable-bbox)  STABLE_BBOX=0; shift ;;
    --bbox-verify)     BBOX_VERIFY=1; shift ;;
    --bbox-cutoff-hz)  require_value "$@"; BBOX_CUTOFF_HZ="$2"; shift 2 ;;
    --lambda-shape)    require_value "$@"; LAMBDA_SHAPE="$2"; shift 2 ;;
    --lambda-exp)      require_value "$@"; LAMBDA_EXP="$2"; shift 2 ;;
    --skip-deca-patches) SKIP_DECA_PATCHES=1; shift ;;
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

: "${CUDA_VISIBLE_DEVICES:=0}"
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

mkdir -p "${ROOT}"
ROOT="$(cd "${ROOT}" && pwd)"

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

DECA_PRECOMPUTED_BBOX_ARG=""
if [[ "${STABLE_BBOX}" == "1" ]]; then
  echo "[preprocess 1.5/5] stable bbox (FlashAvatar PR#7 port)"
  python preprocess/stable_bbox.py \
      --source "${DATA_DIR}" --fps "${FPS}" --cutoff_hz "${BBOX_CUTOFF_HZ}"
  if [[ "${BBOX_VERIFY}" == "1" ]]; then
    echo "[preprocess 1.5b] bbox verify mp4"
    python preprocess/bbox_verify.py \
        --source "${DATA_DIR}" --fps "${FPS}"
  fi
  # The DECA fork patched by tools/patches/apply_deca_stable_bbox.py accepts
  # `--precomputed-bbox`. Without the patch the flag is unknown and DECA
  # would error out, so the per-subject script enables it only when
  # the npz exists AND --no-stable-bbox was NOT passed.
  DECA_PRECOMPUTED_BBOX_ARG="--precomputed-bbox ${DATA_DIR}/stable_bbox.npz"
fi

echo "[preprocess 2/5] DECA initial FLAME"
( cd "${DECA_DIR}" && \
  python demos/demo_reconstruct.py \
      -i "${DATA_DIR}/image" \
      --savefolder "${DATA_DIR}/deca" \
      --saveCode True --saveVis False --sample_step 1 --render_orig False \
      ${DECA_PRECOMPUTED_BBOX_ARG} )
if [[ ! -f "${DATA_DIR}/code.json" ]]; then
  echo "ERROR: DECA did not produce ${DATA_DIR}/code.json (step 2 failed silently)" >&2
  exit 1
fi

echo "[preprocess 3/5] face-alignment landmarks"
python preprocess/keypoint_detector.py --path "${DATA_DIR}"

echo "[preprocess 4/5] iris segmentation"
python preprocess/iris.py --path "${DATA_DIR}"

echo "[preprocess 5/5] optimize FLAME parameters"
DECA_OPTIMIZE_EXTRA_ARGS=()
if [[ -n "${LAMBDA_SHAPE}" ]]; then
  DECA_OPTIMIZE_EXTRA_ARGS+=(--lambda_shape "${LAMBDA_SHAPE}")
fi
if [[ -n "${LAMBDA_EXP}" ]]; then
  DECA_OPTIMIZE_EXTRA_ARGS+=(--lambda_exp "${LAMBDA_EXP}")
fi
( cd "${DECA_DIR}" && \
  python optimize.py --path "${DATA_DIR}" \
      --cx "${CX}" --cy "${CY}" --fx "${FX}" --fy "${FY}" --size "${RESIZE}" \
      --n_shape 100 --n_expr 100 --with_translation \
      ${DECA_OPTIMIZE_EXTRA_ARGS[@]+"${DECA_OPTIMIZE_EXTRA_ARGS[@]}"} )

if [[ "${WITH_ALBEDO}" == "1" ]]; then
  echo "[preprocess opt] IntrinsicAnything pseudo-albedo"
  python preprocess/submodules/IntrinsicAnything/inference.py \
      --input_dir  "${DATA_DIR}/image" \
      --model_dir  assets/intrinsic_anything/albedo \
      --output_dir "${DATA_DIR}/albedo" \
      --ddim 100 --batch_size 10 --image_interval 3
fi

echo "Preprocessing done: ${DATA_DIR}/tracked_params.json"
