#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Extract per-frame LHG (Listening Head Generation) FLAME features from a
# video or a pre-extracted image directory. Mirrors the CLI shape of
# demos/_preprocess_subject.sh but produces a different, smaller artifact
# (lhg_features.npz) intended as either teacher data (--mode pseudo-online)
# or as the recorded output of an inference-equivalent pipeline (--mode online).
#
# This script is NOT a substitute for demos/_preprocess_subject.sh — that
# script produces tracked_params.json for HRAvatar AVATAR PERSONAL FIT via
# DECA's full-clip joint optimization. See doc/preprocessing_scope.md for
# why the two pipelines must stay separate.
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash demos/extract_lhg_features.sh \
      --video <path> \
      --intrinsics <hdtf|insta|custom:fx,fy,cx,cy> \
      --output <path> \
      --mode <online|pseudo-online> \
      [options]

Required arguments:
  --video PATH         input mp4/mov OR an outer-cropped image/ directory
                       (when a directory, outer_offset.json is auto-detected)
  --intrinsics VALUE   "hdtf" | "insta" | "custom:fx,fy,cx,cy"
                       (camera intrinsics in the OUTER-CROP coordinate
                        system, i.e. the image_size square)
  --output PATH        output .npz path
  --mode MODE          online       : strictly causal, mirrors LHG inference
                       pseudo-online: causal core + future-info corrections
                                      for one-time anomalies (teacher data)

Optional flags:
  --fps N              source frame rate                   (default 30)
  --image-size N       outer-crop output square size in px (default 512)
                       Must match the intrinsics preset.
  --bbox-scale F       SMIRK 224-crop scale factor around the stable bbox
                       center                             (default 1.6)
  --world-mat PATH     reuse world_mat from an existing tracked_params.json
                       instead of calibrating from this clip
  --world-mat-calibration-frames N
                       N first valid frames used to calibrate world_mat when
                       --world-mat is not supplied         (default 60)
  --flame-scale F      FLAME scale convention              (default 4.0
                       — HRAvatar v1; pass 1.0 for v2)
  --correspondence PATH
                       MediaPipe→FLAME landmark correspondence asset
                       (default assets/lhg/mediapipe_flame_landmarks.npz;
                        build via tools/build_mediapipe_flame_correspondence.py)
  --run-deca-encoder   also invoke DECA coarse encoder for diagnostics
                       (NOT used for any LHG output channel)
  --quiet              suppress progress bar
  -h, --help           print this message and exit

Honored environment variable:
  CUDA_VISIBLE_DEVICES   GPU index for SMIRK / DECA encoder (default 0)

Output schema (lhg_features.npz, see lhg/output.py for the full list):
  expression       (N, 50) float32   SMIRK
  jaw              (N,  3) float32   SMIRK
  eyelid           (N,  2) float32   SMIRK
  global_rot       (N,  3) float32   axis-angle from cv2.solvePnP (EPnP)
  translation      (N,  3) float32   FLAME canonical space (pre flame_scale)
  valid_mask       (N,)    bool      False = MediaPipe missed the frame
  interpolated_mask(N,)    bool      pseudo-online interpolation applied
  rejected_mask    (N,)    bool      Hampel-rejected
  world_mat        (4, 4)  float32   clip-constant camera extrinsics
  intrinsics       (4,)    float32   [fx, fy, cx, cy]
  outer_bbox       (4,)    int32     [xmin, xmax, ymin, ymax] in raw video coord

EOF
}

VIDEO=""
INTRINSICS=""
OUTPUT=""
MODE=""
FPS=30
IMAGE_SIZE=512
BBOX_SCALE=""
WORLD_MAT=""
WORLD_MAT_CAL_FRAMES=""
FLAME_SCALE=""
CORRESPONDENCE=""
RUN_DECA_ENCODER=0
QUIET=0

require_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "missing value for $1" >&2
    usage
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)        require_value "$@"; VIDEO="$2"; shift 2 ;;
    --intrinsics)   require_value "$@"; INTRINSICS="$2"; shift 2 ;;
    --output)       require_value "$@"; OUTPUT="$2"; shift 2 ;;
    --mode)         require_value "$@"; MODE="$2"; shift 2 ;;
    --fps)          require_value "$@"; FPS="$2"; shift 2 ;;
    --image-size|--image_size) require_value "$@"; IMAGE_SIZE="$2"; shift 2 ;;
    --bbox-scale)   require_value "$@"; BBOX_SCALE="$2"; shift 2 ;;
    --world-mat)    require_value "$@"; WORLD_MAT="$2"; shift 2 ;;
    --world-mat-calibration-frames) require_value "$@"; WORLD_MAT_CAL_FRAMES="$2"; shift 2 ;;
    --flame-scale)  require_value "$@"; FLAME_SCALE="$2"; shift 2 ;;
    --correspondence) require_value "$@"; CORRESPONDENCE="$2"; shift 2 ;;
    --run-deca-encoder) RUN_DECA_ENCODER=1; shift ;;
    --quiet)        QUIET=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    --*) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    *)   echo "unexpected positional argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${VIDEO}" || -z "${INTRINSICS}" || -z "${OUTPUT}" || -z "${MODE}" ]]; then
  usage
  exit 2
fi
case "${MODE}" in
  online|pseudo-online) ;;
  *) echo "--mode must be 'online' or 'pseudo-online'" >&2; exit 2 ;;
esac

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXTRA_ARGS=()
[[ -n "${BBOX_SCALE}" ]]            && EXTRA_ARGS+=(--bbox-scale "${BBOX_SCALE}")
[[ -n "${WORLD_MAT}" ]]             && EXTRA_ARGS+=(--world-mat "${WORLD_MAT}")
[[ -n "${WORLD_MAT_CAL_FRAMES}" ]]  && EXTRA_ARGS+=(--world-mat-calibration-frames "${WORLD_MAT_CAL_FRAMES}")
[[ -n "${FLAME_SCALE}" ]]           && EXTRA_ARGS+=(--flame-scale "${FLAME_SCALE}")
[[ -n "${CORRESPONDENCE}" ]]        && EXTRA_ARGS+=(--correspondence "${CORRESPONDENCE}")
[[ "${RUN_DECA_ENCODER}" == "1" ]]  && EXTRA_ARGS+=(--run-deca-encoder)
[[ "${QUIET}" == "1" ]]             && EXTRA_ARGS+=(--quiet)

python -m lhg.extract \
    --video "${VIDEO}" \
    --intrinsics "${INTRINSICS}" \
    --output "${OUTPUT}" \
    --mode "${MODE}" \
    --fps "${FPS}" \
    --image-size "${IMAGE_SIZE}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
