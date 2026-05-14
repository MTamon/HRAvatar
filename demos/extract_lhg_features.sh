#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Stage 2 of the LHG (Listening Head Generation) preprocessing pipeline.
# Extracts per-frame FLAME features (expression / jaw / eyelid via SMIRK,
# global_rot / translation via cv2.solvePnP EPnP) from a video or
# pre-extracted image directory and writes lhg_features.npz.
#
# This is NOT a substitute for demos/_preprocess_subject.sh — that script
# (Stage 1, runs DECA optimize) produces tracked_params.json with the
# clip-constant world_mat / shapecode / intrinsics that THIS script
# consumes via --calibration. Run Stage 1 once per subject; then run
# Stage 2 here per clip. See doc/preprocessing_scope.md for why the two
# pipelines must stay separate.
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash demos/extract_lhg_features.sh \
      --video <path> \
      --calibration <tracked_params.json> \
      --output <path> \
      --mode <online|pseudo-online> \
      [options]

Required arguments:
  --video PATH         input mp4/mov OR an outer-cropped image/ directory
                       (when a directory, outer_offset.json is auto-detected)
  --calibration PATH   Stage 1 tracked_params.json (or _v2 sibling). Stage 2
                       reads world_mat / shapecode / intrinsics from this
                       file and uses them as clip-constants. Run Stage 1
                       once per subject:
                         bash demos/_preprocess_subject.sh \
                              --sbj-root <root> --sbj-name <name> \
                              --video <video> --intrinsics <preset> \
                              --lhg-only
  --output PATH        output .npz path
  --mode MODE          online       : strictly causal, mirrors LHG inference
                       pseudo-online: causal core + future-info corrections
                                      (bidirectional Hampel, dropout
                                      interpolation, quaternion-flip fix,
                                      larger-lookahead FIR LPF). Use for
                                      detector-dropout salvage or
                                      future-info smoothing studies. The
                                      LHG teacher data does NOT come from
                                      this path under the 2026-05-14 grand
                                      design — see demos/build_lhg_teacher.sh.

Optional flags:
  --intrinsics VALUE   override the EPnP intrinsics. Default: read from
                       --calibration. Accepts "hdtf" / "insta" /
                       "custom:fx,fy,cx,cy".
  --fps N              feature-stream FPS                 (default 25)
                       LHG inference runs at 10 FPS but the per-frame
                       feature stream is at this rate.
  --image-size N       outer-crop output square size in px (default 512)
                       Must match the intrinsics preset.
  --bbox-scale F       SMIRK 224-crop scale factor around the stable bbox
                       center                             (default 1.6)
  --flame-scale F      override flame_scale (default: inferred from the
                       calibration file, _v2 → 1.0 else 4.0)
  --correspondence PATH
                       detector-specific FLAME correspondence asset
                       (default assets/lhg/mediapipe_flame_landmarks.npz
                        for mediapipe; assets/lhg/dlib_flame_landmarks.npz
                        for fan)
  --run-deca-encoder   diagnostic: also invoke DECA coarse encoder
                       (NOT used for any LHG output channel)
  --detector {mediapipe|fan}
                       landmark detector                  (default mediapipe)
                       mediapipe : MediaPipe FaceLandmarker 478-pt + iris
                       fan       : face_alignment 68-pt (matches HRAvatar
                                   avatar fit pipeline; per-frame absolute
                                   accuracy is comparable to mediapipe)
  --mediapipe-mode {image|video}
                       MediaPipe running mode             (default video)
                       video : internal Kalman tracker, lower jitter
                       image : per-frame baseline (used for jitter A/B)
  --lookahead L        symmetric FIR LPF one-sided lookahead in frames
                       (default 4 → taps=9, 160ms lag at 25 fps).
                       Set 0 to disable the LPF entirely.
  --lpf-cutoff-hz F    LPF cutoff (Hz) for rotation+translation (default 4)
  --lpf-jaw            also apply the LPF to jaw          (default off)
  --lpf-jaw-cutoff-hz F
                       LPF cutoff (Hz) for jaw            (default 10)
  --lookahead-offline L
                       Stage 3 offline FIR one-sided lookahead
                       (default 12 → taps=25). Ignored in --mode online.
  --camera-convention {hravatar|opencv}
                       coordinate frame for global_rot / translation /
                       world_mat                          (default hravatar)
                       hravatar : X right, Y up, -Z forward (renderer-ready)
                       opencv   : X right, Y down, +Z forward (raw EPnP)
  --backend {epnp|deca_encoder}
                       per-frame translation / global_rot estimator (default
                       epnp). epnp = cv2.solvePnP(EPnP) on shape-aware
                       FLAME landmarks (5/7 system). deca_encoder = reuse
                       offline DECA coarse encoder (2026-05-14 grand
                       design); global_rot = DECA pose[0:3], translation
                       = 0 (Phase B Stage 1).
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
  valid_mask       (N,)    bool      False = detector missed the frame
  interpolated_mask(N,)    bool      pseudo-online interpolation applied
  rejected_mask    (N,)    bool      Hampel-rejected
  world_mat        (4, 4)  float32   clip-constant camera extrinsics from
                                     --calibration
  intrinsics       (4,)    float32   [fx, fy, cx, cy] from --calibration
  outer_bbox       (4,)    int32     [xmin, xmax, ymin, ymax] in raw video coord

EOF
}

VIDEO=""
CALIBRATION=""
OUTPUT=""
MODE=""
INTRINSICS=""
FPS=25
IMAGE_SIZE=512
BBOX_SCALE=""
FLAME_SCALE=""
CORRESPONDENCE=""
RUN_DECA_ENCODER=0
DETECTOR=""
MEDIAPIPE_MODE=""
LOOKAHEAD=""
LPF_CUTOFF_HZ=""
LPF_JAW=0
LPF_JAW_CUTOFF_HZ=""
LOOKAHEAD_OFFLINE=""
CAMERA_CONVENTION=""
BACKEND=""
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
    --calibration)  require_value "$@"; CALIBRATION="$2"; shift 2 ;;
    --output)       require_value "$@"; OUTPUT="$2"; shift 2 ;;
    --mode)         require_value "$@"; MODE="$2"; shift 2 ;;
    --intrinsics)   require_value "$@"; INTRINSICS="$2"; shift 2 ;;
    --fps)          require_value "$@"; FPS="$2"; shift 2 ;;
    --image-size|--image_size) require_value "$@"; IMAGE_SIZE="$2"; shift 2 ;;
    --bbox-scale)   require_value "$@"; BBOX_SCALE="$2"; shift 2 ;;
    --flame-scale)  require_value "$@"; FLAME_SCALE="$2"; shift 2 ;;
    --correspondence) require_value "$@"; CORRESPONDENCE="$2"; shift 2 ;;
    --run-deca-encoder) RUN_DECA_ENCODER=1; shift ;;
    --detector)     require_value "$@"; DETECTOR="$2"; shift 2 ;;
    --mediapipe-mode) require_value "$@"; MEDIAPIPE_MODE="$2"; shift 2 ;;
    --lookahead)    require_value "$@"; LOOKAHEAD="$2"; shift 2 ;;
    --lpf-cutoff-hz) require_value "$@"; LPF_CUTOFF_HZ="$2"; shift 2 ;;
    --lpf-jaw)      LPF_JAW=1; shift ;;
    --lpf-jaw-cutoff-hz) require_value "$@"; LPF_JAW_CUTOFF_HZ="$2"; shift 2 ;;
    --lookahead-offline) require_value "$@"; LOOKAHEAD_OFFLINE="$2"; shift 2 ;;
    --camera-convention) require_value "$@"; CAMERA_CONVENTION="$2"; shift 2 ;;
    --backend)      require_value "$@"; BACKEND="$2"; shift 2 ;;
    --quiet)        QUIET=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    --*) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    *)   echo "unexpected positional argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${VIDEO}" || -z "${CALIBRATION}" || -z "${OUTPUT}" || -z "${MODE}" ]]; then
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
[[ -n "${INTRINSICS}" ]]            && EXTRA_ARGS+=(--intrinsics "${INTRINSICS}")
[[ -n "${BBOX_SCALE}" ]]            && EXTRA_ARGS+=(--bbox-scale "${BBOX_SCALE}")
[[ -n "${FLAME_SCALE}" ]]           && EXTRA_ARGS+=(--flame-scale "${FLAME_SCALE}")
[[ -n "${CORRESPONDENCE}" ]]        && EXTRA_ARGS+=(--correspondence "${CORRESPONDENCE}")
[[ "${RUN_DECA_ENCODER}" == "1" ]]  && EXTRA_ARGS+=(--run-deca-encoder)
[[ -n "${DETECTOR}" ]]              && EXTRA_ARGS+=(--detector "${DETECTOR}")
[[ -n "${MEDIAPIPE_MODE}" ]]        && EXTRA_ARGS+=(--mediapipe-mode "${MEDIAPIPE_MODE}")
[[ -n "${LOOKAHEAD}" ]]             && EXTRA_ARGS+=(--lookahead "${LOOKAHEAD}")
[[ -n "${LPF_CUTOFF_HZ}" ]]         && EXTRA_ARGS+=(--lpf-cutoff-hz "${LPF_CUTOFF_HZ}")
[[ "${LPF_JAW}" == "1" ]]           && EXTRA_ARGS+=(--lpf-jaw)
[[ -n "${LPF_JAW_CUTOFF_HZ}" ]]     && EXTRA_ARGS+=(--lpf-jaw-cutoff-hz "${LPF_JAW_CUTOFF_HZ}")
[[ -n "${LOOKAHEAD_OFFLINE}" ]]     && EXTRA_ARGS+=(--lookahead-offline "${LOOKAHEAD_OFFLINE}")
[[ -n "${CAMERA_CONVENTION}" ]]     && EXTRA_ARGS+=(--camera-convention "${CAMERA_CONVENTION}")
[[ -n "${BACKEND}" ]]               && EXTRA_ARGS+=(--backend "${BACKEND}")
[[ "${QUIET}" == "1" ]]             && EXTRA_ARGS+=(--quiet)

python -m lhg.extract \
    --video "${VIDEO}" \
    --calibration "${CALIBRATION}" \
    --output "${OUTPUT}" \
    --mode "${MODE}" \
    --fps "${FPS}" \
    --image-size "${IMAGE_SIZE}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
