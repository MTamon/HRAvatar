#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Build LHG teacher data from an offline tracked_params.json.
#
# Under the 2026-05-14 LHG grand design (see ``project_lhg_grand_design``
# memory), the LHG model's supervision target is the per-frame FLAME
# parameters produced by the offline pipeline (DECA optimize + SMIRK).
# This script repackages those per-frame entries into the
# ``LHGFeatures`` (``lhg_features.npz``) schema that the rest of the
# LHG training tooling consumes — it is the inverse of
# ``demos/render_lhg_features.sh`` / ``lhg/render_adapter.py``.
#
# Stage 1 (``demos/_preprocess_subject.sh --lhg-only``) must have been
# run already; this script does NOT re-run DECA optimize. The output
# npz pairs with a sidecar ``<output>.shapecode.json`` carrying the
# clip-constant shape parameters that do not fit the LHGFeatures schema.
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash demos/build_lhg_teacher.sh \
      --tracked-params <tracked_params.json> \
      --output <lhg_features_teacher.npz> \
      [options]

Required arguments:
  --tracked-params PATH  Offline Stage 1 tracked_params.json (or _v2 sibling).
                         Per-frame expcode / fullposecode / translation /
                         eyelids become the LHG teacher target.
  --output PATH          Output lhg_features.npz path. A sidecar
                         <output>.shapecode.json is written alongside.

Optional flags:
  --fps F                Feature stream FPS recorded in metadata.
                         Default 25 (LHG convention; see
                         project_lhg_fps_convention).
  --image-size N         Outer-crop square size in pixels (default 512).
                         Only used for the outer_bbox fallback when no
                         outer_offset.json sidecar is present.
  --expression-dim N     Expression dims to keep from offline expcode
                         (default 50, matches SMIRK / LHGFeatures).
  --avatar-checkpoint D  Avatar checkpoint dir containing
                         flame_params_net.pth (e.g.
                         outputs/custom/<avatar>/saved_model/epoch_<E>).
                         When set, expression / jaw / eyelid are
                         recomputed by the avatar's trained SMIRK on
                         the 224 warped crop from stable_bbox.npz —
                         this matches what the renderer consumes at
                         avatar training time. Without this flag the
                         legacy behaviour (DECA optimize expcode) is
                         used for backwards compatibility.
  --subject-dir D        Directory with stable_bbox.npz + image/ etc.
                         (defaults to --tracked-params parent).
                         Only consulted with --avatar-checkpoint.
  --prefer-outer-bbox    Force the outer_512 (stable_bbox.npz + image/)
                         path even when stable_bbox_raw.npz + image_raw/
                         exist. Default: raw-resolution path is
                         preferred (matches data_loader.py).
  --smirk-batch-size N   Forward batch size for SMIRK (default 16).
  --device DEVICE        Compute device (default cuda).
  -h, --help             Print this message and exit.

Output:
  - <output>                   ``lhg_features.npz`` (LHGFeatures schema)
  - <output>.shapecode.json    Sidecar: clip-constant shapecode + provenance

EOF
}

TRACKED_PARAMS=""
OUTPUT=""
FPS=""
IMAGE_SIZE=""
EXPRESSION_DIM=""
AVATAR_CHECKPOINT=""
SUBJECT_DIR=""
PREFER_OUTER_BBOX=0
SMIRK_BATCH_SIZE=""
DEVICE=""

require_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "missing value for $1" >&2
    usage
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tracked-params)  require_value "$@"; TRACKED_PARAMS="$2"; shift 2 ;;
    --output)          require_value "$@"; OUTPUT="$2"; shift 2 ;;
    --fps)             require_value "$@"; FPS="$2"; shift 2 ;;
    --image-size|--image_size) require_value "$@"; IMAGE_SIZE="$2"; shift 2 ;;
    --expression-dim|--expression_dim) require_value "$@"; EXPRESSION_DIM="$2"; shift 2 ;;
    --avatar-checkpoint|--avatar_checkpoint) require_value "$@"; AVATAR_CHECKPOINT="$2"; shift 2 ;;
    --subject-dir|--subject_dir) require_value "$@"; SUBJECT_DIR="$2"; shift 2 ;;
    --prefer-outer-bbox|--prefer_outer_bbox) PREFER_OUTER_BBOX=1; shift ;;
    --smirk-batch-size|--smirk_batch_size) require_value "$@"; SMIRK_BATCH_SIZE="$2"; shift 2 ;;
    --device)          require_value "$@"; DEVICE="$2"; shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    --*) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    *)   echo "unexpected positional argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${TRACKED_PARAMS}" || -z "${OUTPUT}" ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXTRA_ARGS=()
[[ -n "${FPS}" ]]                && EXTRA_ARGS+=(--fps "${FPS}")
[[ -n "${IMAGE_SIZE}" ]]         && EXTRA_ARGS+=(--image-size "${IMAGE_SIZE}")
[[ -n "${EXPRESSION_DIM}" ]]     && EXTRA_ARGS+=(--expression-dim "${EXPRESSION_DIM}")
[[ -n "${AVATAR_CHECKPOINT}" ]]  && EXTRA_ARGS+=(--avatar-checkpoint "${AVATAR_CHECKPOINT}")
[[ -n "${SUBJECT_DIR}" ]]        && EXTRA_ARGS+=(--subject-dir "${SUBJECT_DIR}")
[[ "${PREFER_OUTER_BBOX}" == "1" ]] && EXTRA_ARGS+=(--prefer-outer-bbox)
[[ -n "${SMIRK_BATCH_SIZE}" ]]   && EXTRA_ARGS+=(--smirk-batch-size "${SMIRK_BATCH_SIZE}")
[[ -n "${DEVICE}" ]]             && EXTRA_ARGS+=(--device "${DEVICE}")

python -m lhg.teacher \
    --tracked-params "${TRACKED_PARAMS}" \
    --output "${OUTPUT}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
