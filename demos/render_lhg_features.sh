#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Visual rendering verification for the LHG (Listening Head Generation)
# preprocessing pipeline. Loads a trained HRAvatar checkpoint (the avatar)
# and substitutes the per-frame tracker payload with values from a
# lhg_features.npz produced by demos/extract_lhg_features.sh, then writes
# the rendered MP4 alongside the avatar's other render outputs.
#
# Use this to compare:
#   1. The avatar's training-time tracked_params.json (= reference render)
#   2. lhg_features.npz with --lookahead 0  (LPF off, baseline jitter)
#   3. lhg_features.npz with --lookahead 4  (LPF on, default)
#
# All three runs share the same avatar checkpoint so any visual difference
# isolates the per-frame tracker payload.
# -----------------------------------------------------------------------------
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash demos/render_lhg_features.sh \
      --avatar <model_dir> \
      --source <source_dir> \
      --lhg-features <lhg_features.npz> \
      [--epoch N]

Required arguments:
  --avatar PATH         the trained HRAvatar checkpoint dir (passed to
                        render.py as -m / --model_path)
  --source PATH         the data directory whose tracked_params.json
                        provides the avatar's shapecode and whose
                        image/ directory provides the renderer
                        defaults (passed as -s / --source_path).
                        Typically the same directory the avatar was
                        trained on.
  --lhg-features PATH   the lhg_features.npz produced by
                        demos/extract_lhg_features.sh

Optional flags:
  --epoch N             checkpoint epoch (default -1 = latest)
  --skip-train          skip rendering the train split
  --skip-test           skip rendering the test split (useful when the
                        avatar was trained on the full clip and you
                        want a single rendered MP4)

Honored environment variable:
  CUDA_VISIBLE_DEVICES   GPU index (default 0)

Output:
  <avatar>/{train|test}/ours_<epoch>/{train|test}_<scene>_video.mp4

  When --lhg-features is set, the rendered MP4 reflects the LHG-derived
  per-frame motion. Compare it against the avatar's reference render
  (run without --lhg-features) to evaluate the LHG pipeline visually.

EOF
}

AVATAR=""
SOURCE=""
LHG_FEATURES=""
EPOCH=""
SKIP_TRAIN=0
SKIP_TEST=0

require_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "missing value for $1" >&2
    usage
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --avatar)         require_value "$@"; AVATAR="$2"; shift 2 ;;
    --source)         require_value "$@"; SOURCE="$2"; shift 2 ;;
    --lhg-features)   require_value "$@"; LHG_FEATURES="$2"; shift 2 ;;
    --epoch)          require_value "$@"; EPOCH="$2"; shift 2 ;;
    --skip-train)     SKIP_TRAIN=1; shift ;;
    --skip-test)      SKIP_TEST=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    --*) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    *)   echo "unexpected positional argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${AVATAR}" || -z "${SOURCE}" || -z "${LHG_FEATURES}" ]]; then
  usage
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXTRA_ARGS=()
[[ -n "${EPOCH}" ]]            && EXTRA_ARGS+=(--epoch "${EPOCH}")
[[ "${SKIP_TRAIN}" == "1" ]]   && EXTRA_ARGS+=(--skip_train)
[[ "${SKIP_TEST}" == "1" ]]    && EXTRA_ARGS+=(--skip_test)

python render.py \
    -m "${AVATAR}" \
    -s "${SOURCE}" \
    --lhg-features "${LHG_FEATURES}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
