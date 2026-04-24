#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# HRAvatar — third-party asset downloader.
#
# Assets are split into three categories:
#   (A) FLAME 2020 generic_model.pkl          -- requires credentialed download
#                                                from https://flame.is.tue.mpg.de
#   (B) DECA deca_model.tar                   -- requires Google-Drive login
#   (C) SMIRK SMIRK_em1.pt                     -- direct download
#   (D) face-parsing 79999_iter.pth            -- requires Google-Drive login
#   (E) RobustVideoMatting rvm_resnet50.pth    -- direct GitHub release
#   (F) IntrinsicAnything weights              -- HuggingFace snapshot
#
# Items (A), (B) and (D) require an interactive browser download.  This script
# will place them in the expected locations if they are already in
# ~/Downloads, otherwise it prints a clear message with the URL.
#
# Reference: MTamon/smirk release/cuda128 -- `quick_install.sh` /
# `prepare_demos.sh` follow the same pattern.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

say() { printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!! %s\033[0m\n' "$*"; }

mkdir -p assets/flame_model \
         assets/smirk/pretrained_models \
         assets/intrinsic_anything/albedo \
         preprocess/submodules/DECA/data \
         preprocess/submodules/face-parsing.PyTorch/res/cp \
         preprocess/submodules/RobustVideoMatting

move_from_downloads() {
  local wanted=$1 dest=$2
  if [[ -f "${dest}" ]]; then
    say "Already present: ${dest}"
    return 0
  fi
  if [[ -f "${HOME}/Downloads/${wanted}" ]]; then
    cp "${HOME}/Downloads/${wanted}" "${dest}"
    say "Copied ${wanted} -> ${dest}"
  else
    warn "Manual download required: ${wanted}"
    warn "  Place file at: ${dest}"
  fi
}

# -------------------------- (A) FLAME 2020 -----------------------------------
say "[A] FLAME 2020 generic_model.pkl"
warn "Register + download at https://flame.is.tue.mpg.de/download.php"
warn "Pick 'FLAME 2020' (generic_model.pkl). Rename expected below."
move_from_downloads generic_model.pkl   assets/flame_model/flame2020.pkl
move_from_downloads generic_model.pkl   preprocess/submodules/DECA/data/generic_model2020.pkl

# -------------------------- (B) DECA ----------------------------------------
say "[B] DECA deca_model.tar"
warn "Download: https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view"
move_from_downloads deca_model.tar preprocess/submodules/DECA/data/deca_model.tar

# -------------------------- (C) SMIRK ---------------------------------------
say "[C] SMIRK SMIRK_em1.pt"
if [[ ! -f assets/smirk/pretrained_models/SMIRK_em1.pt ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -c -O assets/smirk/pretrained_models/SMIRK_em1.pt \
      https://github.com/georgeretsi/smirk/releases/download/v1.0/SMIRK_em1.pt \
      || warn "SMIRK download failed; fetch manually from the smirk releases page."
  else
    warn "wget not found. Install wget or download SMIRK_em1.pt manually."
  fi
fi

# -------------------------- (D) face-parsing -------------------------------
say "[D] face-parsing 79999_iter.pth"
warn "Download: https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view"
move_from_downloads 79999_iter.pth \
  preprocess/submodules/face-parsing.PyTorch/res/cp/79999_iter.pth

# -------------------------- (E) RobustVideoMatting --------------------------
say "[E] RobustVideoMatting rvm_resnet50.pth"
if [[ ! -f preprocess/submodules/RobustVideoMatting/rvm_resnet50.pth ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -c -O preprocess/submodules/RobustVideoMatting/rvm_resnet50.pth \
      https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth
  else
    warn "wget not found. Install wget or download rvm_resnet50.pth manually."
  fi
fi

# -------------------------- (F) IntrinsicAnything ---------------------------
say "[F] IntrinsicAnything albedo weights (optional, only needed for HDTF/custom albedo)"
if [[ ! -f assets/intrinsic_anything/albedo/checkpoints/last.ckpt ]]; then
  if python -c 'import huggingface_hub' >/dev/null 2>&1; then
    python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="LittleFrog/IntrinsicAnything",
    repo_type="space",
    allow_patterns=["weights/albedo/*"],
    local_dir="assets/intrinsic_anything/tmp",
)
PY
    mkdir -p assets/intrinsic_anything/albedo
    rsync -a assets/intrinsic_anything/tmp/weights/albedo/ assets/intrinsic_anything/albedo/
    rm -rf assets/intrinsic_anything/tmp
  else
    warn "huggingface_hub not installed; skip IntrinsicAnything weights."
  fi
fi

say "Done. Verify with:"
echo "  ls assets/flame_model/flame2020.pkl"
echo "  ls preprocess/submodules/DECA/data/{deca_model.tar,generic_model2020.pkl}"
echo "  ls assets/smirk/pretrained_models/SMIRK_em1.pt"
