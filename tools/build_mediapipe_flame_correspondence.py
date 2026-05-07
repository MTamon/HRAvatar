"""Build LHG correspondence assets:

* ``assets/lhg/mediapipe_flame_landmarks.npz``  — MediaPipe FaceMesh →
  FLAME mapping (default subset = 16 frontal-face anchors from MICA).
* ``assets/lhg/dlib_flame_landmarks.npz``        — dlib 68-point →
  FLAME mapping (default subset = 31 STATIC landmarks, indices 17-47:
  brows + nose + eyes; SKIPS the pose-dependent face contour 0-16 and
  the speech-dependent lips 48-67). For FAN-based EPnP.

Build both in one run; pass ``--detector {mediapipe,dlib,both}`` to
restrict.

Output format (consumed by ``lhg.correspondence.load``):

* ``mp_indices`` : (K,) int64 — MediaPipe FaceMesh point indices used by
  the LHG EPnP solve.
* ``flame_face_idx`` : (K,) int64 — FLAME triangle index per landmark.
* ``flame_bary``     : (K, 3) float64 — barycentric coords within that
  triangle.
* ``flame_canonical_xyz`` : (K, 3) float64 — landmark position on the
  canonical FLAME mesh (shape=0, expression=0, pose=identity), i.e. the
  average head shape. EPnP consumes this when no per-clip
  shape_param-conditioned 3D positions are available.

Default subset: ``--subset lhg-pnp`` (16 frontal-face points)
-------------------------------------------------------------
This subset is curated specifically for EPnP conditioning, NOT to mirror
``preprocess._smirk_constants.STABLE_LANDMARK_INDICES`` (whose role is
bbox stabilization, not pose recovery). It includes:

* 4 eye corners (33, 133, 263, 362) — rigid orbital rim
* 6 brow points (46, 276, 105, 334, 55, 285) — rigid frontal bone
* 6 nose points (4, 5, 6, 168, 195, 197) — rigid nasal bone

All 16 are present in MICA's 105 correspondences and all sit on the
ANTERIOR (front) of the FLAME canonical mesh (z>0). They span the full
forehead → eye → nose region, giving a 3D conditioning σ_max/σ_min ≈
3.6 — well within the healthy <10 range — without resorting to lateral
points (temples / cheeks) that MICA does not cover.

Why we don't use the bbox subset for PnP
----------------------------------------
``STABLE_LANDMARK_INDICES`` includes 5 lateral points (1, 127, 234,
356, 454) that are NOT in MICA. An earlier revision filled them from
the dlib 68-point face contour shipped under
``assets/flame_model/landmark_embedding.npy``, but the dlib face
contour points map to FLAME canonical positions with z<0 (BACK of the
head), while MediaPipe's lateral landmarks anatomically sit on the
SIDE of the face with z>0. The Z mismatch (~0.07 unit per landmark)
made EPnP misattribute large head rotations as translation. The
fallback is removed; ``--subset stable`` still exposes the smaller
10-point subset (eye + nose intersection) for diagnostic use.

Run
---
    python tools/build_mediapipe_flame_correspondence.py \
        --mica  assets/flame_model/mediapipe_landmark_embedding.npz \
        --flame assets/flame_model/flame2020.pkl \
        --output assets/lhg/mediapipe_flame_landmarks.npz
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


# Two named subsets of MediaPipe FaceMesh indices, both fully covered
# by MICA's 105 correspondences (no fallback needed):
#
# LHG_PNP_LANDMARK_INDICES — 16 frontal-face points specifically curated
# for EPnP conditioning. All sit on the ANTERIOR of the FLAME canonical
# mesh (z>0) and span the full forehead → eye → nose region. Default
# subset for the LHG pipeline.
#
# STABLE_BBOX_INTERSECTION — the 10 STABLE_LANDMARK_INDICES (defined in
# preprocess/_smirk_constants.py for bbox stability) that happen to be
# in MICA. Eye + nose only, near-coplanar, ill-conditioned for PnP. Kept
# for diagnostic comparisons.
LHG_PNP_LANDMARK_INDICES = np.array([
    # Eye corners (rigid orbital rim)
    33, 133, 263, 362,
    # Brow points (rigid frontal bone): outer / middle / inner pairs
    46, 276, 105, 334, 55, 285,
    # Nose (rigid nasal bone)
    4, 5, 6, 168, 195, 197,
], dtype=np.int64)

STABLE_BBOX_INTERSECTION = np.array(
    [33, 133, 362, 263, 1, 4, 5, 6, 168, 195, 197, 234, 454, 127, 356],
    dtype=np.int64,
)

# dlib 68-point RIGID subset for FAN-based EPnP. Excludes:
# * face contour (0-16)        — pose-dependent visible boundary
# * eyelid contours (37,38,40,41,43,44,46,47) — move with blinks
# * lips (48-67)               — move with speech
# Keeps:
# * brows (17-26, 10 pts)            — rigid frontal bone
# * nose (27-35, 9 pts)              — rigid nasal bone
# * eye CORNERS (36,39,42,45, 4 pts) — rigid orbital rim only (NOT lids)
DLIB_RIGID_INDICES = np.array(
    list(range(17, 36)) + [36, 39, 42, 45],
    dtype=np.int64,
)


def _reconstruct_xyz(
    flame_vertices: np.ndarray,
    flame_faces: np.ndarray,
    face_idx: np.ndarray,
    bary: np.ndarray,
) -> np.ndarray:
    tri = flame_faces[face_idx]                # (K, 3)
    v0 = flame_vertices[tri[:, 0]]
    v1 = flame_vertices[tri[:, 1]]
    v2 = flame_vertices[tri[:, 2]]
    return bary[:, 0:1] * v0 + bary[:, 1:2] * v1 + bary[:, 2:3] * v2


def _load_flame(flame_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(flame_path, 'rb') as fp:
        flame_pkl = pickle.load(fp, encoding='latin1')
    v_template = np.asarray(flame_pkl['v_template'], dtype=np.float64)
    faces = np.asarray(flame_pkl['f'], dtype=np.int64)
    return v_template, faces


def _build_mediapipe_asset(
    mica_path: Path, v_template: np.ndarray, faces: np.ndarray, subset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(mica_path, allow_pickle=False) as npz:
        mica_lmk_idx = npz['landmark_indices'].astype(np.int64)
        mica_face_idx = npz['lmk_face_idx'].astype(np.int64)
        mica_bary = npz['lmk_b_coords'].astype(np.float64)

    mica_pos = {int(idx): i for i, idx in enumerate(mica_lmk_idx)}

    if subset == 'all-mica':
        mp_indices = mica_lmk_idx
        flame_face_idx = mica_face_idx
        flame_bary = mica_bary
        rationale = 'full MICA 105 correspondences'
    else:
        wanted = LHG_PNP_LANDMARK_INDICES if subset == 'lhg-pnp' else STABLE_BBOX_INTERSECTION
        keep_mp, keep_face, keep_bary, missing = [], [], [], []
        for w in wanted:
            i = mica_pos.get(int(w))
            if i is None:
                missing.append(int(w))
                continue
            keep_mp.append(int(w))
            keep_face.append(int(mica_face_idx[i]))
            keep_bary.append(mica_bary[i])
        if len(keep_mp) < 4:
            sys.exit(
                f'[error] only {len(keep_mp)} of {len(wanted)} requested '
                f'indices found in MICA (need >=4 for EPnP).')
        mp_indices = np.asarray(keep_mp, dtype=np.int64)
        flame_face_idx = np.asarray(keep_face, dtype=np.int64)
        flame_bary = np.stack(keep_bary, axis=0)
        rationale = f'mediapipe subset={subset}: {len(keep_mp)} of {len(wanted)}'
        if missing:
            rationale += f' (missing: {missing})'

    canonical = _reconstruct_xyz(v_template, faces, flame_face_idx, flame_bary)
    return mp_indices, flame_face_idx, flame_bary, canonical, rationale


def _build_dlib_asset(
    dlib_path: Path, v_template: np.ndarray, faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Returns (dlib_indices (K,), flame_face_idx (K,), flame_bary (K,3),
    canonical_xyz (K,3), rationale).

    Uses ``static_lmk_faces_idx`` / ``static_lmk_bary_coords`` (51-point
    static dlib subset) restricted to the brow + nose + eye range
    (DLIB_STATIC_PNP_RANGE). Lips are excluded because they move with
    speech, defeating per-frame rigid PnP.
    """
    blob = np.load(dlib_path, allow_pickle=True).item()
    static_face_idx = np.asarray(blob['static_lmk_faces_idx']).reshape(-1).astype(np.int64)
    static_bary = np.asarray(blob['static_lmk_bary_coords']).reshape(-1, 3).astype(np.float64)
    if static_face_idx.shape[0] != 51:
        raise ValueError(
            f'expected 51 static dlib landmarks, got {static_face_idx.shape[0]} '
            f'in {dlib_path}')

    # The 51-static array stores indices 17-67 of the full dlib 68; i.e.
    # the static array index 0 corresponds to dlib idx 17. Map back.
    dlib_idx_offset = 17
    lo, hi = DLIB_STATIC_PNP_RANGE
    keep_static_slots = np.arange(lo - dlib_idx_offset, hi - dlib_idx_offset, dtype=np.int64)
    dlib_indices = keep_static_slots + dlib_idx_offset
    flame_face_idx = static_face_idx[keep_static_slots]
    flame_bary = static_bary[keep_static_slots]
    canonical = _reconstruct_xyz(v_template, faces, flame_face_idx, flame_bary)
    rationale = (
        f'dlib static brow+nose+eye = {len(dlib_indices)} pts '
        f'(dlib idx {lo}..{hi-1}; lips/contour excluded)'
    )
    return dlib_indices, flame_face_idx, flame_bary, canonical, rationale


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent

    p = argparse.ArgumentParser()
    p.add_argument(
        '--mica', default=str(repo_root / 'assets/flame_model/mediapipe_landmark_embedding.npz'),
        help='MICA mediapipe_landmark_embedding.npz.',
    )
    p.add_argument(
        '--dlib', default=str(repo_root / 'assets/flame_model/landmark_embedding.npy'),
        help='HRAvatar dlib 68-point FLAME embedding (used for FAN path).',
    )
    p.add_argument(
        '--flame', default=str(repo_root / 'assets/flame_model/flame2020.pkl'),
        help='FLAME 2020 canonical mesh pickle (v_template, f).',
    )
    p.add_argument(
        '--mp-output', default=str(repo_root / 'assets/lhg/mediapipe_flame_landmarks.npz'),
    )
    p.add_argument(
        '--dlib-output', default=str(repo_root / 'assets/lhg/dlib_flame_landmarks.npz'),
    )
    p.add_argument(
        '--mp-subset', default='lhg-pnp', choices=('lhg-pnp', 'stable', 'all-mica'),
        help='MediaPipe subset (default lhg-pnp = 16 frontal anchors).',
    )
    p.add_argument(
        '--detector', default='both', choices=('both', 'mediapipe', 'dlib'),
        help='Which asset(s) to build.',
    )
    args = p.parse_args(argv)

    flame_path = Path(args.flame)
    if not flame_path.is_file():
        sys.exit(f'[error] FLAME pkl not found: {flame_path}')
    v_template, faces = _load_flame(flame_path)

    if args.detector in ('both', 'mediapipe'):
        mica_path = Path(args.mica)
        if not mica_path.is_file():
            sys.exit(f'[error] MICA embedding not found: {mica_path}')
        idx, face_idx, bary, canonical, rationale = _build_mediapipe_asset(
            mica_path, v_template, faces, args.mp_subset,
        )
        out = Path(args.mp_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, mp_indices=idx, flame_face_idx=face_idx,
            flame_bary=bary, flame_canonical_xyz=canonical,
        )
        _print_summary(out, rationale, idx, canonical)

    if args.detector in ('both', 'dlib'):
        dlib_path = Path(args.dlib)
        if not dlib_path.is_file():
            sys.exit(f'[error] dlib embedding not found: {dlib_path}')
        idx, face_idx, bary, canonical, rationale = _build_dlib_asset(
            dlib_path, v_template, faces,
        )
        out = Path(args.dlib_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, mp_indices=idx, flame_face_idx=face_idx,
            flame_bary=bary, flame_canonical_xyz=canonical,
        )
        _print_summary(out, rationale, idx, canonical)
    return 0


def _print_summary(out: Path, rationale: str, idx: np.ndarray, canonical: np.ndarray) -> None:
    print(f'wrote {out}')
    print(f'  mode: {rationale}')
    print(f'  K={len(idx)} landmarks')
    print(f'  indices={idx.tolist()}')
    print(f'  canonical_xyz extent: '
          f'x=[{canonical[:, 0].min():.4f}, {canonical[:, 0].max():.4f}], '
          f'y=[{canonical[:, 1].min():.4f}, {canonical[:, 1].max():.4f}], '
          f'z=[{canonical[:, 2].min():.4f}, {canonical[:, 2].max():.4f}]')
    centered = canonical - canonical.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    print(f'  3D conditioning σ_max/σ_min = {s[0]/s[-1]:.2f}')


if __name__ == '__main__':
    raise SystemExit(main())
