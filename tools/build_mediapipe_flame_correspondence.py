"""Build assets/lhg/mediapipe_flame_landmarks.npz from MICA's
``mediapipe_landmark_embedding.npz`` and the FLAME 2020 canonical mesh.

Output format (consumed by ``lhg.correspondence.load``):

* ``mp_indices`` : (K,) int64 — MediaPipe FaceMesh point indices used by
  the LHG EPnP solve. Defaults to the intersection of MICA's 105
  landmarks with ``preprocess._smirk_constants.STABLE_LANDMARK_INDICES``
  so the same stable subset is used for bbox-following AND PnP. With
  ``--all-mica`` switch the full 105 are exported instead (more PnP
  correspondences but includes points that move with expression /
  blinking, which the bbox follower deliberately excludes).

* ``flame_face_idx`` : (K,) int64 — FLAME triangle index per landmark.
* ``flame_bary``     : (K, 3) float64 — barycentric coords within that
  triangle.
* ``flame_canonical_xyz`` : (K, 3) float64 — landmark position on the
  canonical FLAME mesh (shape=0, expression=0, pose=identity), i.e. the
  average head shape. EPnP consumes this when no per-clip
  shape_param-conditioned 3D positions are available.

Why the intersection (not the full MICA set)
--------------------------------------------
The bbox follower uses STABLE_LANDMARK_INDICES because those points do
not move with expression / blinking, which keeps the SMIRK input crop
stable. Using the same subset for PnP means the 2D landmarks fed into
EPnP are the same ones used to compute the crop center — the head pose
solve sees a clean signal that doesn't drift with mouth opening.

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


# Same anchors as preprocess/_smirk_constants.STABLE_LANDMARK_INDICES.
# Hard-coded here so this script is runnable in isolation (no PYTHONPATH
# fiddling) and so a future indices change doesn't silently break the
# already-built asset.
STABLE_LANDMARK_INDICES = np.array(
    [33, 133, 362, 263, 1, 4, 5, 6, 168, 195, 197, 234, 454, 127, 356],
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


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent

    p = argparse.ArgumentParser()
    p.add_argument(
        '--mica', default=str(repo_root / 'assets/flame_model/mediapipe_landmark_embedding.npz'),
        help='MICA mediapipe_landmark_embedding.npz (105 mediapipe→FLAME barycentric pairs).',
    )
    p.add_argument(
        '--flame', default=str(repo_root / 'assets/flame_model/flame2020.pkl'),
        help='FLAME 2020 canonical mesh pickle (v_template, f).',
    )
    p.add_argument(
        '--output', default=str(repo_root / 'assets/lhg/mediapipe_flame_landmarks.npz'),
        help='Path to write the LHG-ready asset.',
    )
    p.add_argument(
        '--all-mica', action='store_true',
        help='Export the full 105 MICA correspondences instead of the '
             'STABLE_LANDMARK_INDICES intersection. Larger PnP support set '
             'but the extra points are not jitter-resistant.',
    )
    args = p.parse_args(argv)

    mica_path = Path(args.mica)
    flame_path = Path(args.flame)
    out_path = Path(args.output)

    if not mica_path.is_file():
        sys.exit(f'[error] MICA embedding not found: {mica_path}')
    if not flame_path.is_file():
        sys.exit(f'[error] FLAME pkl not found: {flame_path}')

    with np.load(mica_path, allow_pickle=False) as npz:
        mica_lmk_idx = npz['landmark_indices'].astype(np.int64)            # (105,)
        mica_face_idx = npz['lmk_face_idx'].astype(np.int64)               # (105,)
        mica_bary = npz['lmk_b_coords'].astype(np.float64)                 # (105, 3)

    with open(flame_path, 'rb') as fp:
        flame_pkl = pickle.load(fp, encoding='latin1')
    v_template = np.asarray(flame_pkl['v_template'], dtype=np.float64)     # (5023, 3)
    faces = np.asarray(flame_pkl['f'], dtype=np.int64)                      # (9976, 3)

    if args.all_mica:
        mp_indices = mica_lmk_idx
        flame_face_idx = mica_face_idx
        flame_bary = mica_bary
        rationale = 'full MICA 105'
    else:
        # Intersect with STABLE_LANDMARK_INDICES (preserves order of
        # STABLE_LANDMARK_INDICES so callers get a deterministic ordering).
        mica_pos = {int(idx): i for i, idx in enumerate(mica_lmk_idx)}
        keep_mp, keep_face, keep_bary = [], [], []
        for stable_idx in STABLE_LANDMARK_INDICES:
            i = mica_pos.get(int(stable_idx))
            if i is None:
                continue
            keep_mp.append(int(stable_idx))
            keep_face.append(int(mica_face_idx[i]))
            keep_bary.append(mica_bary[i])
        if len(keep_mp) < 4:
            sys.exit(
                f'[error] only {len(keep_mp)} STABLE_LANDMARK_INDICES are '
                f'covered by MICA — too few for EPnP (needs >=4). Either '
                f're-derive the missing barycentric pairs from FLAME canonical '
                f'mesh or rerun with --all-mica.')
        mp_indices = np.asarray(keep_mp, dtype=np.int64)
        flame_face_idx = np.asarray(keep_face, dtype=np.int64)
        flame_bary = np.stack(keep_bary, axis=0)
        rationale = f'STABLE_LANDMARK_INDICES ∩ MICA = {len(keep_mp)} of {len(STABLE_LANDMARK_INDICES)}'

    canonical = _reconstruct_xyz(v_template, faces, flame_face_idx, flame_bary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        mp_indices=mp_indices,
        flame_face_idx=flame_face_idx,
        flame_bary=flame_bary,
        flame_canonical_xyz=canonical,
    )
    print(f'wrote {out_path}')
    print(f'  mode: {rationale}')
    print(f'  K={len(mp_indices)} landmarks')
    print(f'  mp_indices={mp_indices.tolist()}')
    print(f'  canonical_xyz extent: '
          f'x=[{canonical[:, 0].min():.4f}, {canonical[:, 0].max():.4f}], '
          f'y=[{canonical[:, 1].min():.4f}, {canonical[:, 1].max():.4f}], '
          f'z=[{canonical[:, 2].min():.4f}, {canonical[:, 2].max():.4f}]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
