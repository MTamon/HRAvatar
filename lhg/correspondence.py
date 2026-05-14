"""Loader for the MediaPipe→FLAME 3D landmark correspondence asset.

The EPnP solve pairs the stable subset of MediaPipe FaceMesh landmarks
(``preprocess._smirk_constants.STABLE_LANDMARK_INDICES``) with their
positions on the FLAME canonical mesh. The mapping is precomputed and
shipped as ``assets/lhg/mediapipe_flame_landmarks.npz`` to avoid a
runtime dependency on the FLAME canonical mesh + Procrustes derivation.

Format
------
The npz holds three arrays:

* ``mp_indices`` : (K,) int64 — MediaPipe landmark indices (subset of
  the 478-point FaceMesh; MUST be a subset of the same stable indices
  used by ``preprocess/stable_bbox.py`` so the 2D and 3D sides line up).
* ``flame_face_idx`` : (K,) int64 — FLAME face (triangle) index
  containing the corresponding point.
* ``flame_bary`` : (K, 3) float64 — barycentric coordinates within that
  face.

Given a per-frame FLAME vertex tensor ``V`` of shape (V, 3), the 3D
position of MediaPipe landmark ``mp_indices[k]`` is::

    p_k = bary[k, 0] * V[F[face_idx[k], 0]]
        + bary[k, 1] * V[F[face_idx[k], 1]]
        + bary[k, 2] * V[F[face_idx[k], 2]]

where ``F`` is the FLAME face index array (``flame.faces_tensor``).

For a *neutral* (shape_param-independent) approximation that is good
enough for EPnP, the loader can also expose a precomputed
``flame_canonical_xyz`` array of shape (K, 3) — the position of each
landmark on the canonical FLAME mesh with shape_param=0. This is what
EPnP consumes when no per-clip shape_param is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ASSET = Path('./assets/lhg/mediapipe_flame_landmarks.npz')
DEFAULT_FAN_ASSET = Path('./assets/lhg/dlib_flame_landmarks.npz')


def default_asset_for(detector_type: str) -> Path:
    if detector_type == 'fan':
        return DEFAULT_FAN_ASSET
    if detector_type == 'mediapipe':
        return DEFAULT_ASSET
    raise ValueError(
        f'unknown detector_type {detector_type!r}; expected "fan" or "mediapipe"')


@dataclass
class MediaPipeFLAMECorrespondence:
    mp_indices: np.ndarray            # (K,) int64
    flame_face_idx: np.ndarray        # (K,) int64
    flame_bary: np.ndarray            # (K, 3) float64
    flame_canonical_xyz: np.ndarray | None  # (K, 3) float64 or None

    @property
    def num_points(self) -> int:
        return int(self.mp_indices.shape[0])

    def reconstruct_xyz(
        self,
        flame_vertices: np.ndarray,
        flame_faces: np.ndarray,
    ) -> np.ndarray:
        """Recover 3D landmark positions for a given vertex tensor.

        Parameters
        ----------
        flame_vertices : (V, 3) FLAME vertices in canonical space.
        flame_faces    : (F, 3) FLAME triangle vertex-index array.
        """
        tri = flame_faces[self.flame_face_idx]                # (K, 3)
        v0 = flame_vertices[tri[:, 0]]
        v1 = flame_vertices[tri[:, 1]]
        v2 = flame_vertices[tri[:, 2]]
        b = self.flame_bary
        return (b[:, 0:1] * v0 + b[:, 1:2] * v1 + b[:, 2:3] * v2)


def load(path: str | Path = DEFAULT_ASSET) -> MediaPipeFLAMECorrespondence:
    """Load the MediaPipe→FLAME correspondence asset.

    Raises a clear error when the asset is missing so callers know to
    run the asset preparation step. The asset itself is built either
    by porting MICA's
    ``mediapipe_landmark_embedding.npz`` (preferred) or by deriving the
    barycentric mapping from the FLAME canonical mesh (see
    ``tools/build_mediapipe_flame_correspondence.py`` once available).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'MediaPipe→FLAME correspondence asset not found: {path}\n'
            f'Build it via tools/build_mediapipe_flame_correspondence.py '
            f'or copy MICA\'s mediapipe_landmark_embedding.npz and '
            f're-export with the LHG stable subset; see lhg/correspondence.py '
            f'for the expected format.')
    with np.load(path) as npz:
        canonical = npz['flame_canonical_xyz'] if 'flame_canonical_xyz' in npz.files else None
        return MediaPipeFLAMECorrespondence(
            mp_indices=npz['mp_indices'].astype(np.int64),
            flame_face_idx=npz['flame_face_idx'].astype(np.int64),
            flame_bary=npz['flame_bary'].astype(np.float64),
            flame_canonical_xyz=(
                canonical.astype(np.float64) if canonical is not None else None
            ),
        )


DEFAULT_FLAME_MODEL_PATH = Path('./assets/FLAME2020/generic_model.pkl')


def build_shape_aware_landmarks(
    correspondence: MediaPipeFLAMECorrespondence,
    shapecode: np.ndarray,
    flame_model_path: str | Path = DEFAULT_FLAME_MODEL_PATH,
) -> np.ndarray:
    """Compute the per-subject 3D landmark positions from a Stage 1 shapecode.

    EPnP fits a fixed 3D template to the observed 2D points to recover
    pose + depth. If the template is the FLAME *neutral* mesh but the
    subject's true mesh is e.g. 3x narrower (large shape coefficients),
    EPnP compensates by pushing the camera ~3x farther away, producing
    a systematically biased ``tvec``. Applying the Stage 1 shapecode
    here removes that bias.

    The computation evaluates a single FLAME forward pass at
    ``expression=0`` and ``pose=0`` (so the result is invariant to
    per-frame motion), then maps the per-vertex tensor through the
    landmark barycentric mapping. pose blendshape and LBS contributions
    are zero in this canonical state, so plain ``v_template +
    shapedirs @ shapecode`` is exact.
    """
    import pickle

    path = Path(flame_model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f'FLAME model not found: {path}. Required for shape-aware EPnP. '
            f'Place ``generic_model.pkl`` from FLAME 2020 release at this '
            f'path, or pass ``flame_model_path`` explicitly.')
    with open(path, 'rb') as f:
        fm = pickle.load(f, encoding='latin1')

    def _to_np(x):
        return np.asarray(x.r) if hasattr(x, 'r') else np.asarray(x)

    v_template = _to_np(fm['v_template'])           # (V, 3)
    shapedirs = _to_np(fm['shapedirs'])             # (V, 3, 300+100)
    faces = _to_np(fm['f']).astype(np.int64)        # (F, 3)

    shape_coeffs = np.asarray(shapecode, dtype=np.float64).reshape(-1)
    n_shape = shape_coeffs.size
    if n_shape > shapedirs.shape[2]:
        raise ValueError(
            f'shapecode length {n_shape} exceeds shapedirs capacity '
            f'{shapedirs.shape[2]}')
    shape_blend = np.einsum('vsd,d->vs', shapedirs[:, :, :n_shape], shape_coeffs)
    verts = (v_template + shape_blend).astype(np.float64)

    tri = faces[correspondence.flame_face_idx]      # (K, 3)
    v0 = verts[tri[:, 0]]
    v1 = verts[tri[:, 1]]
    v2 = verts[tri[:, 2]]
    b = correspondence.flame_bary
    return (b[:, 0:1] * v0 + b[:, 1:2] * v1 + b[:, 2:3] * v2).astype(np.float64)
