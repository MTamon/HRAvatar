"""Convert ``lhg_features.npz`` into a ``tracked_params.json``-shaped dict.

HRAvatar's renderer (``scene.data_loader.TrackedData``) reads
``tracked_params.json`` from disk and consumes per-frame entries with
the keys::

    expcode       (1, 100)   FLAME expression (DECA + SMIRK fork uses up to
                             100 dims; SMIRK only fills the first 50, the
                             rest are zero-padded)
    fullposecode  (1, 15)    [global_rot(3), neck(3), jaw(3), eye_pose(6)]
    eyelids       (2,)       upper / lower eyelid blend
    translation   (1, 3)     FLAME canonical-space delta around world_mat
    world_mat     (4, 4)     per-frame copy of the clip-constant extrinsics

plus the top-level keys::

    shapecode    (1, 100)    clip-constant FLAME shape parameter
    intrinsics   (4,)        [fx, fy, cx, cy]
    world_mat    (4, 4)      clip-constant extrinsics

This adapter substitutes the per-frame entries with values from
``lhg_features.npz`` while keeping the top-level shape / intrinsics
from a base ``tracked_params.json`` (typically the avatar's training
calibration). The base file's shape is intentionally preserved
because HRAvatar bakes the shape into the avatar at training time
and the renderer ignores any per-frame override.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .output import LHGFeatures


def _resolve_frame_key(basename: str, base: dict) -> str | None:
    """Find the matching key in the base ``tracked_params.json``.

    The base may store keys either as full basenames (``"00042.png"``)
    or as numeric stems (``"00042"``); we accept either.
    """
    if basename in base:
        return basename
    stem = Path(basename).stem
    if stem in base:
        return stem
    return None


def features_to_tracked_params(
    features: LHGFeatures,
    base_tracked_params: dict | None = None,
    avatar_shapecode: list | np.ndarray | None = None,
) -> dict:
    """Build the ``tracked_params.json``-shaped dict the renderer
    expects.

    Parameters
    ----------
    features : the loaded LHGFeatures from ``lhg_features.npz``.
    base_tracked_params : optional dict from a separate avatar
        ``tracked_params.json``. When given, its top-level
        ``shapecode`` is used and per-frame entries are STARTED FROM
        the base (so any extra keys the renderer might consume — e.g.
        ``cam`` from DECA's optimize output — survive).
    avatar_shapecode : alternative way to supply the avatar's
        shapecode without passing the whole tracked_params dict.
        Used by the demo when the base file is unavailable.
    """
    if base_tracked_params is None:
        out: dict = {}
    else:
        out = copy.deepcopy(base_tracked_params)

    # Top-level constants. world_mat / intrinsics come from the LHG
    # features (which baked the Stage 1 calibration). shapecode comes
    # from the avatar (renderer ignores per-frame shape anyway).
    wm_list = features.world_mat.astype(np.float64).tolist()
    out['world_mat'] = wm_list
    out['intrinsics'] = list(map(float, features.intrinsics.tolist()))

    if avatar_shapecode is not None:
        shape = np.asarray(avatar_shapecode, dtype=np.float32).reshape(-1)
        out['shapecode'] = [shape.astype(float).tolist()]
    elif 'shapecode' not in out:
        raise ValueError(
            'features_to_tracked_params: base_tracked_params has no '
            'shapecode and no avatar_shapecode override was provided. '
            'Pass the avatar\'s shapecode explicitly so the renderer '
            'can build the FLAME mesh.')

    # Per-frame entries. We don't know whether the renderer's existing
    # per-frame dict has extra keys (e.g. ``cam`` from DECA), so for
    # each frame we PATCH the existing entry where possible and CREATE
    # a fresh one when the basename is missing.
    n = features.frame_basenames.shape[0]
    expression_dim = features.expression.shape[1]    # 50 for SMIRK
    for i in range(n):
        basename = str(features.frame_basenames[i])
        key = _resolve_frame_key(basename, out) or basename

        if key not in out:
            out[key] = {}

        # expcode: pad SMIRK's 50d expression up to 100d so HRAvatar's
        # default n_expr=100 receives a valid-length vector. The
        # renderer slices [:n_expr] so any padding is harmless.
        expcode_100 = np.zeros(100, dtype=np.float32)
        expcode_100[:expression_dim] = features.expression[i]
        out[key]['expcode'] = [expcode_100.astype(float).tolist()]

        # fullposecode: [global_rot, neck, jaw, eye_pose]. Layout
        # matches HRAvatar's ``GaussianHeadModel`` which slices indices
        # 0:3 → global_rot, 3:6 → neck, 6:9 → jaw, 9:15 → eye_pose
        # (eye_pose = [eye_l(3), eye_r(3)]).
        fullposecode = np.zeros(15, dtype=np.float32)
        fullposecode[0:3] = features.global_rot[i]
        fullposecode[3:6] = features.neck_pose[i]
        fullposecode[6:9] = features.jaw[i]
        fullposecode[9:15] = features.eye_pose[i]
        out[key]['fullposecode'] = [fullposecode.astype(float).tolist()]

        out[key]['eyelids'] = list(map(float, features.eyelid[i].tolist()))
        out[key]['translation'] = [
            list(map(float, features.translation[i].tolist())),
        ]
        out[key]['world_mat'] = wm_list

    return out


def load_lhg_features_as_tracked_params(
    npz_path: str | Path,
    base_tracked_params_path: str | Path | None = None,
) -> dict:
    """Convenience wrapper: load both files and call
    ``features_to_tracked_params``.

    The base file is OPTIONAL. When omitted, the resulting dict has
    no ``shapecode`` and the caller must arrange one (typically by
    pointing the renderer at a base directory whose
    ``tracked_params.json`` does carry shapecode, then calling
    ``features_to_tracked_params(features, base)`` directly).
    """
    npz_path = Path(npz_path)
    features = LHGFeatures.read(npz_path)

    base_dict: dict | None = None
    if base_tracked_params_path is not None:
        base_path = Path(base_tracked_params_path)
        if not base_path.is_file():
            raise FileNotFoundError(
                f'base tracked_params.json not found: {base_path}')
        with open(base_path) as fp:
            base_dict = json.load(fp)

    return features_to_tracked_params(features, base_tracked_params=base_dict)
