"""Loader for the Stage 1 calibration artifact (``tracked_params.json``).

Stage 1 (``demos/_preprocess_subject.sh --lhg-only``) runs DECA's
clip-wide joint optimization on a one-time calibration clip and
writes ``tracked_params.json``. Stage 2 (``lhg/extract.py``) consumes
THREE clip-constants from this file:

* ``world_mat`` (4x4): the camera extrinsics. Per-frame ``global_rot``
  and ``translation`` are emitted as DELTAS around this reference, so
  the downstream LHG model sees only motion (not absolute pose).
* ``shapecode`` (100,): FLAME shape parameters baked at training time.
  The renderer ignores any per-frame override so this is the
  authoritative shape used downstream.
* ``intrinsics`` (4,): ``[fx, fy, cx, cy]`` pinned to the outer-cropped
  ``image_size`` square. Stage 2 overrides ``LHGConfig.intrinsics``
  with these values, making the EPnP solve consistent with whatever
  intrinsics preset was used to bake ``world_mat``.

Per-frame entries (``expcode`` / ``fullposecode`` / ``translation``)
are deliberately ignored by Stage 2 — Stage 2's per-frame estimates
are NEW from MediaPipe + SMIRK + EPnP and are NOT to be replaced by
the avatar-fit's joint-optimized values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StageOneCalibration:
    """Clip-constants extracted from a Stage 1 ``tracked_params.json``."""

    world_mat: np.ndarray             # (4, 4) float32
    shapecode: np.ndarray             # (100,) float32 (or 1, 100 from raw json)
    intrinsics: np.ndarray            # (4,) float32  [fx, fy, cx, cy]
    flame_scale: float                # 4.0 (v1) or 1.0 (v2)
    source_path: Path
    # Phase B-2 baseline: DECA encoder cam (orthographic) clip-mean
    # reference values, used by the online deca_encoder backend (Phase
    # B Stage 2) to lift per-frame translation off zero via a cam→z
    # proxy. None when the calibration has no per-frame DECA cam and
    # no top-level lhg_baseline (e.g. legacy tracked_params.json that
    # predates Phase B-2).
    deca_cam_scale_ref: float | None = None
    deca_cam_tx_ref: float | None = None
    deca_cam_ty_ref: float | None = None

    def intrinsics_tuple(self) -> tuple[float, float, float, float]:
        fx, fy, cx, cy = self.intrinsics.tolist()
        return float(fx), float(fy), float(cx), float(cy)


def _resolve_world_mat(payload: dict, path: Path) -> np.ndarray:
    """Resolve ``world_mat`` regardless of which schema variant the
    avatar-fit json used.

    Three known layouts:
    * top-level ``world_mat`` (legacy demo data)
    * ``frames[0].world_mat`` (DECA optimize.py with ``--frames`` array)
    * per-image-key dicts with ``world_mat`` (current default)
    """
    if 'world_mat' in payload:
        wm = np.asarray(payload['world_mat'], dtype=np.float64)
    elif 'frames' in payload and payload['frames']:
        wm = np.asarray(payload['frames'][0]['world_mat'], dtype=np.float64)
    else:
        frame_keys = [
            k for k in payload
            if isinstance(k, str) and (
                k.endswith('.png') or k.endswith('.jpg') or k.endswith('.bmp')
            )
        ]
        if not frame_keys:
            # Fall back: any value that itself looks like a per-frame entry.
            frame_keys = [
                k for k, v in payload.items()
                if isinstance(v, dict) and 'world_mat' in v
            ]
        if not frame_keys:
            raise RuntimeError(
                f'world_mat not found in {path}: expected world_mat at the '
                f'top level, frames[0].world_mat, or per-image-key dicts')
        wm = np.asarray(payload[frame_keys[0]]['world_mat'], dtype=np.float64)
    if wm.shape == (3, 4):
        wm = np.vstack([wm, [0.0, 0.0, 0.0, 1.0]])
    if wm.shape != (4, 4):
        raise ValueError(
            f'world_mat in {path} has shape {wm.shape}; expected (3,4) or (4,4)')
    return wm.astype(np.float64)


def _resolve_intrinsics(payload: dict, path: Path) -> np.ndarray:
    """Resolve ``intrinsics`` (``[fx, fy, cx, cy]``).

    Either at the top level (current DECA optimize.py output) or, in
    rare legacy variants, embedded inside the first per-frame entry.
    """
    if 'intrinsics' in payload:
        intr = payload['intrinsics']
    else:
        for v in payload.values():
            if isinstance(v, dict) and 'intrinsics' in v:
                intr = v['intrinsics']
                break
        else:
            raise RuntimeError(
                f'intrinsics not found in {path}: expected top-level '
                f'"intrinsics" array of length 4')
    arr = np.asarray(intr, dtype=np.float32).reshape(-1)
    if arr.size != 4:
        raise ValueError(
            f'intrinsics in {path} has length {arr.size}; expected 4 '
            f'([fx, fy, cx, cy])')
    return arr


def _resolve_shapecode(payload: dict, path: Path) -> np.ndarray:
    """Resolve ``shapecode``. Stored at the top level by DECA optimize.py
    (clip-constant). Returns a flat ``(D,)`` float32 array even if the
    json holds ``(1, D)``.
    """
    if 'shapecode' not in payload:
        raise RuntimeError(
            f'shapecode not found at the top level of {path}; '
            f'this should be the clip-constant FLAME shape parameter '
            f'baked by DECA optimize.py.')
    arr = np.asarray(payload['shapecode'], dtype=np.float32)
    return arr.reshape(-1)


def _resolve_lhg_baseline(payload: dict) -> tuple[float | None, float | None, float | None]:
    """Resolve the DECA cam clip-mean reference (Phase B-2).

    Three layouts are accepted, in order of preference:
    1. top-level ``lhg_baseline`` dict (written by the Phase B-2 patch
       to DECA optimize.py)
    2. per-frame ``cam`` arrays — compute the clip mean on the fly
       (back-compat for tracked_params.json files generated before
       the Phase B-2 patch landed)
    3. nothing — returns ``(None, None, None)`` so the online
       deca_encoder backend gracefully falls back to its rotation-only
       Stage 1 behaviour.
    """
    if 'lhg_baseline' in payload and isinstance(payload['lhg_baseline'], dict):
        d = payload['lhg_baseline']
        if 'deca_cam_scale_ref' in d:
            return (
                float(d.get('deca_cam_scale_ref')),
                float(d.get('deca_cam_tx_ref', 0.0)),
                float(d.get('deca_cam_ty_ref', 0.0)),
            )

    cams = []
    for v in payload.values():
        if isinstance(v, dict) and 'cam' in v:
            try:
                arr = np.asarray(v['cam'], dtype=np.float64).reshape(-1)
                if arr.size >= 3:
                    cams.append(arr[:3])
            except Exception:
                continue
    if cams:
        cam_arr = np.stack(cams, axis=0)
        return (
            float(cam_arr[:, 0].mean()),
            float(cam_arr[:, 1].mean()),
            float(cam_arr[:, 2].mean()),
        )
    return (None, None, None)


def load_stage_one_calibration(path: str | Path) -> StageOneCalibration:
    """Read the Stage 1 ``tracked_params.json`` (or v2 sibling) and
    extract the clip-constants Stage 2 consumes.

    The ``flame_scale`` is inferred from the file name (``_v2`` ->
    1.0; otherwise 4.0), matching ``scene/data_loader.py``.

    Additionally resolves the optional Phase B-2 baseline (DECA cam
    clip-mean reference) used by the online deca_encoder backend.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Stage 1 calibration not found: {p}')
    with open(p) as fp:
        payload = json.load(fp)
    flame_scale = 1.0 if p.stem.endswith('_v2') else 4.0
    cam_s_ref, cam_tx_ref, cam_ty_ref = _resolve_lhg_baseline(payload)
    return StageOneCalibration(
        world_mat=_resolve_world_mat(payload, p).astype(np.float32),
        shapecode=_resolve_shapecode(payload, p),
        intrinsics=_resolve_intrinsics(payload, p),
        flame_scale=flame_scale,
        source_path=p,
        deca_cam_scale_ref=cam_s_ref,
        deca_cam_tx_ref=cam_tx_ref,
        deca_cam_ty_ref=cam_ty_ref,
    )
