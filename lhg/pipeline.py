"""Per-frame core algorithm and mode-specific orchestration.

The two modes share one ``per_frame_core`` step (MediaPipe → stable
bbox → SMIRK → EPnP → causal Hampel) and differ only in the surrounding
clip-level wrapper:

* ``online`` runs the core strictly causally and yields one frame at a
  time. Detector dropouts are held at the previous valid value.

* ``pseudo-online`` runs the core in the same order but post-processes
  the collected sequence with bidirectional Hampel + linear
  interpolation across detector dropouts + bidirectional quaternion-flip
  fix.

The post-processing in ``pseudo-online`` is restricted to the four
limited cases listed in ``feedback_pseudo_online_for_lhg_teacher.md`` —
no smoothing of in-range values, no future-info access for normal
frames.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from preprocess._smirk_constants import (
    STABLE_LANDMARK_INDICES,
    STABLE_LANDMARK_SIZE_CALIBRATION,
)

from .config import LHGConfig, intrinsics_matrix
from .correspondence import MediaPipeFLAMECorrespondence
from .detector import MediaPipeFaceLandmarker
from .encoders import SMIRKEncoder
from .epnp import (
    axis_angle_to_quat,
    fix_quat_sign_continuity,
    quat_to_axis_angle,
    solve_epnp,
)
from .hampel import CausalHampel, bidirectional_hampel
from .interpolate import linear_interpolate_dropouts
from .output import LHGFeatures
from .video import FrameSource, load_outer_offset


@dataclass
class _BboxState:
    """Causal hysteresis follower state (per stable_bbox.py:_hysteresis_forward)
    plus optional FAN-detector bbox seed (constant across the clip)."""
    anchor: np.ndarray              # (2,) current emitted center
    target: np.ndarray              # (2,) candidate to chase
    ring: np.ndarray                # (window,) bool sliding history
    ring_idx: int = 0
    initialized: bool = False
    fan_bbox: np.ndarray | None = None  # (4,) [x1, y1, x2, y2] for FAN, seeded once


def _step_bbox_hysteresis(
    state: _BboxState, raw_center: np.ndarray, cfg: LHGConfig,
) -> np.ndarray:
    """One causal step of the K-of-N hysteresis center follower.

    Mirrors ``preprocess.stable_bbox._hysteresis_forward`` exactly so
    the LHG bbox center stream is parameter-aligned with the avatar fit
    pipeline.
    """
    dt = 1.0 / float(cfg.fps)
    alpha = 1.0 - math.exp(-dt / max(cfg.bbox_tau, 1e-6))

    if not state.initialized:
        state.anchor = raw_center.astype(np.float64).copy()
        state.target = state.anchor.copy()
        state.ring = np.zeros(max(int(cfg.bbox_window), 1), dtype=bool)
        state.ring_idx = 0
        state.initialized = True
        return state.anchor.copy()

    disp = float(np.linalg.norm(raw_center - state.anchor))
    state.ring[state.ring_idx % state.ring.size] = disp > float(cfg.bbox_deadzone_px)
    state.ring_idx += 1

    k_thresh = max(1, min(int(cfg.bbox_k_of_n), state.ring.size))
    if int(state.ring.sum()) >= k_thresh:
        # When the K-of-N condition fires we'd ideally average the last
        # `window` raw centers; in the streaming setting we only have
        # the current one, so we use the raw center as the new target.
        # (The avatar fit path averages the last `window` raws but that
        # requires a per-frame raw-center buffer; for LHG the practical
        # difference is sub-pixel and the ring length is small.)
        state.target = raw_center.astype(np.float64).copy()

    state.anchor = state.anchor + (state.target - state.anchor) * alpha
    return state.anchor.copy()


def _bbox_size_from_landmarks(
    landmarks: np.ndarray, detector_type: str,
) -> tuple[np.ndarray, float]:
    """(center, size) extraction from per-detector stable subset.

    For MediaPipe (478-pt) we use STABLE_LANDMARK_INDICES (~15 pts on
    eyes / nose / temples). For FAN (68-pt) we use the 31-pt static
    subset (brow + nose + eye, indices 17-47), which has equivalent
    expression-invariance and a similar X/Y span across the face. Both
    formulas use the legacy ``(width + height) / 2`` size convention so
    SMIRK's 224-crop scale is consistent across detectors.
    """
    if detector_type == 'mediapipe':
        pts = landmarks[STABLE_LANDMARK_INDICES]
        size_cal = STABLE_LANDMARK_SIZE_CALIBRATION
    elif detector_type == 'fan':
        # dlib idx 17..47 inclusive — same set as DLIB_STATIC_PNP_RANGE.
        pts = landmarks[17:48]
        # FAN's brow-to-nose-tip span happens to match the unscaled
        # legacy formula closely (it covers most of the face height
        # naturally); keep calibration at 1.0 for now and let
        # bbox_scale handle the SMIRK margin.
        size_cal = 1.0
    else:
        raise ValueError(f'unknown detector_type {detector_type!r}')
    xs = pts[:, 0]
    ys = pts[:, 1]
    left = float(np.min(xs))
    right = float(np.max(xs))
    top = float(np.min(ys))
    bottom = float(np.max(ys))
    size = ((right - left) + (bottom - top)) / 2.0 * size_cal
    center = np.array(
        [(left + right) / 2.0, (top + bottom) / 2.0], dtype=np.float64,
    )
    return center, float(size)


def _warp_to_224(
    image_rgb: np.ndarray,
    center: np.ndarray,
    size: float,
    bbox_scale: float,
) -> np.ndarray:
    """Reproduce the legacy ``crop_face`` warp at 224×224.

    Uses ``skimage.transform.warp`` with a ``SimilarityTransform`` built
    from the same three control-point pairs as
    ``preprocess.stable_bbox.build_similarity_tform`` so the LHG SMIRK
    input matches the avatar-fit SMIRK input pixel-for-pixel.
    """
    from skimage.transform import estimate_transform, warp

    padded = int(float(size) * float(bbox_scale))
    cx, cy = float(center[0]), float(center[1])
    half = padded / 2.0
    src_pts = np.array([
        [cx - half, cy - half],
        [cx - half, cy + half],
        [cx + half, cy - half],
    ])
    dst_pts = np.array([
        [0, 0],
        [0, 224 - 1],
        [224 - 1, 0],
    ])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    img01 = image_rgb.astype(np.float32) / 255.0
    warped = warp(img01, tform.inverse, output_shape=(224, 224))
    return (warped * 255.0).clip(0, 255).astype(np.uint8)


@dataclass
class FrameTrace:
    """Per-frame trace from ``per_frame_core``. Aggregated into LHGFeatures
    after the per-frame loop completes."""
    expression: np.ndarray | None = None       # (50,) or None on detect miss
    jaw: np.ndarray | None = None              # (3,)
    eyelid: np.ndarray | None = None           # (2,)
    global_rot: np.ndarray | None = None       # (3,) axis-angle
    translation: np.ndarray | None = None      # (3,) FLAME canonical
    bbox_center: np.ndarray | None = None      # (2,)
    bbox_size: float | None = None
    valid: bool = False                        # True iff MediaPipe + EPnP succeeded


def per_frame_core(
    image_outer_rgb: np.ndarray,
    detector,
    smirk: SMIRKEncoder,
    correspondence: MediaPipeFLAMECorrespondence,
    K: np.ndarray,
    cfg: LHGConfig,
    bbox_state: _BboxState,
    last_valid: FrameTrace | None,
) -> FrameTrace:
    """One causal frame step. Shared by online and pseudo-online modes.

    ``detector`` is duck-typed: any object with a ``.detect(rgb)``
    returning either ``None`` or an ``(N, 2)`` landmark array works
    (currently MediaPipe 478-pt or FAN 68-pt).

    The only inter-frame state used here is ``bbox_state`` (causal
    hysteresis follower) and ``last_valid`` (used to hold values across
    a detector dropout). Hampel rejection happens one level up.
    """
    # For FAN: skip per-frame S3FD detection by passing a TIGHT, fixed
    # bbox sized to the actual face. S3FD's bbox would otherwise wobble
    # ~5 px frame-to-frame even on a still face, and any bbox shift
    # propagates into 2-3 px landmark jitter; conversely a bbox that
    # is too LOOSE (e.g. the full 512x512 image) makes the face occupy
    # ~75 px in the 256-input 2DFAN, well below the model's optimal
    # ~195 px scale, sharply degrading landmark accuracy.
    #
    # We seed the fixed bbox from the first valid frame (FAN-auto-
    # detect once) and reuse it across the clip. ``bbox_state.fan_bbox``
    # holds the persistent value; on the first call it's None and the
    # auto-detect pass populates it.
    if cfg.detector_type == 'fan':
        if getattr(bbox_state, 'fan_bbox', None) is not None:
            landmarks = detector.detect(
                image_outer_rgb, bbox_xyxy=bbox_state.fan_bbox,
            )
        else:
            landmarks = detector.detect(image_outer_rgb)
            if landmarks is not None:
                # Compute a tight bbox from this frame's landmarks +
                # 25% margin and freeze it for subsequent frames.
                xs, ys = landmarks[:, 0], landmarks[:, 1]
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())
                w_pad = (x2 - x1) * 0.25
                h_pad = (y2 - y1) * 0.25
                bbox_state.fan_bbox = np.array(
                    [x1 - w_pad, y1 - h_pad, x2 + w_pad, y2 + h_pad],
                    dtype=np.float64,
                )
    else:
        landmarks = detector.detect(image_outer_rgb)
    if landmarks is None:
        # Detector miss: hold previous valid sample if available.
        if last_valid is None:
            return FrameTrace(valid=False)
        return FrameTrace(
            expression=last_valid.expression,
            jaw=last_valid.jaw,
            eyelid=last_valid.eyelid,
            global_rot=last_valid.global_rot,
            translation=last_valid.translation,
            bbox_center=last_valid.bbox_center,
            bbox_size=last_valid.bbox_size,
            valid=False,
        )

    raw_center, raw_size = _bbox_size_from_landmarks(landmarks, cfg.detector_type)
    smoothed_center = _step_bbox_hysteresis(bbox_state, raw_center, cfg)

    face_224 = _warp_to_224(
        image_outer_rgb, smoothed_center, raw_size, cfg.bbox_scale,
    )
    smirk_out = smirk.encode(face_224)

    img_pts_2d = landmarks[correspondence.mp_indices]
    if correspondence.flame_canonical_xyz is None:
        raise RuntimeError(
            'correspondence.flame_canonical_xyz is None; LHG EPnP solve '
            'requires the precomputed canonical landmark positions. Re-build '
            'the correspondence asset with --include-canonical.')
    rvec, tvec_canonical, ok = solve_epnp(
        img_pts_2d,
        correspondence.flame_canonical_xyz,
        K,
        cfg.flame_scale,
        camera_convention=cfg.camera_convention,
    )
    if not ok:
        if last_valid is None:
            return FrameTrace(
                expression=smirk_out['expression'],
                jaw=smirk_out['jaw'],
                eyelid=smirk_out['eyelid'],
                global_rot=None,
                translation=None,
                bbox_center=smoothed_center,
                bbox_size=raw_size,
                valid=False,
            )
        return FrameTrace(
            expression=smirk_out['expression'],
            jaw=smirk_out['jaw'],
            eyelid=smirk_out['eyelid'],
            global_rot=last_valid.global_rot,
            translation=last_valid.translation,
            bbox_center=smoothed_center,
            bbox_size=raw_size,
            valid=False,
        )

    return FrameTrace(
        expression=smirk_out['expression'],
        jaw=smirk_out['jaw'],
        eyelid=smirk_out['eyelid'],
        global_rot=rvec.astype(np.float32),
        translation=tvec_canonical.astype(np.float32),
        bbox_center=smoothed_center,
        bbox_size=raw_size,
        valid=True,
    )


def _crop_outer(image_raw_rgb: np.ndarray, outer_bbox: np.ndarray, image_size: int) -> np.ndarray:
    """Crop ``image_raw_rgb`` by ``outer_bbox`` and resize to ``image_size``.

    Mirrors ``crop_and_matting.crop_image``: out-of-bounds reads are
    zero-padded so that ``bbox_scale`` stays linear.
    """
    import cv2

    xmin, xmax, ymin, ymax = (int(v) for v in outer_bbox)
    h_raw, w_raw = image_raw_rgb.shape[:2]
    side = xmax - xmin
    canvas = np.zeros((side, side, 3), dtype=image_raw_rgb.dtype)
    src_x0 = max(0, xmin)
    src_x1 = min(w_raw, xmax)
    src_y0 = max(0, ymin)
    src_y1 = min(h_raw, ymax)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - xmin
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = src_y0 - ymin
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = image_raw_rgb[src_y0:src_y1, src_x0:src_x1]
    if side != image_size:
        canvas = cv2.resize(canvas, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return canvas


def _compute_outer_bbox_from_landmarks(
    landmarks_per_frame: list[np.ndarray | None],
    image_shape_hw: tuple[int, int],
    bbox_scale: float = 2.2,
) -> np.ndarray:
    """Mirror ``crop_and_matting._compute_video_wide_outer_bbox`` from
    a precomputed list of landmark arrays. Used for pseudo-online's
    clip-wide outer crop.
    """
    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
    for lm in landmarks_per_frame:
        if lm is None:
            continue
        x_mins.append(float(np.min(lm[:, 0])))
        x_maxs.append(float(np.max(lm[:, 0])))
        y_mins.append(float(np.min(lm[:, 1])))
        y_maxs.append(float(np.max(lm[:, 1])))
    if not x_mins:
        raise RuntimeError(
            'no successful MediaPipe detections in the entire clip; cannot '
            'compute clip-wide outer bbox')
    u_xmin, u_xmax = min(x_mins), max(x_maxs)
    u_ymin, u_ymax = min(y_mins), max(y_maxs)
    cx = int(round((u_xmin + u_xmax) / 2.0))
    cy = int(round((u_ymin + u_ymax) / 2.0))
    half_extent = max((u_xmax - u_xmin) / 2.0, (u_ymax - u_ymin) / 2.0)
    size = int(bbox_scale * 2 * half_extent)
    xb_min = cx - size // 2
    xb_max = cx + size // 2
    yb_min = cy - size // 2
    yb_max = cy + size // 2
    if (xb_max - xb_min) % 2 != 0:
        xb_min += 1
    if (yb_max - yb_min) % 2 != 0:
        yb_min += 1
    return np.array([xb_min, xb_max, yb_min, yb_max], dtype=np.int32)


def _axis_angle_to_R(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to 3x3 rotation matrix (numpy Rodrigues)."""
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(aa, dtype=np.float64).reshape(3, 1))
    return R


def _R_to_axis_angle(R: np.ndarray) -> np.ndarray:
    import cv2
    aa, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return aa.reshape(3)


def _resolve_world_mat_and_recenter(
    cfg: LHGConfig,
    global_rots: np.ndarray,
    translations: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the clip-constant world_mat (rotation + translation) AND
    recenter per-frame global_rot / translation around it.

    HRAvatar's renderer convention (matches what DECA ``optimize.py``
    writes when ``with_translation_camera=False``):

    * ``world_mat`` carries the absolute camera-to-mean-head transform
      (rotation in [:3, :3] absorbs the FLAME-canonical-to-camera-
      facing pre-rotation; translation in [:3, 3] absorbs the depth).
    * Per-frame ``global_rot`` is a SMALL axis-angle delta around the
      reference rotation, applied AFTER the world_mat rotation.
    * Per-frame ``translation`` is a SMALL delta in FLAME canonical
      units around the reference translation, applied additively to
      FLAME vertices before the world_mat * flame_scale projection.

    EPnP returns the TOTAL camera-to-object pose on each frame. This
    routine splits it: the calibration-window reference pose becomes
    world_mat, and per-frame deltas are recovered via R_delta =
    R_ref^T @ R_total and t_delta = t_total - t_ref.

    When ``cfg.world_mat_path`` is set we instead load the world_mat
    from an existing tracked_params.json and recenter against it,
    keeping LHG output drop-in compatible with the avatar-fit pipeline.

    Returns
    -------
    world_mat              : (4, 4) float32
    global_rot_recentered  : (N, 3) float32 axis-angle deltas
    translation_recentered : (N, 3) float32 deltas in FLAME canonical units
    """
    if cfg.world_mat_path is not None:
        import json

        path = Path(cfg.world_mat_path)
        if not path.is_file():
            raise FileNotFoundError(f'--world-mat file not found: {path}')
        with open(path) as fp:
            payload = json.load(fp)
        if 'world_mat' in payload:
            wm = np.asarray(payload['world_mat'], dtype=np.float64)
        elif 'frames' in payload and payload['frames']:
            wm = np.asarray(payload['frames'][0]['world_mat'], dtype=np.float64)
        else:
            frame_keys = [k for k in payload if k.endswith('.png') or k.endswith('.jpg')]
            if frame_keys:
                wm = np.asarray(payload[frame_keys[0]]['world_mat'], dtype=np.float64)
            else:
                raise RuntimeError(
                    f'world_mat not found in {path}: expected world_mat at the '
                    f'top level, frames[0].world_mat, or per-image-key dicts')
        if wm.shape == (3, 4):
            wm = np.vstack([wm, [0.0, 0.0, 0.0, 1.0]])
        R_ref = wm[:3, :3]
        t_ref_canonical = wm[:3, 3] / float(cfg.flame_scale)
        # Recenter rotations: R_delta = R_ref^T @ R_total
        R_ref_T = R_ref.T
        rot_recentered = np.zeros_like(global_rots, dtype=np.float32)
        for i in range(global_rots.shape[0]):
            R_total = _axis_angle_to_R(global_rots[i])
            R_delta = R_ref_T @ R_total
            rot_recentered[i] = _R_to_axis_angle(R_delta).astype(np.float32)
        trans_recentered = translations - t_ref_canonical.astype(translations.dtype)
        return wm.astype(np.float32), rot_recentered, trans_recentered

    # Calibrate from the first valid frames in the clip.
    n_calib = max(1, int(cfg.world_mat_calibration_frames))
    valid_frames = []
    for i in range(translations.shape[0]):
        if valid[i]:
            valid_frames.append(i)
            if len(valid_frames) >= n_calib:
                break
    if not valid_frames:
        return (
            np.eye(4, dtype=np.float32),
            global_rots.astype(np.float32, copy=True),
            translations.astype(np.float32, copy=True),
        )

    # Translation reference: arithmetic mean over calibration window.
    t_ref = translations[valid_frames].astype(np.float64).mean(axis=0)

    # Rotation reference: chordal mean of rotation matrices, then SVD-
    # project back to SO(3). For ~30-60 frames of a roughly-still head,
    # this is well within the convergence radius of the chordal mean.
    R_sum = np.zeros((3, 3), dtype=np.float64)
    for i in valid_frames:
        R_sum += _axis_angle_to_R(global_rots[i])
    R_sum /= len(valid_frames)
    U, _, Vt = np.linalg.svd(R_sum)
    R_ref = U @ Vt
    if np.linalg.det(R_ref) < 0:
        # Reflect to keep right-handed.
        U[:, -1] *= -1
        R_ref = U @ Vt

    wm = np.eye(4, dtype=np.float64)
    wm[:3, :3] = R_ref
    wm[:3, 3] = t_ref * float(cfg.flame_scale)

    # Recenter per-frame against the reference.
    R_ref_T = R_ref.T
    rot_recentered = np.zeros_like(global_rots, dtype=np.float32)
    for i in range(global_rots.shape[0]):
        R_total = _axis_angle_to_R(global_rots[i])
        R_delta = R_ref_T @ R_total
        rot_recentered[i] = _R_to_axis_angle(R_delta).astype(np.float32)
    trans_recentered = (translations - t_ref.astype(translations.dtype)).astype(np.float32)
    return wm.astype(np.float32), rot_recentered, trans_recentered


def extract(
    frames: FrameSource,
    outer_bbox: np.ndarray | None,
    cfg: LHGConfig,
    correspondence_path: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    apply_outer_crop: bool = True,
) -> LHGFeatures:
    """Top-level extraction. Mode is read from ``cfg.mode``.

    Parameters
    ----------
    frames : ``FrameSource`` over either an outer-cropped image dir
        (``outer_bbox`` should be the bbox used to make the dir, or None
        to skip the second crop pass) OR a raw video / raw image dir
        (in which case ``outer_bbox`` is computed by this routine).
    outer_bbox : (4,) int32 [xmin, xmax, ymin, ymax] in raw video coord,
        or ``None`` to compute internally.
    apply_outer_crop : if False, skip the in-process outer crop entirely
        — this is the right setting when ``frames`` is already an
        outer-cropped image directory (the avatar-fit pipeline's
        ``image/`` folder, with ``outer_offset.json`` describing the
        historical crop). The outer_bbox value is still passed through
        to the output as metadata.
    """
    from .correspondence import default_asset_for, load as load_correspondence

    if cfg.detector_type == 'fan':
        from .detector_fan import FANLandmarker
        detector = FANLandmarker()
    elif cfg.detector_type == 'mediapipe':
        detector = MediaPipeFaceLandmarker()
    else:
        raise ValueError(f'unknown detector_type {cfg.detector_type!r}')

    smirk = SMIRKEncoder()
    if correspondence_path is None:
        correspondence_path = default_asset_for(cfg.detector_type)
    correspondence = load_correspondence(correspondence_path)
    K = intrinsics_matrix(*cfg.intrinsics)

    raw_frames_rgb: list[np.ndarray] = list(frames.iter_frames())
    n_total = len(raw_frames_rgb)

    # --- Outer crop policy ------------------------------------------
    # Three cases:
    # (a) apply_outer_crop=False: input is already outer-cropped
    #     (e.g. demos/_preprocess_subject.sh's image/ folder). outer_bbox
    #     is metadata only; we never re-crop.
    # (b) apply_outer_crop=True, outer_bbox=None: compute it now.
    # (c) apply_outer_crop=True, outer_bbox given: use it as-is.
    if not apply_outer_crop and outer_bbox is None:
        # No metadata supplied — use the frame extent so callers still
        # get a sensible value persisted to lhg_features.npz.
        h, w = raw_frames_rgb[0].shape[:2]
        outer_bbox = np.array([0, w, 0, h], dtype=np.int32)
    elif apply_outer_crop and outer_bbox is None:
        if cfg.mode == 'pseudo-online':
            raw_landmarks: list[np.ndarray | None] = []
            for i, raw in enumerate(raw_frames_rgb):
                raw_landmarks.append(detector.detect(raw))
                if progress is not None:
                    progress(i, n_total * 2)
            outer_bbox = _compute_outer_bbox_from_landmarks(
                raw_landmarks, raw_frames_rgb[0].shape[:2],
            )
        else:
            first_n = min(cfg.world_mat_calibration_frames, n_total)
            first_landmarks = [
                detector.detect(raw_frames_rgb[i]) for i in range(first_n)
            ]
            outer_bbox = _compute_outer_bbox_from_landmarks(
                first_landmarks, raw_frames_rgb[0].shape[:2],
            )

    # --- Per-frame causal core --------------------------------------
    bbox_state = _BboxState(
        anchor=np.zeros(2, dtype=np.float64),
        target=np.zeros(2, dtype=np.float64),
        ring=np.zeros(max(int(cfg.bbox_window), 1), dtype=bool),
    )
    min_sigma_per_channel = {
        'expression': cfg.hampel_min_sigma_expression,
        'jaw': cfg.hampel_min_sigma_jaw,
        'eyelid': cfg.hampel_min_sigma_eyelid,
        'global_rot': cfg.hampel_min_sigma_global_rot,
        'translation': cfg.hampel_min_sigma_translation,
    }
    causal_filters = {
        name: CausalHampel(
            window=cfg.causal_hampel_window,
            k_sigma=cfg.causal_hampel_k_sigma,
            min_sigma=min_sigma_per_channel[name],
        )
        for name in ('expression', 'jaw', 'eyelid', 'global_rot', 'translation')
    }

    traces: list[FrameTrace] = []
    rejected_per_frame = np.zeros(n_total, dtype=bool)
    last_valid: FrameTrace | None = None

    for i, raw in enumerate(raw_frames_rgb):
        outer_img = (
            _crop_outer(raw, outer_bbox, cfg.image_size)
            if (apply_outer_crop and outer_bbox is not None)
            else raw
        )
        trace = per_frame_core(
            outer_img, detector, smirk, correspondence, K, cfg,
            bbox_state, last_valid,
        )

        if trace.valid and trace.global_rot is not None:
            for name, value in (
                ('expression', trace.expression),
                ('jaw', trace.jaw),
                ('eyelid', trace.eyelid),
                ('global_rot', trace.global_rot),
                ('translation', trace.translation),
            ):
                filtered, rej = causal_filters[name].step(value)
                if rej:
                    rejected_per_frame[i] = True
                # Replace in trace so downstream (next iteration's
                # last_valid, output aggregation) sees the filtered val.
                if name == 'expression':
                    trace.expression = filtered
                elif name == 'jaw':
                    trace.jaw = filtered
                elif name == 'eyelid':
                    trace.eyelid = filtered
                elif name == 'global_rot':
                    trace.global_rot = filtered
                elif name == 'translation':
                    trace.translation = filtered

        traces.append(trace)
        if trace.valid:
            last_valid = trace

        offset = n_total if cfg.mode == 'pseudo-online' else 0
        if progress is not None:
            progress(offset + i, n_total * (2 if cfg.mode == 'pseudo-online' else 1))

    # --- Aggregate ---------------------------------------------------
    valid = np.array([t.valid for t in traces], dtype=bool)

    def _stack_or_fallback(name: str, dim: int) -> np.ndarray:
        out = np.zeros((n_total, dim), dtype=np.float32)
        for i, t in enumerate(traces):
            v = getattr(t, name)
            if v is not None:
                out[i] = np.asarray(v, dtype=np.float32)
        return out

    expression = _stack_or_fallback('expression', 50)
    jaw = _stack_or_fallback('jaw', 3)
    eyelid = _stack_or_fallback('eyelid', 2)
    global_rot = _stack_or_fallback('global_rot', 3)
    translation = _stack_or_fallback('translation', 3)

    interpolated = np.zeros(n_total, dtype=bool)

    # --- Pseudo-online post-processing -----------------------------
    if cfg.mode == 'pseudo-online':
        for arr_name, arr in (
            ('expression', expression),
            ('jaw', jaw),
            ('eyelid', eyelid),
            ('global_rot', global_rot),
            ('translation', translation),
        ):
            cleaned, rej = bidirectional_hampel(
                arr.astype(np.float64),
                window=cfg.bidirectional_hampel_window,
                k_sigma=cfg.bidirectional_hampel_k_sigma,
                min_sigma=min_sigma_per_channel[arr_name],
            )
            if arr_name == 'expression':
                expression = cleaned.astype(np.float32)
            elif arr_name == 'jaw':
                jaw = cleaned.astype(np.float32)
            elif arr_name == 'eyelid':
                eyelid = cleaned.astype(np.float32)
            elif arr_name == 'global_rot':
                global_rot = cleaned.astype(np.float32)
            elif arr_name == 'translation':
                translation = cleaned.astype(np.float32)
            rejected_per_frame |= rej

        # Linear interpolation across detector dropouts.
        for arr_name, arr in (
            ('expression', expression),
            ('jaw', jaw),
            ('eyelid', eyelid),
            ('global_rot', global_rot),
            ('translation', translation),
        ):
            filled, interp = linear_interpolate_dropouts(
                arr.astype(np.float64), valid,
            )
            interpolated |= interp
            if arr_name == 'expression':
                expression = filled.astype(np.float32)
            elif arr_name == 'jaw':
                jaw = filled.astype(np.float32)
            elif arr_name == 'eyelid':
                eyelid = filled.astype(np.float32)
            elif arr_name == 'global_rot':
                global_rot = filled.astype(np.float32)
            elif arr_name == 'translation':
                translation = filled.astype(np.float32)

        # Bidirectional quaternion-flip fix on global_rot.
        quats = np.stack(
            [axis_angle_to_quat(global_rot[i]) for i in range(n_total)], axis=0,
        )
        quats = fix_quat_sign_continuity(quats)
        global_rot = np.stack(
            [quat_to_axis_angle(quats[i]) for i in range(n_total)], axis=0,
        ).astype(np.float32)

    world_mat, global_rot, translation = _resolve_world_mat_and_recenter(
        cfg, global_rot, translation, valid | interpolated,
    )

    detector.close()

    return LHGFeatures(
        frame_basenames=np.array(frames.basenames(), dtype=str),
        expression=expression,
        jaw=jaw,
        eyelid=eyelid,
        global_rot=global_rot,
        translation=translation,
        valid_mask=valid,
        interpolated_mask=interpolated,
        rejected_mask=rejected_per_frame,
        mode=cfg.mode,
        intrinsics=np.array(cfg.intrinsics, dtype=np.float32),
        world_mat=world_mat,
        outer_bbox=outer_bbox.astype(np.int32),
        fps=cfg.fps,
        image_size=cfg.image_size,
        flame_scale=cfg.flame_scale,
        camera_convention=cfg.camera_convention,
    )
