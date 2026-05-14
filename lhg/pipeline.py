"""Per-frame core algorithm and mode-specific orchestration.

The two modes share one ``per_frame_core`` step (detector → stable
bbox → SMIRK → EPnP → causal Hampel) and differ only in the
surrounding clip-level wrapper:

* ``online`` runs the core strictly causally and yields one frame at a
  time. Detector dropouts are held at the previous valid value. After
  the per-frame loop, a symmetric (zero-phase) FIR LPF is applied to
  rotation/translation (and optionally jaw) — mathematically
  equivalent to a streaming filter with lookahead ``L`` frames, since
  we have the full clip in hand at extraction time.

* ``pseudo-online`` (Stage 3, gated in ``lhg.extract``) runs the core
  in the same order but post-processes the collected sequence with
  bidirectional Hampel + linear interpolation across detector
  dropouts + bidirectional quaternion-flip fix + a wider LPF.

Camera/world-mat handling
-------------------------
The Stage 1 calibration (DECA optimize.py joint fit) provides the
``world_mat`` and ``shapecode`` clip-constants. Per-frame
``global_rot`` and ``translation`` are emitted as DELTAS around
``world_mat``, so the downstream LHG model sees only motion (not
absolute pose). The per-frame EPnP could not reproduce the
calibration's clip-wide accuracy in any case, so this layered design
splits the work cleanly: calibration handles absolute pose, Stage 2
handles per-frame deltas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from preprocess._smirk_constants import (
    STABLE_LANDMARK_INDICES,
    STABLE_LANDMARK_SIZE_CALIBRATION,
)

from .calibration import StageOneCalibration, load_stage_one_calibration
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
from .lpf import apply_offline_zero_phase
from .output import LHGFeatures
from .video import FrameSource


@dataclass
class _BboxState:
    """Causal hysteresis follower state plus per-detector bbox tracker.

    ``anchor`` / ``target`` / ``ring`` / ``ring_idx`` / ``initialized``
    follow ``preprocess.stable_bbox._hysteresis_forward`` for the
    SMIRK 224-crop center. ``fan_bbox`` is the *next-frame* xyxy bbox
    used to bypass FAN's internal S3FD on subsequent frames; it's
    updated at the END of each iteration from the current frame's
    detected landmarks (with 25% padding) so subject motion is
    followed without re-running S3FD.
    """
    anchor: np.ndarray              # (2,) current emitted center
    target: np.ndarray              # (2,) candidate to chase
    ring: np.ndarray                # (window,) bool sliding history
    ring_idx: int = 0
    initialized: bool = False
    fan_bbox: np.ndarray | None = None  # (4,) [x1, y1, x2, y2] for FAN


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


def _fan_bbox_from_landmarks(
    landmarks: np.ndarray, padding_frac: float = 0.25,
) -> np.ndarray:
    """Tight FAN-input bbox from the current frame's 68 landmarks.

    The bbox is inflated by ``padding_frac`` (default 25%) so the next
    frame's face — which may have moved by a few px due to head /
    body motion — still falls comfortably inside. The 25% margin is
    much smaller than the full 512x512 frame, keeping the face at
    ~195 px in FAN's 256-input (the model's optimal scale).
    """
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    w_pad = (x2 - x1) * padding_frac
    h_pad = (y2 - y1) * padding_frac
    return np.array(
        [x1 - w_pad, y1 - h_pad, x2 + w_pad, y2 + h_pad],
        dtype=np.float64,
    )


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
    valid: bool = False                        # True iff detector + EPnP succeeded


def _detect_landmarks(
    image_outer_rgb: np.ndarray,
    detector,
    bbox_state: _BboxState,
    cfg: LHGConfig,
    frame_index: int,
) -> np.ndarray | None:
    """Wrapper that dispatches to the per-detector calling convention.

    For ``mediapipe`` in video mode we synthesize an integer
    ``timestamp_ms`` from ``frame_index / cfg.fps`` so the internal
    Kalman tracker sees monotonically-increasing timestamps. For
    ``fan`` we feed the previous frame's tight-bbox-with-padding to
    bypass S3FD; the first frame triggers FAN's internal S3FD seed.
    """
    if cfg.detector_type == 'mediapipe':
        if cfg.mediapipe_running_mode == 'video':
            timestamp_ms = int(frame_index * 1000.0 / float(cfg.fps))
            return detector.detect(image_outer_rgb, timestamp_ms=timestamp_ms)
        return detector.detect(image_outer_rgb)
    if cfg.detector_type == 'fan':
        bbox = getattr(bbox_state, 'fan_bbox', None)
        return detector.detect(image_outer_rgb, bbox_xyxy=bbox)
    raise ValueError(f'unknown detector_type {cfg.detector_type!r}')


def per_frame_core(
    image_outer_rgb: np.ndarray,
    detector,
    smirk: SMIRKEncoder,
    correspondence: MediaPipeFLAMECorrespondence,
    K: np.ndarray,
    cfg: LHGConfig,
    bbox_state: _BboxState,
    last_valid: FrameTrace | None,
    frame_index: int,
    epnp_object_points: np.ndarray | None = None,
) -> FrameTrace:
    """One causal frame step. Shared by online and pseudo-online modes.

    The only inter-frame state is ``bbox_state`` (causal hysteresis
    follower + FAN bbox tracker) and ``last_valid`` (used to hold
    values across a detector dropout). Hampel rejection happens one
    level up.
    """
    landmarks = _detect_landmarks(
        image_outer_rgb, detector, bbox_state, cfg, frame_index,
    )

    if landmarks is None:
        # Detector miss: hold previous valid sample if available.
        # Note: bbox_state.fan_bbox is intentionally NOT updated this
        # frame, so the next frame retries with the same prior bbox.
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

    # FAN bbox tracking: update from THIS frame's landmarks so the
    # NEXT frame's FAN call has a tight bbox at this frame's face
    # location. Even with subject motion the face will be inside the
    # 25%-padded bbox at the next frame.
    if cfg.detector_type == 'fan':
        bbox_state.fan_bbox = _fan_bbox_from_landmarks(landmarks)

    raw_center, raw_size = _bbox_size_from_landmarks(landmarks, cfg.detector_type)
    smoothed_center = _step_bbox_hysteresis(bbox_state, raw_center, cfg)

    face_224 = _warp_to_224(
        image_outer_rgb, smoothed_center, raw_size, cfg.bbox_scale,
    )
    smirk_out = smirk.encode(face_224)

    img_pts_2d = landmarks[correspondence.mp_indices]
    object_points = (
        epnp_object_points
        if epnp_object_points is not None
        else correspondence.flame_canonical_xyz
    )
    if object_points is None:
        raise RuntimeError(
            'No 3D landmark positions available for EPnP. Either pass '
            '``epnp_object_points`` (preferred — built from Stage 1 '
            'shapecode via build_shape_aware_landmarks) or ensure the '
            'correspondence asset carries ``flame_canonical_xyz`` (neutral '
            'mesh fallback).')
    rvec, tvec_canonical, ok = solve_epnp(
        img_pts_2d,
        object_points,
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


def _axis_angle_to_R(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to 3x3 rotation matrix (numpy Rodrigues)."""
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(aa, dtype=np.float64).reshape(3, 1))
    return R


def _R_to_axis_angle(R: np.ndarray) -> np.ndarray:
    import cv2
    aa, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return aa.reshape(3)


def _recenter_against_world_mat(
    world_mat: np.ndarray,
    flame_scale: float,
    global_rots: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-frame total pose (R_total, t_total) into deltas
    around the calibration ``world_mat``.

    HRAvatar's renderer (and DECA's ``optimize.py`` ``projection``)
    consume per-frame pose as a SMALL delta around ``world_mat``:

    * ``world_mat`` carries the absolute camera-to-mean-head
      transform — rotation in [:3, :3] absorbs the FLAME-canonical-
      to-camera-facing pre-rotation, translation in [:3, 3] absorbs
      the depth (in scaled FLAME units).
    * Per-frame ``global_rot`` is a SMALL axis-angle delta around
      that reference, applied AFTER the world_mat rotation.
    * Per-frame ``translation`` is a SMALL delta in FLAME canonical
      units (i.e. AFTER dividing the world_mat translation by
      ``flame_scale``).

    EPnP returns the TOTAL camera-to-object pose on each frame; this
    routine splits off the calibration reference so the model only
    sees the residual motion.
    """
    R_ref = world_mat[:3, :3].astype(np.float64)
    t_ref_canonical = world_mat[:3, 3].astype(np.float64) / float(flame_scale)
    R_ref_T = R_ref.T
    rot_recentered = np.zeros_like(global_rots, dtype=np.float32)
    for i in range(global_rots.shape[0]):
        R_total = _axis_angle_to_R(global_rots[i])
        R_delta = R_ref_T @ R_total
        rot_recentered[i] = _R_to_axis_angle(R_delta).astype(np.float32)
    trans_recentered = (
        translations.astype(np.float64) - t_ref_canonical
    ).astype(np.float32)
    return rot_recentered, trans_recentered


def extract(
    frames: FrameSource,
    outer_bbox: np.ndarray | None,
    cfg: LHGConfig,
    calibration: StageOneCalibration | None = None,
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
        outer-cropped image directory. The outer_bbox value is still
        passed through to the output as metadata.
    calibration : optional Stage 1 calibration. When given (or when
        ``cfg.calibration_path`` is set, in which case it's loaded
        here), the world_mat / shapecode / intrinsics from this
        artifact override per-frame computation. Required for
        production use; only ``None`` is accepted for diagnostic /
        unit-test paths that supply a synthetic intrinsics tuple
        directly via ``cfg``.
    """
    from .correspondence import default_asset_for, load as load_correspondence

    # Resolve calibration.
    if calibration is None and cfg.calibration_path:
        calibration = load_stage_one_calibration(cfg.calibration_path)
    if calibration is None:
        raise ValueError(
            'extract() requires either a `calibration` object or '
            '`cfg.calibration_path`. Run Stage 1 first '
            '(`bash demos/_preprocess_subject.sh --lhg-only ...`) and '
            'pass the produced tracked_params.json via `--calibration`.')

    # The calibration is the source of truth for these clip-constants.
    cfg.intrinsics = calibration.intrinsics_tuple()
    cfg.flame_scale = float(calibration.flame_scale)
    world_mat = calibration.world_mat.astype(np.float64)

    if cfg.detector_type == 'fan':
        from .detector_fan import FANLandmarker
        detector = FANLandmarker()
    elif cfg.detector_type == 'mediapipe':
        detector = MediaPipeFaceLandmarker(
            running_mode=cfg.mediapipe_running_mode,
        )
    else:
        raise ValueError(f'unknown detector_type {cfg.detector_type!r}')

    smirk = SMIRKEncoder()
    if correspondence_path is None:
        correspondence_path = default_asset_for(cfg.detector_type)
    correspondence = load_correspondence(correspondence_path)
    K = intrinsics_matrix(*cfg.intrinsics)

    # Build shape-aware EPnP object points from the Stage 1 shapecode.
    # The neutral correspondence asset would bias EPnP's depth estimate by
    # the ratio of (neutral face width) / (subject face width); see
    # build_shape_aware_landmarks for the derivation.
    from .correspondence import build_shape_aware_landmarks
    epnp_object_points = build_shape_aware_landmarks(
        correspondence, calibration.shapecode,
    )

    raw_frames_rgb: list[np.ndarray] = list(frames.iter_frames())
    n_total = len(raw_frames_rgb)
    if n_total == 0:
        raise RuntimeError('no frames provided to lhg.extract')

    # --- Outer crop policy ------------------------------------------
    # (a) apply_outer_crop=False: input is already outer-cropped (e.g.
    #     demos/_preprocess_subject.sh's image/ folder). outer_bbox is
    #     metadata only; we never re-crop.
    # (b) apply_outer_crop=True, outer_bbox=None: use the FULL frame
    #     extent. We don't compute a clip-wide bbox here anymore;
    #     callers running on a raw video should pre-crop with
    #     crop_and_matting.py first (Stage 1 handles this).
    # (c) apply_outer_crop=True, outer_bbox given: use as-is.
    if not apply_outer_crop and outer_bbox is None:
        h, w = raw_frames_rgb[0].shape[:2]
        outer_bbox = np.array([0, w, 0, h], dtype=np.int32)
    elif apply_outer_crop and outer_bbox is None:
        h, w = raw_frames_rgb[0].shape[:2]
        outer_bbox = np.array([0, w, 0, h], dtype=np.int32)

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
            bbox_state, last_valid, frame_index=i,
            epnp_object_points=epnp_object_points,
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
        for j, t in enumerate(traces):
            v = getattr(t, name)
            if v is not None:
                out[j] = np.asarray(v, dtype=np.float32)
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
            [axis_angle_to_quat(global_rot[j]) for j in range(n_total)], axis=0,
        )
        quats = fix_quat_sign_continuity(quats)
        global_rot = np.stack(
            [quat_to_axis_angle(quats[j]) for j in range(n_total)], axis=0,
        ).astype(np.float32)

    # --- Recenter against the calibration world_mat ----------------
    # Done BEFORE the LPF so the filter operates on small deltas
    # (numerically better-conditioned than the absolute axis-angle
    # which can wrap around π for large head turns).
    global_rot, translation = _recenter_against_world_mat(
        world_mat, cfg.flame_scale, global_rot, translation,
    )

    # --- Symmetric FIR LPF on rotation/translation (and optional jaw)
    # The same ``apply_offline_zero_phase`` is used in both online
    # (lookahead from cfg.lpf_lookahead) and pseudo-online (larger
    # lookahead from cfg.lpf_offline_lookahead). At extraction time
    # we have the full clip in hand, so the streaming filter math is
    # bit-equivalent to the offline edge-padded zero-phase filter.
    if cfg.mode == 'pseudo-online':
        L = int(cfg.lpf_offline_lookahead)
    else:
        L = int(cfg.lpf_lookahead)

    if L > 0:
        global_rot = apply_offline_zero_phase(
            global_rot.astype(np.float64), L, cfg.lpf_cutoff_hz, cfg.fps,
        ).astype(np.float32)
        translation = apply_offline_zero_phase(
            translation.astype(np.float64), L, cfg.lpf_cutoff_hz, cfg.fps,
        ).astype(np.float32)
        if cfg.lpf_jaw:
            jaw = apply_offline_zero_phase(
                jaw.astype(np.float64), L, cfg.lpf_jaw_cutoff_hz, cfg.fps,
            ).astype(np.float32)

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
        world_mat=world_mat.astype(np.float32),
        outer_bbox=outer_bbox.astype(np.int32),
        fps=cfg.fps,
        image_size=cfg.image_size,
        flame_scale=cfg.flame_scale,
        camera_convention=cfg.camera_convention,
    )
