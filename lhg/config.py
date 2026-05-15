"""Runtime configuration for the LHG feature extraction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Mode = Literal['online', 'pseudo-online']
OnlineBackend = Literal['epnp', 'deca_encoder']


@dataclass
class LHGConfig:
    """All runtime knobs for ``lhg.pipeline.extract``.

    Stage layout
    ------------
    Stage 1 (offline calibration, ``demos/_preprocess_subject.sh
    --lhg-only``) writes ``tracked_params.json`` whose ``world_mat``,
    ``shapecode``, and ``intrinsics`` are read here via
    ``calibration_path`` and used as clip-constants.

    Stage 2 (this config governs) is the per-frame online pipeline:
    detector → stable bbox → SMIRK → EPnP → causal Hampel → symmetric
    FIR LPF on rotation/translation (and optionally jaw).

    Stage 3 (``--mode pseudo-online``, future) re-runs Stage 2 then
    applies bidirectional Hampel + linear interpolation +
    ``apply_offline_zero_phase`` with a larger lookahead.

    The defaults mirror ``preprocess/stable_bbox.py`` (deadzone,
    K-of-N, tau) so the LHG feature stream stays parameter-aligned
    with the rest of the preprocessing stack.
    """

    mode: Mode
    fps: float

    # Path to the Stage 1 ``tracked_params.json`` (or ``_v2`` sibling).
    # Required: Stage 2 reads world_mat / shapecode / intrinsics from
    # this file. The previous "calibration window inside extract()"
    # path has been removed because per-frame EPnP cannot reproduce
    # DECA optimize.py's clip-wide joint accuracy.
    calibration_path: str | None = None

    # Landmark detector for both the stable bbox and the EPnP solve.
    # Default 'mediapipe' uses MediaPipe FaceLandmarker (478pt + iris)
    # in VIDEO running mode. Stable subset is precomputed in
    # ``preprocess._smirk_constants.STABLE_LANDMARK_INDICES`` and the
    # MICA-derived FLAME barycentric is shipped in
    # ``assets/lhg/mediapipe_flame_landmarks.npz``.
    #
    # 'fan' (face_alignment 68pt) is retained for the Phase 4 A/B test
    # and for users who want a detector identical to HRAvatar's avatar
    # fit pipeline. FAN's per-frame absolute accuracy is comparable to
    # MediaPipe (both are EPnP-precision-limited); the historical
    # "FAN gives 5-10x lower jitter" statement only held for a
    # seed-once-bbox setup that fails when the subject moves.
    detector_type: str = 'mediapipe'

    # MediaPipe running mode. 'video' enables the internal Kalman
    # tracker (lower per-frame jitter, slightly faster) and is the
    # production default. 'image' is the per-frame baseline used in
    # the Stage 2 jitter A/B comparison.
    mediapipe_running_mode: str = 'video'

    # Camera intrinsics in the OUTER-CROP coordinate system (the same
    # space the EPnP solve runs in). When ``calibration_path`` is set,
    # these are OVERWRITTEN by the calibration's intrinsics so the
    # per-frame EPnP and the avatar-fit ``world_mat`` agree.
    intrinsics: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    image_size: int = 512

    # Stable bbox center hysteresis (mirrors stable_bbox.py defaults).
    bbox_scale: float = 1.6
    bbox_deadzone_px: float = 4.0
    bbox_window: int = 5
    bbox_k_of_n: int = 3
    bbox_tau: float = 0.25

    # Hampel filter (outlier rejection only — no smoothing of in-range
    # values). ``causal_*`` runs in both modes; ``bidirectional_*`` is
    # an extra pass that only ``pseudo-online`` performs.
    #
    # ``hampel_min_sigma_*`` floors the local MAD-derived sigma so the
    # threshold never collapses to ~0 on very stable signals AND so that
    # natural head/face motion (which can briefly produce per-frame
    # deltas much larger than the local MAD estimate around a quiet
    # baseline) does NOT get misclassified as an outlier.
    causal_hampel_window: int = 5
    causal_hampel_k_sigma: float = 3.0
    bidirectional_hampel_window: int = 11
    bidirectional_hampel_k_sigma: float = 3.0
    hampel_min_sigma_expression: float = 0.30
    hampel_min_sigma_jaw: float = 0.10
    hampel_min_sigma_eyelid: float = 0.30
    hampel_min_sigma_global_rot: float = 0.30
    hampel_min_sigma_translation: float = 0.05

    # Symmetric (zero-phase) FIR LPF on rotation/translation. Applied
    # in both online (StreamingSymmetricFIR with lookahead L=4 default
    # → 160ms@25fps lag) and pseudo-online (apply_offline_zero_phase
    # with L=lpf_offline_lookahead). Filter design is identical
    # between the two modes; only the tap count differs.
    #
    # ``lpf_lookahead`` is the ONLINE filter's one-sided lookahead.
    # Set to 0 to disable LPF entirely (pass-through).
    lpf_lookahead: int = 4
    lpf_cutoff_hz: float = 4.0

    # Optional LPF on jaw with a separate cutoff. Default OFF — jaw
    # carries syllable-rate (~5-8 Hz) information that LHG models use
    # for lip-sync, so the cutoff is set high (10 Hz) when opted in
    # to attenuate detector-noise above the speech band without
    # eating into syllabic content.
    lpf_jaw: bool = False
    lpf_jaw_cutoff_hz: float = 10.0

    # Pseudo-online (Stage 3) only: lookahead for the offline filter.
    # 12 → taps=25, matches the user's existing offline FIR
    # configuration (cutoff 4 Hz @ 25 fps).
    lpf_offline_lookahead: int = 12

    # FLAME convention. v1 uses ``flame_scale=4.0`` (HRAvatar default);
    # v2 uses 1.0. The output ``translation`` channel is always stored
    # in FLAME-canonical space (i.e. before multiplication by
    # ``flame_scale``), regardless of which convention was used to
    # solve the PnP. Auto-overridden by Stage 1 calibration when
    # provided.
    flame_scale: float = 4.0

    # Diagnostic switches. The DECA encoder is NOT used for any output
    # channel; it is invoked only when ``run_deca_encoder=True`` so its
    # per-frame ``cam`` / ``pose`` can be persisted alongside the SMIRK
    # output for offline debugging.
    run_deca_encoder: bool = False

    # Path to the avatar checkpoint directory containing
    # ``flame_params_net.pth`` (typically
    # ``outputs/custom/<avatar>/saved_model/epoch_<E>``). When set, the
    # online SMIRKEncoder loads these per-subject trained weights so
    # the online output matches the avatar's renderer-time SMIRK
    # output (the same weights ``lhg.teacher`` uses to produce the
    # offline teacher target). When ``None`` (default), SMIRKEncoder
    # falls back to the pretrained initialization. Under the
    # 2026-05-15 grand design clarification, both paths should
    # ordinarily share the same SMIRK weights so the only differences
    # between online and teacher come from online-only constraints
    # (causality, no clip-wide joint optimization).
    avatar_checkpoint: str | None = None

    # EPnP backend refinements (no effect when online_backend is not
    # 'epnp'). Both are clip-constant or per-frame corrections layered
    # on top of the base EPnP solve:
    #
    # * epnp_expression_aware: rebuild the EPnP 3D landmark template
    #   per frame from SMIRK's expression output (instead of the
    #   shape-only template). Targets the per-frame JITTER caused by
    #   fitting an expression-neutral template to an expressive face.
    #
    # * correct_translation_offset: measure the clip-mean difference
    #   between the EPnP translation and the calibration's DECA
    #   optimize translation, then subtract it. Targets the SYSTEMATIC
    #   translation bias from EPnP's 16-pt landmark centroid differing
    #   from DECA optimize's 68-pt centroid. The offset is a clip
    #   constant (avatar-specific), consistent with the grand design's
    #   "maximize clip-constants" tenet.
    epnp_expression_aware: bool = False
    correct_translation_offset: bool = False

    # Online backend selects which per-frame translation/global_rot
    # estimator the pipeline runs:
    #
    # * 'epnp' (legacy, default for backwards compatibility) — solve
    #   cv2.solvePnP(EPnP) per frame against the shape-aware FLAME
    #   landmark template. Independent of any DECA/offline coupling.
    #
    # * 'deca_encoder' (2026-05-14 grand design) — re-use the offline
    #   pipeline's DECA encoder. ``global_rot`` comes from
    #   ``pose[0:3]``; translation is currently 0 (Stage 1 of the new
    #   backend) and will be lifted to a cam→z proxy once the
    #   subject's clip-mean cam scale is available (Phase B-2).
    online_backend: OnlineBackend = 'epnp'

    # Camera coordinate convention for the global_rot / translation /
    # world_mat outputs. Default 'hravatar' produces values directly
    # consumable by HRAvatar's renderer (X right, Y up, -Z forward).
    # 'opencv' keeps the raw EPnP output (X right, Y down, +Z forward)
    # — only useful for pipelines that already have their own
    # OpenCV→OpenGL conversion downstream.
    camera_convention: str = 'hravatar'


INTRINSICS_PRESETS = {
    # outer-crop 512x512, source-resolution-agnostic. Same numbers as
    # ``demos/_preprocess_subject.sh``.
    'hdtf': (1539.67462, 1508.93280, 261.442628, 253.231895),
    'insta': (1536.00, 1536.00, 256.00, 256.00),
}


def parse_intrinsics(spec: str) -> tuple[float, float, float, float]:
    """Resolve ``hdtf`` / ``insta`` / ``custom:fx,fy,cx,cy`` like the
    avatar-fit shell wrapper does."""
    if spec in INTRINSICS_PRESETS:
        return INTRINSICS_PRESETS[spec]
    if spec.startswith('custom:'):
        nums = spec[len('custom:'):].split(',')
        if len(nums) != 4:
            raise ValueError(
                f'custom intrinsics must be "custom:fx,fy,cx,cy", got {spec!r}')
        return tuple(float(x) for x in nums)
    raise ValueError(
        f'unknown intrinsics spec {spec!r}; '
        f'use one of {list(INTRINSICS_PRESETS)} or custom:fx,fy,cx,cy')


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
