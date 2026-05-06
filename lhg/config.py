"""Runtime configuration for the LHG feature extraction pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Mode = Literal['online', 'pseudo-online']


@dataclass
class LHGConfig:
    """All runtime knobs for ``lhg.pipeline.extract``.

    The defaults mirror the same values used by
    ``preprocess/stable_bbox.py`` (deadzone, K-of-N, tau) so the LHG
    feature stream stays parameter-aligned with the rest of the
    preprocessing stack.
    """

    mode: Mode
    fps: float

    # Camera intrinsics in the OUTER-CROP coordinate system (the same
    # space the EPnP solve runs in). Use one of the named presets via
    # ``intrinsics_preset`` or pass (fx, fy, cx, cy) directly.
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
    # threshold never collapses to ~0 on very stable signals. Tuned to
    # the natural per-frame noise floor of each channel; rejection then
    # only fires for genuinely large jumps. Set to 0.0 for legacy
    # MAD-only behaviour (will reject sub-noise variation as outliers).
    causal_hampel_window: int = 5
    causal_hampel_k_sigma: float = 3.0
    bidirectional_hampel_window: int = 11
    bidirectional_hampel_k_sigma: float = 3.0
    hampel_min_sigma_expression: float = 0.05    # SMIRK exp std floor
    hampel_min_sigma_jaw: float = 0.005          # rad/axis-angle floor
    hampel_min_sigma_eyelid: float = 0.02        # 0-1 normalised floor
    hampel_min_sigma_global_rot: float = 0.005   # rad floor
    hampel_min_sigma_translation: float = 0.005  # FLAME canonical units

    # FLAME convention. v1 uses ``flame_scale=4.0`` (HRAvatar default);
    # v2 uses 1.0. The output ``translation`` channel is always stored
    # in FLAME-canonical space (i.e. before multiplication by
    # ``flame_scale``), regardless of which convention was used to
    # solve the PnP.
    flame_scale: float = 4.0

    # When ``world_mat_path`` is None and ``world_mat_calibration_frames``
    # > 0, the first N frames are used to compute the clip-constant
    # ``world_mat`` from the per-frame translation mean.
    world_mat_path: str | None = None
    world_mat_calibration_frames: int = 60

    # Diagnostic switches. The DECA encoder is NOT used for any output
    # channel; it is invoked only when ``run_deca_encoder=True`` so its
    # per-frame ``cam`` / ``pose`` can be persisted alongside the SMIRK
    # output for offline debugging.
    run_deca_encoder: bool = False


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
