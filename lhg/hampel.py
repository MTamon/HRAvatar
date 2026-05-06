"""Hampel filter (outlier rejection only, NO smoothing of in-range values).

Two flavours:

* ``CausalHampel`` — forward-only sliding window. Used in both online
  and pseudo-online modes. When the current sample is more than
  ``k_sigma`` MADs away from the local median, replace it with the
  most recent in-range value. This deliberately does NOT touch
  in-range samples (so the noise floor is left to the L3 LHG model
  temporal regularization to handle, per
  ``feedback_no_one_euro_in_lhg_pipeline``).

* ``bidirectional_hampel`` — centered-window two-pass version. Used
  ONLY in pseudo-online mode for one-time anomalies that aren't
  reachable causally (post-hoc replay of the whole clip).
"""
from __future__ import annotations

from collections import deque

import numpy as np


_MAD_TO_SIGMA = 1.4826  # consistency factor for normal-distribution MAD


class CausalHampel:
    """Per-channel forward-only Hampel filter.

    State is one ``deque`` of the last ``window`` values, plus the most
    recent in-range value (used as the replacement when the current
    sample is rejected).

    The first frame seeds the filter; the second-through-Nth frames
    accumulate the window before any rejection is allowed.

    Rejection threshold floor
    -------------------------
    A naive ``k * MAD`` threshold collapses to ~0 on very stable signals
    (e.g. talking-head ``translation`` whose MAD is sub-millimeter), so
    every micro-fluctuation gets flagged as an outlier and the output
    becomes a step function holding the first frame value. To avoid
    this we floor sigma at ``min_sigma`` (absolute, in the channel's
    own units) — set per-channel to roughly the L1 noise floor of
    that channel. The default 0.0 disables the floor (legacy Hampel).

    Multi-dim rejection rule
    ------------------------
    For multi-dim channels (expression 50d, jaw 3d, etc.) we reject
    the whole frame only when the FRACTION of dims out of range
    exceeds ``oor_frac`` (default 0.5 = strict majority). Single-dim
    transient noise no longer kills the frame; a genuine tracking
    glitch typically blows out most/all dims at once and is still
    caught. This matches the "physically implausible change" intent
    of the L2 layer in the LHG jitter design.
    """

    def __init__(
        self,
        window: int = 5,
        k_sigma: float = 3.0,
        min_sigma: float | np.ndarray = 0.0,
        oor_frac: float = 0.5,
    ):
        if window < 3:
            raise ValueError(f'window must be >= 3, got {window}')
        if not 0.0 < oor_frac <= 1.0:
            raise ValueError(f'oor_frac must be in (0, 1], got {oor_frac}')
        self.window = int(window)
        self.k_sigma = float(k_sigma)
        self.min_sigma = np.asarray(min_sigma, dtype=np.float64)
        self.oor_frac = float(oor_frac)
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)
        self._last_inrange: np.ndarray | None = None

    def reset(self) -> None:
        self._buf.clear()
        self._last_inrange = None

    def step(self, value: np.ndarray) -> tuple[np.ndarray, bool]:
        """Process one frame's value (1-D array). Returns
        ``(value_or_replacement, rejected)``.
        """
        v = np.asarray(value, dtype=np.float64)
        if self._last_inrange is None:
            self._buf.append(v.copy())
            self._last_inrange = v.copy()
            return v.astype(value.dtype, copy=False), False

        # Need a full window to make a defensible MAD estimate.
        if len(self._buf) < self.window - 1:
            self._buf.append(v.copy())
            self._last_inrange = v.copy()
            return v.astype(value.dtype, copy=False), False

        stack = np.stack(list(self._buf) + [v], axis=0)  # (window, D)
        median = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - median), axis=0)
        sigma = np.maximum(mad * _MAD_TO_SIGMA, self.min_sigma)
        threshold = self.k_sigma * sigma
        oor = np.abs(v - median) > np.maximum(threshold, 1e-9)
        # Reject when a STRICT MAJORITY of dims is OOR (default
        # oor_frac=0.5). Single-dim transients don't kill the frame;
        # a genuine tracking glitch blows out most dims simultaneously.
        oor_count = int(np.sum(oor))
        rejected = oor_count >= max(1, int(np.ceil(oor.size * self.oor_frac)))

        if rejected:
            replaced = self._last_inrange.copy()
            self._buf.append(replaced.copy())
            return replaced.astype(value.dtype, copy=False), True

        self._buf.append(v.copy())
        self._last_inrange = v.copy()
        return v.astype(value.dtype, copy=False), False


def bidirectional_hampel(
    series: np.ndarray,
    window: int = 11,
    k_sigma: float = 3.0,
    min_sigma: float | np.ndarray = 0.0,
    oor_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered-window Hampel for offline replay of a full series.

    Parameters
    ----------
    series : (N, D) float
    window : odd integer >= 3 (forced odd; centered around each sample)
    k_sigma : MAD threshold
    min_sigma : floor on the local sigma estimate to avoid spurious
        rejections on very stable signals (see ``CausalHampel`` for the
        same rationale).

    Returns
    -------
    cleaned : (N, D) — outliers replaced by linear interpolation between
              the nearest in-range neighbours on each side. If no in-range
              neighbour exists on one side, the closest available is
              repeated (boundary).
    rejected_mask : (N,) bool — True where the original sample was OOR.
    """
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f'series must be (N, D), got {s.shape}')
    n, d = s.shape

    w = int(window)
    if w < 3:
        raise ValueError(f'window must be >= 3, got {w}')
    if w % 2 == 0:
        w += 1
    half = w // 2

    if not 0.0 < oor_frac <= 1.0:
        raise ValueError(f'oor_frac must be in (0, 1], got {oor_frac}')
    floor = np.asarray(min_sigma, dtype=np.float64)
    rejected = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = s[lo:hi]
        median = np.median(win, axis=0)
        mad = np.median(np.abs(win - median), axis=0)
        sigma = np.maximum(mad * _MAD_TO_SIGMA, floor)
        threshold = k_sigma * sigma
        oor = np.abs(s[i] - median) > np.maximum(threshold, 1e-9)
        oor_count = int(np.sum(oor))
        if oor_count >= max(1, int(np.ceil(oor.size * oor_frac))):
            rejected[i] = True

    cleaned = s.copy()
    if not rejected.any():
        return cleaned, rejected

    valid = np.where(~rejected)[0]
    if valid.size == 0:
        # Pathological: every frame is an outlier. Leave untouched and
        # let the caller decide.
        return cleaned, rejected

    for i in np.where(rejected)[0]:
        # Find nearest valid neighbours.
        left_idx = valid[valid < i]
        right_idx = valid[valid > i]
        if left_idx.size and right_idx.size:
            l = int(left_idx[-1])
            r = int(right_idx[0])
            t = (i - l) / float(r - l)
            cleaned[i] = (1.0 - t) * s[l] + t * s[r]
        elif left_idx.size:
            cleaned[i] = s[int(left_idx[-1])]
        else:
            cleaned[i] = s[int(right_idx[0])]
    return cleaned, rejected
