"""Linear interpolation across detector dropouts (pseudo-online only).

In ``online`` mode a missing detection is held at the previous valid
sample (causal). In ``pseudo-online`` mode we replay the whole clip
and can interpolate between the nearest valid neighbours.
"""
from __future__ import annotations

import numpy as np


def linear_interpolate_dropouts(
    series: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace samples where ``valid_mask`` is False with linear
    interpolation between the surrounding True samples.

    Parameters
    ----------
    series : (N, D) — per-frame value array.
    valid_mask : (N,) bool — True where the sample is valid.

    Returns
    -------
    interpolated : (N, D) — interpolated array.
    interpolated_mask : (N,) bool — True where a sample was created
        by interpolation (valid_mask was False AND a value was filled in).

    Behaviour at boundaries:

    * Leading invalid run is filled by repeating the first valid sample.
    * Trailing invalid run is filled by repeating the last valid sample.
    * If every sample is invalid, the input is returned unchanged.
    """
    s = np.asarray(series, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if s.shape[0] != valid.shape[0]:
        raise ValueError(
            f'length mismatch: series {s.shape[0]} vs valid_mask {valid.shape[0]}')
    n = s.shape[0]

    out = s.copy()
    interpolated_mask = np.zeros(n, dtype=bool)

    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return out, interpolated_mask

    invalid_idx = np.where(~valid)[0]
    for i in invalid_idx:
        left = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]
        if left.size and right.size:
            l = int(left[-1])
            r = int(right[0])
            t = (i - l) / float(r - l)
            out[i] = (1.0 - t) * s[l] + t * s[r]
        elif left.size:
            out[i] = s[int(left[-1])]
        else:
            out[i] = s[int(right[0])]
        interpolated_mask[i] = True
    return out, interpolated_mask
