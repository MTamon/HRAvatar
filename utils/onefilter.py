"""One-Euro filter for temporal smoothing of per-frame tracker parameters.

Inspired by `MTamon/Gaussian-HS`_, which applies One-Euro filtering to DECA
outputs to remove per-frame tracker jitter that vanilla `tracked_params.json`
inherits from per-frame FAN / SMIRK regression.

The One-Euro filter (Casiez et al., 2012) is an adaptive low-pass filter
parameterised by:

* ``min_cutoff`` (Hz) - cutoff at zero velocity. Lower => smoother static.
* ``beta``               - speed coefficient. Higher => more responsive when fast.
* ``d_cutoff`` (Hz)      - derivative low-pass cutoff. Default 1.0 is fine.

It is **causal** but the offline ``smooth_sequence`` helper runs a forward +
backward pass to remove phase delay (zero-phase, equivalent in spirit to
``scipy.signal.filtfilt``).

This module is dependency-free (numpy only). All public functions accept
either a ``np.ndarray`` of shape ``(N, ...)`` or a list of arrays / tensors
and return the same container shape with smoothed values.

.. _MTamon/Gaussian-HS: https://github.com/MTamon/Gaussian-HS
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


_TWO_PI = 2.0 * math.pi


def _alpha(cutoff_hz: float, fps: float) -> float:
    """Standard One-Euro alpha for a given cutoff and sampling rate."""
    if cutoff_hz <= 0.0:
        return 0.0
    tau = 1.0 / (_TWO_PI * cutoff_hz)
    te = 1.0 / max(fps, 1e-6)
    return 1.0 / (1.0 + tau / te)


class OneEuroFilter:
    """Per-channel One-Euro filter operating on numpy arrays.

    The filter keeps state across calls. ``reset()`` clears it. For offline
    sequence smoothing prefer :func:`smooth_sequence` which runs forward +
    backward and then averages, removing causal phase lag.
    """

    def __init__(
        self,
        fps: float,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = float(fps)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()

        dt = 1.0 / self.fps
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, self.fps)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = np.empty_like(cutoff)
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = 1.0 / (_TWO_PI * np.maximum(cutoff, 1e-12))
            a = 1.0 / (1.0 + tau / dt)

        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def smooth_sequence(
    seq: np.ndarray,
    fps: float,
    min_cutoff: float = 1.0,
    beta: float = 0.0,
    d_cutoff: float = 1.0,
    bidirectional: bool = True,
) -> np.ndarray:
    """Offline smoothing of an (N, ...) sequence with the One-Euro filter.

    ``bidirectional=True`` runs a forward and a reverse-time pass and averages
    them so the output has zero net phase lag (analogous to ``filtfilt``).
    """
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim == 0:
        return arr.copy()
    if arr.shape[0] < 2:
        return arr.copy()

    def _run(x: np.ndarray) -> np.ndarray:
        flt = OneEuroFilter(fps=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = flt(x[i])
        return out

    fwd = _run(arr)
    if not bidirectional:
        return fwd.astype(arr.dtype, copy=False)
    bwd = _run(arr[::-1])[::-1]
    return (0.5 * (fwd + bwd)).astype(arr.dtype, copy=False)


def smooth_sequence_list(
    items: Sequence[np.ndarray] | Iterable[np.ndarray],
    fps: float,
    min_cutoff: float = 1.0,
    beta: float = 0.0,
    d_cutoff: float = 1.0,
    bidirectional: bool = True,
) -> list[np.ndarray]:
    """Stack a list of per-frame arrays, smooth, and split back."""
    items_list = list(items)
    if len(items_list) < 2:
        return [np.asarray(x).copy() for x in items_list]
    stacked = np.stack([np.asarray(x) for x in items_list], axis=0)
    smoothed = smooth_sequence(
        stacked,
        fps=fps,
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff,
        bidirectional=bidirectional,
    )
    return [smoothed[i] for i in range(smoothed.shape[0])]


class CausalOneEuroBuffer:
    """Tiny stateful wrapper used at render-time per (model, sequence) pair.

    Unlike :class:`OneEuroFilter` this dispatches one filter per leaf tensor
    shape, keyed by an explicit channel name. Useful when SMIRK encoder
    outputs come in a dict and we want one filter per key.
    """

    def __init__(
        self,
        fps: float,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.fps = fps
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._filters: dict[str, OneEuroFilter] = {}

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._filters.clear()
        elif key in self._filters:
            self._filters[key].reset()

    def filter(self, key: str, x: np.ndarray) -> np.ndarray:
        flt = self._filters.get(key)
        if flt is None:
            flt = OneEuroFilter(
                fps=self.fps,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff,
            )
            self._filters[key] = flt
        return flt(x)
