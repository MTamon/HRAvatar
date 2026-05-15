"""Symmetric (zero-phase) FIR low-pass filter for the LHG pipeline.

Both online (Stage 2) and offline (Stage 3) paths use the SAME filter
design (linear-phase symmetric FIR via ``scipy.signal.firwin``); they
differ only in the lookahead ``L``:

* Online: ``L=4`` default → ``taps = 9``, lookahead delay ``= L / fps``
  (160 ms @ 25 fps). ``StreamingSymmetricFIR.step(value)`` returns
  ``None`` for the first ``L`` calls (warming the lookahead buffer)
  and starts emitting filtered output ``L`` frames late from then on.
* Offline: ``L=12`` default → ``taps = 25``, applied to a complete
  series via ``apply_offline_zero_phase`` with edge padding.

The two paths share filter coefficients and frequency response, so the
distribution gap between training (offline) and inference (online)
collapses to "the offline series saw more taps and therefore a sharper
roll-off" — a quantitative, not qualitative, difference.

Why symmetric FIR rather than the One-Euro filter
-------------------------------------------------
Per ``feedback_no_one_euro_in_lhg_pipeline``: One-Euro is a causal
nonlinear filter whose group delay depends on signal speed, making the
preprocessing-induced timing offset hyperparameter-dependent and
hard to characterize for paper metrics. Symmetric FIR has an integer
group delay of exactly ``L`` samples, identical to the lookahead the
LHG model itself can be specified to consume — so the preprocessing
filter and the model lookahead can share the same buffer, removing the
extra latency entirely.

Channel scope (default)
-----------------------
LPF is applied ONLY to ``global_rot`` and ``translation`` (per-frame
EPnP-noise-bound). ``expression`` and ``eyelid`` pass through verbatim
(SMIRK timing is critical for lip-sync and blinks). ``jaw`` passes
through unless the caller opts in (``--lpf-jaw``); its cutoff defaults
to 10 Hz so syllable-rate motion (5-8 Hz) is preserved while
detector-noise above the speech band is attenuated.
"""
from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


def design_fir(
    lookahead: int, cutoff_hz: float, fps: float, window: str = 'hamming',
) -> np.ndarray:
    """Design a linear-phase symmetric FIR low-pass filter.

    Parameters
    ----------
    lookahead : int
        One-sided lookahead in frames. ``taps = 2 * lookahead + 1``.
        ``lookahead == 0`` returns the identity filter ``[1.0]`` so
        callers can disable smoothing without a separate code path.
    cutoff_hz : float
        Cutoff frequency in Hz. Must satisfy ``0 < cutoff_hz < fps/2``;
        anything outside this band returns the identity filter to match
        ``preprocess.stable_bbox.fir_lowpass_offline`` behaviour.
    fps : float
        Sample rate (frames / s) of the input series.
    window : str
        ``scipy.signal.firwin`` window argument. ``'hamming'`` (default)
        matches ``preprocess/stable_bbox.py``.

    Returns
    -------
    coef : (taps,) float64 — symmetric FIR coefficients summing to 1.0.
    """
    L = int(lookahead)
    if L < 0:
        raise ValueError(f'lookahead must be >= 0, got {L}')
    if L == 0:
        return np.array([1.0], dtype=np.float64)

    nyq = float(fps) / 2.0
    if cutoff_hz <= 0.0 or cutoff_hz >= nyq:
        return np.array([1.0], dtype=np.float64)

    from scipy import signal

    taps = 2 * L + 1
    coef = signal.firwin(taps, cutoff_hz / nyq, window=window)
    return np.asarray(coef, dtype=np.float64)


def apply_offline_zero_phase(
    series: np.ndarray, lookahead: int, cutoff_hz: float, fps: float,
    window: str = 'hamming',
) -> np.ndarray:
    """Zero-phase symmetric FIR LPF for an entire (N, D) or (N,) series.

    Edge-padded ``valid``-mode convolution — same pattern as
    ``preprocess.stable_bbox.fir_lowpass_offline`` extended to vector-
    valued samples. Output length matches the input.

    Parameters
    ----------
    series : (N,) or (N, D) float
    lookahead, cutoff_hz, fps, window : see ``design_fir``.

    Returns
    -------
    filtered : same shape as ``series``, dtype ``float64``.
    """
    s = np.asarray(series, dtype=np.float64)
    if s.size == 0:
        return s.copy()
    coef = design_fir(lookahead, cutoff_hz, fps, window=window)
    if coef.size == 1:
        return s.copy()
    half = (coef.size - 1) // 2
    if s.shape[0] <= coef.size:
        return s.copy()

    if s.ndim == 1:
        padded = np.concatenate([
            np.full(half, s[0]),
            s,
            np.full(half, s[-1]),
        ])
        return np.convolve(padded, coef, mode='valid')

    if s.ndim != 2:
        raise ValueError(f'series must be 1-D or 2-D, got shape {s.shape}')

    out = np.empty_like(s)
    for d in range(s.shape[1]):
        col = s[:, d]
        padded = np.concatenate([
            np.full(half, col[0]),
            col,
            np.full(half, col[-1]),
        ])
        out[:, d] = np.convolve(padded, coef, mode='valid')
    return out


class StreamingSymmetricFIR:
    """Online-friendly symmetric FIR with explicit lookahead semantics.

    Usage::

        f = StreamingSymmetricFIR(lookahead=4, cutoff_hz=4.0, fps=25.0)
        for x in stream:
            y = f.step(x)        # None for the first ``lookahead`` calls
            if y is not None:
                emit(y)          # filtered value of the frame that
                                 # arrived ``lookahead`` calls ago
        for y in f.flush():
            emit(y)              # trailing samples, edge-padded right

    Semantics (Option B in the design notes)
    ----------------------------------------
    ``.step(x_i)`` (called for input frame ``i``, 0-indexed) returns
    the FIR output for frame ``(i - lookahead)`` if ``i >= lookahead``;
    otherwise returns ``None``. For ``i < 2*lookahead`` we don't yet
    have the full ``2*lookahead+1``-sample kernel window; the missing
    samples on the LEFT are filled by repeating ``x_0``. This is the
    same edge-padding policy ``apply_offline_zero_phase`` uses.

    Total outputs across the lifetime of the filter equals the number
    of inputs: ``step()`` emits ``N - lookahead`` values during the
    main loop, and ``flush()`` emits the remaining ``lookahead`` (or
    ``N`` total when ``N <= lookahead``) by edge-padding the RIGHT.

    For an identity filter (``lookahead == 0``) the input is returned
    verbatim on every call and ``flush()`` is empty.
    """

    def __init__(
        self,
        lookahead: int,
        cutoff_hz: float,
        fps: float,
        window: str = 'hamming',
    ):
        self.lookahead = int(lookahead)
        self.cutoff_hz = float(cutoff_hz)
        self.fps = float(fps)
        self._coef = design_fir(self.lookahead, self.cutoff_hz, self.fps, window)
        # ``taps`` is always odd: 1 (identity) when lookahead==0,
        # otherwise 2L+1.
        self._taps = int(self._coef.size)
        self._buf: Deque[np.ndarray] = deque(maxlen=self._taps)
        self._n_seen = 0

    @property
    def taps(self) -> int:
        return self._taps

    def reset(self) -> None:
        self._buf.clear()
        self._n_seen = 0

    def _kernel_apply(self) -> np.ndarray:
        """Apply the kernel to the current ``_buf``. When ``_buf`` has
        fewer than ``_taps`` elements (i.e. we're at the leading edge),
        pad on the LEFT by repeating ``_buf[0]`` to reach ``_taps``.
        """
        n_buf = len(self._buf)
        if n_buf < self._taps:
            pad = self._taps - n_buf
            first = self._buf[0]
            samples = np.stack(
                [first] * pad + list(self._buf), axis=0,
            )
        else:
            samples = np.stack(list(self._buf), axis=0)
        return (self._coef[:, None] * samples).sum(axis=0)

    def step(self, value: np.ndarray) -> np.ndarray | None:
        """Push one input sample. Returns the LPF output for the frame
        that arrived ``lookahead`` calls ago, or ``None`` while still
        warming up.
        """
        v = np.asarray(value, dtype=np.float64)
        if self._taps == 1:
            return v.copy()

        self._buf.append(v.copy())
        self._n_seen += 1

        if self._n_seen <= self.lookahead:
            return None

        return self._kernel_apply()

    def flush(self) -> list[np.ndarray]:
        """Emit any outputs that ``step()`` did not yet produce.

        For ``N`` total inputs and lookahead ``L``, ``step()`` emitted
        ``max(0, N - L)`` outputs in the main loop; flush() must emit
        the remaining ``N - max(0, N - L) = min(N, L)``. Each flushed
        output is computed by appending an extra copy of the last
        input to the buffer (right-edge padding).
        """
        if self._taps == 1 or self._n_seen == 0:
            return []
        emitted_in_main = max(0, self._n_seen - self.lookahead)
        n_remaining = self._n_seen - emitted_in_main
        last = self._buf[-1]
        outputs: list[np.ndarray] = []
        for _ in range(n_remaining):
            self._buf.append(last.copy())
            outputs.append(self._kernel_apply())
        return outputs
