"""Visualize per-frame jitter of one or more ``lhg_features.npz`` clips.

For each input npz the script plots three figures:

* ``jitter_static.png``       — per-frame value (raw signal)
* ``jitter_velocity.png``     — first difference (per-frame Δ)
* ``jitter_acceleration.png`` — second difference

The X axis is frame index. Each figure has 9 subplots arranged as a
3×3 grid: rows are channels (rotation, translation, jaw), columns are
axes (x, y, z). Multiple input npzs are overlaid on the same axes for
A/B comparison (e.g. ``--lookahead 0`` vs ``--lookahead 4`` vs
``--lookahead 12``).

Usage::

    python -m scripts.plot_lhg_jitter \
        --inputs run0=lhg_features_L0.npz \
                 run4=lhg_features_L4.npz \
                 run12=lhg_features_L12.npz \
        --output-dir verification/

    # or as a script:
    python scripts/plot_lhg_jitter.py --inputs ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


CHANNELS = ('global_rot', 'translation', 'jaw')
AXES = ('x', 'y', 'z')


def _load_series(path: Path) -> dict[str, np.ndarray]:
    """Return {channel_name: (N, D)} for the channels we plot."""
    with np.load(path, allow_pickle=False) as npz:
        out = {
            'global_rot': npz['global_rot'].astype(np.float64),
            'translation': npz['translation'].astype(np.float64),
            'jaw': npz['jaw'].astype(np.float64),
            'valid_mask': npz['valid_mask'].astype(bool),
        }
    return out


def _diff(arr: np.ndarray, order: int) -> np.ndarray:
    """Numerical derivative by repeated first differencing.

    Result is padded back to the original length with leading zeros so
    the X-axis stays aligned across the three figures.
    """
    if order == 0:
        return arr
    out = arr.astype(np.float64).copy()
    for _ in range(order):
        d = np.diff(out, axis=0)
        out = np.concatenate(
            [np.zeros((1, *out.shape[1:]), dtype=np.float64), d], axis=0,
        )
    return out


def _make_figure(
    series_per_label: dict[str, dict[str, np.ndarray]],
    order: int,
    title: str,
):
    import matplotlib.pyplot as plt

    fig, axarr = plt.subplots(
        len(CHANNELS), len(AXES), figsize=(15, 9), sharex=True,
    )
    fig.suptitle(title, fontsize=14)
    for r, ch in enumerate(CHANNELS):
        for c, ax_name in enumerate(AXES):
            ax = axarr[r, c]
            for label, series in series_per_label.items():
                arr = series[ch]
                if arr.shape[1] <= c:
                    continue
                d = _diff(arr, order)[:, c]
                ax.plot(d, label=label, linewidth=1.0, alpha=0.85)
            ax.set_title(f'{ch}.{ax_name}', fontsize=10)
            ax.grid(True, alpha=0.3)
            if r == len(CHANNELS) - 1:
                ax.set_xlabel('frame')
            if c == 0:
                ax.set_ylabel({0: 'value', 1: 'Δ', 2: 'Δ²'}[order])
    # Shared legend at the top of the figure.
    handles, labels = axarr[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _print_rms_summary(
    series_per_label: dict[str, dict[str, np.ndarray]],
) -> None:
    """Print per-label velocity/acceleration RMS so the user can verify
    that jitter monotonically decreases with larger lookahead without
    looking at the figures.
    """
    print()
    print('Velocity / acceleration RMS (lower = smoother):')
    header = '  {:<14s}'.format('label')
    for ch in CHANNELS:
        header += '  {:>22s}'.format(f'{ch} v / a')
    print(header)
    for label, series in series_per_label.items():
        line = '  {:<14s}'.format(label)
        for ch in CHANNELS:
            arr = series[ch].astype(np.float64)
            v = np.diff(arr, axis=0)
            a = np.diff(v, axis=0) if v.shape[0] >= 2 else np.zeros_like(v)
            v_rms = float(np.sqrt(np.mean(v * v))) if v.size else float('nan')
            a_rms = float(np.sqrt(np.mean(a * a))) if a.size else float('nan')
            line += '  {:>10.5f} / {:<9.5f}'.format(v_rms, a_rms)
        print(line)
    print()


def _parse_inputs(items: Iterable[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in items:
        if '=' in raw:
            label, path = raw.split('=', 1)
        else:
            label, path = Path(raw).stem, raw
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f'lhg_features.npz not found: {p}')
        out[label] = p
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Visualize per-frame jitter for lhg_features.npz files.')
    parser.add_argument(
        '--inputs', nargs='+', required=True,
        help='One or more "label=path" pairs (or just a path; the file '
             'stem is used as the label). Each input is overlaid on '
             'the same axes for A/B comparison.',
    )
    parser.add_argument(
        '--output-dir', default='verification',
        help='Directory to write the three PNG figures (default: verification/).',
    )
    parser.add_argument(
        '--no-summary', action='store_true',
        help='Skip the per-label velocity/acceleration RMS summary print.',
    )
    args = parser.parse_args(argv)

    paths = _parse_inputs(args.inputs)
    series_per_label: dict[str, dict[str, np.ndarray]] = {
        label: _load_series(path) for label, path in paths.items()
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    titles = {
        0: 'LHG features — static (per-frame value)',
        1: 'LHG features — velocity (1st difference)',
        2: 'LHG features — acceleration (2nd difference)',
    }
    suffixes = {
        0: 'jitter_static.png',
        1: 'jitter_velocity.png',
        2: 'jitter_acceleration.png',
    }

    for order in (0, 1, 2):
        fig = _make_figure(series_per_label, order, titles[order])
        out = out_dir / suffixes[order]
        fig.savefig(out, dpi=120)
        print(f'wrote {out}')

    if not args.no_summary:
        _print_rms_summary(series_per_label)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
