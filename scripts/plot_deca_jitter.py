"""Quantitative jitter comparison: DECA encoder (per-frame coarse,
``code.json``) vs DECA optimize (clip-wide joint, ``tracked_params.json``).

For each channel the script plots three figures aligned on the same
frame index:

* ``deca_jitter_static.png``       — per-frame value
* ``deca_jitter_velocity.png``     — first difference (Δ per frame)
* ``deca_jitter_acceleration.png`` — second difference

Each figure has a 3×3 grid: rows are channels (global_rot, jaw,
exp[:3]), columns are dims (x/y/z or first three coefficients).
Both series are overlaid on the same axes so the relative jitter is
visible at a glance, and the per-channel RMS of velocity and
acceleration is printed for each series — that is the number to
compare numerically against the visual impression.

Usage::

    python scripts/plot_deca_jitter.py \
        --subject_dir data/subjects/MK6cP \
        --output_dir  /tmp/deca_jitter

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _resolve_keys(payload: dict) -> list[str]:
    keys = [k for k in payload if isinstance(k, str) and (
        k.endswith('.png') or k.endswith('.jpg') or k.endswith('.bmp')
        or k.endswith('.jpeg'))]
    return sorted(keys)


def _load_optimize_series(path: Path) -> dict[str, np.ndarray]:
    """Extract per-frame channels from a DECA optimize tracked_params.json."""
    payload = json.loads(path.read_text())
    keys = _resolve_keys(payload)
    n = len(keys)

    global_rot = np.zeros((n, 3), dtype=np.float64)
    jaw = np.zeros((n, 3), dtype=np.float64)
    exp = np.zeros((n, 3), dtype=np.float64)  # first 3 coeffs
    translation = np.zeros((n, 3), dtype=np.float64)

    for i, k in enumerate(keys):
        fr = payload[k]
        if 'fullposecode' in fr:
            arr = np.asarray(fr['fullposecode']).reshape(-1)
            if arr.size >= 3:
                global_rot[i] = arr[:3]
            if arr.size >= 9:
                jaw[i] = arr[6:9]
        if 'expcode' in fr:
            arr = np.asarray(fr['expcode']).reshape(-1)
            if arr.size >= 3:
                exp[i] = arr[:3]
        if 'translation' in fr:
            arr = np.asarray(fr['translation']).reshape(-1)
            if arr.size >= 3:
                translation[i] = arr[:3]

    return {
        'global_rot': global_rot,
        'jaw': jaw,
        'exp': exp,
        'translation': translation,
        'keys': keys,
    }


def _load_encoder_series(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    """Extract per-frame channels from a DECA encoder code.json.

    Uses the same frame key order as the optimize series so the two
    can be plotted on aligned X axes.
    """
    payload = json.loads(path.read_text())
    n = len(keys)

    global_rot = np.zeros((n, 3), dtype=np.float64)
    jaw = np.zeros((n, 3), dtype=np.float64)
    exp = np.zeros((n, 3), dtype=np.float64)
    cam = np.zeros((n, 3), dtype=np.float64)

    for i, k in enumerate(keys):
        if k not in payload:
            continue
        fr = payload[k]
        if 'pose' in fr:
            arr = np.asarray(fr['pose']).reshape(-1)
            if arr.size >= 3:
                global_rot[i] = arr[:3]
            if arr.size >= 6:
                jaw[i] = arr[3:6]
        if 'exp' in fr:
            arr = np.asarray(fr['exp']).reshape(-1)
            if arr.size >= 3:
                exp[i] = arr[:3]
        if 'cam' in fr:
            arr = np.asarray(fr['cam']).reshape(-1)
            if arr.size >= 3:
                cam[i] = arr[:3]

    return {
        'global_rot': global_rot,
        'jaw': jaw,
        'exp': exp,
        'cam': cam,
    }


def _diff(arr: np.ndarray, order: int) -> np.ndarray:
    """Pad-leading-zero first/second differencing."""
    if order == 0:
        return arr
    out = arr.astype(np.float64).copy()
    for _ in range(order):
        d = np.zeros_like(out)
        d[1:] = out[1:] - out[:-1]
        out = d
    return out


def _rms(arr: np.ndarray) -> float:
    """Element-wise RMS across (N, D) — single scalar."""
    return float(np.sqrt(np.mean(arr ** 2)))


def _plot_figure(
    title: str,
    order: int,
    optimize: dict[str, np.ndarray],
    encoder: dict[str, np.ndarray],
    output_path: Path,
) -> dict[str, dict[str, float]]:
    """Write a 3x3 figure (rows: channels, cols: dims) and return RMS values."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    channels = ('global_rot', 'jaw', 'exp')
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    fig.suptitle(f'{title} (order={order})', fontsize=12)

    rms_table: dict[str, dict[str, float]] = {}
    for r, ch in enumerate(channels):
        opt_diff = _diff(optimize[ch], order)
        enc_diff = _diff(encoder[ch], order)
        rms_opt = _rms(opt_diff)
        rms_enc = _rms(enc_diff)
        rms_table[ch] = {'optimize': rms_opt, 'encoder': rms_enc}
        for c in range(3):
            ax = axes[r, c]
            ax.plot(opt_diff[:, c], color='tab:blue', linewidth=0.6,
                    alpha=0.85, label='DECA optimize')
            ax.plot(enc_diff[:, c], color='tab:red', linewidth=0.6,
                    alpha=0.85, label='DECA encoder')
            if c == 0:
                ax.set_ylabel(f'{ch}', fontsize=10)
            if r == 0:
                ax.set_title(f'dim {c}', fontsize=10)
            ax.grid(alpha=0.3)
            if r == 0 and c == 2:
                ax.legend(loc='upper right', fontsize=8)
        axes[r, -1].text(
            1.02, 0.5,
            f'RMS:\noptimize={rms_opt:.4f}\nencoder ={rms_enc:.4f}',
            transform=axes[r, -1].transAxes, fontsize=8, va='center')

    for c in range(3):
        axes[-1, c].set_xlabel('frame', fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return rms_table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--subject_dir', required=True, type=str,
                   help='preprocessed subject dir '
                        '(needs tracked_params.json + code.json)')
    p.add_argument('--output_dir', required=True, type=str,
                   help='directory to write deca_jitter_{static,velocity,acceleration}.png')
    args = p.parse_args()

    subject = Path(args.subject_dir).resolve()
    tracked = subject / 'tracked_params.json'
    code = subject / 'code.json'
    if not tracked.is_file():
        raise SystemExit(f'tracked_params.json not found in {subject}')
    if not code.is_file():
        raise SystemExit(f'code.json not found in {subject}')

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    optimize = _load_optimize_series(tracked)
    keys = optimize.pop('keys')
    encoder = _load_encoder_series(code, keys)

    print(f'loaded {len(keys)} frames')
    print(f'  optimize channels: {list(optimize)}')
    print(f'  encoder  channels: {list(encoder)}')
    print()

    titles = [
        ('static', 0, 'static'),
        ('velocity', 1, 'velocity (Δ per frame)'),
        ('acceleration', 2, 'acceleration (Δ² per frame)'),
    ]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for fname, order, label in titles:
        path = out_dir / f'deca_jitter_{fname}.png'
        rms = _plot_figure(label, order, optimize, encoder, path)
        summary[fname] = rms
        print(f'wrote {path}')

    print()
    print('=== RMS summary ===')
    print(f'{"channel":<12} {"order":<14} {"optimize":<12} {"encoder":<12}')
    for fname, _, _ in titles:
        for ch in ('global_rot', 'jaw', 'exp'):
            row = summary[fname][ch]
            print(f'{ch:<12} {fname:<14} {row["optimize"]:<12.4f} {row["encoder"]:<12.4f}')


if __name__ == '__main__':
    main()
