"""Render an overlay video to visually verify `stable_bbox.npz`.

The user asked for a moving comparison rather than still frames — looking
at each crop in motion is the only reliable way to tell whether the
remaining wobble is encoder-stage residual or upstream bbox jitter.

What's drawn per frame
----------------------
- **Cyan rectangle**  : the legacy bbox = all-landmarks min/max (the source
  of the original "breathing" jitter). Reconstructed from the cached raw
  MediaPipe landmarks during this pass — we don't depend on the npz holding
  legacy series.
- **Yellow rectangle**: the stable-subset bbox **before** FIR LPF (raw
  series stored in `stable_bbox.npz`). Shows what the calibration alone buys.
- **Green rectangle** : the stable-subset bbox **after** FIR LPF (the
  series that DECA / SMIRK actually consume). Should sit very still even
  when the mouth opens or the subject blinks.
- **Red dots**        : the 15 stable landmark indices for sanity.
- **HUD**             : frame index, detection flag, raw vs smoothed size,
  and `Δsize` (frame-to-frame absolute difference, in source-image px).

A `bbox_verify_stats.csv` is written alongside the video with per-frame
numerical values so jitter can be quantified, not just eyeballed.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from glob import glob
from pathlib import Path

# Allow `python preprocess/bbox_verify.py` from the repo root.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from preprocess._smirk_constants import STABLE_LANDMARK_INDICES
from preprocess.stable_bbox import extract_bbox_center_size
from utils.general_utils import natural_sort_key, run_mediapipe


# Colours are BGR for cv2.
LEGACY_COLOR = (255, 200, 0)     # cyan-ish
RAW_STABLE_COLOR = (0, 200, 255) # amber
SMOOTH_COLOR = (0, 220, 0)       # green
LANDMARK_COLOR = (0, 0, 220)     # red
HUD_BG = (0, 0, 0)
HUD_FG = (255, 255, 255)


def _list_frames(image_dir: Path) -> list[Path]:
    paths = sorted(
        glob(f'{image_dir}/*.jpg') + glob(f'{image_dir}/*.png'),
        key=natural_sort_key,
    )
    return [Path(p) for p in paths]


def _draw_bbox(frame, center, size, color, scale: float = 1.4, label: str = ''):
    """Draw a `scale * size` square centred at `center`, axis-aligned. The
    drawn rectangle matches what `crop_face` would extract for warping."""
    half = (float(size) * float(scale)) / 2.0
    cx, cy = float(center[0]), float(center[1])
    p0 = (int(round(cx - half)), int(round(cy - half)))
    p1 = (int(round(cx + half)), int(round(cy + half)))
    cv2.rectangle(frame, p0, p1, color, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, (int(round(cx)), int(round(cy))), color,
                   markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    if label:
        cv2.putText(frame, label, (p0[0] + 4, p0[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_hud(frame, lines: list[str]):
    """Top-left translucent bar with one line per entry."""
    pad = 6
    line_h = 18
    h = pad * 2 + line_h * len(lines)
    w = max(180, max((len(s) for s in lines), default=0) * 8 + pad * 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, s in enumerate(lines):
        y = pad + line_h * (i + 1) - 4
        cv2.putText(frame, s, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_FG, 1, cv2.LINE_AA)


def make_verify_video(
    image_dir: Path, npz_path: Path, output_video: Path, fps: float,
    csv_path: Path | None = None,
) -> None:
    image_dir = Path(image_dir)
    npz_path = Path(npz_path)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    npz = np.load(npz_path, allow_pickle=False)
    smooth_centers = npz['center']
    smooth_sizes = npz['size']
    raw_centers = npz['raw_center']
    raw_sizes = npz['raw_size']
    detected = npz['detected']
    expected_basenames = npz['frame_basenames']

    frames = _list_frames(image_dir)
    if len(frames) != len(expected_basenames):
        raise RuntimeError(
            f'frame count mismatch: {len(frames)} on disk vs '
            f'{len(expected_basenames)} in {npz_path}. Did you regenerate '
            f'image/ without rerunning stable_bbox?')

    # Probe frame 0 for video dimensions.
    h0, w0 = np.array(Image.open(frames[0]).convert('RGB')).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, float(fps), (w0, h0))
    if not writer.isOpened():
        raise RuntimeError(f'failed to open video writer for {output_video}')

    # MediaPipe is needed again here to draw landmarks and recover the
    # legacy all-landmarks bbox (so the comparison is faithful, not
    # extrapolated). Identical detector config to stable_bbox.compute.
    from preprocess.stable_bbox import _make_detector
    detector = _make_detector()
    last_landmarks: np.ndarray | None = None

    csv_rows: list[dict] = []
    prev_smooth_size: float | None = None
    prev_raw_size: float | None = None

    for i, fp in enumerate(tqdm(frames, desc='bbox_verify')):
        rgb = np.array(Image.open(fp).convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        kpt = run_mediapipe(rgb, detector)
        if kpt is None:
            kpt = last_landmarks
            redet = False
        else:
            kpt = kpt[..., :2]
            last_landmarks = kpt
            redet = True

        # Re-derive the legacy (all-landmarks min/max) bbox for visual
        # comparison; it is intentionally NOT stored in the npz.
        if kpt is not None:
            legacy_c, legacy_s = extract_bbox_center_size(
                kpt, use_stable_subset=False, size_calibration=1.0,
            )
            _draw_bbox(bgr, legacy_c, legacy_s, LEGACY_COLOR,
                       label='legacy(all-lmk)')

            for idx in STABLE_LANDMARK_INDICES:
                px, py = int(round(kpt[idx, 0])), int(round(kpt[idx, 1]))
                cv2.circle(bgr, (px, py), 2, LANDMARK_COLOR, -1, cv2.LINE_AA)

        _draw_bbox(bgr, raw_centers[i], float(raw_sizes[i]),
                   RAW_STABLE_COLOR, label='stable raw')
        _draw_bbox(bgr, smooth_centers[i], float(smooth_sizes[i]),
                   SMOOTH_COLOR, label='stable + FIR')

        d_raw = (abs(float(raw_sizes[i]) - prev_raw_size)
                 if prev_raw_size is not None else 0.0)
        d_smooth = (abs(float(smooth_sizes[i]) - prev_smooth_size)
                    if prev_smooth_size is not None else 0.0)

        _draw_hud(bgr, [
            f'f={i:05d}  det={int(bool(detected[i]))}  redet={int(redet)}',
            f'raw   size={float(raw_sizes[i]):7.2f} px   d|s|={d_raw:6.3f}',
            f'smooth size={float(smooth_sizes[i]):7.2f} px   d|s|={d_smooth:6.3f}',
            f'center=({float(smooth_centers[i,0]):7.2f},'
            f'{float(smooth_centers[i,1]):7.2f})',
        ])

        writer.write(bgr)

        csv_rows.append({
            'frame': i,
            'basename': fp.name,
            'detected': int(bool(detected[i])),
            'raw_center_x': float(raw_centers[i, 0]),
            'raw_center_y': float(raw_centers[i, 1]),
            'raw_size': float(raw_sizes[i]),
            'smooth_center_x': float(smooth_centers[i, 0]),
            'smooth_center_y': float(smooth_centers[i, 1]),
            'smooth_size': float(smooth_sizes[i]),
            'd_raw_size': d_raw,
            'd_smooth_size': d_smooth,
        })
        prev_raw_size = float(raw_sizes[i])
        prev_smooth_size = float(smooth_sizes[i])

    writer.release()

    if csv_path is None:
        csv_path = output_video.with_suffix('.csv')
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    print(f'[bbox_verify] wrote {output_video} ({len(frames)} frames)')
    print(f'[bbox_verify] wrote {csv_path}')


def _resolve_image_dir(source: Path) -> Path:
    for c in [source / 'image', source / 'images']:
        if c.is_dir() and any(c.iterdir()):
            return c
    raise FileNotFoundError(f'no image/ or images/ directory under {source}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Render an overlay video comparing legacy / stable-raw / '
                    'stable-smoothed bbox per frame.')
    parser.add_argument('--source', type=str, required=True,
                        help='Dataset root containing image/ + stable_bbox.npz.')
    parser.add_argument('--fps', type=float, required=True,
                        help='Output video FPS (typically the source video FPS).')
    parser.add_argument('--npz', type=str, default=None,
                        help='Path to stable_bbox.npz (default: '
                             '<source>/stable_bbox.npz).')
    parser.add_argument('--output', type=str, default=None,
                        help='Output mp4 path (default: '
                             '<source>/bbox_verify.mp4).')
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    image_dir = _resolve_image_dir(source)
    npz_path = Path(args.npz) if args.npz else source / 'stable_bbox.npz'
    output = Path(args.output) if args.output else source / 'bbox_verify.mp4'

    make_verify_video(image_dir, npz_path, output, fps=float(args.fps))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
