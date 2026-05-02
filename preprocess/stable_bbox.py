"""Pre-compute a temporally stabilized 224-crop bbox sequence.

Why this exists
---------------
Both the DECA preprocessing path (`submodules/DECA/demos/demo_reconstruct.py`)
and the SMIRK online encoder consumed at training time
(`scene/data_loader.py`) crop a 224x224 face patch per frame from
MediaPipe / FAN landmarks. The default `crop_face` derives the bbox from the
**min/max of every landmark**, which means mouth opening, blinking, and
per-frame detection noise leak into the crop `size`. SMIRK's weak-perspective
camera then absorbs that into the depth `Z`, and the rendered FLAME mesh
visibly "breathes" at the ears / scalp.

Upstream MTamon/smirk@release/cuda128 (PR #7) fixed this with two ideas:

1. A **stable landmark subset** (eye corners, nose bridge, temples — 15
   points that don't move with speech/blink) — so mouth/blink no longer leak
   into bbox `size` at all. The size source is decoupled from the expression.
2. A **zero-phase FIR low-pass** on the 1-D `size` series (offline, two-pass).
   Removes residual high-frequency detector noise without phase delay.

This module ports both ideas to HRAvatar and persists the result so that the
DECA preprocessor (offline) and the data_loader (online, every iteration)
both consume the *same* stabilized bbox sequence — i.e. the FLAME tracker
input no longer wobbles with the mouth.

Output format
-------------
`stable_bbox.npz` is written to the dataset root next to `image/`,
containing four parallel float64 / object arrays:

    frame_basenames : (N,) <U..  basename of the source image (e.g. "00042.png")
    center          : (N, 2)     bbox center in source-image pixel space
    size            : (N,)       smoothed size in source-image pixels
                                (after stable-subset extraction + FIR LPF)
    tform           : (N, 3, 3)  skimage SimilarityTransform.params,
                                  ready to consume as `warp(img, tform.inverse)`
    detected        : (N,)       True if MediaPipe detected a face on this
                                  frame; False means the previous valid
                                  landmark was reused.

Plus a small JSON sidecar `stable_bbox.meta.json` recording the parameters
used (cutoff, taps, scale, image_size, fps, calibration) so consumers can
sanity-check that the file matches the active config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from glob import glob
from pathlib import Path

# Allow `python preprocess/stable_bbox.py` from the repo root.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
from skimage.transform import estimate_transform
from tqdm import tqdm

from preprocess._smirk_constants import (
    STABLE_LANDMARK_INDICES,
    STABLE_LANDMARK_SIZE_CALIBRATION,
)
from utils.general_utils import natural_sort_key, run_mediapipe


WINDOW_SECONDS = 2.0  # FIR window length in seconds; matches FlashAvatar's
                     # default of 61 taps at 30 fps. Not exposed as a CLI
                     # flag — taps is auto-derived from fps so the *time*
                     # window is invariant across capture rates.


@dataclass
class StableBboxConfig:
    fps: float
    cutoff_hz: float = 2.5
    scale: float = 1.4
    image_size: int = 224
    use_stable_subset: bool = True
    size_calibration: float = STABLE_LANDMARK_SIZE_CALIBRATION
    smooth_center: bool = False
    center_cutoff_hz: float | None = None


# --- Geometry helpers (mirror MTamon/smirk utils/bbox_tracker.py) ----------

def extract_bbox_center_size(
    landmarks: np.ndarray,
    use_stable_subset: bool = True,
    size_calibration: float = STABLE_LANDMARK_SIZE_CALIBRATION,
) -> tuple[np.ndarray, float]:
    """Return ``(center_xy, size)`` with the same formula as `crop_face`.

    `size` follows the legacy ``(right - left + bottom - top) / 2`` convention,
    optionally rescaled by `size_calibration` when the stable subset is used
    (so that the resulting crop visually matches the legacy `scale=1.4` output).
    """
    if use_stable_subset:
        pts = landmarks[STABLE_LANDMARK_INDICES]
        cal = float(size_calibration)
    else:
        pts = landmarks
        cal = 1.0
    xs = pts[:, 0]
    ys = pts[:, 1]
    left = float(np.min(xs))
    right = float(np.max(xs))
    top = float(np.min(ys))
    bottom = float(np.max(ys))
    size = ((right - left) + (bottom - top)) / 2.0 * cal
    center = np.array(
        [(left + right) / 2.0, (top + bottom) / 2.0], dtype=np.float64,
    )
    return center, float(size)


def build_similarity_tform(
    center: np.ndarray, size: float,
    scale: float = 1.4, image_size: int = 224,
):
    """Reproduce the three control-point pairs `crop_face` uses, for a given
    (center, size). Drop-in compatible with `tform.params` / `tform.inverse`."""
    padded = int(float(size) * float(scale))
    cx, cy = float(center[0]), float(center[1])
    half = padded / 2.0
    src_pts = np.array([
        [cx - half, cy - half],
        [cx - half, cy + half],
        [cx + half, cy - half],
    ])
    dst_pts = np.array([
        [0, 0],
        [0, image_size - 1],
        [image_size - 1, 0],
    ])
    return estimate_transform('similarity', src_pts, dst_pts)


# --- FIR helpers -----------------------------------------------------------

def compute_taps(fps: float, window_seconds: float = WINDOW_SECONDS) -> int:
    """Auto-derive the FIR tap count from fps so the time-domain window is
    constant. `window_seconds=2.0` reproduces FlashAvatar's `taps=61` at
    30 fps; group delay = (taps-1)/(2*fps) seconds, cancelled by the
    zero-phase edge-padded `valid` convolution."""
    n = int(round(float(window_seconds) * float(fps)))
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1
    return n


def fir_lowpass_offline(
    series: np.ndarray, fps: float, cutoff_hz: float, taps: int,
    window: str = 'hamming',
) -> np.ndarray:
    """Zero-phase symmetric FIR low-pass for a 1-D series.

    Designed via `scipy.signal.firwin` (linear phase, no ringing with
    Hamming/Hann). `taps` is forced odd so group delay is integer; the
    input is edge-padded by `(taps-1)//2` samples on both sides and convolved
    in `valid` mode, recovering the original length with zero net phase.

    No-op fallback (returns a copy) when the cutoff is outside (0, fps/2)
    or when the series is shorter than the filter — same behaviour as
    SMIRK's upstream so callers never silently get NaNs / a shorter series.
    """
    from scipy import signal

    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError('series must be 1-D')
    if s.size == 0:
        return s.copy()

    nyq = float(fps) / 2.0
    if cutoff_hz >= nyq or cutoff_hz <= 0.0:
        return s.copy()

    taps = int(taps)
    if taps % 2 == 0:
        taps += 1
    if taps >= s.size:
        return s.copy()

    coef = signal.firwin(taps, cutoff_hz / nyq, window=window)
    half = (taps - 1) // 2
    padded = np.concatenate([np.full(half, s[0]), s, np.full(half, s[-1])])
    return np.convolve(padded, coef, mode='valid')


# --- Pipeline --------------------------------------------------------------

def _make_detector():
    """Build a MediaPipe FaceLandmarker matching `scene/data_loader.py`."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    asset = './assets/smirk/face_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=asset)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _list_frames(image_dir: Path) -> list[Path]:
    paths = sorted(
        glob(f'{image_dir}/*.jpg') + glob(f'{image_dir}/*.png'),
        key=natural_sort_key,
    )
    return [Path(p) for p in paths]


def compute_stable_bbox(
    image_dir: Path, output_path: Path, cfg: StableBboxConfig,
) -> dict:
    """Two-pass bbox stabilization. Returns a dict with the arrays for use
    by callers that want the result in-memory; also persists `output_path`
    (npz) and `output_path.with_suffix('.meta.json')`."""
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    frames = _list_frames(image_dir)
    if not frames:
        raise FileNotFoundError(f'no jpg/png frames under {image_dir}')

    detector = _make_detector()

    # Pass 1: per-frame MediaPipe + raw (center, size) extraction.
    # We propagate the previous valid landmark when MediaPipe fails on a
    # frame, mirroring `scene/data_loader.py`'s `landmark` global behaviour.
    raw_centers: list[np.ndarray] = []
    raw_sizes: list[float] = []
    detected_flags: list[bool] = []
    last_landmarks: np.ndarray | None = None

    for fp in tqdm(frames, desc='stable_bbox/pass-1'):
        img = np.array(Image.open(fp).convert('RGB'))
        kpt = run_mediapipe(img, detector)
        if kpt is None:
            if last_landmarks is None:
                raise RuntimeError(
                    f'MediaPipe failed on the first frame {fp}; trim the '
                    f'video so frame 0 contains a clearly visible face.')
            kpt = last_landmarks
            detected_flags.append(False)
        else:
            kpt = kpt[..., :2]
            last_landmarks = kpt
            detected_flags.append(True)

        c, s = extract_bbox_center_size(
            kpt,
            use_stable_subset=cfg.use_stable_subset,
            size_calibration=cfg.size_calibration,
        )
        raw_centers.append(c)
        raw_sizes.append(s)

    raw_centers_np = np.stack(raw_centers, axis=0)
    raw_sizes_np = np.asarray(raw_sizes, dtype=np.float64)

    # Pass 2: zero-phase FIR LPF on the size series. Centers are kept raw
    # by default (head translation is real motion we want to track) unless
    # the operator explicitly asks via `--smooth-center`.
    taps = compute_taps(cfg.fps)
    smooth_sizes = fir_lowpass_offline(
        raw_sizes_np, fps=cfg.fps, cutoff_hz=cfg.cutoff_hz, taps=taps,
    )
    smooth_centers = raw_centers_np.copy()
    if cfg.smooth_center:
        cc = (cfg.center_cutoff_hz
              if cfg.center_cutoff_hz is not None else cfg.cutoff_hz)
        smooth_centers[:, 0] = fir_lowpass_offline(
            raw_centers_np[:, 0], fps=cfg.fps, cutoff_hz=cc, taps=taps,
        )
        smooth_centers[:, 1] = fir_lowpass_offline(
            raw_centers_np[:, 1], fps=cfg.fps, cutoff_hz=cc, taps=taps,
        )

    # Pre-build per-frame similarity transforms so consumers can drop them
    # directly into `warp(img, tform_inverse)`.
    tform_params = np.zeros((len(frames), 3, 3), dtype=np.float64)
    for i in range(len(frames)):
        t = build_similarity_tform(
            smooth_centers[i], float(smooth_sizes[i]),
            scale=cfg.scale, image_size=cfg.image_size,
        )
        tform_params[i] = t.params

    basenames = np.array([f.name for f in frames])
    detected_arr = np.asarray(detected_flags, dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        frame_basenames=basenames,
        center=smooth_centers,
        size=smooth_sizes,
        tform=tform_params,
        detected=detected_arr,
        # Keep raw series too for diagnostics and the verify video. Cheap
        # (a few hundred kB).
        raw_center=raw_centers_np,
        raw_size=raw_sizes_np,
    )

    meta = asdict(cfg)
    meta['taps'] = taps
    meta['n_frames'] = int(len(frames))
    meta['n_detected'] = int(detected_arr.sum())
    meta_path = output_path.with_suffix('.meta.json')
    meta_path.write_text(json.dumps(meta, indent=2))

    print(
        f'[stable_bbox] wrote {output_path} '
        f'({len(frames)} frames, {meta["n_detected"]} detected, '
        f'taps={taps}, cutoff={cfg.cutoff_hz} Hz)')

    return {
        'frame_basenames': basenames,
        'center': smooth_centers,
        'size': smooth_sizes,
        'tform': tform_params,
        'detected': detected_arr,
        'raw_center': raw_centers_np,
        'raw_size': raw_sizes_np,
        'taps': taps,
    }


def _resolve_image_dir(source: Path) -> Path:
    """`crop_and_matting.py` writes to either `image/` or `images/`."""
    candidates = [source / 'image', source / 'images']
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    raise FileNotFoundError(
        f'no image/ or images/ directory under {source}; '
        f'run preprocess/crop_and_matting.py first')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Compute a temporally stabilized 224-crop bbox sequence '
                    '(stable-landmark subset + zero-phase FIR LPF on size).')
    parser.add_argument('--source', type=str, required=True,
                        help='Dataset root (containing image/ or images/).')
    parser.add_argument('--fps', type=float, required=True,
                        help='Source video FPS. Required for cutoff '
                             'normalisation against Nyquist.')
    parser.add_argument('--cutoff_hz', type=float, default=2.5,
                        help='FIR cutoff for the size series (default 2.5).')
    parser.add_argument('--scale', type=float, default=1.4,
                        help='bbox scale factor; matches data_loader.py '
                             'crop_face(..., scale=1.4) (default 1.4).')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Output crop size; matches the SMIRK encoder '
                             'input (default 224).')
    parser.add_argument('--all_landmarks', action='store_true',
                        help='Opt out of the stable subset and use all 478 '
                             'MediaPipe landmarks (legacy bbox source). '
                             'Use only when the stable subset misbehaves.')
    parser.add_argument('--smooth_center', action='store_true',
                        help='Also FIR-smooth the bbox center. Off by '
                             'default — head translation should track real '
                             'motion faithfully.')
    parser.add_argument('--center_cutoff_hz', type=float, default=None,
                        help='Center-channel cutoff (default: same as '
                             '--cutoff_hz when --smooth_center is set).')
    parser.add_argument('--output', type=str, default=None,
                        help='Output npz path. Defaults to '
                             '<source>/stable_bbox.npz.')
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    image_dir = _resolve_image_dir(source)
    output = Path(args.output) if args.output else source / 'stable_bbox.npz'

    cfg = StableBboxConfig(
        fps=float(args.fps),
        cutoff_hz=float(args.cutoff_hz),
        scale=float(args.scale),
        image_size=int(args.image_size),
        use_stable_subset=not args.all_landmarks,
        smooth_center=bool(args.smooth_center),
        center_cutoff_hz=(
            float(args.center_cutoff_hz)
            if args.center_cutoff_hz is not None else None
        ),
    )
    compute_stable_bbox(image_dir, output, cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
