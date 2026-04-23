#!/usr/bin/env python3
"""demo 3 — Overlay detection / tracking onto a video.

Runs an offline pipeline that, frame-by-frame:
  1. detects the face (FAN 2D from face-alignment),
  2. extracts features:
       - ``flame_mesh``  : DECA FLAME mesh projected + rendered back onto the
                           original frame via DECA's own rasterizer.
       - ``landmarks``   : 68-point face-alignment landmarks (red dots).
       - ``both``        : FLAME mesh + landmarks overlaid together.
  3. alpha-blends the feature onto the frame,
  4. writes an mp4.

No HRAvatar model is loaded.  This demo only depends on the DECA stack
already installed via ``bash setup.sh``.

Usage
-----
.. code-block:: bash

    python demos/demo_3_overlay_tracking.py \\
        --input  /data/raw/alice.mp4 \\
        --output /tmp/alice_overlay.mp4 \\
        --mode   flame_mesh \\
        --alpha  0.55

Reference: the overlay style and flag naming mirror
``MTamon/smirk@release/cuda128/demos/demo_video.py`` (--overlay /
--overlay_alpha).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

# Make the DECA submodule importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
DECA_DIR = REPO_ROOT / "preprocess" / "submodules" / "DECA"
sys.path.insert(0, str(DECA_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, type=str, help="source video (mp4/mov)")
    p.add_argument("--output", required=True, type=str, help="output mp4 path")
    p.add_argument(
        "--mode",
        choices=("flame_mesh", "landmarks", "both"),
        default="flame_mesh",
        help="what to overlay on each frame",
    )
    p.add_argument("--alpha", type=float, default=0.55, help="overlay alpha [0, 1]")
    p.add_argument("--fps", type=float, default=0.0, help="output fps (0 = copy input)")
    p.add_argument("--max_frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument(
        "--rasterizer_type",
        default="standard",
        choices=("standard", "pytorch3d"),
        help="DECA rasterizer backend",
    )
    p.add_argument(
        "--no_crop",
        action="store_true",
        help="treat the input as already face-cropped (skip FAN detection)",
    )
    return p.parse_args()


def load_deca(device: str, rasterizer_type: str):
    from decalib.deca import DECA  # noqa: WPS433
    from decalib.utils.config import cfg as deca_cfg  # noqa: WPS433

    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = rasterizer_type
    return DECA(config=deca_cfg, device=device)


def build_crop_transform(bbox, orig_h, orig_w, target=224):
    """Compute the affine that DECA's TestData uses internally."""
    left, top, right, bottom = bbox
    old_size = max(right - left, bottom - top) * 1.25
    center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    size = int(old_size)
    src = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ],
        dtype=np.float32,
    )
    dst = np.array([[0, 0], [0, target - 1], [target - 1, 0]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    return M, np.array([left, top, right, bottom])


def tensor_to_np(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def render_mesh_on_frame(deca, codedict, frame_bgr, tform_inv, device, alpha):
    """Render the FLAME mesh at the frame's original resolution and alpha-blend."""
    h, w = frame_bgr.shape[:2]
    original_t = (
        torch.from_numpy(frame_bgr[..., ::-1].astype(np.float32) / 255.0)
        .permute(2, 0, 1)[None]
        .to(device)
    )
    _, visdict = deca.decode(
        codedict,
        render_orig=True,
        original_image=original_t,
        tform=tform_inv,
    )
    shape_image = tensor_to_np(visdict["shape_images"][0])
    # DECA outputs RGB; convert to BGR for cv2 blending.
    shape_image = cv2.cvtColor(shape_image, cv2.COLOR_RGB2BGR)
    # The rendered face region sits on a gray background.  Use the alpha
    # channel exposed via `ops['alpha_images']` if present; otherwise
    # derive a mask by thresholding against the neutral gray.
    mask_key = "alpha_images"
    if mask_key in visdict:
        mask = tensor_to_np(visdict[mask_key][0])[..., :1].astype(np.float32) / 255.0
    else:
        gray = cv2.cvtColor(shape_image, cv2.COLOR_BGR2GRAY)
        mask = (np.abs(gray.astype(np.int32) - 127) > 5).astype(np.float32)[..., None]
    mask = np.clip(mask * alpha, 0, 1).astype(np.float32)
    blended = frame_bgr.astype(np.float32) * (1 - mask) + shape_image.astype(np.float32) * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_landmarks(frame_bgr, lmks_xy):
    out = frame_bgr.copy()
    for x, y in lmks_xy:
        cv2.circle(out, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
    return out


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    # face-alignment lazily imports torch; silence its warnings.
    import face_alignment  # noqa: WPS433

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device=args.device
    )

    reader = imageio.get_reader(args.input, "ffmpeg")
    meta = reader.get_meta_data()
    fps_in = meta.get("fps", 30.0)
    fps_out = args.fps if args.fps > 0 else fps_in
    writer = imageio.get_writer(args.output, fps=fps_out, codec="libx264", quality=8)

    need_deca = args.mode in ("flame_mesh", "both")
    deca = load_deca(args.device, args.rasterizer_type) if need_deca else None

    total = meta.get("nframes", None)
    pbar = tqdm(reader, total=total, desc="overlay")
    count = 0
    for frame_rgb in pbar:
        if args.max_frames and count >= args.max_frames:
            break
        count += 1
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]

        lmks = fa.get_landmarks_from_image(frame_rgb)
        if not lmks:
            writer.append_data(frame_rgb)
            continue
        lmks = lmks[0]

        out_bgr = frame_bgr
        if args.mode in ("flame_mesh", "both") and deca is not None:
            # Build a crop-affine centered on the detection (same as DECA's TestData).
            bbox = [
                float(lmks[:, 0].min()),
                float(lmks[:, 1].min()),
                float(lmks[:, 0].max()),
                float(lmks[:, 1].max()),
            ]
            if args.no_crop:
                # Pretend the whole frame is the crop: resize to 224 directly.
                target = 224
                crop = cv2.resize(frame_bgr, (target, target))
                M = np.array(
                    [
                        [(target - 1) / (w - 1), 0, 0],
                        [0, (target - 1) / (h - 1), 0],
                    ],
                    dtype=np.float32,
                )
            else:
                M, _ = build_crop_transform(bbox, h, w)
                crop = cv2.warpAffine(frame_bgr, M, (224, 224))
            inp = (
                torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
                .permute(2, 0, 1)[None]
                .to(args.device)
            )
            with torch.no_grad():
                codedict = deca.encode(inp)
            # DECA expects a 3x3 tform (homogeneous); wrap and invert.
            tform = np.vstack([M, [0, 0, 1]]).astype(np.float32)
            tform_t = torch.from_numpy(tform).to(args.device)
            tform_inv = torch.inverse(tform_t).transpose(0, 1)[None]
            out_bgr = render_mesh_on_frame(
                deca, codedict, out_bgr, tform_inv, args.device, args.alpha
            )

        if args.mode in ("landmarks", "both"):
            out_bgr = draw_landmarks(out_bgr, lmks)

        writer.append_data(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))

    writer.close()
    reader.close()
    print(f"wrote {count} frames -> {args.output}")


if __name__ == "__main__":
    main()
