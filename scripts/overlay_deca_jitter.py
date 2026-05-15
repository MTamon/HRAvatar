#!/usr/bin/env python3
"""Overlay DECA encoder (per-frame coarse) and DECA optimize (clip-wide
joint) FLAME meshes on the same image to visualize the per-frame
jitter that the encoder produces relative to the optimized output.

For each frame we project the SAME FLAME canonical mesh twice:

* in **blue** with the parameters DECA optimize wrote into
  ``tracked_params.json`` (smooth, clip-wide joint Adam),
* in **red** with the parameters DECA encoder wrote into ``code.json``
  (per-frame coarse, no temporal smoothness).

shape is taken from the clip-constant ``tracked_params.shapecode`` for
BOTH renders so the only thing that differs between the two colour
channels is expression and pose. The camera (``world_mat``,
``intrinsics``) is also shared — we are not trying to use DECA
encoder's orthographic ``cam`` as a real perspective camera; we just
want to see how much the per-frame coarse pose wiggles around the
clip-wide-smooth pose under a fixed viewpoint.

DECA encoder gives only ``pose[0:6] = [global_rot(3), jaw(3)]``, so
neck and eye-pose slots are zero-filled. ``translation`` and
``eyelids`` are also zero on the encoder side — we are isolating the
"rotation + jaw + expression" channels where the wiggle is most
visible.

Reuses the FLAME forward + projection helpers from
``demos/demo_3_overlay_tracking.py`` so the geometry is bit-equivalent
with the rest of the HRAvatar visualization tooling.

Example
-------
::

    python scripts/overlay_deca_jitter.py \\
        --subject_dir data/subjects/MK6cP \\
        --output      /tmp/MK6cP_deca_jitter.mp4

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse demo_3's FLAME forward and projection routines so the overlay
# geometry matches the rest of the HRAvatar tooling bit-for-bit.
from demos.demo_3_overlay_tracking import (  # noqa: E402
    build_flame,
    pose_flame,
    project,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--subject_dir", required=True, type=str,
                   help="preprocessed subject dir (contains tracked_params.json + code.json + image/)")
    p.add_argument("--output", required=True, type=str,
                   help="output mp4 path")
    p.add_argument("--n_shape", type=int, default=100,
                   help="shape dim used by the tracker (default 100)")
    p.add_argument("--n_expr", type=int, default=100,
                   help="expression dim to use when building FLAME (default 100; "
                        "covers HRAvatar's optimize default and pads DECA "
                        "encoder's 50d output with zeros)")
    p.add_argument("--flame_scale", type=float, default=4.0,
                   help="scale applied to FLAME output (HRAvatar default; 1.0 for _v2)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max_frames", type=int, default=0, help="0 = process all")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--vertex_stride", type=int, default=30,
                   help="draw every Nth vertex (default 30 of 5023)")
    p.add_argument("--vertex_radius", type=int, default=1)
    p.add_argument("--encoder_color", default="0,0,255",
                   help="BGR triple for DECA encoder dots (default red)")
    p.add_argument("--optimize_color", default="255,0,0",
                   help="BGR triple for DECA optimize dots (default blue)")
    return p.parse_args()


def _resolve_frame_key(name: str, stem: str, payload: dict) -> str | None:
    if name in payload:
        return name
    if stem in payload:
        return stem
    return None


def _pad_or_truncate(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Reshape arr to (target_dim,), zero-padding or truncating."""
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    if flat.shape[0] == target_dim:
        return flat
    if flat.shape[0] < target_dim:
        return np.concatenate(
            [flat, np.zeros(target_dim - flat.shape[0], dtype=np.float32)])
    return flat[:target_dim]


def _expand_deca_pose_to_full(pose_6: np.ndarray) -> np.ndarray:
    """DECA encoder emits ``pose = [global_rot(3), jaw(3)]`` (6-d).
    Expand to HRAvatar's 15-d ``fullposecode``
    ``[global_rot(3), neck(3), jaw(3), eye_l(3), eye_r(3)]`` by
    zero-filling neck and eyes."""
    pose_6 = np.asarray(pose_6, dtype=np.float32).reshape(-1)
    full = np.zeros(15, dtype=np.float32)
    if pose_6.shape[0] >= 3:
        full[:3] = pose_6[:3]
    if pose_6.shape[0] >= 6:
        full[6:9] = pose_6[3:6]
    return full


def main() -> None:
    args = parse_args()
    subject = Path(args.subject_dir).resolve()
    tracked = subject / "tracked_params.json"
    code = subject / "code.json"
    if not tracked.is_file():
        raise SystemExit(
            f"tracked_params.json not found in {subject}. "
            f"Run demos/_preprocess_subject.sh first.")
    if not code.is_file():
        raise SystemExit(
            f"code.json not found in {subject}. "
            f"This script needs DECA's per-frame coarse output "
            f"(written by preprocess/submodules/DECA/demos/demo_reconstruct.py "
            f"with --saveCode True; the avatar-fit pipeline writes it as a "
            f"matter of course).")

    tp = json.loads(tracked.read_text())
    cd = json.loads(code.read_text())
    intrinsics = tp.get("intrinsics")
    if intrinsics is None:
        raise SystemExit("intrinsics missing from tracked_params.json")

    global_shape = torch.tensor(
        tp["shapecode"], dtype=torch.float32, device=args.device
    )[:, : args.n_shape]

    image_dir = subject / "image"
    if not image_dir.exists():
        image_dir = subject / "images"
    image_paths = sorted(
        list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.bmp")))
    if args.max_frames:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise SystemExit(f"no images under {image_dir}")

    img0 = cv2.imread(str(image_paths[0]))
    if img0 is None:
        raise SystemExit(f"failed to read {image_paths[0]}")
    image_h, image_w = img0.shape[:2]

    flame = build_flame(args.device, args.n_shape, args.n_expr)
    expr_dirs_dim = int(flame["expression_dirs"].shape[2])

    enc_color = tuple(int(c) for c in args.encoder_color.split(","))
    opt_color = tuple(int(c) for c in args.optimize_color.split(","))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    writer = imageio.get_writer(
        args.output, fps=args.fps, codec="libx264", quality=8)

    skipped = 0
    for p in tqdm(image_paths, desc="overlay"):
        frame_bgr = cv2.imread(str(p))
        if frame_bgr is None:
            continue
        key_tp = _resolve_frame_key(p.name, p.stem, tp)
        key_cd = _resolve_frame_key(p.name, p.stem, cd)
        if key_tp is None or key_cd is None:
            writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            skipped += 1
            continue

        fr_tp = tp[key_tp]
        fr_cd = cd[key_cd]
        world_mat = np.asarray(fr_tp["world_mat"], dtype=np.float32)

        # --- DECA optimize (clip-wide joint) ---
        exp_tp_arr = _pad_or_truncate(fr_tp["expcode"], expr_dirs_dim)
        exp_tp = torch.tensor(exp_tp_arr, dtype=torch.float32,
                              device=args.device).unsqueeze(0)
        pose_tp = torch.tensor(
            fr_tp["fullposecode"], dtype=torch.float32, device=args.device)
        eyelids_tp = (
            torch.tensor(fr_tp["eyelids"], dtype=torch.float32,
                         device=args.device)
            if "eyelids" in fr_tp else None
        )
        translation_tp = (
            torch.tensor(fr_tp["translation"], dtype=torch.float32,
                         device=args.device)
            if "translation" in fr_tp else None
        )

        # --- DECA encoder (per-frame coarse) ---
        exp_cd_arr = _pad_or_truncate(fr_cd["exp"], expr_dirs_dim)
        exp_cd = torch.tensor(exp_cd_arr, dtype=torch.float32,
                              device=args.device).unsqueeze(0)
        pose_cd_arr = _expand_deca_pose_to_full(fr_cd["pose"])
        pose_cd = torch.tensor(pose_cd_arr, dtype=torch.float32,
                               device=args.device).unsqueeze(0)
        # Translation and eyelids are intentionally zero for the
        # encoder render: we want to isolate the rotation / expression
        # / jaw jitter rather than introduce a separate translation
        # bias from DECA's orthographic ``cam``.

        with torch.no_grad():
            v_tp = pose_flame(
                flame, global_shape, exp_tp, pose_tp, eyelids_tp,
                translation_tp, args.flame_scale)
            v_cd = pose_flame(
                flame, global_shape, exp_cd, pose_cd, None, None,
                args.flame_scale)

        uv_tp, _ = project(
            v_tp[0].detach().cpu().numpy().astype(np.float32),
            world_mat, intrinsics, image_w, image_h)
        uv_cd, _ = project(
            v_cd[0].detach().cpu().numpy().astype(np.float32),
            world_mat, intrinsics, image_w, image_h)

        out = frame_bgr.copy()
        for uv, color in ((uv_cd, enc_color), (uv_tp, opt_color)):
            for i in range(0, uv.shape[0], args.vertex_stride):
                x, y = uv[i]
                if 0 <= x < image_w and 0 <= y < image_h:
                    cv2.circle(out, (int(x), int(y)),
                               args.vertex_radius, color, -1, cv2.LINE_AA)

        # Legend
        cv2.putText(out, "DECA encoder (per-frame coarse)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, enc_color, 1, cv2.LINE_AA)
        cv2.putText(out, "DECA optimize (clip-wide joint)", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, opt_color, 1, cv2.LINE_AA)

        writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    writer.close()
    msg = f"wrote {len(image_paths)} frames -> {args.output}"
    if skipped:
        msg += f" (skipped {skipped} frames without tracking)"
    print(msg)


if __name__ == "__main__":
    main()
