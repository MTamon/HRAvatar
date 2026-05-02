#!/usr/bin/env python3
"""demo 3 — Visualize the features that HRAvatar's renderer actually receives.

Unlike DECA's raw per-frame output, the tensors fed into HRAvatar's
renderer come from ``preprocess/submodules/DECA/optimize.py`` — a
photometric + landmark + temporal refinement of DECA's initial guess.
The optimised state lives in ``tracked_params.json`` and contains, for
every frame:

* ``shapecode``     — one per video (identity, 100-dim by default)
* ``fullposecode``  — 15-dim: global 3 + neck 3 + jaw 3 + eye 6
* ``expcode``       — expression (n_expr, default 100)
* ``eyelids``       — 2-dim (left, right)
* ``translation``   — 3-dim (world-space)
* ``world_mat``     — 4×4 world-to-camera matrix
* ``intrinsics``    — ``[fx, fy, cx, cy]``

This demo:

  1. reads ``tracked_params.json`` (running the preprocessing pipeline
     first if ``--run_preprocess`` is set);
  2. re-implements the same LBS + offset chain HRAvatar uses at inference
     (``scene.gaussian_head_model.GaussianHeadModel.lbs_v2``) to produce
     the posed FLAME vertices that the Gaussian splats are attached to;
  3. projects those vertices with the exact camera the data loader
     builds (``w2c = diag(1,-1,-1,1) @ world_mat``, fx=fo=image_w/(2·tan(½·fov)));
  4. overlays the result on the original frame and writes an mp4.

This is what a downstream Listening-Head-Generation model must reproduce
if it wants to drive HRAvatar at inference time.  If the overlay here
tracks the input correctly, HRAvatar will render correctly too.

Modes (``--mode``)
------------------
``vertices``   — every FLAME vertex drawn as a cyan dot (like
                 ``MTamon/smirk@release/cuda128 --show_vertices``).
``wireframe``  — triangle edges projected to 2D.
``landmarks``  — 68-point face-alignment landmarks saved at
                 ``<subject_dir>/keypoint/*.npy`` (the landmarks the
                 optimizer was fit against).
``params_card``— numerical summary card in the top-left of each frame
                 (first six coefficients of shape/exp, full pose in
                 degrees, eyelids, translation).
``all``        — vertices + landmarks + params_card together.

Example
-------
::

    # Preprocess alice once, then just visualize.
    bash demos/_preprocess_subject.sh \\
        --sbj-root /data/subjects \\
        --sbj-name alice \\
        --video /data/raw/alice.mp4 \\
        --intrinsics hdtf
    python demos/demo_3_overlay_tracking.py \\
        --subject_dir /data/subjects/alice \\
        --output      /tmp/alice_features.mp4 \\
        --mode        all
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

# HRAvatar's own FLAME + LBS helpers — this demo reuses them so the
# result is bit-exact with what the renderer consumes.
from scene.flame import (  # noqa: E402
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    load_flame_mesh,
)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--subject_dir", required=True, type=str,
                   help="preprocessed subject dir (contains tracked_params.json + image/)")
    p.add_argument("--output", required=True, type=str, help="output mp4 path")
    p.add_argument("--mode",
                   choices=("vertices", "wireframe", "landmarks", "params_card", "all"),
                   default="all")
    p.add_argument("--n_shape", type=int, default=100,
                   help="shape dim used when the subject was tracked (default 100)")
    p.add_argument("--n_expr", type=int, default=100,
                   help="expression dim used when the subject was tracked (default 100)")
    p.add_argument("--flame_scale", type=float, default=4.0,
                   help="scale applied to FLAME output (HRAvatar default; 1.0 for _v2)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max_frames", type=int, default=0, help="0 = process all")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--vertex_radius", type=int, default=1)
    p.add_argument("--vertex_stride", type=int, default=1,
                   help="draw every Nth vertex (1 = all 5023)")
    p.add_argument("--vertex_color", type=str, default="0,255,255",
                   help="BGR triple, default cyan")
    p.add_argument("--alpha", type=float, default=0.8,
                   help="overlay alpha for vertices/wireframe (0..1)")
    return p.parse_args()


# ----------------------------------------------------------------------
# FLAME posing — bit-for-bit parity with GaussianHeadModel.lbs_v2 on the
# flame_vertexes branch (no teeth/mouth-interior extras, no Gaussians).
# ----------------------------------------------------------------------
def build_flame(device: str, n_shape: int, n_expr: int) -> dict[str, torch.Tensor]:
    """Replicate scene.__init__.load_head_info's FLAME setup.  Key step:
    ``lbs_weights`` is padded with a zero column so the LBS matmul matches
    HRAvatar's ``num_joints = J_regressor.shape[1] + 1`` convention."""
    mesh = load_flame_mesh(n_shape=n_shape, n_expr=n_expr,
                           use_add_teeth=False, use_add_mouth_interior=False,
                           file_name="assets/flame_model/flame2020.pkl")
    out = {}
    for k in ("v_template", "triangles", "shape_dirs", "expression_dirs",
              "pose_dirs", "J_regressor", "lbs_weights", "parents",
              "l_eyelid", "r_eyelid"):
        t = mesh[k]
        out[k] = (t if isinstance(t, torch.Tensor) else torch.as_tensor(t)).to(device)
    out["parents"] = out["parents"].long()
    out["triangles"] = out["triangles"].long()
    # lbs_weights: (V, 5) -> (V, 6) with a zero column for the virtual 6th joint.
    pad = torch.zeros_like(out["lbs_weights"][:, :1])
    out["lbs_weights"] = torch.cat([out["lbs_weights"], pad], dim=1)
    return out


def pose_flame(
    flame: dict[str, torch.Tensor],
    shape_params: torch.Tensor,        # (1, n_shape)
    expression_params: torch.Tensor,   # (1, n_expr)
    full_pose_params: torch.Tensor,    # (1, 15)
    eyelid_params: torch.Tensor | None,  # (1, 2) or None
    translation: torch.Tensor | None,    # (1, 1, 3) or None
    flame_scale: float,
) -> torch.Tensor:
    """Mirror of scene.gaussian_head_model.GaussianHeadModel.lbs_v2 on the
    pure-FLAME branch.  Returns world-space vertices (B, V, 3) after the
    flame_scale (default 4.0) has been applied."""
    device = shape_params.device
    B = shape_params.shape[0]

    v_shaped = (flame["v_template"].unsqueeze(0).expand(B, -1, -1).contiguous()
                + blend_shapes(shape_params, flame["shape_dirs"]))
    # Joints depend on shape only (matches GaussianHeadModel.flame_joint_center).
    J = torch.einsum("bvk,jv->bjk", v_shaped, flame["J_regressor"])

    v = v_shaped + blend_shapes(expression_params, flame["expression_dirs"])

    rot = batch_rodrigues(full_pose_params.view(-1, 3)).view(B, -1, 3, 3)
    pose_feat = (rot[:, 1:] - torch.eye(3, device=device)).view(B, -1)
    v = v + blend_shapes(pose_feat, flame["pose_dirs"])

    if eyelid_params is not None:
        v = v + flame["l_eyelid"].expand(B, -1, -1) * eyelid_params[:, 0:1, None]
        v = v + flame["r_eyelid"].expand(B, -1, -1) * eyelid_params[:, 1:2, None]

    _, A = batch_rigid_transform(rot, J, flame["parents"])
    # Extend with identity for the "free" joint LBS weights slot.
    A = torch.cat([A, torch.eye(4, device=device)[None, None].expand(B, 1, -1, -1)], dim=1)
    W = flame["lbs_weights"].unsqueeze(0).expand(B, -1, -1)
    T = torch.matmul(W, A.reshape(B, A.shape[1], 16)).view(B, -1, 4, 4)

    ones = torch.ones((B, v.shape[1], 1), device=device, dtype=v.dtype)
    vh = torch.cat([v, ones], dim=-1)
    v_world = torch.matmul(T, vh.unsqueeze(-1))[:, :, :3, 0]

    if translation is not None:
        v_world = v_world + translation

    return v_world * flame_scale


# ----------------------------------------------------------------------
# Camera: reproduce data_loader._load_camera.
# ----------------------------------------------------------------------
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def project(verts_world: np.ndarray,
            world_mat: np.ndarray,
            intrinsics: list[float],
            image_w: int,
            image_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) pixel coords and z_cam for every vertex.  Mirrors the
    HRAvatar data loader: w2c = diag(1,-1,-1,1) @ world_mat, then pin-hole
    with fo = image_w / (2 tan(½·fovx)), fovx = 2·arctan2(cx, fx)."""
    w2c = FLIP_YZ @ world_mat
    N = verts_world.shape[0]
    vh = np.concatenate([verts_world, np.ones((N, 1), dtype=np.float32)], axis=1)
    cam = (w2c @ vh.T).T[:, :3]

    fx = float(intrinsics[0]); cx = float(intrinsics[2])
    fovx = 2.0 * np.arctan2(cx, fx)
    fo = image_w / (2.0 * np.tan(fovx / 2.0))
    z = np.where(np.abs(cam[:, 2]) < 1e-6, np.sign(cam[:, 2]) * 1e-6 + 1e-6, cam[:, 2])
    u = fo * cam[:, 0] / z + image_w / 2.0
    v = fo * cam[:, 1] / z + image_h / 2.0
    uv = np.stack([u, v], axis=-1)
    return uv, z


# ----------------------------------------------------------------------
# Drawing helpers.
# ----------------------------------------------------------------------
def draw_vertices(img: np.ndarray, uv: np.ndarray, color, radius: int, stride: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for i in range(0, uv.shape[0], stride):
        x, y = uv[i]
        if 0 <= x < w and 0 <= y < h:
            if radius <= 1:
                out[int(y), int(x)] = color
            else:
                cv2.circle(out, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    return out


def draw_wireframe(img: np.ndarray, uv: np.ndarray, z: np.ndarray,
                   triangles: np.ndarray, color) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    # Only draw triangles in front of the camera.
    in_front = z < 0  # cv OpenCV-ish convention after FLIP_YZ ⇒ front is z<0
    for a, b, c in triangles:
        if not (in_front[a] and in_front[b] and in_front[c]):
            continue
        pa = (int(uv[a, 0]), int(uv[a, 1]))
        pb = (int(uv[b, 0]), int(uv[b, 1]))
        pc = (int(uv[c, 0]), int(uv[c, 1]))
        cv2.line(out, pa, pb, color, 1, cv2.LINE_AA)
        cv2.line(out, pb, pc, color, 1, cv2.LINE_AA)
        cv2.line(out, pc, pa, color, 1, cv2.LINE_AA)
    return out


def draw_landmarks(img: np.ndarray, lmks_xy: np.ndarray) -> np.ndarray:
    out = img.copy()
    for x, y in lmks_xy:
        cv2.circle(out, (int(round(float(x))), int(round(float(y)))),
                   2, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def draw_params_card(img: np.ndarray, params: dict) -> np.ndarray:
    out = img.copy()
    box = out[5:210, 5:360]
    box[:] = (box * 0.25 + 0).astype(np.uint8)
    lines = [
        f"shape[:6]   {np.array2string(params['shape'][:6], precision=2)}",
        f"exp[:6]     {np.array2string(params['exp'][:6], precision=2)}",
        f"pose_global {np.array2string(np.degrees(params['pose'][:3]), precision=1)} deg",
        f"pose_neck   {np.array2string(np.degrees(params['pose'][3:6]), precision=1)} deg",
        f"pose_jaw    {np.array2string(np.degrees(params['pose'][6:9]), precision=1)} deg",
        f"eyelids     {np.array2string(params['eyelids'], precision=2)}",
        f"translation {np.array2string(params['translation'], precision=2)}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 25 + 27 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------
# Main driver.
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    subject = Path(args.subject_dir).resolve()
    tracked = subject / "tracked_params.json"
    v2 = subject / "tracked_params_v2.json"
    if not tracked.exists() and v2.exists():
        tracked = v2
        args.flame_scale = 1.0
    if not tracked.exists():
        raise SystemExit(f"tracked_params.json not found in {subject}. "
                         f"Run demos/_preprocess_subject.sh first.")

    tp = json.loads(tracked.read_text())
    intrinsics = tp.get("intrinsics")
    if intrinsics is None:
        raise SystemExit("intrinsics missing from tracked_params.json — "
                         "was optimize.py run with --cx/--cy/--fx/--fy?")
    global_shape = torch.tensor(tp["shapecode"], dtype=torch.float32,
                                device=args.device)[:, : args.n_shape]

    image_dir = subject / "image"
    if not image_dir.exists():
        image_dir = subject / "images"
    image_paths = sorted(list(image_dir.glob("*.png")) +
                         list(image_dir.glob("*.jpg")) +
                         list(image_dir.glob("*.bmp")))
    if args.max_frames:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise SystemExit(f"no images under {image_dir}")

    # Peek at the first image for w/h.
    img0 = cv2.imread(str(image_paths[0]))
    if img0 is None:
        raise SystemExit(f"failed to read {image_paths[0]}")
    image_h, image_w = img0.shape[:2]

    flame = build_flame(args.device, args.n_shape, args.n_expr)
    triangles = flame["triangles"].detach().cpu().numpy()
    color = tuple(int(c) for c in args.vertex_color.split(","))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8)

    # face-alignment keypoints are saved as a single dict in keypoint.json.
    landmark_json = subject / "keypoint.json"
    landmarks = json.loads(landmark_json.read_text()) if landmark_json.exists() else {}

    for p in tqdm(image_paths, desc="overlay"):
        frame_bgr = cv2.imread(str(p))
        if frame_bgr is None:
            continue
        name = p.name
        key = p.stem if p.stem in tp else name
        if key not in tp:
            # Missing tracking for this frame — still write raw frame.
            writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            continue
        fr = tp[key]

        exp = torch.tensor(fr["expcode"], dtype=torch.float32,
                           device=args.device)[:, : args.n_expr]
        pose = torch.tensor(fr["fullposecode"], dtype=torch.float32,
                            device=args.device)
        eyelids = (torch.tensor(fr["eyelids"], dtype=torch.float32,
                                device=args.device) if "eyelids" in fr else None)
        translation = (torch.tensor(fr["translation"], dtype=torch.float32,
                                    device=args.device) if "translation" in fr else None)

        with torch.no_grad():
            v_world = pose_flame(flame, global_shape, exp, pose, eyelids,
                                 translation, args.flame_scale)

        v_world_np = v_world[0].detach().cpu().numpy().astype(np.float32)
        world_mat = np.asarray(fr["world_mat"], dtype=np.float32)
        uv, z = project(v_world_np, world_mat, intrinsics, image_w, image_h)

        out = frame_bgr
        if args.mode in ("vertices", "all"):
            layer = draw_vertices(out, uv, color, args.vertex_radius, args.vertex_stride)
            out = cv2.addWeighted(layer, args.alpha, out, 1.0 - args.alpha, 0.0)
        if args.mode in ("wireframe",):
            layer = draw_wireframe(out, uv, z, triangles, color)
            out = cv2.addWeighted(layer, args.alpha, out, 1.0 - args.alpha, 0.0)
        if args.mode in ("landmarks", "all"):
            lmk = landmarks.get(p.name) or landmarks.get(p.stem)
            if lmk is not None:
                out = draw_landmarks(out, np.asarray(lmk, dtype=np.float32).reshape(-1, 2))
        if args.mode in ("params_card", "all"):
            out = draw_params_card(out, {
                "shape": global_shape[0].detach().cpu().numpy(),
                "exp": exp[0].detach().cpu().numpy(),
                "pose": pose[0].detach().cpu().numpy(),
                "eyelids": (eyelids[0].detach().cpu().numpy()
                            if eyelids is not None else np.zeros(2)),
                "translation": (translation[0, 0].detach().cpu().numpy()
                                if translation is not None else np.zeros(3)),
            })

        writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    writer.close()
    print(f"wrote {len(image_paths)} frames -> {args.output}")


if __name__ == "__main__":
    main()
