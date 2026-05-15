#!/usr/bin/env python3
"""demo 4 — Side-by-side diagnostic of HRAvatar render vs FLAME input.

Purpose
-------
After training, look at ``outputs/<name>/<split>/ours_<E>/<name>_<split>_video.mp4``
and you may notice ghosting / a translucent doubled face that drifts
across frames. To localise the cause, this demo writes a 2x2 panel mp4
that lines up, frame-by-frame:

    +------------------+------------------+
    | HRAvatar render  | GT + FLAME mesh  |
    +------------------+------------------+
    | GT (raw frame)   | params + deltas  |
    +------------------+------------------+

If the FLAME mesh tracks the GT cleanly while the render ghosts, the
fault is downstream of ``tracked_params.json`` (Gaussian fit / training
loss / data loader). If the FLAME mesh itself wobbles or the params card
shows large per-frame ``delta`` jumps, the fault is upstream
(``preprocess/submodules/DECA/optimize.py`` produced unstable per-frame
parameters).

Implementation note
-------------------
The middle FLAME mesh is rendered with the same LBS + projection chain
that ``GaussianHeadModel.lbs_v2`` uses at runtime. We import
``scene/flame.py`` directly (bypassing ``scene/__init__.py``) so this
diagnostic does not pull in the heavy training-only deps (plyfile,
gaussian_model, etc.).

Usage
-----
::

    python demos/demo_4_render_vs_flame.py \\
        --model_path outputs/custom/MK6cN \\
        --subject_dir data/subjects/MK6cN \\
        --split train --epoch 15

    # 200-frame sample, every 2nd frame, custom output:
    python demos/demo_4_render_vs_flame.py \\
        --model_path outputs/custom/MK6cN \\
        --subject_dir data/subjects/MK6cN \\
        --frames 0:400 --every 2 \\
        --output /tmp/MK6cN_diagnose.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
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


# ----------------------------------------------------------------------
# Lazy load of scene/flame.py — sidesteps scene/__init__.py's heavy
# imports (plyfile, gaussian_model, etc.) that the diagnostic does not
# need. Lazy so that ``--help`` works even if FLAME-side deps (iopath,
# pytorch3d-style obj loader) are not installed in the active env.
# ----------------------------------------------------------------------
_SCENE_FLAME_MOD = None


def _scene_flame():
    global _SCENE_FLAME_MOD
    if _SCENE_FLAME_MOD is not None:
        return _SCENE_FLAME_MOD
    spec = importlib.util.spec_from_file_location(
        "scene_flame_only", REPO_ROOT / "scene" / "flame.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to locate scene/flame.py under {REPO_ROOT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scene_flame_only"] = mod
    spec.loader.exec_module(mod)
    _SCENE_FLAME_MOD = mod
    return mod


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_path", required=True, type=str,
                   help="HRAvatar output dir, e.g. outputs/custom/MK6cN")
    p.add_argument("--subject_dir", required=True, type=str,
                   help="preprocessed subject dir (contains tracked_params.json + image/)")
    p.add_argument("--split", default="train", choices=("train", "test"))
    p.add_argument("--epoch", type=int, default=15,
                   help="checkpoint epoch (matches the ours_<E> dir under model_path)")
    p.add_argument("--output", type=str, default="",
                   help="output mp4 (default: <model_path>/<split>/ours_<E>/diagnose_<name>.mp4)")
    p.add_argument("--frames", type=str, default="",
                   help="frame range like '0:400' (sliced into the split set)")
    p.add_argument("--every", type=int, default=1,
                   help="stride within frames (default 1 = every frame)")
    p.add_argument("--test_set_num", type=int, default=500,
                   help="must match the value used at training (default 500)")
    p.add_argument("--n_shape", type=int, default=100)
    p.add_argument("--n_expr", type=int, default=100)
    p.add_argument("--flame_scale", type=float, default=4.0,
                   help="HRAvatar default; auto-set to 1.0 when tracked_params_v2.json is used")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--mesh_alpha", type=float, default=0.7,
                   help="overlay alpha for the FLAME wireframe (0..1)")
    p.add_argument("--mesh_color", type=str, default="0,255,255",
                   help="BGR triple for FLAME wireframe (default cyan)")
    p.add_argument("--delta_pose_warn_deg", type=float, default=3.0,
                   help="highlight pose delta in red above this many deg/frame")
    p.add_argument("--delta_translation_warn", type=float, default=0.05,
                   help="highlight translation delta in red above this many world units/frame")
    p.add_argument("--trained_attrs", action="store_true",
                   help="Use the trained Gaussian point cloud + adapted "
                        "shape_dirs/expression_dirs/pose_dirs/lbs_weights from "
                        "<model_path>/saved_model/attributes_params.pth instead "
                        "of the standard FLAME mesh. This shows what the "
                        "*renderer actually sees* — the per-frame deformed "
                        "Gaussian point set — which can drift far from the "
                        "standard FLAME mesh when shape_param is in the "
                        "alien-head regime and the training has compensated "
                        "via shape_dirs. Requires plyfile (lazy import).")
    p.add_argument("--trained_xyz_stride", type=int, default=10,
                   help="when --trained_attrs is on, draw every Nth Gaussian "
                        "center as a dot (default 10 = ~1/10 of points). "
                        "Lower for denser visualization, higher for speed.")
    return p.parse_args()


# ----------------------------------------------------------------------
# FLAME posing — bit-for-bit parity with GaussianHeadModel.lbs_v2 on the
# flame_vertexes branch (mirrors demo_3_overlay_tracking.py).
# ----------------------------------------------------------------------
def build_flame(device: str, n_shape: int, n_expr: int) -> dict:
    sf = _scene_flame()
    mesh = sf.load_flame_mesh(n_shape=n_shape, n_expr=n_expr,
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
    pad = torch.zeros_like(out["lbs_weights"][:, :1])
    out["lbs_weights"] = torch.cat([out["lbs_weights"], pad], dim=1)
    return out


def pose_flame(flame, shape_params, expression_params, full_pose_params,
               eyelid_params, translation, flame_scale):
    sf = _scene_flame()
    device = shape_params.device
    B = shape_params.shape[0]

    v_shaped = (flame["v_template"].unsqueeze(0).expand(B, -1, -1).contiguous()
                + sf.blend_shapes(shape_params, flame["shape_dirs"]))
    J = torch.einsum("bvk,jv->bjk", v_shaped, flame["J_regressor"])

    v = v_shaped + sf.blend_shapes(expression_params, flame["expression_dirs"])

    rot = sf.batch_rodrigues(full_pose_params.view(-1, 3)).view(B, -1, 3, 3)
    pose_feat = (rot[:, 1:] - torch.eye(3, device=device)).view(B, -1)
    v = v + sf.blend_shapes(pose_feat, flame["pose_dirs"])

    if eyelid_params is not None:
        v = v + flame["l_eyelid"].expand(B, -1, -1) * eyelid_params[:, 0:1, None]
        v = v + flame["r_eyelid"].expand(B, -1, -1) * eyelid_params[:, 1:2, None]

    _, A = sf.batch_rigid_transform(rot, J, flame["parents"])
    A = torch.cat(
        [A, torch.eye(4, device=device)[None, None].expand(B, 1, -1, -1)], dim=1)
    W = flame["lbs_weights"].unsqueeze(0).expand(B, -1, -1)
    T = torch.matmul(W, A.reshape(B, A.shape[1], 16)).view(B, -1, 4, 4)

    ones = torch.ones((B, v.shape[1], 1), device=device, dtype=v.dtype)
    vh = torch.cat([v, ones], dim=-1)
    v_world = torch.matmul(T, vh.unsqueeze(-1))[:, :, :3, 0]

    if translation is not None:
        v_world = v_world + translation

    return v_world * flame_scale


# ----------------------------------------------------------------------
# Trained-attrs path: load Gaussian _xyz + adapted shape_dirs etc.
# Mirror of GaussianHeadModel.lbs_v2 (scene/gaussian_head_model.py:239-301)
# without the gradient-only branches (eval mode only).
# ----------------------------------------------------------------------
def _load_gaussian_xyz(ply_path: Path, device: str) -> torch.Tensor:
    try:
        from plyfile import PlyData
    except ImportError as e:
        raise SystemExit(
            "--trained_attrs requires plyfile. Install with: pip install plyfile"
        ) from e
    plydata = PlyData.read(str(ply_path))
    vx = plydata["vertex"]
    xyz = np.stack([np.asarray(vx["x"]), np.asarray(vx["y"]),
                    np.asarray(vx["z"])], axis=-1).astype(np.float32)
    return torch.tensor(xyz, dtype=torch.float32, device=device)


def _resolve_saved_model_dir(model_path: Path, epoch: int) -> Path:
    """Locate the directory holding attributes_params.pth + point_cloud.ply.

    HRAvatar saves checkpoints under one of these layouts:
      A) <model_path>/saved_model/epoch_<E>/{attributes_params.pth, point_cloud.ply}
      B) <model_path>/saved_model/{attributes_params.pth, point_cloud.ply}
    Try the per-epoch directory first, then fall back to the flat one.
    """
    candidates = [
        model_path / "saved_model" / f"epoch_{epoch}",
        model_path / "saved_model",
    ]
    for d in candidates:
        if (d / "attributes_params.pth").exists() and (d / "point_cloud.ply").exists():
            return d
    raise SystemExit(
        "trained attrs not found in any of:\n  "
        + "\n  ".join(str(d / "attributes_params.pth") for d in candidates))


def build_trained_flame(model_path: Path, epoch: int, device: str) -> dict:
    """Read attributes_params.pth + point_cloud.ply produced by HRAvatar.

    This returns a flame-like dict for ``pose_flame_trained``. Unlike the
    standard FLAME path, ``v_template`` here is the **Gaussian point cloud**
    (the learnable ``_xyz``), and the deformation bases are the trained
    ones — so the resulting per-frame point cloud is what the renderer
    actually splats.
    """
    saved_dir = _resolve_saved_model_dir(model_path, epoch)
    attrs_path = saved_dir / "attributes_params.pth"
    ply_path = saved_dir / "point_cloud.ply"

    attrs = torch.load(str(attrs_path), map_location="cpu", weights_only=False)
    xyz = _load_gaussian_xyz(ply_path, device)

    def _t(key):
        v = attrs[key]
        return v.detach().to(device) if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=device)

    flame_t = {
        "v_template": xyz,                     # (N, 3) — Gaussian centers
        "shape_dirs": _t("shape_dirs"),        # (N, 3, n_shape)
        "expression_dirs": _t("expression_dirs"),
        "pose_dirs": _t("pose_dirs"),
        "lbs_weights": _t("lbs_weights"),       # (N, n_joints+1)
        "parents": _t("rot_parents").long(),
        "flame_joint_center": _t("flame_joint_center"),  # (1, n_joints, 3)
        "r_eyelid_dirs": _t("r_eyelid_dirs"),
        "l_eyelid_dirs": _t("l_eyelid_dirs"),
        "shape_param_baked": _t("shape_param"),  # for sanity check vs tracked
        "flame_scale": float(attrs["flame_scale"].item()
                             if isinstance(attrs["flame_scale"], torch.Tensor)
                             else attrs["flame_scale"]),
    }
    return flame_t


def pose_flame_trained(flame_t: dict,
                       shape_param: torch.Tensor,
                       expression_param: torch.Tensor,
                       full_pose_param: torch.Tensor,
                       eyelid_param: torch.Tensor | None,
                       translation_param: torch.Tensor | None) -> torch.Tensor:
    """Mirror of ``GaussianHeadModel.lbs_v2`` (eval branch) using trained
    attrs. Returns world-space point cloud (B, N, 3) after flame_scale.
    """
    sf = _scene_flame()
    device = shape_param.device
    B = shape_param.shape[0]

    xyz = flame_t["v_template"]
    shape_dirs = flame_t["shape_dirs"]
    expression_dirs = flame_t["expression_dirs"]
    pose_dirs = flame_t["pose_dirs"]
    lbs_weights = flame_t["lbs_weights"]
    parents = flame_t["parents"]
    J = flame_t["flame_joint_center"]
    r_eyelid_dirs = flame_t["r_eyelid_dirs"]
    l_eyelid_dirs = flame_t["l_eyelid_dirs"]

    # The trained dirs may have fewer leading dims than tracked_params
    # provides. HRAvatar typically slices expression to 50 (SMIRK-compat)
    # while tracked_params.json keeps 100. Truncate per-call so the
    # einsum dims line up.
    n_shape_t = shape_dirs.shape[-1]
    n_expr_t = expression_dirs.shape[-1]
    if shape_param.shape[-1] > n_shape_t:
        shape_param = shape_param[..., :n_shape_t]
    if expression_param.shape[-1] > n_expr_t:
        expression_param = expression_param[..., :n_expr_t]

    xyz_canonical = xyz.unsqueeze(0).expand(B, -1, -1).contiguous()
    xyz_canonical = xyz_canonical + sf.blend_shapes(shape_param, shape_dirs)
    shape_offsets = sf.blend_shapes(expression_param, expression_dirs)
    xyz_shaped = xyz_canonical + shape_offsets

    rot_mats = sf.batch_rodrigues(full_pose_param.view(-1, 3)).view(B, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:] - torch.eye(3, device=device)).view(B, -1)
    pose_offsets = sf.blend_shapes(pose_feature, pose_dirs)
    xyz_posed = xyz_shaped + pose_offsets

    if eyelid_param is not None:
        xyz_posed = xyz_posed + r_eyelid_dirs.unsqueeze(0).expand(B, -1, -1) * eyelid_param[:, 1:2, None]
        xyz_posed = xyz_posed + l_eyelid_dirs.unsqueeze(0).expand(B, -1, -1) * eyelid_param[:, 0:1, None]

    if J.dim() == 2:
        J = J.unsqueeze(0)
    J_b = J.expand(B, -1, -1)
    _, A = sf.batch_rigid_transform(rot_mats, J_b, parents)
    A = torch.cat(
        [A, torch.eye(4, device=device)[None, None].expand(B, 1, -1, -1)], dim=1)

    # Match GaussianHeadModel.forward:386-387 — relu + row-normalize.
    lbs_w = torch.relu(lbs_weights)
    lbs_w = lbs_w / (lbs_w.sum(dim=-1, keepdim=True) + 1e-5)
    W = lbs_w.unsqueeze(0).expand(B, -1, -1)
    T = torch.matmul(W, A.reshape(B, A.shape[1], 16)).view(B, -1, 4, 4)

    ones = torch.ones((B, xyz_posed.shape[1], 1), device=device, dtype=xyz_posed.dtype)
    vh = torch.cat([xyz_posed, ones], dim=-1)
    v_world = torch.matmul(T, vh.unsqueeze(-1))[:, :, :3, 0]

    if translation_param is not None:
        v_world = v_world + translation_param

    return v_world * flame_t["flame_scale"]


def draw_dots(img: np.ndarray, uv: np.ndarray, z: np.ndarray,
              color, stride: int, radius: int = 1) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    in_front = z < 0
    for i in range(0, uv.shape[0], stride):
        if not in_front[i]:
            continue
        x, y = uv[i]
        if 0 <= x < w and 0 <= y < h:
            if radius <= 1:
                out[int(y), int(x)] = color
            else:
                cv2.circle(out, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------
# Camera projection — reproduces scene.data_loader._load_camera.
# ----------------------------------------------------------------------
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def project(verts_world, world_mat, intrinsics, image_w, image_h):
    w2c = FLIP_YZ @ world_mat
    N = verts_world.shape[0]
    vh = np.concatenate([verts_world, np.ones((N, 1), dtype=np.float32)], axis=1)
    cam = (w2c @ vh.T).T[:, :3]

    fx = float(intrinsics[0])
    cx = float(intrinsics[2])
    fovx = 2.0 * np.arctan2(cx, fx)
    fo = image_w / (2.0 * np.tan(fovx / 2.0))
    z = np.where(np.abs(cam[:, 2]) < 1e-6,
                 np.sign(cam[:, 2]) * 1e-6 + 1e-6, cam[:, 2])
    u = fo * cam[:, 0] / z + image_w / 2.0
    v = fo * cam[:, 1] / z + image_h / 2.0
    uv = np.stack([u, v], axis=-1)
    return uv, z


# ----------------------------------------------------------------------
# Drawing helpers.
# ----------------------------------------------------------------------
def draw_wireframe(img, uv, z, triangles, color):
    h, w = img.shape[:2]
    out = img.copy()
    in_front = z < 0
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


def draw_landmarks(img, lmks_xy):
    out = img.copy()
    for x, y in lmks_xy:
        cv2.circle(out, (int(round(float(x))), int(round(float(y)))),
                   2, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def draw_params_card_with_delta(canvas_hw, params, prev,
                                pose_warn_deg, translation_warn):
    H, W = canvas_hw
    out = np.zeros((H, W, 3), dtype=np.uint8)

    pose_deg = np.degrees(params["pose"])
    eyel = params["eyelids"]
    tr = params["translation"]
    if prev is not None:
        dpose_deg = np.degrees(params["pose"] - prev["pose"])
        dtr = tr - prev["translation"]
    else:
        dpose_deg = np.zeros_like(pose_deg)
        dtr = np.zeros_like(tr)
    d_global = float(np.linalg.norm(dpose_deg[:3]))
    d_neck = float(np.linalg.norm(dpose_deg[3:6]))
    d_jaw = float(np.linalg.norm(dpose_deg[6:9]))
    d_t = float(np.linalg.norm(dtr))

    def color_for(d, warn):
        return (60, 60, 255) if d > warn else (60, 220, 60)

    lines = [
        ("shape[:6]    ", np.array2string(params["shape"][:6], precision=2),
         (220, 220, 220)),
        ("exp[:6]      ", np.array2string(params["exp"][:6], precision=2),
         (220, 220, 220)),
        ("pose_global  ",
         f"{np.array2string(pose_deg[:3], precision=1)} deg  d={d_global:.2f}",
         color_for(d_global, pose_warn_deg)),
        ("pose_neck    ",
         f"{np.array2string(pose_deg[3:6], precision=1)} deg  d={d_neck:.2f}",
         color_for(d_neck, pose_warn_deg)),
        ("pose_jaw     ",
         f"{np.array2string(pose_deg[6:9], precision=1)} deg  d={d_jaw:.2f}",
         color_for(d_jaw, pose_warn_deg)),
        ("pose_eye_L   ", np.array2string(pose_deg[9:12], precision=1) + " deg",
         (220, 220, 220)),
        ("pose_eye_R   ", np.array2string(pose_deg[12:15], precision=1) + " deg",
         (220, 220, 220)),
        ("eyelids      ", np.array2string(eyel, precision=2),
         (220, 220, 220)),
        ("translation  ",
         f"{np.array2string(tr, precision=2)}  d={d_t:.3f}",
         color_for(d_t, translation_warn)),
    ]
    y = 38
    for label, val, col in lines:
        cv2.putText(out, label + val, (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        y += 36
    legend = "(green=stable, red=>warn threshold)"
    cv2.putText(out, legend, (16, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return out


def add_panel_label(img, text):
    out = img.copy()
    h, w = out.shape[:2]
    bar = max(28, h // 18)
    cv2.rectangle(out, (0, 0), (w, bar), (0, 0, 0), -1)
    cv2.putText(out, text, (10, int(bar * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def resize_to(img, H, W):
    if img.shape[:2] != (H, W):
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img


# ----------------------------------------------------------------------
# Frame indexing — mirrors scene/data_loader.py:117-123 split logic.
# ----------------------------------------------------------------------
def parse_frame_range(spec, n):
    if not spec:
        return 0, n
    if ":" in spec:
        a, b = spec.split(":", 1)
        return int(a or 0), int(b or n)
    v = int(spec)
    return v, v + 1


def list_split_imagepaths(subject_dir, split, test_set_num):
    image_dir = subject_dir / "image"
    if not image_dir.exists():
        image_dir = subject_dir / "images"
    paths = sorted(list(image_dir.glob("*.png")) +
                   list(image_dir.glob("*.jpg")) +
                   list(image_dir.glob("*.bmp")))
    if not paths:
        raise SystemExit(f"no images under {image_dir}")
    n = len(paths)
    train_set_len = max(0, n - test_set_num)
    return paths[:train_set_len] if split == "train" else paths[train_set_len:]


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    subject = Path(args.subject_dir).resolve()
    model = Path(args.model_path).resolve()

    tracked = subject / "tracked_params.json"
    v2 = subject / "tracked_params_v2.json"
    if not tracked.exists() and v2.exists():
        tracked = v2
        args.flame_scale = 1.0
    if not tracked.exists():
        raise SystemExit(f"tracked_params.json not found in {subject}")

    tp = json.loads(tracked.read_text())
    intrinsics = tp.get("intrinsics")
    if intrinsics is None:
        raise SystemExit("intrinsics missing from tracked_params.json")
    global_shape = torch.tensor(tp["shapecode"], dtype=torch.float32,
                                device=args.device)[:, : args.n_shape]

    image_paths = list_split_imagepaths(subject, args.split, args.test_set_num)
    if not image_paths:
        raise SystemExit(f"empty {args.split} split for {subject}")

    renders_dir = model / args.split / f"ours_{args.epoch}" / "renders"
    gt_dir = model / args.split / f"ours_{args.epoch}" / "gt"
    if not renders_dir.exists():
        raise SystemExit(f"renders dir not found: {renders_dir}")

    n_split = len(image_paths)
    n_renders = len(list(renders_dir.glob("*.png")))
    n = min(n_split, n_renders)
    if n_split != n_renders:
        print(f"[warn] split has {n_split} frames but renders has {n_renders}; "
              f"using min={n}")

    a, b = parse_frame_range(args.frames, n)
    a = max(0, a); b = min(n, b)
    if b <= a:
        raise SystemExit(f"empty range {a}:{b}")
    indices = list(range(a, b, max(1, args.every)))

    output = args.output
    if not output:
        scene_name = model.name
        output = str(model / args.split / f"ours_{args.epoch}" /
                     f"diagnose_{scene_name}.mp4")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    color = tuple(int(c) for c in args.mesh_color.split(","))
    if args.trained_attrs:
        trained_flame = build_trained_flame(model, args.epoch, args.device)
        flame = None
        triangles = None
        baked_shape = trained_flame["shape_param_baked"].detach().cpu().numpy().reshape(-1)
        tracked_shape = global_shape[0].detach().cpu().numpy()
        sh_match = np.allclose(baked_shape[: len(tracked_shape)],
                               tracked_shape, atol=1e-4)
        n_shape_t = trained_flame["shape_dirs"].shape[-1]
        n_expr_t = trained_flame["expression_dirs"].shape[-1]
        print(f"[trained_attrs] shape_param baked into checkpoint matches "
              f"tracked_params: {sh_match}")
        print(f"[trained_attrs] Gaussian point count: "
              f"{trained_flame['v_template'].shape[0]}")
        print(f"[trained_attrs] trained dirs dims  shape={n_shape_t}  "
              f"expression={n_expr_t}  "
              f"(tracked_params has shape={args.n_shape}, expr={args.n_expr}; "
              f"larger params are truncated per-call)")
        print(f"[trained_attrs] flame_scale baked = "
              f"{trained_flame['flame_scale']:.4f}")
    else:
        trained_flame = None
        flame = build_flame(args.device, args.n_shape, args.n_expr)
        triangles = flame["triangles"].detach().cpu().numpy()

    img0 = cv2.imread(str(image_paths[0]))
    if img0 is None:
        raise SystemExit(f"failed to read {image_paths[0]}")
    H, W = img0.shape[:2]

    landmark_json = subject / "keypoint.json"
    landmarks_dict = (json.loads(landmark_json.read_text())
                      if landmark_json.exists() else {})

    writer = imageio.get_writer(output, fps=args.fps,
                                codec="libx264", quality=8)

    prev_params = None
    for idx in tqdm(indices, desc="diagnose"):
        img_path = image_paths[idx]
        rend_path = renders_dir / f"{idx:05d}.png"
        gt_saved_path = gt_dir / f"{idx:05d}.png"

        gt_frame = cv2.imread(str(img_path))
        if gt_frame is None:
            continue

        rend = cv2.imread(str(rend_path))
        rend = resize_to(rend if rend is not None
                         else np.zeros((H, W, 3), np.uint8), H, W)

        gt_saved = cv2.imread(str(gt_saved_path)) if gt_saved_path.exists() else None
        gt_saved = resize_to(gt_saved if gt_saved is not None else gt_frame, H, W)

        # ---------- FLAME mesh + params ----------
        name = img_path.name
        key = img_path.stem if img_path.stem in tp else name
        if key not in tp:
            mesh_panel = gt_frame.copy()
            params_panel = np.zeros((H, W, 3), np.uint8)
            cv2.putText(params_panel, f"no tracked_params for {name}",
                        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (60, 60, 255), 1, cv2.LINE_AA)
        else:
            fr = tp[key]
            exp = torch.tensor(fr["expcode"], dtype=torch.float32,
                               device=args.device)[:, : args.n_expr]
            pose = torch.tensor(fr["fullposecode"], dtype=torch.float32,
                                device=args.device)
            eyelids = (torch.tensor(fr["eyelids"], dtype=torch.float32,
                                    device=args.device)
                       if "eyelids" in fr else None)
            translation = (torch.tensor(fr["translation"], dtype=torch.float32,
                                        device=args.device)
                           if "translation" in fr else None)

            with torch.no_grad():
                if args.trained_attrs:
                    v_world = pose_flame_trained(
                        trained_flame, global_shape, exp, pose,
                        eyelids, translation)
                else:
                    v_world = pose_flame(flame, global_shape, exp, pose,
                                         eyelids, translation, args.flame_scale)
            v_world_np = v_world[0].detach().cpu().numpy().astype(np.float32)
            world_mat = np.asarray(fr["world_mat"], dtype=np.float32)
            uv, z = project(v_world_np, world_mat, intrinsics, W, H)

            mesh_panel = gt_frame.copy()
            if args.trained_attrs:
                # Trained mode: visualise the actual deformed Gaussian
                # centers as dots (no triangulation available for the
                # learned point cloud).
                layer = draw_dots(mesh_panel, uv, z, color,
                                  args.trained_xyz_stride, radius=1)
                mesh_panel = cv2.addWeighted(layer, args.mesh_alpha, mesh_panel,
                                             1.0 - args.mesh_alpha, 0.0)
            else:
                wf = draw_wireframe(mesh_panel, uv, z, triangles, color)
                mesh_panel = cv2.addWeighted(wf, args.mesh_alpha, mesh_panel,
                                             1.0 - args.mesh_alpha, 0.0)
            lmk = landmarks_dict.get(name) or landmarks_dict.get(img_path.stem)
            if lmk is not None:
                mesh_panel = draw_landmarks(
                    mesh_panel,
                    np.asarray(lmk, dtype=np.float32).reshape(-1, 2))

            cur_params = {
                "shape": global_shape[0].detach().cpu().numpy(),
                "exp": exp[0].detach().cpu().numpy(),
                "pose": pose[0].detach().cpu().numpy(),
                "eyelids": (eyelids[0].detach().cpu().numpy()
                            if eyelids is not None else np.zeros(2)),
                "translation": (translation[0, 0].detach().cpu().numpy()
                                if translation is not None else np.zeros(3)),
            }
            params_panel = draw_params_card_with_delta(
                (H, W), cur_params, prev_params,
                args.delta_pose_warn_deg, args.delta_translation_warn)
            prev_params = cur_params

        # ---------- 2x2 layout ----------
        rend_l = add_panel_label(rend, f"HRAvatar render  idx={idx}  {img_path.name}")
        if args.trained_attrs:
            mesh_label = "GT + trained Gaussian point cloud (post-LBS)"
        else:
            mesh_label = "GT + standard FLAME mesh (DECA optimize view)"
        mesh_l = add_panel_label(mesh_panel, mesh_label)
        gt_l = add_panel_label(gt_saved, "GT (saved by render.py)")
        prm_l = add_panel_label(params_panel, "tracked_params + delta vs prev frame")

        top = np.concatenate([rend_l, mesh_l], axis=1)
        bot = np.concatenate([gt_l, prm_l], axis=1)
        canvas = np.concatenate([top, bot], axis=0)
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    writer.close()
    print(f"wrote {len(indices)} frames -> {output}")


if __name__ == "__main__":
    main()
