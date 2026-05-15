#!/usr/bin/env python3
"""Overlay an ``lhg_features.npz`` on top of the source frames.

Use this to verify that the LHG online extraction path actually
captured the speaker's motion: the FLAME mesh rendered from the
per-frame features in the npz should land on the speaker's face. If
mesh / landmark / bbox overlays line up with the input video, we
treat the features as "correctly understood / correctly used" by the
extraction pipeline.

Modes (``--mode``)
------------------
``mesh``         — FLAME mesh vertex projected with the npz's pose +
                   shape + expression. Drawn as small dots so per-frame
                   jitter is visible.
``wireframe``    — FLAME mesh triangle edges projected to 2D. Far
                   easier to read than dots: the face contour and the
                   eye / nose / mouth structure are drawn as lines, so
                   whether the mesh sits on the face is obvious.
``bbox``         — ``stable_bbox.npz`` (center, size) → outer-crop
                   rectangle. Visualises the SMIRK 224 crop region.
``landmarks``    — face-alignment 68 pt landmarks read from
                   ``keypoint.json`` (the same landmarks DECA optimize
                   was fit against).
``params_card``  — numerical summary (rotation magnitude in deg,
                   translation z, expression norm, etc.) in the top-left.
``all``          — mesh + bbox + landmarks + params_card together (default).

``--face-only`` restricts the mesh / wireframe to the FLAME ``face``
region (1787 of 5023 vertices, from ``assets/flame_model/FLAME_masks.pkl``)
so the cheeks-ears-scalp-neck vertices that fall outside the visible
face are dropped.

This script reuses ``demos.demo_3_overlay_tracking``'s FLAME forward
and projection helpers, so the overlay geometry matches the rest of
the HRAvatar visualization tooling. Camera convention is HRAvatar
``(-Z forward)`` which the project helper handles via the
``diag(1, -1, -1, 1)`` pre-multiply on ``world_mat``.

Example
-------

    python scripts/overlay_online_features.py \\
        --lhg-features data/subjects/MK6cP/lhg_features.npz \\
        --source       data/subjects/MK6cP \\
        --output       /tmp/MK6cP_overlay.mp4 \\
        --mode all

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

# Reuse demo_3's FLAME forward + projection so the overlay geometry
# matches what the avatar fit pipeline produces.
from demos.demo_3_overlay_tracking import (  # noqa: E402
    build_flame,
    pose_flame,
    project,
    draw_vertices,
)
from lhg.output import LHGFeatures  # noqa: E402


def _draw_wireframe(
    img: np.ndarray,
    uv: np.ndarray,
    triangles: np.ndarray,
    color: tuple,
) -> np.ndarray:
    """Draw triangle edges of a projected mesh.

    Unlike ``demos.demo_3_overlay_tracking.draw_wireframe``, this does
    NOT z-cull: that helper assumes ``z < 0`` is front-facing, but the
    ``project()`` helper returns a positive camera-frame z (it applies
    ``FLIP_YZ`` to world_mat first), so every triangle would be
    discarded. ``--face-only`` already restricts triangles to the
    front of the face, so a plain edge draw with an in-frame check is
    both correct and fast.
    """
    h, w = img.shape[:2]
    out = img.copy()
    for a, b, c in triangles:
        pa = (int(round(uv[a, 0])), int(round(uv[a, 1])))
        pb = (int(round(uv[b, 0])), int(round(uv[b, 1])))
        pc = (int(round(uv[c, 0])), int(round(uv[c, 1])))
        # skip triangles entirely outside the frame
        if all(not (0 <= x < w and 0 <= y < h) for x, y in (pa, pb, pc)):
            continue
        cv2.line(out, pa, pb, color, 1, cv2.LINE_AA)
        cv2.line(out, pb, pc, color, 1, cv2.LINE_AA)
        cv2.line(out, pc, pa, color, 1, cv2.LINE_AA)
    return out


def _load_face_vertex_mask(n_vertices: int) -> np.ndarray:
    """Boolean (n_vertices,) mask of the FLAME ``face`` region.

    Reads ``assets/flame_model/FLAME_masks.pkl``; the ``face`` entry
    is a list of 1787 vertex indices covering the front of the face
    (excludes ears / scalp / neck). Used by --face-only to drop the
    mesh vertices that project outside the visible face.
    """
    import pickle

    masks_path = REPO_ROOT / 'assets' / 'flame_model' / 'FLAME_masks.pkl'
    if not masks_path.is_file():
        raise FileNotFoundError(
            f'FLAME_masks.pkl not found at {masks_path}; cannot use '
            f'--face-only.')
    with open(masks_path, 'rb') as f:
        masks = pickle.load(f, encoding='latin1')
    face_idx = np.asarray(masks['face'], dtype=np.int64)
    mask = np.zeros(n_vertices, dtype=bool)
    mask[face_idx[face_idx < n_vertices]] = True
    return mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--lhg-features', required=True, type=str,
                   help='Path to lhg_features.npz to visualize')
    p.add_argument('--source', required=True, type=str,
                   help='Subject dir containing image/, stable_bbox.npz, '
                        'keypoint.json (optional)')
    p.add_argument('--output', required=True, type=str,
                   help='Output mp4 path')
    p.add_argument('--mode',
                   choices=('all', 'mesh', 'wireframe', 'landmarks',
                            'bbox', 'params_card'),
                   default='all')
    p.add_argument('--face-only', action='store_true',
                   help='Restrict mesh / wireframe to the FLAME face '
                        'region (drops ears / scalp / neck vertices).')
    p.add_argument('--shapecode-json', default=None, type=str,
                   help='Path to a shapecode sidecar JSON (the file '
                        'lhg.teacher writes alongside the npz). Defaults '
                        'to <lhg-features>.shapecode.json then '
                        '<source>/tracked_params.json. zeros(100) fallback '
                        'with a warning if neither is available.')
    p.add_argument('--n_shape', type=int, default=100)
    p.add_argument('--n_expr', type=int, default=100,
                   help='FLAME expression dim. LHGFeatures.expression is '
                        '50d, gets zero-padded to this width.')
    p.add_argument('--vertex_stride', type=int, default=30,
                   help='Draw every Nth vertex (5023 total)')
    p.add_argument('--vertex_radius', type=int, default=1)
    p.add_argument('--fps', type=float, default=0.0,
                   help='Output fps; 0 = use features.fps')
    p.add_argument('--max_frames', type=int, default=0,
                   help='0 = process all')
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--mesh_color', default='0,255,0',
                   help='BGR triple for mesh dots (default green)')
    p.add_argument('--bbox_color', default='255,255,0',
                   help='BGR triple for bbox rectangle (default cyan)')
    p.add_argument('--landmark_color', default='0,0,255',
                   help='BGR triple for landmark dots (default red)')
    return p.parse_args()


def _resolve_shapecode(
    features: LHGFeatures,
    args: argparse.Namespace,
) -> np.ndarray:
    """Return a (n_shape,) float32 shapecode for FLAME forward.

    Priority: --shapecode-json > <features>.shapecode.json sidecar
    > <source>/tracked_params.json. Falls back to zeros + warning."""
    candidates: list[Path] = []
    if args.shapecode_json is not None:
        candidates.append(Path(args.shapecode_json))
    candidates.append(Path(args.lhg_features).with_suffix('.shapecode.json'))
    candidates.append(Path(args.source) / 'tracked_params.json')

    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f'[overlay] could not parse {path}: {exc}', file=sys.stderr)
            continue
        # Try the sidecar layout first
        if 'avatar_shapecode' in payload:
            arr = np.asarray(payload['avatar_shapecode'], dtype=np.float32).reshape(-1)
            print(f'[overlay] shapecode loaded from {path} (avatar_shapecode)')
            return arr[: args.n_shape]
        # tracked_params.json layout: top-level shapecode (1, n_shape)
        if 'shapecode' in payload:
            arr = np.asarray(payload['shapecode'], dtype=np.float32).reshape(-1)
            print(f'[overlay] shapecode loaded from {path} (shapecode)')
            return arr[: args.n_shape]

    print('[overlay] WARNING: no shapecode source found, falling back to '
          'zeros(n_shape) — mesh shape will not match the subject.',
          file=sys.stderr)
    return np.zeros(args.n_shape, dtype=np.float32)


def _resolve_image_paths(source_dir: Path) -> list[Path]:
    image_dir = source_dir / 'image'
    if not image_dir.exists():
        image_dir = source_dir / 'images'
    paths = (
        sorted(image_dir.glob('*.png'))
        + sorted(image_dir.glob('*.jpg'))
        + sorted(image_dir.glob('*.bmp'))
    )
    return paths


def _load_bbox_table(source_dir: Path) -> dict[str, tuple[int, int, int, int]] | None:
    """Load per-frame (x0, y0, x1, y1) bbox from stable_bbox.npz.

    Returns ``None`` when the file is missing OR the coord_system is
    not the outer_512 one (raw-coord boxes don't make sense to draw on
    image/ frames).
    """
    npz_path = source_dir / 'stable_bbox.npz'
    if not npz_path.is_file():
        return None
    with np.load(npz_path, allow_pickle=False) as npz:
        if 'coord_system' in npz.files:
            coord = str(npz['coord_system'])
            if coord != 'outer_512':
                print(f'[overlay] stable_bbox.npz coord_system={coord}; '
                      f'bbox overlay skipped (only outer_512 supported)',
                      file=sys.stderr)
                return None
        basenames = [str(b) for b in npz['frame_basenames']]
        centers = npz['center'].astype(np.float64)
        sizes = npz['size'].astype(np.float64)
    out: dict[str, tuple[int, int, int, int]] = {}
    # Use ``scale * size`` to approximate the 224-crop bbox in the
    # 512x512 outer frame. scale matches stable_bbox.py's default
    # SMIRK margin (1.6) — kept fixed here because the overlay is
    # purely diagnostic.
    scale = 1.6
    for i, b in enumerate(basenames):
        half = float(sizes[i]) * scale / 2.0
        cx, cy = centers[i]
        out[b] = (
            int(round(cx - half)),
            int(round(cy - half)),
            int(round(cx + half)),
            int(round(cy + half)),
        )
    return out


def _load_keypoint_table(source_dir: Path) -> dict[str, np.ndarray] | None:
    """Load ``keypoint.json`` if present, else None.

    Format: per-frame dict mapping image basename to (68, 2) ndarray.
    """
    path = source_dir / 'keypoint.json'
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        print(f'[overlay] could not parse {path}: {exc}', file=sys.stderr)
        return None
    return {
        str(k): np.asarray(v, dtype=np.float32).reshape(-1, 2)
        for k, v in payload.items()
    }


def _resolve_frame_basename(image_path: Path, table: dict) -> str | None:
    if image_path.name in table:
        return image_path.name
    if image_path.stem in table:
        return image_path.stem
    return None


def _build_full_pose(features: LHGFeatures, i: int, device: str) -> torch.Tensor:
    """Compose (1, 15) full_pose = [global_rot(3), neck(3), jaw(3), eye_pose(6)]
    from the LHGFeatures schema."""
    full = np.zeros(15, dtype=np.float32)
    full[0:3] = features.global_rot[i]
    full[3:6] = features.neck_pose[i]
    full[6:9] = features.jaw[i]
    full[9:15] = features.eye_pose[i]
    return torch.from_numpy(full).unsqueeze(0).to(device)


def _expand_expression(features: LHGFeatures, i: int, dim: int, device: str) -> torch.Tensor:
    arr = features.expression[i]
    if arr.shape[0] == dim:
        out = arr.astype(np.float32)
    elif arr.shape[0] < dim:
        out = np.zeros(dim, dtype=np.float32)
        out[: arr.shape[0]] = arr
    else:
        out = arr[:dim].astype(np.float32)
    return torch.from_numpy(out).unsqueeze(0).to(device)


def _draw_params_card(img: np.ndarray, features: LHGFeatures, i: int) -> np.ndarray:
    out = img.copy()
    box = out[5:175, 5:340]
    box[:] = (box * 0.25 + 0).astype(np.uint8)
    gr = features.global_rot[i]
    nk = features.neck_pose[i]
    jw = features.jaw[i]
    tr = features.translation[i]
    eyel = features.eyelid[i]
    exp_norm = float(np.linalg.norm(features.expression[i]))
    lines = [
        f'global_rot [{np.degrees(gr[0]):+6.1f}, {np.degrees(gr[1]):+6.1f}, {np.degrees(gr[2]):+6.1f}] deg',
        f'neck       [{np.degrees(nk[0]):+6.1f}, {np.degrees(nk[1]):+6.1f}, {np.degrees(nk[2]):+6.1f}] deg',
        f'jaw        [{np.degrees(jw[0]):+6.1f}, {np.degrees(jw[1]):+6.1f}, {np.degrees(jw[2]):+6.1f}] deg',
        f'translation[{tr[0]:+.3f}, {tr[1]:+.3f}, {tr[2]:+.3f}]',
        f'eyelid     [{eyel[0]:+.2f}, {eyel[1]:+.2f}]',
        f'exp |L2|   {exp_norm:.2f}',
    ]
    for j, line in enumerate(lines):
        cv2.putText(out, line, (10, 25 + 24 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    features = LHGFeatures.read(args.lhg_features)
    cam_conv = str(features.camera_convention)
    if cam_conv != 'hravatar':
        raise SystemExit(
            f'lhg_features.camera_convention={cam_conv!r}; only "hravatar" '
            f'is supported (project() assumes a HRAvatar-convention '
            f'world_mat). Re-extract with --camera-convention hravatar.')

    source_dir = Path(args.source).resolve()
    image_paths = _resolve_image_paths(source_dir)
    if not image_paths:
        raise SystemExit(f'no images under {source_dir / "image"}')

    if args.max_frames:
        image_paths = image_paths[: args.max_frames]

    img0 = cv2.imread(str(image_paths[0]))
    if img0 is None:
        raise SystemExit(f'failed to read {image_paths[0]}')
    image_h, image_w = img0.shape[:2]

    flame = build_flame(args.device, args.n_shape, args.n_expr)
    shapecode_np = _resolve_shapecode(features, args)
    shape = torch.from_numpy(shapecode_np).unsqueeze(0).to(args.device)

    triangles = flame['triangles'].detach().cpu().numpy()
    n_vertices = int(flame['v_template'].shape[0])

    # Optional FLAME face-region restriction for mesh / wireframe.
    face_mask = None
    face_triangles = triangles
    if args.face_only:
        face_mask = _load_face_vertex_mask(n_vertices)
        # keep only triangles whose three vertices are all in the face
        # region (so the wireframe doesn't trail edges out to the ears)
        keep = face_mask[triangles].all(axis=1)
        face_triangles = triangles[keep]

    draws_mesh = args.mode in ('all', 'mesh')
    draws_wireframe = args.mode == 'wireframe'
    # wireframe mode behaves like 'all' but swaps the mesh dots for
    # triangle edges — so bbox / landmarks / params_card come along.
    composite = args.mode in ('all', 'wireframe')
    draws_bbox = composite or args.mode == 'bbox'
    draws_landmarks = composite or args.mode == 'landmarks'
    draws_params_card = composite or args.mode == 'params_card'
    bbox_table = _load_bbox_table(source_dir) if draws_bbox else None
    landmark_table = _load_keypoint_table(source_dir) if draws_landmarks else None

    mesh_color = tuple(int(c) for c in args.mesh_color.split(','))
    bbox_color = tuple(int(c) for c in args.bbox_color.split(','))
    landmark_color = tuple(int(c) for c in args.landmark_color.split(','))

    world_mat = features.world_mat.astype(np.float32)
    intrinsics = features.intrinsics.tolist()
    flame_scale = float(features.flame_scale)
    fps = float(args.fps) if args.fps > 0 else float(features.fps)

    # Map LHGFeatures.frame_basenames to ordinal positions so we can
    # look up per-frame channels by image basename.
    basename_to_idx = {
        str(b): i for i, b in enumerate(features.frame_basenames)
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    writer = imageio.get_writer(args.output, fps=fps, codec='libx264', quality=8)

    skipped = 0
    for p in tqdm(image_paths, desc='overlay'):
        frame_bgr = cv2.imread(str(p))
        if frame_bgr is None:
            continue
        out = frame_bgr.copy()

        idx = basename_to_idx.get(p.name) or basename_to_idx.get(p.stem)
        if idx is None:
            writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            skipped += 1
            continue

        if draws_mesh or draws_wireframe:
            exp = _expand_expression(features, idx, args.n_expr, args.device)
            pose = _build_full_pose(features, idx, args.device)
            eyelid = torch.from_numpy(
                features.eyelid[idx].astype(np.float32)
            ).unsqueeze(0).to(args.device)
            translation = torch.from_numpy(
                features.translation[idx].astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(args.device)

            with torch.no_grad():
                v_world = pose_flame(
                    flame, shape, exp, pose, eyelid, translation, flame_scale)
            v_np = v_world[0].detach().cpu().numpy().astype(np.float32)
            uv, _ = project(v_np, world_mat, intrinsics, image_w, image_h)

            if draws_wireframe:
                out = _draw_wireframe(out, uv, face_triangles, mesh_color)
            else:
                # mesh dots; restrict to the face region if requested
                if face_mask is not None:
                    uv_draw = uv[face_mask]
                else:
                    uv_draw = uv
                out = draw_vertices(
                    out, uv_draw, mesh_color,
                    args.vertex_radius, args.vertex_stride)

        if draws_bbox and bbox_table is not None:
            key = _resolve_frame_basename(p, bbox_table)
            if key is not None:
                x0, y0, x1, y1 = bbox_table[key]
                cv2.rectangle(out, (x0, y0), (x1, y1), bbox_color, 1, cv2.LINE_AA)

        if draws_landmarks and landmark_table is not None:
            key = _resolve_frame_basename(p, landmark_table)
            if key is not None:
                lmk = landmark_table[key]
                # demo_3's draw_landmarks hardcodes red; inline a draw
                # loop so the colour is configurable.
                for x, y in lmk:
                    cv2.circle(out, (int(round(float(x))), int(round(float(y)))),
                               2, landmark_color, -1, cv2.LINE_AA)

        if draws_params_card:
            out = _draw_params_card(out, features, idx)

        writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    writer.close()
    msg = f'wrote {len(image_paths)} frames -> {args.output}'
    if skipped:
        msg += f' (skipped {skipped} frames not in lhg_features)'
    print(msg)


if __name__ == '__main__':
    main()
