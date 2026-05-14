"""Convert an offline ``tracked_params.json`` into a teacher ``lhg_features.npz``.

This is the inverse of ``lhg.render_adapter.features_to_tracked_params``:
the offline pipeline (``demos/_preprocess_subject.sh`` + DECA
``optimize.py``) produces ``tracked_params.json`` with per-frame
``expcode`` / ``fullposecode`` / ``translation`` / ``eyelids`` entries
plus the clip-constants ``world_mat`` / ``shapecode`` / ``intrinsics``.
The LHG model treats these per-frame values as the supervision target
(see ``project_lhg_grand_design``), so they need to be re-packaged into
the ``LHGFeatures`` schema (``lhg/output.py``) that the rest of the LHG
training tooling consumes.

Layout mapping (mirror of ``render_adapter.features_to_tracked_params``):

    LHGFeatures.expression       <-  tracked_params[frame].expcode[0, :50]
    LHGFeatures.jaw              <-  tracked_params[frame].fullposecode[0, 6:9]
    LHGFeatures.eyelid           <-  tracked_params[frame].eyelids
    LHGFeatures.global_rot       <-  tracked_params[frame].fullposecode[0, 0:3]
    LHGFeatures.translation      <-  tracked_params[frame].translation[0]
    LHGFeatures.neck_pose        <-  tracked_params[frame].fullposecode[0, 3:6]
    LHGFeatures.eye_pose         <-  tracked_params[frame].fullposecode[0, 9:15]
    LHGFeatures.world_mat        <-  tracked_params.world_mat (clip-constant)
    LHGFeatures.intrinsics       <-  tracked_params.intrinsics
    LHGFeatures.outer_bbox       <-  sidecar outer_offset.json (or fallback)
    LHGFeatures.valid_mask       <-  all True (offline always emits a value)
    LHGFeatures.interpolated_mask <- all False (no dropout interpolation offline)
    LHGFeatures.rejected_mask    <-  all False (no Hampel reject offline)
    LHGFeatures.mode             <-  'offline_teacher'

``shapecode`` does not have a slot in ``LHGFeatures``. It is written
alongside the npz as a sidecar JSON file ``<output>.shapecode.json`` so
downstream callers can recover it without round-tripping through the
original ``tracked_params.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _natural_sorted_frame_keys(payload: dict) -> list[str]:
    """Return the frame keys of ``tracked_params.json`` in natural order.

    Frame keys are per-image filenames like ``"00042.png"`` (or stems
    like ``"00042"`` in some legacy variants). All other top-level keys
    (``world_mat`` / ``shapecode`` / ``intrinsics`` / etc.) are filtered
    out. The natural sort here matches ``natsorted`` in the rest of the
    repo: numeric stems compare numerically.
    """
    image_keys = [
        k for k in payload
        if isinstance(k, str) and (
            k.endswith('.png') or k.endswith('.jpg') or k.endswith('.bmp')
            or k.endswith('.jpeg')
        )
    ]
    if not image_keys:
        # legacy stem-only keys
        image_keys = [
            k for k, v in payload.items()
            if isinstance(v, dict) and (
                'expcode' in v or 'fullposecode' in v or 'translation' in v
            )
        ]
    return sorted(image_keys, key=lambda k: (len(k), k))


def _resolve_outer_bbox(
    tracked_params_path: Path, image_size: int,
) -> np.ndarray:
    """Look up ``outer_offset.json`` alongside the tracked_params.json.

    The bbox layout is ``[xmin, xmax, ymin, ymax]`` to match
    ``LHGFeatures.outer_bbox``. When the sidecar is missing, fall back
    to ``[0, image_size, 0, image_size]`` (the entire outer crop) — the
    same convention ``lhg/extract.py`` uses for image-directory inputs
    without a sidecar.
    """
    sidecar = tracked_params_path.parent / 'outer_offset.json'
    if sidecar.is_file():
        with open(sidecar) as fp:
            d = json.load(fp)
        return np.array(
            [int(d['x_min']), int(d['x_max']), int(d['y_min']), int(d['y_max'])],
            dtype=np.int32,
        )
    return np.array([0, image_size, 0, image_size], dtype=np.int32)


def build_features_from_tracked_params(
    tracked_params_path: str | Path,
    fps: float = 25.0,
    image_size: int = 512,
    expression_dim: int = 50,
):
    """Read ``tracked_params.json`` and return an ``LHGFeatures`` instance.

    Parameters
    ----------
    tracked_params_path : path to a Stage 1 ``tracked_params.json`` (or
        ``_v2`` sibling).
    fps : feature stream FPS to record in metadata (LHG convention is
        25 FPS, see ``project_lhg_fps_convention``).
    image_size : outer-crop square size (512 in HRAvatar default).
        Only used for the ``outer_bbox`` fallback when no
        ``outer_offset.json`` sidecar is present.
    expression_dim : number of expression dims to keep from the offline
        ``expcode`` (HRAvatar stores 100d, SMIRK consumes the first
        50d). Default 50 matches ``LHGFeatures.expression`` shape.
    """
    # Import here so the CLI can be imported as a library without
    # forcing the full lhg dependency tree.
    from lhg.calibration import load_stage_one_calibration
    from lhg.output import LHGFeatures

    tracked_params_path = Path(tracked_params_path)
    calibration = load_stage_one_calibration(tracked_params_path)
    with open(tracked_params_path) as fp:
        payload = json.load(fp)

    frame_keys = _natural_sorted_frame_keys(payload)
    if not frame_keys:
        raise RuntimeError(
            f'No per-frame entries found in {tracked_params_path}. Expected '
            f'keys like "00000.png" with expcode/fullposecode/translation.')

    n = len(frame_keys)
    expression = np.zeros((n, expression_dim), dtype=np.float32)
    jaw = np.zeros((n, 3), dtype=np.float32)
    eyelid = np.zeros((n, 2), dtype=np.float32)
    global_rot = np.zeros((n, 3), dtype=np.float32)
    translation = np.zeros((n, 3), dtype=np.float32)
    neck_pose = np.zeros((n, 3), dtype=np.float32)
    eye_pose = np.zeros((n, 6), dtype=np.float32)

    for i, key in enumerate(frame_keys):
        entry = payload[key]

        if 'expcode' in entry:
            arr = np.asarray(entry['expcode'], dtype=np.float32).reshape(-1)
            take = min(expression_dim, arr.shape[0])
            expression[i, :take] = arr[:take]

        if 'fullposecode' in entry:
            # FLAME pose layout: [global_rot(3), neck(3), jaw(3), eye_l(3), eye_r(3)]
            arr = np.asarray(entry['fullposecode'], dtype=np.float32).reshape(-1)
            if arr.shape[0] >= 3:
                global_rot[i] = arr[:3]
            if arr.shape[0] >= 6:
                neck_pose[i] = arr[3:6]
            if arr.shape[0] >= 9:
                jaw[i] = arr[6:9]
            if arr.shape[0] >= 15:
                eye_pose[i] = arr[9:15]

        if 'eyelids' in entry:
            arr = np.asarray(entry['eyelids'], dtype=np.float32).reshape(-1)
            take = min(2, arr.shape[0])
            eyelid[i, :take] = arr[:take]

        if 'translation' in entry:
            arr = np.asarray(entry['translation'], dtype=np.float32).reshape(-1)
            take = min(3, arr.shape[0])
            translation[i, :take] = arr[:take]

    outer_bbox = _resolve_outer_bbox(tracked_params_path, image_size)

    return LHGFeatures(
        frame_basenames=np.array(frame_keys, dtype=str),
        expression=expression,
        jaw=jaw,
        eyelid=eyelid,
        global_rot=global_rot,
        translation=translation,
        neck_pose=neck_pose,
        eye_pose=eye_pose,
        valid_mask=np.ones(n, dtype=bool),
        interpolated_mask=np.zeros(n, dtype=bool),
        rejected_mask=np.zeros(n, dtype=bool),
        mode='offline_teacher',
        intrinsics=calibration.intrinsics.astype(np.float32),
        world_mat=calibration.world_mat.astype(np.float32),
        outer_bbox=outer_bbox,
        fps=float(fps),
        image_size=int(image_size),
        flame_scale=float(calibration.flame_scale),
        camera_convention='hravatar',
        metadata={
            'avatar_shapecode': calibration.shapecode.astype(np.float32).tolist(),
            'source_tracked_params': str(tracked_params_path),
            'generator': 'lhg.teacher',
        },
    )


def write_shapecode_sidecar(
    features, output_npz_path: str | Path,
) -> Path:
    """Write ``<output>.shapecode.json`` next to the npz.

    The shapecode does not fit the ``LHGFeatures`` schema (which is
    fixed for the LHG model's training tooling), but is needed by
    downstream renderers / cross-reenactment scripts. The sidecar JSON
    keeps schema compatibility while making the shape readily
    accessible.
    """
    output_npz_path = Path(output_npz_path)
    sidecar = output_npz_path.with_suffix('.shapecode.json')
    payload = {
        'avatar_shapecode': features.metadata.get('avatar_shapecode', []),
        'source_tracked_params': features.metadata.get(
            'source_tracked_params', ''),
        'flame_scale': features.flame_scale,
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, 'w') as fp:
        json.dump(payload, fp, indent=2)
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m lhg.teacher',
        description='Build a teacher lhg_features.npz from an offline '
                    'tracked_params.json (DECA optimize output).',
    )
    p.add_argument(
        '--tracked-params', required=True,
        help='Path to the offline tracked_params.json (Stage 1 output) '
             'whose per-frame entries become the teacher target.',
    )
    p.add_argument(
        '--output', required=True,
        help='Output lhg_features.npz path. A sidecar '
             '<output>.shapecode.json is written alongside.',
    )
    p.add_argument(
        '--fps', type=float, default=25.0,
        help='Feature stream FPS recorded in metadata (default 25). '
             'LHG convention is 25 FPS — see project_lhg_fps_convention.',
    )
    p.add_argument(
        '--image-size', type=int, default=512,
        help='Outer-crop square size in pixels (default 512). Only used '
             'for the outer_bbox fallback when no outer_offset.json '
             'sidecar is present alongside tracked_params.json.',
    )
    p.add_argument(
        '--expression-dim', type=int, default=50,
        help='Number of expression dims to keep from the offline expcode '
             '(default 50, matching SMIRK / LHGFeatures.expression).',
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    features = build_features_from_tracked_params(
        tracked_params_path=args.tracked_params,
        fps=args.fps,
        image_size=args.image_size,
        expression_dim=args.expression_dim,
    )
    features.write(args.output)
    sidecar = write_shapecode_sidecar(features, args.output)
    print(
        f'wrote {args.output}  '
        f'(N={features.frame_basenames.shape[0]}, mode={features.mode}, '
        f'shapecode-sidecar={sidecar})'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
