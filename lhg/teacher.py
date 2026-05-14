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

    LHGFeatures.expression       <-  see "expression source" below
    LHGFeatures.jaw              <-  see "expression source" below
    LHGFeatures.eyelid           <-  see "expression source" below
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

Expression source
-----------------
Two modes, selected by the presence of ``--avatar-checkpoint``:

* **Default** (no ``--avatar-checkpoint``): expression / jaw / eyelid are
  read from ``tracked_params[frame].expcode`` / ``fullposecode[6:9]`` /
  ``eyelids`` — the DECA optimize joint-Adam output. Backwards-compatible
  with the 5/14 ``[add] LHG teacher data builder`` commit.

* **Avatar-trained SMIRK** (``--avatar-checkpoint <dir>`` set): expression
  / jaw / eyelid are recomputed per-frame by running the avatar's own
  trained ``FlameParamsNetSmirk`` (loaded from
  ``<dir>/flame_params_net.pth``) on the 224x224 warped face crop derived
  from ``stable_bbox.npz``. This is what the renderer actually consumes
  at avatar training time (``scene/gaussian_head_model.py:364-382``
  overwrites the external ``expression_param`` / ``jaw_params`` /
  ``eyelid_param`` with the SMIRK output when ``warped_image is not
  None``), so it is the right teacher target for the 2026-05-14 grand
  design (memory ``project_lhg_grand_design``).

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


def _warp_frame_to_224(
    image_path: Path, tform_3x3: np.ndarray,
) -> np.ndarray:
    """Warp one image with the stable_bbox tform to a 224x224 uint8 crop.

    Bit-equivalent with ``scene/data_loader.py:_load_images`` (L354 +
    L375): float ``warp(..., preserve_range=True)`` followed by the
    ``((*255).astype(uint8))/255.0`` two-step quantisation. The result
    is what HRAvatar's SMIRK module receives at avatar training time,
    so feeding this through the same SMIRK at teacher-build time
    reproduces the renderer's per-frame expression/jaw/eyelid output.
    """
    import cv2
    from skimage.transform import warp, SimilarityTransform

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f'could not read image {image_path}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img01 = img_rgb.astype(np.float32) / 255.0
    tform = SimilarityTransform(matrix=np.asarray(tform_3x3, dtype=np.float64))
    warped01 = warp(img01, tform.inverse, output_shape=(224, 224),
                    preserve_range=True)
    # Two-step quantisation matching data_loader.py:375. This keeps the
    # SMIRK input numerically identical to what avatar training saw.
    warped_u8 = (warped01 * 255.0).clip(0, 255).astype(np.uint8)
    return warped_u8


def _load_stable_bbox(
    subject_dir: Path, prefer_raw: bool = True,
) -> tuple[dict[str, np.ndarray], str, Path]:
    """Resolve the stable_bbox tform table for SMIRK warp.

    Returns ``(basename_to_tform, coord_system, source_image_dir)``.

    Branching matches ``scene/data_loader.py:_load_images`` (L321-354):
    prefer ``stable_bbox_raw.npz`` + ``image_raw/`` when available (the
    raw-resolution path the avatar's renderer uses when image_raw/
    exists), fall back to ``stable_bbox.npz`` + ``image/`` otherwise.
    """
    raw_npz = subject_dir / 'stable_bbox_raw.npz'
    raw_dir = subject_dir / 'image_raw'
    outer_npz = subject_dir / 'stable_bbox.npz'
    outer_dir = subject_dir / 'image'

    if prefer_raw and raw_npz.is_file() and raw_dir.is_dir():
        npz_path = raw_npz
        source_dir = raw_dir
    elif outer_npz.is_file() and outer_dir.is_dir():
        npz_path = outer_npz
        source_dir = outer_dir
    elif raw_npz.is_file() and raw_dir.is_dir():
        npz_path = raw_npz
        source_dir = raw_dir
    else:
        raise FileNotFoundError(
            f'No stable_bbox.npz / stable_bbox_raw.npz with matching '
            f'image dir found under {subject_dir}. Stage 1 of the '
            f'avatar fit pipeline must have run already (it writes '
            f'these files).')

    with np.load(npz_path, allow_pickle=False) as npz:
        basenames = [str(b) for b in npz['frame_basenames']]
        tforms = npz['tform'].astype(np.float64)
        if 'coord_system' in npz.files:
            coord_system = str(npz['coord_system'])
        else:
            coord_system = 'outer_512' if 'outer' in npz_path.stem else 'raw'

    basename_to_tform = {b: tforms[i] for i, b in enumerate(basenames)}
    return basename_to_tform, coord_system, source_dir


def _recompute_expression_jaw_eyelid_via_smirk(
    subject_dir: Path,
    avatar_checkpoint: Path,
    frame_basenames: list[str],
    *,
    batch_size: int = 16,
    device: str = 'cuda',
    prefer_raw: bool = True,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Re-run the avatar's trained SMIRK module on every frame and
    return ``(expression, jaw, eyelid, info)``.

    Where:
      * ``expression`` (N, 50) float32
      * ``jaw`` (N, 3) float32
      * ``eyelid`` (N, 2) float32
      * ``info`` dict with diagnostic metadata (coord_system,
        source_dir, batch_size, n_frames).

    The 224 warp is reproduced from ``stable_bbox.npz`` exactly the
    way ``scene/data_loader.py`` does at avatar-training time, so the
    SMIRK output here matches what the renderer received during avatar
    training.
    """
    from lhg.encoders import SMIRKEncoder

    bb_map, coord_system, source_dir = _load_stable_bbox(
        subject_dir, prefer_raw=prefer_raw)

    # Cross-check frame ordering between tracked_params and stable_bbox.
    missing = [b for b in frame_basenames if b not in bb_map]
    if missing:
        head = ', '.join(missing[:3])
        raise RuntimeError(
            f'{len(missing)} frame(s) listed in tracked_params.json have no '
            f'matching tform in stable_bbox(_raw).npz (first missing: {head}). '
            f'Re-run Stage 1 with --no-stable-bbox disabled, or check that '
            f'tracked_params and stable_bbox were written from the same clip.')

    encoder = SMIRKEncoder(device=device, avatar_checkpoint=avatar_checkpoint)

    n = len(frame_basenames)
    expression = np.zeros((n, 50), dtype=np.float32)
    jaw = np.zeros((n, 3), dtype=np.float32)
    eyelid = np.zeros((n, 2), dtype=np.float32)

    i = 0
    while i < n:
        bsz = min(batch_size, n - i)
        batch = np.empty((bsz, 224, 224, 3), dtype=np.uint8)
        for j in range(bsz):
            basename = frame_basenames[i + j]
            tform = bb_map[basename]
            batch[j] = _warp_frame_to_224(source_dir / basename, tform)
        out = encoder.encode_batch(batch)
        expression[i:i + bsz] = out['expression']
        jaw[i:i + bsz] = out['jaw']
        eyelid[i:i + bsz] = out['eyelid']
        i += bsz
        if progress_cb is not None:
            progress_cb(i, n)

    info = {
        'coord_system': coord_system,
        'source_dir': str(source_dir),
        'batch_size': int(batch_size),
        'n_frames': int(n),
    }
    return expression, jaw, eyelid, info


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
    avatar_checkpoint: str | Path | None = None,
    subject_dir: str | Path | None = None,
    prefer_raw_bbox: bool = True,
    smirk_batch_size: int = 16,
    device: str = 'cuda',
    progress_cb=None,
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
    avatar_checkpoint : path to the avatar's checkpoint directory
        containing ``flame_params_net.pth``. When provided, expression
        / jaw / eyelid are recomputed by running the avatar's trained
        SMIRK module on the 224x224 warped face crop (matches the
        renderer's actual per-frame consumption). When ``None``
        (default) the legacy behaviour applies: expression / jaw /
        eyelid come from ``tracked_params.json`` (DECA optimize).
    subject_dir : directory holding ``stable_bbox.npz`` /
        ``stable_bbox_raw.npz`` and ``image/`` / ``image_raw/``.
        Defaults to the parent directory of ``tracked_params_path``.
        Only consulted when ``avatar_checkpoint`` is set.
    prefer_raw_bbox : when True (default), prefer
        ``stable_bbox_raw.npz`` + ``image_raw/`` over ``stable_bbox.npz``
        + ``image/`` (matches ``scene/data_loader.py`` branching).
    smirk_batch_size : forward batch size when recomputing SMIRK.
    device : compute device for SMIRK (default ``'cuda'``).
    progress_cb : optional callable ``(i, n) -> None`` invoked during
        the SMIRK recompute loop.
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

    expression_source = 'tracked_params_expcode'
    smirk_info: dict | None = None
    if avatar_checkpoint is not None:
        # Recompute expression / jaw / eyelid using the avatar's
        # trained SMIRK module — this is what the renderer actually
        # consumes (see ``project_lhg_grand_design`` memory). The
        # other channels (global_rot / neck / eye_pose / translation /
        # shape / world_mat / intrinsics) continue to come from
        # tracked_params.json because DECA optimize handles those
        # and the renderer reads them directly.
        sd = (
            Path(subject_dir) if subject_dir is not None
            else tracked_params_path.parent
        )
        smirk_exp, smirk_jaw, smirk_eyelid, smirk_info = (
            _recompute_expression_jaw_eyelid_via_smirk(
                subject_dir=sd,
                avatar_checkpoint=Path(avatar_checkpoint),
                frame_basenames=frame_keys,
                batch_size=smirk_batch_size,
                device=device,
                prefer_raw=prefer_raw_bbox,
                progress_cb=progress_cb,
            )
        )
        # SMIRK natively emits a 50-d expression vector; pad / trim
        # to ``expression_dim`` to stay schema-compatible.
        if smirk_exp.shape[1] == expression_dim:
            expression = smirk_exp
        elif smirk_exp.shape[1] < expression_dim:
            expression = np.zeros((n, expression_dim), dtype=np.float32)
            expression[:, :smirk_exp.shape[1]] = smirk_exp
        else:
            expression = smirk_exp[:, :expression_dim].astype(np.float32)
        jaw = smirk_jaw
        eyelid = smirk_eyelid
        expression_source = f'avatar_smirk@{avatar_checkpoint}'

    metadata = {
        'avatar_shapecode': calibration.shapecode.astype(np.float32).tolist(),
        'source_tracked_params': str(tracked_params_path),
        'generator': 'lhg.teacher',
        'expression_source': expression_source,
    }
    if smirk_info is not None:
        metadata['smirk_coord_system'] = smirk_info['coord_system']
        metadata['smirk_source_dir'] = smirk_info['source_dir']
        metadata['smirk_batch_size'] = smirk_info['batch_size']

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
        metadata=metadata,
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
    p.add_argument(
        '--avatar-checkpoint', default=None,
        help='Path to the avatar checkpoint directory containing '
             '``flame_params_net.pth`` (typically '
             '``outputs/custom/<avatar>/saved_model/epoch_<E>``). When '
             'set, expression / jaw / eyelid are recomputed by running '
             'the avatar\'s trained SMIRK module on the 224x224 warped '
             'crop from stable_bbox.npz — this matches what the renderer '
             'consumes at avatar training time, and is the right teacher '
             'target under the 2026-05-14 grand design. When omitted, '
             'expression / jaw / eyelid come from tracked_params.json '
             '(DECA optimize joint output) for backwards compatibility.',
    )
    p.add_argument(
        '--subject-dir', default=None,
        help='Directory holding stable_bbox.npz / stable_bbox_raw.npz and '
             'image/ / image_raw/ (defaults to the parent of '
             '--tracked-params). Consulted only with --avatar-checkpoint.',
    )
    p.add_argument(
        '--prefer-outer-bbox', action='store_true',
        help='Use stable_bbox.npz + image/ even when stable_bbox_raw.npz + '
             'image_raw/ exist. By default the raw-resolution pair is '
             'preferred (matches scene/data_loader.py:_load_images).',
    )
    p.add_argument(
        '--smirk-batch-size', type=int, default=16,
        help='Forward batch size when recomputing SMIRK (default 16).',
    )
    p.add_argument(
        '--device', default='cuda',
        help='Compute device for SMIRK forward (default cuda).',
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    progress_cb = None
    if args.avatar_checkpoint is not None:
        try:
            from tqdm import tqdm
            bar_state = {'bar': None}

            def progress_cb(i: int, n: int) -> None:
                if bar_state['bar'] is None or bar_state['bar'].total != n:
                    if bar_state['bar'] is not None:
                        bar_state['bar'].close()
                    bar_state['bar'] = tqdm(total=n, desc='teacher/smirk')
                bar = bar_state['bar']
                delta = i - bar.n
                if delta > 0:
                    bar.update(delta)
        except ImportError:
            progress_cb = None

    features = build_features_from_tracked_params(
        tracked_params_path=args.tracked_params,
        fps=args.fps,
        image_size=args.image_size,
        expression_dim=args.expression_dim,
        avatar_checkpoint=args.avatar_checkpoint,
        subject_dir=args.subject_dir,
        prefer_raw_bbox=not args.prefer_outer_bbox,
        smirk_batch_size=args.smirk_batch_size,
        device=args.device,
        progress_cb=progress_cb,
    )
    features.write(args.output)
    sidecar = write_shapecode_sidecar(features, args.output)
    print(
        f'wrote {args.output}  '
        f'(N={features.frame_basenames.shape[0]}, mode={features.mode}, '
        f'expression_source={features.metadata.get("expression_source")}, '
        f'shapecode-sidecar={sidecar})'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
