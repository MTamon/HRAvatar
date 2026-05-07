"""CLI entry point for the LHG feature extraction pipeline.

Usage::

    python -m lhg.extract \
        --video data/HDTF/elijah/elijah.mp4 \
        --intrinsics hdtf \
        --output data/HDTF/elijah/lhg_features.npz \
        --mode pseudo-online

The shell wrapper ``demos/extract_lhg_features.sh`` is the recommended
caller (it forwards the same flags and applies $CUDA_VISIBLE_DEVICES).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m lhg.extract',
        description='Extract per-frame LHG FLAME features from a video '
                    'or image directory.',
    )
    p.add_argument(
        '--video', required=True,
        help='Path to mp4/mov OR an image directory. Image directories are '
             'assumed to be the outer-cropped image/ folder produced by '
             'demos/_preprocess_subject.sh; outer_offset.json is auto-detected.',
    )
    p.add_argument(
        '--intrinsics', required=True,
        help='hdtf | insta | custom:fx,fy,cx,cy. Pinned to the OUTER-CROP '
             'coordinate system (image_size square).',
    )
    p.add_argument(
        '--output', required=True,
        help='Output .npz path.',
    )
    p.add_argument(
        '--mode', required=True, choices=('online', 'pseudo-online'),
        help='online: strictly causal, mirrors LHG inference. '
             'pseudo-online: causal core + future-info corrections for '
             'detector dropouts and one-time anomalies (teacher data).',
    )
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--bbox-scale', type=float, default=1.6)
    p.add_argument(
        '--world-mat', default=None,
        help='Optional path to a tracked_params.json whose world_mat will be '
             'reused. When omitted, world_mat is calibrated from the first '
             '--world-mat-calibration-frames valid frames of this clip.',
    )
    p.add_argument('--world-mat-calibration-frames', type=int, default=60)
    p.add_argument('--flame-scale', type=float, default=4.0)
    p.add_argument(
        '--correspondence', default=None,
        help='Path to mediapipe_flame_landmarks.npz. Defaults to '
             'assets/lhg/mediapipe_flame_landmarks.npz.',
    )
    p.add_argument('--run-deca-encoder', action='store_true',
                   help='Diagnostic only: also invoke DECA coarse encoder.')
    p.add_argument(
        '--detector', choices=('fan', 'mediapipe'), default='fan',
        help='Landmark detector. fan (default) = face_alignment 68-pt, '
             'matches the avatar-fit pipeline and gives ~5-10x lower '
             'per-frame EPnP noise than mediapipe. mediapipe = MediaPipe '
             'FaceMesh 478-pt, faster but with weaker EPnP precision.',
    )
    p.add_argument(
        '--camera-convention', choices=('hravatar', 'opencv'), default='hravatar',
        help='Coordinate frame for global_rot / translation / world_mat. '
             'hravatar (default) is directly consumable by HRAvatar\'s '
             'renderer. opencv keeps the raw EPnP output for downstream '
             'pipelines that have their own conversion.',
    )
    p.add_argument('--quiet', action='store_true', help='Suppress progress bar.')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from lhg.config import LHGConfig, parse_intrinsics
    from lhg.pipeline import extract
    from lhg.video import FrameSource, load_outer_offset

    fx, fy, cx, cy = parse_intrinsics(args.intrinsics)
    cfg = LHGConfig(
        mode=args.mode,
        fps=args.fps,
        intrinsics=(fx, fy, cx, cy),
        image_size=args.image_size,
        bbox_scale=args.bbox_scale,
        flame_scale=args.flame_scale,
        world_mat_path=args.world_mat,
        world_mat_calibration_frames=args.world_mat_calibration_frames,
        run_deca_encoder=args.run_deca_encoder,
        camera_convention=args.camera_convention,
        detector_type=args.detector,
    )

    frames = FrameSource(args.video)
    is_image_dir = Path(args.video).is_dir()
    outer_bbox = (
        load_outer_offset(Path(args.video)) if is_image_dir else None
    )
    # When the input is the avatar-fit pipeline's pre-cropped image/
    # folder, frames are already in outer-crop space; outer_offset.json
    # describes the historical raw->outer crop and is metadata only.
    apply_outer_crop = not (is_image_dir and outer_bbox is not None)

    progress_cb = None
    if not args.quiet:
        from tqdm import tqdm
        bar_state = {'bar': None}

        def cb(i: int, n: int) -> None:
            if bar_state['bar'] is None or bar_state['bar'].total != n:
                if bar_state['bar'] is not None:
                    bar_state['bar'].close()
                bar_state['bar'] = tqdm(total=n, desc=f'lhg/{cfg.mode}')
            bar = bar_state['bar']
            delta = (i + 1) - bar.n
            if delta > 0:
                bar.update(delta)
        progress_cb = cb

    features = extract(
        frames=frames,
        outer_bbox=outer_bbox,
        cfg=cfg,
        correspondence_path=args.correspondence,
        progress=progress_cb,
        apply_outer_crop=apply_outer_crop,
    )
    frames.close()
    features.write(args.output)
    print(f'wrote {args.output}  '
          f'(N={features.frame_basenames.shape[0]}, mode={features.mode}, '
          f'valid={int(features.valid_mask.sum())})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
