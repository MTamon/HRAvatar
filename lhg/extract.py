"""CLI entry point for the LHG feature extraction pipeline.

Usage::

    # Stage 1 (one-time per subject, runs DECA optimize):
    bash demos/_preprocess_subject.sh \
        --sbj-root data/lhg --sbj-name foo \
        --video raw/foo.mp4 --intrinsics hdtf \
        --lhg-only

    # Stage 2 (this entry point):
    python -m lhg.extract \
        --video data/lhg/foo/image \
        --calibration data/lhg/foo/tracked_params.json \
        --output data/lhg/foo/lhg_features.npz \
        --mode online

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
        '--calibration', required=True,
        help='Path to the Stage 1 tracked_params.json (or _v2 sibling). '
             'Stage 2 reads world_mat, shapecode, and intrinsics from this '
             'file and uses them as clip-constants.',
    )
    p.add_argument(
        '--output', required=True,
        help='Output .npz path.',
    )
    p.add_argument(
        '--mode', required=True, choices=('online', 'pseudo-online'),
        help='online: strictly causal, mirrors LHG inference. '
             'pseudo-online: causal core + future-info corrections '
             '(bidirectional Hampel, linear interpolation across detector '
             'dropouts, bidirectional quaternion-flip fix, larger-lookahead '
             'FIR LPF). Use this for detector-dropout salvage / future-info '
             'smoothing studies, NOT for LHG teacher generation — under the '
             '2026-05-14 grand design the teacher comes from the offline '
             'pipeline via lhg/teacher.py instead.',
    )
    p.add_argument(
        '--intrinsics', default=None,
        help='Optional override for the EPnP intrinsics. By default the '
             'intrinsics from --calibration are used. Accepts "hdtf" / '
             '"insta" / "custom:fx,fy,cx,cy".',
    )
    p.add_argument('--fps', type=float, default=25.0,
                   help='Feature stream FPS (default 25). LHG inference '
                        'runs at 10 FPS but the per-frame feature stream '
                        'is at this rate.')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--bbox-scale', type=float, default=1.6)
    p.add_argument('--flame-scale', type=float, default=None,
                   help='Override flame_scale. By default inferred from the '
                        'calibration file name (_v2 → 1.0, otherwise 4.0).')
    p.add_argument(
        '--correspondence', default=None,
        help='Path to the detector-specific FLAME correspondence asset. '
             'Defaults to assets/lhg/mediapipe_flame_landmarks.npz for '
             'mediapipe, assets/lhg/dlib_flame_landmarks.npz for fan.',
    )
    p.add_argument('--run-deca-encoder', action='store_true',
                   help='Diagnostic only: also invoke DECA coarse encoder.')
    p.add_argument(
        '--detector', choices=('mediapipe', 'fan'), default='mediapipe',
        help='Landmark detector. mediapipe (default) = MediaPipe '
             'FaceLandmarker 478-pt + iris; fan = face_alignment 68-pt '
             '(matches HRAvatar avatar fit pipeline). Per-frame absolute '
             'accuracy is comparable; the choice mainly affects detector '
             'consistency between train/inference.',
    )
    p.add_argument(
        '--mediapipe-mode', choices=('image', 'video'), default='video',
        help='MediaPipe running mode. "video" (default) enables the '
             'internal Kalman tracker. "image" is per-frame; used as a '
             'jitter A/B baseline.',
    )
    p.add_argument(
        '--lookahead', type=int, default=4,
        help='Symmetric FIR LPF one-sided lookahead in frames '
             '(default 4 → taps=9, 160ms lag at 25 fps). Set 0 to '
             'disable the LPF entirely (pass-through).',
    )
    p.add_argument(
        '--lpf-cutoff-hz', type=float, default=4.0,
        help='LPF cutoff (Hz) for rotation and translation (default 4).',
    )
    p.add_argument(
        '--lpf-jaw', action='store_true',
        help='Also apply the LPF to jaw. Off by default — jaw carries '
             'syllable-rate (5-8 Hz) information for lip-sync.',
    )
    p.add_argument(
        '--lpf-jaw-cutoff-hz', type=float, default=10.0,
        help='LPF cutoff (Hz) for jaw (default 10), only used when '
             '--lpf-jaw is set. 10 Hz preserves syllable rate while '
             'still attenuating detector noise above the speech band.',
    )
    p.add_argument(
        '--lookahead-offline', type=int, default=12,
        help='pseudo-online offline FIR one-sided lookahead '
             '(default 12 → taps=25). Ignored in --mode online.',
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

    from lhg.calibration import load_stage_one_calibration
    from lhg.config import LHGConfig, parse_intrinsics
    from lhg.pipeline import extract
    from lhg.video import FrameSource, load_outer_offset

    calibration = load_stage_one_calibration(args.calibration)

    if args.intrinsics is not None:
        fx, fy, cx, cy = parse_intrinsics(args.intrinsics)
    else:
        fx, fy, cx, cy = calibration.intrinsics_tuple()

    flame_scale = (
        float(args.flame_scale)
        if args.flame_scale is not None else float(calibration.flame_scale)
    )

    cfg = LHGConfig(
        mode=args.mode,
        fps=args.fps,
        calibration_path=str(calibration.source_path),
        intrinsics=(fx, fy, cx, cy),
        image_size=args.image_size,
        bbox_scale=args.bbox_scale,
        flame_scale=flame_scale,
        run_deca_encoder=args.run_deca_encoder,
        camera_convention=args.camera_convention,
        detector_type=args.detector,
        mediapipe_running_mode=args.mediapipe_mode,
        lpf_lookahead=args.lookahead,
        lpf_cutoff_hz=args.lpf_cutoff_hz,
        lpf_jaw=args.lpf_jaw,
        lpf_jaw_cutoff_hz=args.lpf_jaw_cutoff_hz,
        lpf_offline_lookahead=args.lookahead_offline,
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
        calibration=calibration,
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
