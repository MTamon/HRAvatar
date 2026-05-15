# Preprocessing scope: avatar fit vs. LHG feature extraction

This repo hosts two distinct preprocessing pipelines that share several
modules but have **different output formats and incompatible algorithmic
guarantees**. Pick the right entry point for the right task.

For LHG-specific workflow details (per-stage commands, full flag list,
output schema, render verification), see [`lhg/README.md`](../lhg/README.md).
This file describes the high-level scope split and why the two pipelines
must stay separate.

## TL;DR

| Use case | Entry point | Output | Algorithm |
|---|---|---|---|
| HRAvatar avatar personal fit (one-time) | `demos/_preprocess_subject.sh` (no `--lhg-only`) | `tracked_params.json`, segmentation, image dirs, `mask/`, `albedo/` (optional) | Full-clip joint Adam optimization (DECA `optimize.py`) plus matting + albedo |
| LHG **Stage 1 calibration** (one-time per subject) | `demos/_preprocess_subject.sh --lhg-only` | `tracked_params.json` (only `world_mat` / `shapecode` / `intrinsics` consumed downstream), outer-cropped `image/`, `outer_offset.json` | Same DECA `optimize.py` joint fit, with matting + albedo SKIPPED |
| LHG **Stage 2 online feature extraction** (per clip) | `demos/extract_lhg_features.sh --mode online` | `lhg_features.npz` (per-frame 11-dim FLAME params) | Strictly causal per-frame pipeline: MediaPipe video mode → SMIRK → cv2.solvePnP EPnP → causal Hampel → symmetric FIR LPF on rotation/translation (and optional jaw) |
| LHG **Stage 3 pseudo-online teacher data** (per clip, future) | `demos/extract_lhg_features.sh --mode pseudo-online` | Same schema as Stage 2 | Same per-frame core + bidirectional Hampel + linear interpolation across detector dropouts + bidirectional quaternion-flip + larger-lookahead FIR LPF |

## Why the avatar fit and LHG paths cannot be merged

DECA `optimize.py` performs a clip-wide joint Adam optimization with
temporal smoothness regularizers (`exp[1:]-exp[:-1]`,
`pose[1:]-pose[:-1]`, `translation[1:]-translation[:-1]`). The optimizer
sees the whole clip at once and trades off across frames. This is fine
when the goal is **fitting an avatar to a subject once**: the avatar
parameters (`shape_param`, learned Gaussians, etc.) are baked from the
fit and the per-frame parameters are discarded after training.

For LHG the per-frame FLAME parameters are the **target signal** the
model has to predict at inference. If the teacher data is a clip-wide
joint optimum, the LHG model will try to learn that signal — but its
inference pipeline cannot do clip-wide joint optimization (it only sees
the past), so the learned target is unreachable. Concretely: the smoothing
regularizer leaks the future, the optimizer's final-frame state depends on
the clip length, and the resulting trajectory has properties (e.g. zero
group delay between channels, post-hoc residual minimization) that no
causal extractor can match.

## How the LHG pipeline splits the work across three stages

Per-frame EPnP cannot reproduce the avatar-fit's clip-wide accuracy
(the gap is roughly 10×). Rather than try, the LHG pipeline splits the
work cleanly: **calibration handles absolute pose, Stage 2 handles
per-frame deltas**.

* **Stage 1** (one-time per subject) runs the full joint fit and writes
  `tracked_params.json`. Stage 2/3 read **only the clip-constants**
  (`world_mat`, `shapecode`, `intrinsics`) — per-frame entries from
  Stage 1 are deliberately ignored, since reproducing them online is
  infeasible.

* **Stage 2** (per clip, online) runs a strictly causal per-frame pipeline
  whose *every* step has a real-time analogue. Per-frame `global_rot`
  / `translation` are emitted as **deltas around `world_mat`**, so the
  LHG model only has to learn motion (not absolute pose). A
  symmetric FIR LPF (zero-phase + lookahead) is applied to rotation
  and translation at extraction time; in production, an equivalent
  streaming filter (`lhg.lpf.StreamingSymmetricFIR`) runs with the same
  coefficients, so extracted and inferred values are bit-equivalent.
  `expression` / `eyelid` are SMIRK pass-through (LPF would compromise
  lip-sync timing). `jaw` is opt-in via `--lpf-jaw` with a separate
  cutoff (default 10 Hz, preserves syllable-rate motion).

* **Stage 3** (per clip, pseudo-online teacher data, currently gated)
  runs the same per-frame core, then post-processes the collected
  sequence with these future-info corrections — limited to events
  that don't repeat at inference:
    - Linear interpolation across frames where the face detector dropped
    - Bidirectional Hampel filter for one-time outliers (centered window)
    - Bidirectional `global_rot` quaternion-flip detection and sign fix
    - Larger-lookahead FIR LPF (default `--lookahead-offline 12`,
      taps=25) on rotation/translation. **Same filter design** as
      Stage 2 (firwin + cutoff), only the lookahead differs — so the
      train/inference distribution mismatch is localized to filter
      sharpness, not filter type.
  
  Normal-state jitter floor receives the **same** treatment as `online`
  (no extra bidirectional smoothing on `expression` / `eyelid`). The two
  outputs are otherwise expected to agree within tight `np.allclose`
  tolerances on non-anomalous frames.

The clip-wide outer crop is **NOT recomputed** by Stage 2/3 — Stage 1
already produced `outer_offset.json` and the `image/` folder is in that
coordinate system, so Stage 2/3 just read the metadata and consume the
pre-cropped frames.

## Output schema

`lhg_features.npz` carries the per-frame channels the LHG model
generates (`expression` 50d, `jaw` 3d, `eyelid` 2d, `global_rot` 3d,
`translation` 3d) plus the clip-constant context (`world_mat`,
`intrinsics`, `outer_bbox`) and validity masks (`valid_mask`,
`interpolated_mask`, `rejected_mask`). See `lhg/output.py` for the
canonical fields and `lhg/render_adapter.py` for the
`tracked_params.json`-compatible mapping used by the rendering verification
demo.

## Cross-references

* [`lhg/README.md`](../lhg/README.md) — per-stage commands, CLI flags,
  troubleshooting, jitter visualization
* `feedback_lhg_stage_separation` — Stage 1/2/3 responsibility split
* `feedback_pseudo_online_for_lhg_teacher` — what Stage 3 may / may not do
* `feedback_no_one_euro_in_lhg_pipeline` — symmetric FIR LPF scope
  (rotation / translation default; jaw opt-in; expression / eyelid
  pass-through)
* `feedback_online_translation_via_epnp` — why per-frame
  translation/global_rot uses cv2.solvePnP (EPnP), not Adam
* `feedback_use_mediapipe_not_fan` — why the LHG path defaults to
  MediaPipe FaceLandmarker (video running mode); FAN is opt-in via
  `--detector fan`
* `project_renderer_input_facts` — `shape_code` is baked, `world_mat` is
  clip-constant; only the 11-dim per-frame channels need to be generated
* `project_translation_depth_decoupling` — bbox size and translation z
  are decoupled in the offline pipeline (FAN runs on outer-512)
* `project_lhg_fps_convention` — 25 FPS feature stream / 10 FPS LHG
  inference; lookahead time conversions all use 25 FPS
