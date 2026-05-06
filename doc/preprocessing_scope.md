# Preprocessing scope: avatar fit vs. LHG feature extraction

This repo hosts two distinct preprocessing pipelines that share several
modules but have **different output formats and incompatible algorithmic
guarantees**. Pick the right entry point for the right task.

## TL;DR

| Use case | Entry point | Output | Algorithm |
|---|---|---|---|
| HRAvatar avatar personal fit (one-time) | `demos/_preprocess_subject.sh` | `tracked_params.json`, segmentation, image dirs | Full-clip joint Adam optimization (DECA `optimize.py`) |
| Listening Head Generation (LHG) feature extraction (offline teacher data OR online inference) | `demos/extract_lhg_features.sh` | `lhg_features.npz` (per-frame 11-dim FLAME params) | Per-frame causal pipeline; pseudo-online mode adds future-info corrections only for one-time setup and outliers |

## Why they cannot be the same

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

## What `demos/_preprocess_subject.sh` does

For each subject:

1. ffmpeg frame extraction + outer crop + matting
   (`preprocess/crop_and_matting.py`)
2. Stable inner-bbox computation (`preprocess/stable_bbox.py`) — feeds
   SMIRK encoder warp at training time and DECA `--precomputed-bbox`
3. DECA initial FLAME (`demos/demo_reconstruct.py`)
4. FAN 68pt landmarks (`preprocess/keypoint_detector.py`)
5. Iris segmentation (`preprocess/iris.py`)
6. **DECA full-clip joint optimize** (`optimize.py`)
   — produces `tracked_params.json`

Step 6 is the irreducible difference vs. the LHG path. The main loop
runs to its `--max_iters` cap (no early-stop, see
`tools/patches/apply_deca_optimize_iters.py`); only the iris loop is
short-circuited by `--early-stop-rel-tol`.

## What `demos/extract_lhg_features.sh` does (Phase 2)

For each clip, extract per-frame FLAME parameters under one of two modes:

* `--mode online` — strictly causal pipeline. MediaPipe FaceLandmarker →
  stable_bbox causal hysteresis → SMIRK encoder (exp/jaw/eyelid) +
  EPnP solve (global_rot/translation) → causal Hampel for outliers.
  No future-information access at any step. This is what the LHG model
  must reproduce at inference time.
* `--mode pseudo-online` — base pipeline identical to `online`, plus
  these future-info corrections **only for events that don't occur at
  inference**:
    - Clip-wide fixed outer crop (computed from full-clip face union)
    - Linear interpolation across frames where the face detector dropped
    - Bidirectional Hampel filter for local outliers in bbox center/size
      and landmarks
    - Bidirectional global_rot quaternion-flip detection and sign fix

  Normal-state jitter floor receives the **same** treatment as `online`
  (no extra bidirectional smoothing). The two outputs are otherwise
  expected to agree within tight `np.allclose` tolerances on
  non-anomalous frames.

Output is `lhg_features.npz` with the per-frame channels the LHG model
generates (`expression` 50d, `jaw` 3d, `eyelid` 2d, `global_rot` 3d,
`translation` 3d) plus the clip-constant context (`world_mat`,
`intrinsics`, `outer_bbox`) and validity masks.

## Cross-references

* `feedback_pseudo_online_for_lhg_teacher` — why teacher data must mirror
  the online pipeline
* `feedback_no_one_euro_in_lhg_pipeline` — why preprocessing must not
  apply temporal LPF (would contaminate LHG timing metrics)
* `feedback_online_translation_via_epnp` — why per-frame
  translation/global_rot uses cv2.solvePnP (EPnP), not Adam
* `feedback_use_mediapipe_not_fan` — why the LHG path uses MediaPipe
  FaceLandmarker even though `_preprocess_subject.sh` uses FAN
* `project_renderer_input_facts` — `shape_code` is baked, `world_mat` is
  clip-constant; only the 11-dim per-frame channels need to be generated
* `project_translation_depth_decoupling` — bbox size and translation z
  are decoupled in the offline pipeline (FAN runs on outer-512)
