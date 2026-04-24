# HRAvatar demos (Python 3.11 + PyTorch 2.9 + CUDA 12.8)

Three demos are provided, mirroring the demo layout used by the companion
MTamon branches (`MTamon/DECA@release/cuda128`, `MTamon/smirk@release/cuda128`).

| # | Script                              | Purpose                                                                 |
|---|-------------------------------------|-------------------------------------------------------------------------|
| 1 | `demo_1_train_subject.sh`           | End-to-end training of HRAvatar on a single subject's video.            |
| 2 | `demo_2_cross_reenactment.sh`       | Drive a trained HRAvatar with a **different** person's extracted FLAME. |
| 3 | `demo_3_overlay_tracking.py`        | Visualize the **post-optimize** features HRAvatar's renderer actually receives (FLAME vertices + landmarks + params card), overlaid on the source frames. |

All three demos assume `bash setup.sh` and `bash download_assets.sh` have
been executed successfully and `conda activate HRAvatar` is live.

---

## Data layout

Place your videos under a single directory tree.  After the preprocessing
step the structure becomes:

```text
/data/subjects/
└── alice/
    ├── video.mp4                <- input video (required)
    ├── image/                   <- frames (created by crop_and_matting)
    │   ├── 000000.png
    │   └── ...
    ├── matted/                  <- alpha-matted images
    ├── mask/                    <- binary masks
    ├── deca/                    <- DECA per-frame FLAME predictions
    ├── keypoints/               <- 2D face landmarks
    ├── iris/                    <- iris segmentation
    ├── albedo/                  <- IntrinsicAnything pseudo-GT (optional)
    └── tracked_params.npz       <- optimised FLAME params (from optimize.py)
```

The cross-reenactment demo uses the same layout for the *source* subject
whose expressions are transferred to the avatar of the *target* subject.

---

## 1. Train HRAvatar on your own subject

`demos/demo_1_train_subject.sh`

```bash
# Arguments:
#   $1  path to the directory that will hold the subject folder
#   $2  subject name (the sub-directory under $1)
#   $3  path to the input video file (mp4/mov)
#   $4  intrinsics preset: "hdtf" | "insta" | "custom:fx,fy,cx,cy"
#
bash demos/demo_1_train_subject.sh \
    /data/subjects alice /data/raw/alice.mp4 hdtf
```

The script performs these steps:

1. `preprocess/crop_and_matting.py` — frame extraction, matting, 512² crop.
2. `preprocess/submodules/DECA/demos/demo_reconstruct.py` — initial FLAME
   parameters (shape/expression/pose) with DECA.
3. `preprocess/keypoint_detector.py` — 68-point landmarks with face-alignment.
4. `preprocess/iris.py` — iris segmentation with FDLite.
5. `preprocess/submodules/DECA/optimize.py` — photometric FLAME fit.
6. *(optional, enabled with `--with_albedo`)*
   `preprocess/submodules/IntrinsicAnything/inference.py` — albedo pseudo-GT.
7. `train.py` — HRAvatar training (15 epochs by default, `--epochs` override).
8. `render.py` — self-reenactment renders + metrics.

Outputs land in `outputs/custom/<subject_name>/`.

## 2. Offline cross-reenactment

`demos/demo_2_cross_reenactment.sh`

Drive a trained HRAvatar with FLAME parameters extracted from a *different*
person's video.  The script runs the same preprocessing pipeline on the
source video (to get `tracked_params.npz` + landmarks) and then invokes
`render.py --corss_source_paths ...` to render frames with the target
subject's Gaussian head but the source subject's expression trajectory.

**Feature-extraction backend.**  HRAvatar already bundles both DECA
(`preprocess/submodules/DECA`) and SMIRK (`net_modules/flame_params_net_smirk.py`).
Follow the project's convention:

* **DECA** — canonical per-frame FLAME extractor (used in the training
  pipeline).  This is the default in `demo_2_cross_reenactment.sh`.
* **SMIRK** — used only as an online expression encoder **inside** the
  trained HRAvatar (`--with_param_net_smirk` in `arguments/__init__.py`).
  It is not used for offline feature extraction in the reference code.

```bash
# Arguments:
#   $1  pre-processed source subject path (the SOURCE of the expression)
#   $2  trained target HRAvatar model_path
#   $3  feature-extraction backend: "deca" (default) | "smirk"
#
bash demos/demo_2_cross_reenactment.sh \
    /data/subjects/alice \
    outputs/custom/bob \
    deca
```

Results are written to:

```text
outputs/custom/bob/test_cross_reenactment/ours_<N>/alice_reenactment_bob/
├── 00000.png  ...  NNNNN.png
└── alice_reenactment_bob_video.mp4
```

## 3. Visualize HRAvatar's actual renderer input

`demos/demo_3_overlay_tracking.py`

This is the **offline feature inspector**.  Use it to verify that a
downstream Listening-Head-Generation model is producing the exact feature
shape HRAvatar's renderer expects.

Important: the features HRAvatar ingests are **not** raw DECA output —
they come from `preprocess/submodules/DECA/optimize.py`, a photometric
+ landmark + temporal refinement whose result is stored in
`tracked_params.json`.  The demo:

1. reads `tracked_params.json` produced by the preprocessing pipeline;
2. re-implements the same LBS + offset chain HRAvatar uses internally
   (`scene.gaussian_head_model.GaussianHeadModel.lbs_v2`, including
   the `num_joints = J_regressor.shape[1] + 1` 6-joint augmentation and
   the `lbs_weights` zero-column pad from `scene/__init__.py`);
3. projects the resulting vertices with exactly the camera the data
   loader builds (`w2c = diag(1,-1,-1,1) @ world_mat`, pinhole
   `fo = image_w / (2·tan(½·fovx))` with `fovx = 2·arctan2(cx, fx)`);
4. overlays the result on the original frames and writes an mp4.

No HRAvatar checkpoint is loaded — only the FLAME assets installed via
`download_assets.sh`.  If the overlay tracks the input here, HRAvatar
will reproduce it.

```bash
# 1) preprocess once (reuse the helper used by demo 1)
bash demos/_preprocess_subject.sh /data/subjects alice /data/raw/alice.mp4 hdtf

# 2) inspect the features fed to the renderer
python demos/demo_3_overlay_tracking.py \
    --subject_dir /data/subjects/alice \
    --output      /tmp/alice_features.mp4 \
    --mode        all
```

Modes: `vertices` (cyan dots at every projected FLAME vertex),
`wireframe` (triangle edges), `landmarks` (the 68-pt FAN points the
optimiser was fit against, read from `keypoint.json`), `params_card`
(numerical summary of `shape[:6]`, `exp[:6]`, global/neck/jaw pose in
degrees, eyelids, translation), `all` (vertices + landmarks + card).

The overlay style mirrors
`MTamon/smirk@release/cuda128/demos/demo_video.py --show_vertices`.
