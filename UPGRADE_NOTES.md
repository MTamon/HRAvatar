# HRAvatar — Upgrade notes (Python 3.11 + PyTorch 2.9.1 + CUDA 12.8)

This branch (`claude/update-pytorch-cuda-deps-d9xCb`) migrates the original
HRAvatar code from the Python 3.10 / PyTorch 1.13.1 / CUDA 11.7 stack to
Python 3.11 / PyTorch 2.9.1 / CUDA 12.8.

The target pin set is aligned with the companion forks under the
[`MTamon`](https://github.com/MTamon) GitHub account, which carry the
same migration for the DECA / SMIRK / FLARE stacks:

* https://github.com/MTamon/FLARE            (main)
* https://github.com/MTamon/DECA             (release/cuda128)
* https://github.com/MTamon/smirk            (release/cuda128)

## Summary

| Area                | old (main)                               | new (this branch)                         |
|---------------------|------------------------------------------|-------------------------------------------|
| Python              | 3.10.14                                  | 3.11                                      |
| PyTorch             | 1.13.1 + cu117                           | 2.9.1 + cu128                             |
| torchvision         | 0.14.1                                   | 0.24.1                                    |
| numpy               | 1.23.5                                   | 2.2.6                                     |
| scipy               | (unpinned)                               | 1.16.3                                    |
| kornia              | 0.7.3                                    | 0.8.2                                     |
| opencv-python       | 4.8.0.76                                 | 4.12.0.88                                 |
| mediapipe           | 0.10.10                                  | 0.10.14                                   |
| pillow              | (transitive)                             | 12.0.0                                    |
| pytorch-lightning   | 1.5.2                                    | 2.5.2                                     |
| torchmetrics        | 1.4.0.post0                              | 1.6.0                                     |
| pytorch3d           | 0.7.2 (conda)                            | 0.7.8 (built from source)                 |
| jax / jaxlib        | 0.4.30                                   | 0.4.30 (kept — mediapipe dep)             |
| chumpy              | 0.70 (PyPI)                              | git mattloper/chumpy (numpy-2 compatible) |
| scikit-image        | 0.22.0                                   | 0.25.2                                    |
| transformers        | 4.44.0                                   | 4.44.2                                    |
| huggingface-hub     | 0.24.5                                   | 0.25.2                                    |
| nvdiffrast          | pip                                      | git (NVlabs main)                         |

## Deviations from the MTamon reference branches (reported per request)

The MTamon cuda128 branches do **not** need every package HRAvatar uses.
These are HRAvatar-specific additions or small deviations:

| Package                      | MTamon value               | this branch | Reason |
|------------------------------|----------------------------|-------------|--------|
| `pytorch3d`                  | **absent** (DECA uses its own rasterizer) | `0.7.8` (git) | `scene/gaussian_head_model.py` imports `pytorch3d.ops`; `utils/loss_utils.py` imports `pytorch3d`.  Built from source since no cu128 wheel is published. |
| `pyshtools`                  | absent                     | `4.13.1`    | `utils/sh_utils.py::rotateSH` (used by `run_shells/filter_envmap.sh`). |
| `plyfile`                    | absent                     | `1.0.3`     | `scene/gaussian_model.py::save_ply/load_ply`. |
| `roma`                       | absent                     | `1.5.1`     | rotation utilities. |
| `carvekit-colab`             | absent                     | `4.1.2`     | background removal in preprocessing. |
| `diffusers`                  | absent                     | `0.30.3`    | IntrinsicAnything pipeline. |
| `torchfile`                  | absent                     | `0.1.0`     | legacy checkpoint compatibility. |
| `taming-transformers` + `clip` | absent                   | git HEAD    | IntrinsicAnything encoder. |
| `numba`                      | 0.62.1                     | 0.62.1      | same |
| `opencv-python`              | 4.12.0.88                  | 4.12.0.88   | same |
| `pip install --no-deps`      | yes                        | yes         | followed exactly (see `setup.sh`). |

**None of the HRAvatar-specific pins conflict with the MTamon DECA / SMIRK
cuda128 sets.**  They sit on top of the shared block.

We deliberately picked the DECA/SMIRK pin set (torch 2.9.1, mediapipe
0.10.14, protobuf 4.25.5) and **not** FLARE's (torch 2.9.0, mediapipe
0.10.11, protobuf 3.20.3) because (a) HRAvatar loads SMIRK weights
(`assets/smirk/pretrained_models/SMIRK_em1.pt`) and (b) the two pin sets
are mutually exclusive.

## Code changes required by the new stack

* `utils/sh_utils.py` — `np.int` → `np.int64` (removed in numpy 1.20+).
* `preprocess/submodules/DECA/decalib/utils/util.py` — same `np.int` fix
  in the `get_regional_weight_mask` helper.
* `train.py`, `scene/gaussian_head_model.py`,
  `preprocess/submodules/DECA/decalib/deca.py` — explicit
  `weights_only=False` on `torch.load` calls that unpickle arbitrary
  Python objects (required since PyTorch 2.6 flipped the default).
* `net_modules/flame_params_net_smirk.py` — explicit
  `weights_only=False` on `torch.load` for `SMIRK_em1.pt` (and any
  fine-tuned checkpoint).  The SMIRK weights were produced under a
  pre-2.x PyTorch and although the HRAvatar-consumed state dict is
  tensors-only, we do not want to risk the default `weights_only=True`
  failing on a legacy asset we cannot regenerate.

These are intentionally the minimum set of patches; no functional changes.

## Files added or rewritten by this branch

* `environment.yml`  — rewritten (conda only owns python + cuda toolkit).
* `requirements.txt` — NEW.  Authoritative pin set, applied with `--no-deps`.
* `setup.sh`         — NEW.  Ordered installer mirroring
  `MTamon/DECA:install_128.sh`.
* `download_assets.sh` — NEW.  Fetches FLAME / DECA / SMIRK / RVM /
  face-parsing / IntrinsicAnything weights.
* `demos/README.md`                        — NEW.
* `demos/_preprocess_subject.sh`           — NEW (internal helper).
* `demos/demo_1_train_subject.sh`          — NEW.  End-to-end training on
  a single subject's video.
* `demos/demo_2_cross_reenactment.sh`      — NEW.  Offline feature
  extraction (DECA) + cross-reenactment with `render.py
  --corss_source_paths`.
* `demos/demo_3_overlay_tracking.py`       — NEW.  Offline **feature
  inspector** for the tensors HRAvatar's renderer actually consumes:
  reads `tracked_params.json` (post-`optimize.py`), re-runs the same
  LBS + offset chain as `GaussianHeadModel.lbs_v2` (including the
  6-joint augmentation and the zero-column `lbs_weights` pad from
  `scene/__init__.py`), projects with the exact camera built by
  `data_loader._load_camera`, and overlays vertices / landmarks /
  params summary on the source frames.  No HRAvatar checkpoint is
  loaded — intended for downstream Listening-Head-Generation models.
* `UPGRADE_NOTES.md`                       — THIS FILE.

## Quick start

```bash
bash setup.sh                             # creates conda env "HRAvatar"
conda activate HRAvatar
bash download_assets.sh                   # places required checkpoints
bash demos/demo_1_train_subject.sh \
     /data/subjects alice /data/raw/alice.mp4 hdtf
```

See `demos/README.md` for the cross-reenactment and overlay demos.
