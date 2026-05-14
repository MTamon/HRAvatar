"""Per-frame SMIRK + (optional) DECA encoder wrappers.

The SMIRK encoder produces the three time-varying expression channels
the LHG model has to predict (``expression``, ``jaw``, ``eyelid``).
The DECA encoder is optional and only used for offline diagnostics; its
outputs are NOT part of the LHG feature set.

Both encoders consume a 224x224 face crop produced by warping the
outer-cropped frame with ``preprocess.stable_bbox.build_similarity_tform``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Allow ``python -m lhg.extract`` from the repo root; mirrors the
# convention in ``preprocess/stable_bbox.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

if TYPE_CHECKING:
    import torch


class SMIRKEncoder:
    """Wrapper around ``net_modules.flame_params_net_smirk.FlameParamsNetSmirk``.

    Two operating modes:

    * Pretrained init (``avatar_checkpoint=None``, default): the model
      is created with its native initial weights (SMIRK pretrained
      backbone + freshly-initialised expression head). Used by the
      LHG online extraction path, where we want a generic SMIRK that
      can run on any frame.

    * Avatar-trained weights (``avatar_checkpoint=<dir>``): the model
      loads ``<dir>/flame_params_net.pth`` exactly the way
      ``scene/gaussian_head_model.py:469-472`` does at render time.
      This is the SMIRK the avatar's renderer actually consumes
      (HRAvatar trains a per-subject SMIRK alongside the Gaussian
      avatar), so it is the right encoder to use for LHG *teacher*
      generation — the per-frame expression/jaw/eyelid it produces
      are bit-equivalent to what the avatar's renderer reads off
      ``flame_params_net(warped_image)`` during avatar training.
    """

    def __init__(
        self,
        exp_dim: int = 50,
        device: str = 'cuda',
        avatar_checkpoint: str | Path | None = None,
    ):
        import torch
        from net_modules.flame_params_net_smirk import FlameParamsNetSmirk

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.exp_dim = exp_dim
        self.model = FlameParamsNetSmirk(exp_dim=exp_dim).to(self.device)
        if avatar_checkpoint is not None:
            ckpt_dir = Path(avatar_checkpoint)
            pth = ckpt_dir / 'flame_params_net.pth'
            if not pth.is_file():
                raise FileNotFoundError(
                    f'flame_params_net.pth not found under {ckpt_dir}. '
                    f'Expected the avatar checkpoint directory written by '
                    f'``scene/gaussian_head_model.py:save_model`` (e.g. '
                    f'``outputs/custom/<avatar>/saved_model/epoch_<E>``).')
            state_dict = torch.load(
                str(pth), map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            self.avatar_checkpoint = str(ckpt_dir)
        else:
            self.avatar_checkpoint = None
        self.model.eval()

    @staticmethod
    def to_input(face_224_rgb: np.ndarray) -> 'torch.Tensor':
        """Convert a uint8 (224, 224, 3) RGB crop into the SMIRK input
        tensor (1, 3, 224, 224) with ImageNet-style normalization to
        [0, 1]. SMIRK's pretrained checkpoint expects 0-1 input."""
        import torch

        if face_224_rgb.dtype != np.uint8:
            raise TypeError('face_224_rgb must be uint8')
        if face_224_rgb.shape != (224, 224, 3):
            raise ValueError(
                f'face_224_rgb must be (224, 224, 3), got {face_224_rgb.shape}')
        arr = face_224_rgb.astype(np.float32) / 255.0
        tens = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tens

    @staticmethod
    def to_input_batch(faces_224_rgb: np.ndarray) -> 'torch.Tensor':
        """Batched version of :meth:`to_input`.

        Input shape ``(B, 224, 224, 3) uint8`` → output ``(B, 3, 224, 224)``
        float32 in [0, 1]. Used by teacher generation where we encode
        many frames in a single forward pass.
        """
        import torch

        if faces_224_rgb.dtype != np.uint8:
            raise TypeError('faces_224_rgb must be uint8')
        if faces_224_rgb.ndim != 4 or faces_224_rgb.shape[1:] != (224, 224, 3):
            raise ValueError(
                f'faces_224_rgb must be (B, 224, 224, 3), got {faces_224_rgb.shape}')
        arr = faces_224_rgb.astype(np.float32) / 255.0
        tens = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return tens

    def encode(self, face_224_rgb: np.ndarray) -> dict:
        """Returns a dict with numpy arrays for ``expression`` (50d),
        ``jaw`` (3d), ``eyelid`` (2d).
        """
        import torch

        tens = self.to_input(face_224_rgb).to(self.device)
        with torch.no_grad():
            out = self.model(tens)
        return {
            'expression': out['expression_params'].squeeze(0).cpu().numpy().astype(np.float32),
            'jaw': out['jaw_params'].squeeze(0).cpu().numpy().astype(np.float32),
            'eyelid': out['eyelid_params'].squeeze(0).cpu().numpy().astype(np.float32),
        }

    def encode_batch(self, faces_224_rgb: np.ndarray) -> dict:
        """Batched encode for teacher generation. Returns a dict with
        per-batch numpy arrays for ``expression`` (B, 50), ``jaw`` (B, 3),
        ``eyelid`` (B, 2)."""
        import torch

        tens = self.to_input_batch(faces_224_rgb).to(self.device)
        with torch.no_grad():
            out = self.model(tens)
        return {
            'expression': out['expression_params'].cpu().numpy().astype(np.float32),
            'jaw': out['jaw_params'].cpu().numpy().astype(np.float32),
            'eyelid': out['eyelid_params'].cpu().numpy().astype(np.float32),
        }


class DECAEncoder:
    """Optional wrapper around DECA's coarse encoder for diagnostics.

    Loaded lazily because the DECA module imports a CUDA rasterizer that
    is heavy to compile. Only instantiated when
    ``LHGConfig.run_deca_encoder=True``.
    """

    def __init__(self, device: str = 'cuda'):
        import torch

        # DECA uses an in-package config + global yaml; resolve the
        # package root the same way the avatar fit pipeline does.
        deca_root = _REPO_ROOT / 'preprocess' / 'submodules' / 'DECA'
        if not deca_root.is_dir():
            raise FileNotFoundError(
                f'DECA submodule not found at {deca_root}. Initialize the '
                f'submodule before enabling --run-deca-encoder.')
        if str(deca_root) not in sys.path:
            sys.path.append(str(deca_root))

        from decalib.deca import DECA
        from decalib.utils.config import cfg as deca_cfg

        deca_cfg.model.use_tex = False
        deca_cfg.model.extract_tex = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DECA(config=deca_cfg, device=self.device)
        self.model.eval()

    @staticmethod
    def to_input(face_224_rgb: np.ndarray) -> 'torch.Tensor':
        # DECA's ``encode()`` accepts a (B, 3, 224, 224) tensor in [0, 1].
        return SMIRKEncoder.to_input(face_224_rgb)

    def encode(self, face_224_rgb: np.ndarray) -> dict:
        import torch

        tens = self.to_input(face_224_rgb).to(self.device)
        with torch.no_grad():
            code = self.model.encode(tens, use_detail=False)
        return {
            'cam': code['cam'].squeeze(0).cpu().numpy().astype(np.float32),
            'pose': code['pose'].squeeze(0).cpu().numpy().astype(np.float32),
            'shape': code['shape'].squeeze(0).cpu().numpy().astype(np.float32),
            'exp': code['exp'].squeeze(0).cpu().numpy().astype(np.float32),
        }
