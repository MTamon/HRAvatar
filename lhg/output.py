"""Writer for ``lhg_features.npz``.

The output format is the contract between this preprocessing step and
the downstream LHG model. Keep changes here additive (new optional
fields) — never rename or repurpose an existing field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LHGFeatures:
    """In-memory representation of one clip's LHG features.

    Camera convention
    -----------------
    ``global_rot`` and ``translation`` are in OpenCV camera coordinates
    (camera looks down +Z, X right, Y down) — the natural output of
    cv2.solvePnP. ``world_mat`` is the camera-extrinsic translation
    in those same coordinates, with ``world_mat[2, 3] = +mean(tvec_z)
    * flame_scale``. This is NOT directly compatible with HRAvatar's
    renderer convention (camera looks down -Z; ``world_mat[2, 3]`` is
    negative for an object in front of the camera).

    Downstream consumers feeding LHG features into HRAvatar's renderer
    must apply the OpenCV→HRAvatar sign-flip on Y and Z (and the
    corresponding rotation conversion). Storing the raw OpenCV output
    keeps the EPnP semantics transparent and lets each consumer pick
    the convention conversion appropriate for its renderer.
    """

    frame_basenames: np.ndarray         # (N,) <U..
    expression: np.ndarray              # (N, 50) float32
    jaw: np.ndarray                     # (N, 3) float32
    eyelid: np.ndarray                  # (N, 2) float32
    global_rot: np.ndarray              # (N, 3) float32 axis-angle, OpenCV
    translation: np.ndarray             # (N, 3) float32 FLAME canonical, OpenCV
    valid_mask: np.ndarray              # (N,) bool — False = MediaPipe miss
    interpolated_mask: np.ndarray       # (N,) bool — True = filled in pseudo-online
    rejected_mask: np.ndarray           # (N,) bool — True = Hampel-rejected
    mode: str                           # 'online' / 'pseudo-online'
    intrinsics: np.ndarray              # (4,) [fx, fy, cx, cy]
    world_mat: np.ndarray               # (4, 4) float32, OpenCV camera convention
    outer_bbox: np.ndarray              # (4,) int32 [xmin, xmax, ymin, ymax] in raw video coord
    fps: float
    image_size: int
    flame_scale: float
    camera_convention: str = 'opencv'   # 'opencv' or 'hravatar' (currently always 'opencv')
    metadata: dict = field(default_factory=dict)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame_basenames=self.frame_basenames,
            expression=self.expression.astype(np.float32),
            jaw=self.jaw.astype(np.float32),
            eyelid=self.eyelid.astype(np.float32),
            global_rot=self.global_rot.astype(np.float32),
            translation=self.translation.astype(np.float32),
            valid_mask=self.valid_mask.astype(bool),
            interpolated_mask=self.interpolated_mask.astype(bool),
            rejected_mask=self.rejected_mask.astype(bool),
            mode=np.array(self.mode),
            intrinsics=self.intrinsics.astype(np.float32),
            world_mat=self.world_mat.astype(np.float32),
            outer_bbox=self.outer_bbox.astype(np.int32),
            fps=np.array(self.fps, dtype=np.float32),
            image_size=np.array(self.image_size, dtype=np.int32),
            flame_scale=np.array(self.flame_scale, dtype=np.float32),
            camera_convention=np.array(self.camera_convention),
        )

    @classmethod
    def read(cls, path: str | Path) -> 'LHGFeatures':
        with np.load(path, allow_pickle=False) as npz:
            return cls(
                frame_basenames=npz['frame_basenames'],
                expression=npz['expression'],
                jaw=npz['jaw'],
                eyelid=npz['eyelid'],
                global_rot=npz['global_rot'],
                translation=npz['translation'],
                valid_mask=npz['valid_mask'],
                interpolated_mask=npz['interpolated_mask'],
                rejected_mask=npz['rejected_mask'],
                mode=str(npz['mode'].item()),
                intrinsics=npz['intrinsics'],
                world_mat=npz['world_mat'],
                outer_bbox=npz['outer_bbox'],
                fps=float(npz['fps']),
                image_size=int(npz['image_size']),
                flame_scale=float(npz['flame_scale']),
                camera_convention=(
                    str(npz['camera_convention'].item())
                    if 'camera_convention' in npz.files else 'opencv'
                ),
            )
