"""FAN-based 68-point landmark detector wrapper.

Wraps ``face_alignment.FaceAlignment`` (the same library the avatar fit
pipeline uses in ``preprocess/keypoint_detector.py``). Returns 68 dlib-
ordered landmarks per frame; the caller pairs the static subset
(indices 17-47, brow + nose + eye) with the FLAME barycentric in
``assets/lhg/dlib_flame_landmarks.npz`` for per-frame EPnP.

Why FAN over MediaPipe for the LHG path
---------------------------------------
EPnP per-frame depth precision is bottlenecked by the spread and the
2D accuracy of the landmark set. MediaPipe FaceMesh's MICA-mapped
subset offers no anatomically correct LATERAL points (temples /
cheeks); the dlib face contour points DO give that lateral spread but
are pose-dependent. FAN's 68 points include the same dlib face
contour AND the static brow/nose/eye subset, AND the per-landmark 2D
accuracy is comparable to MediaPipe (~1-2 px). Switching the
detector lets us reuse the avatar-fit ``static_lmk_*`` FLAME
embedding and recover per-frame pose with much smaller noise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class FANLandmarker:
    """Lazy wrapper around ``face_alignment.FaceAlignment``.

    The face_alignment library does its own face detection (S3FD by
    default) before running landmark prediction. For LHG we feed an
    OUTER-CROPPED frame (the face is already centered), so the internal
    detector finds the face on the first try and the cost is dominated
    by the 2D-FAN-4 hourglass. On a 512x512 input expect ~10-25 ms on
    GPU. See ``onnx_export`` in this module for an optional ONNX path
    that drops to ~3-5 ms.
    """

    def __init__(self, device: str | None = None, flip_input: bool = False):
        import face_alignment
        import torch

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self._fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=flip_input,
            device=device,
        )

    def detect(
        self,
        image_rgb: np.ndarray,
        bbox_xyxy: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Run detection on a uint8 (H, W, 3) RGB image.

        Parameters
        ----------
        image_rgb : (H, W, 3) uint8 RGB.
        bbox_xyxy : optional fixed face bbox ``[x1, y1, x2, y2]`` in
            pixel coordinates. When supplied, face_alignment SKIPS its
            per-frame S3FD detection and uses this bbox to crop the
            input for the 2DFAN-4 hourglass. Skipping per-frame detection
            removes the dominant source of jitter (S3FD's bbox shifts
            ~5 px frame-to-frame, which propagates into 2-3 px landmark
            jitter even on a still face). Pass the SAME bbox every
            frame for a still subject; for a moving subject pass a
            slightly enlarged bbox that comfortably covers the head's
            motion range.

        Returns
        -------
        landmarks : (68, 2) float64 in pixel coordinates, or ``None``
                    when face_alignment finds no face.
        """
        if image_rgb.dtype != np.uint8:
            raise TypeError(f'image_rgb must be uint8, got {image_rgb.dtype}')
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f'image_rgb must be (H, W, 3), got {image_rgb.shape}')

        detected_faces = None
        if bbox_xyxy is not None:
            # face_alignment expects bbox as [x1, y1, x2, y2, score].
            box = list(map(float, np.asarray(bbox_xyxy).reshape(4)))
            detected_faces = [box + [1.0]]

        result = self._fa.get_landmarks_from_image(
            image_rgb, detected_faces=detected_faces,
        )
        if not result:
            return None
        return np.asarray(result[0], dtype=np.float64)

    def close(self) -> None:
        # face_alignment doesn't expose a close hook; releasing the
        # reference triggers GC, which releases the underlying torch
        # modules + CUDA memory.
        self._fa = None
