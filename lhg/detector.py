"""MediaPipe FaceLandmarker wrapper for the LHG pipeline.

Reuses the same detector configuration as ``preprocess/stable_bbox.py``
and ``scene/data_loader.py`` so the LHG path produces stable subset
landmarks consistent with the rest of the codebase.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Importing the asset path lazily so that callers that only need the
# config dataclasses don't pay the mediapipe init cost at import time.
_DEFAULT_ASSET = Path('./assets/smirk/face_landmarker.task')


class MediaPipeFaceLandmarker:
    """Thin wrapper around mediapipe.tasks.vision.FaceLandmarker.

    Returns the 478 (x, y) landmark array in the input image's pixel
    coordinate system. Z is dropped on purpose — EPnP reconstructs depth
    from the 2D-3D correspondence and we don't want a second source of
    truth on per-frame Z that callers might accidentally read.
    """

    def __init__(self, asset_path: str | Path = _DEFAULT_ASSET):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        asset_path = Path(asset_path)
        if not asset_path.is_file():
            raise FileNotFoundError(
                f'MediaPipe FaceLandmarker asset not found: {asset_path}\n'
                f'Run the asset download step from the repo README, or pass '
                f'an explicit asset_path to MediaPipeFaceLandmarker().')

        base_options = python.BaseOptions(model_asset_path=str(asset_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Run detection on a uint8 (H, W, 3) RGB image.

        Returns
        -------
        landmarks : (478, 2) float64 array in pixel coordinates, or
                    ``None`` when no face was detected.
        """
        import mediapipe as mp

        if image_rgb.dtype != np.uint8:
            raise TypeError(f'image_rgb must be uint8, got {image_rgb.dtype}')
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f'image_rgb must be (H, W, 3), got {image_rgb.shape}')

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)
        if not result.face_landmarks:
            return None

        h, w = image_rgb.shape[:2]
        face = result.face_landmarks[0]
        out = np.empty((478, 2), dtype=np.float64)
        for i, lm in enumerate(face):
            out[i, 0] = lm.x * w
            out[i, 1] = lm.y * h
        return out

    def close(self) -> None:
        if hasattr(self._detector, 'close'):
            self._detector.close()
