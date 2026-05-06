"""Video / image-folder frame iteration helper.

The LHG pipeline accepts either:

* A pre-extracted image directory (the same ``image/`` folder produced
  by ``demos/_preprocess_subject.sh``), in which case the outer crop
  has already been applied and ``outer_bbox`` is read from the existing
  ``outer_offset.json`` sidecar.
* A raw mp4/mov file. In this case the pipeline performs its own outer
  crop in-process (``--mode online`` uses a fixed bbox computed from
  the first ``world_mat_calibration_frames``; ``--mode pseudo-online``
  uses the clip-wide union, identical to ``crop_and_matting.py``).
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Iterator

import numpy as np


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


class FrameSource:
    """Unified interface over either a video file or an image directory."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if path.is_dir():
            self._mode = 'dir'
            self._dir = path
            files = sorted(
                glob(f'{path}/*.jpg') + glob(f'{path}/*.png'),
                key=lambda p: natural_key(Path(p).name),
            )
            if not files:
                raise FileNotFoundError(f'no jpg/png frames under {path}')
            self._files = [Path(f) for f in files]
            self._cap = None
            self._n = len(self._files)
        elif path.is_file():
            import cv2
            self._mode = 'video'
            self._dir = None
            self._files = None
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise FileNotFoundError(f'cv2.VideoCapture failed on {path}')
            self._n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            raise FileNotFoundError(f'not a file or directory: {path}')

    def __len__(self) -> int:
        return self._n

    def basenames(self) -> list[str]:
        if self._mode == 'dir':
            return [p.name for p in self._files]
        # Video frames don't have a natural basename — synthesize one
        # so downstream consumers can index by frame.
        width = max(5, len(str(self._n)))
        return [f'{i:0{width}d}.png' for i in range(self._n)]

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield uint8 RGB frames in source order."""
        import cv2
        if self._mode == 'dir':
            for p in self._files:
                img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise RuntimeError(f'cv2.imread failed on {p}')
                yield cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            while True:
                ok, frame_bgr = self._cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def load_outer_offset(image_dir: Path) -> np.ndarray | None:
    """Try to read ``outer_offset.json`` next to ``image/`` and return
    the [xmin, xmax, ymin, ymax] outer bbox in raw video coordinates.
    Returns ``None`` if the sidecar doesn't exist.

    Schema follows ``preprocess/crop_and_matting.py``: x_min, x_max,
    y_min, y_max, outer_size, image_size, source_image_dir, source_width,
    source_height.
    """
    image_dir = Path(image_dir)
    candidates = [
        image_dir.parent / 'outer_offset.json',
        image_dir / 'outer_offset.json',
    ]
    for c in candidates:
        if c.is_file():
            with open(c) as fp:
                payload = json.load(fp)
            xmin = int(payload['x_min'])
            xmax = int(payload['x_max'])
            ymin = int(payload['y_min'])
            ymax = int(payload['y_max'])
            return np.array([xmin, xmax, ymin, ymax], dtype=np.int32)
    return None
