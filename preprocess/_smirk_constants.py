"""Constants ported from MTamon/smirk@release/cuda128 utils/bbox_tracker.py.

Kept in a tiny separate module so that both `preprocess/stable_bbox.py`
(offline preprocess) and `scene/data_loader.py` (online training-time
crop) can import the exact same values without pulling in scipy.
"""
from __future__ import annotations

import numpy as np


# MediaPipe FaceMesh landmark indices that do not move with speech / blink.
# Covers the face's horizontal and vertical extent using only points anchored
# to skull geometry (eye corners, nose bridge, temples). Mouth, jaw outline,
# eyebrows, and eyelids are intentionally excluded so that mouth opening and
# blinking do not leak into the bbox `size` signal.
#
#   33, 133   : right eye outer / inner corner
#   362, 263  : left eye inner / outer corner
#   1, 4, 5   : nose tip and bottom-of-nose
#   6, 168    : glabella (between brows, on bone) and nose root
#   195, 197  : nose bridge
#   234, 454  : right / left temple
#   127, 356  : just below the temples on the hairline
STABLE_LANDMARK_INDICES = np.array(
    [33, 133, 362, 263, 1, 4, 5, 6, 168, 195, 197, 234, 454, 127, 356],
    dtype=np.int64,
)

# The stable subset omits the mouth/jaw/eyebrows/forehead, so the (top, bottom)
# extent over the subset is roughly one third of the full-face extent (it is
# dominated by "nose root to nose tip"). Horizontal extent is comparable to
# the full-face width since 234 / 454 are temple points at the widest part of
# the face. Using the legacy `(width + height) / 2` size formula on the stable
# subset therefore yields a `size` about 60% of what the all-landmarks formula
# produced, and the legacy `scale=1.4` no longer covers the whole face.
#
# The calibration below rescales the stable-subset size so that the effective
# crop matches what the legacy pipeline was trained on, keeping `scale`
# semantics unchanged across modes. Tuned upstream in MTamon/smirk to make the
# average crop visually match the legacy bbox.
STABLE_LANDMARK_SIZE_CALIBRATION = 1.55
