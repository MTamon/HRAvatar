"""LHG (Listening Head Generation) feature extraction pipeline.

Per-frame extraction of the 11-dim FLAME parameter vector that an LHG
model has to predict at inference time:

* expression (50d), jaw (3d), eyelid (2d) from the SMIRK encoder
* global_rot (3d, axis-angle), translation (3d, FLAME canonical space)
  from cv2.solvePnP (EPnP) over MediaPipe FaceLandmarker stable subset

Two run modes share one per-frame core: ``online`` is strictly causal
(LHG inference parity) and ``pseudo-online`` adds clip-wide outer crop,
linear interpolation across detector dropouts, bidirectional Hampel for
local outliers, and bidirectional quaternion-flip detection — i.e.
corrections that do not happen at inference time but are safe to apply
to teacher data.

See ``doc/preprocessing_scope.md`` for why this is a separate pipeline
from ``demos/_preprocess_subject.sh`` (avatar personal fit).
"""
