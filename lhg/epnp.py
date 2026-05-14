"""cv2.solvePnP (EPnP) wrapper for per-frame translation / global_rot.

Per ``feedback_online_translation_via_epnp``: Adam-based per-frame
optimization is too slow for a 30 fps inference budget. EPnP is a
1-shot analytical solver (~1-2 ms/frame) that produces equivalent
per-frame quality once paired with a clean stable subset of
correspondences and a precomputed FLAME 3D landmark table.

Camera convention
-----------------
``solve_epnp`` accepts a ``camera_convention`` flag that controls the
output coordinate frame:

* ``'opencv'`` (raw EPnP) — X right, Y down, +Z forward.
* ``'hravatar'`` (default) — X right, Y up, -Z forward. This is the
  convention HRAvatar's renderer (and DECA's ``optimize.py``
  ``projection``) consume directly: ``world_mat[:3, 3]`` has a
  NEGATIVE Z component for an object in front of the camera, and the
  projection formula ``i = fx * x/z + cx`` produces a horizontally
  consistent mapping with the input image when z<0.

The conversion is M = diag(1, -1, -1) applied to both translation and
rotation:
* tvec_hr = (tvec_x, -tvec_y, -tvec_z)
* For axis-angle: axis_hr = (axis_x, -axis_y, -axis_z), angle unchanged
  (equivalent to R_hr = M @ R_cv @ M).
"""
from __future__ import annotations

from typing import Literal

import numpy as np


CameraConvention = Literal['opencv', 'hravatar']

# Diagonal sign-flip matrix for OpenCV → HRAvatar conversion.
# M = diag(1, -1, -1) maps OpenCV camera coordinates to HRAvatar:
# the y and z axes flip (OpenCV: y-down, +z-forward; HRAvatar: y-up,
# -z-forward). M is its own inverse.
_CV_TO_HR_DIAG = np.array([1.0, -1.0, -1.0], dtype=np.float64)


def convert_pose_opencv_to_hravatar(
    rvec_cv: np.ndarray, tvec_cv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an OpenCV (rvec, tvec) pair to HRAvatar/OpenGL convention.

    The OpenCV solver returns ``(R_cv, t_cv)`` such that a point in
    object frame maps to OpenCV-camera coordinates via
    ``p_cv = R_cv @ p_obj + t_cv``. The same physical transform in
    HRAvatar-camera coordinates is ``p_hr = M @ p_cv = (M @ R_cv) @
    p_obj + M @ t_cv``, where ``M = diag(1, -1, -1)``. Therefore:

        R_hr = M @ R_cv        (pre-multiply, NOT conjugation)
        t_hr = M @ t_cv

    The object frame itself is unchanged (FLAME canonical is already
    y-up); only the destination camera frame switches. Note this is
    NOT the symmetric ``M @ R_cv @ M`` form — that would apply when
    both the object frame AND the camera frame are rotating, which is
    not the case here.
    """
    import cv2

    rvec_cv = np.asarray(rvec_cv, dtype=np.float64).reshape(3)
    tvec_cv = np.asarray(tvec_cv, dtype=np.float64).reshape(3)
    R_cv, _ = cv2.Rodrigues(rvec_cv)
    M = np.diag(_CV_TO_HR_DIAG)
    R_hr = M @ R_cv
    rvec_hr, _ = cv2.Rodrigues(R_hr)
    tvec_hr = tvec_cv * _CV_TO_HR_DIAG
    return rvec_hr.reshape(3), tvec_hr


def solve_epnp(
    image_points_2d: np.ndarray,
    object_points_3d: np.ndarray,
    K: np.ndarray,
    flame_scale: float,
    camera_convention: CameraConvention = 'hravatar',
) -> tuple[np.ndarray, np.ndarray, bool] | tuple[None, None, bool]:
    """Solve PnP for a single frame.

    Parameters
    ----------
    image_points_2d : (K, 2) float64 — pixel coordinates in the same
        space the intrinsics ``K`` are pinned to (outer-cropped frame).
    object_points_3d : (K, 3) float64 — corresponding FLAME
        canonical-space landmark positions in METERS-equivalent units
        (``flame_scale`` factor will be applied here so that the EPnP
        solve runs in the same scale used by the intrinsics — i.e. the
        rendered scale, where 1 unit ~= 25cm * flame_scale).
    K : (3, 3) float64 — camera intrinsics matrix.
    flame_scale : float — HRAvatar's ``flame_scale`` constant (4.0 for
        v1, 1.0 for v2). Object points are multiplied by this before
        solving.

    Returns
    -------
    rvec : (3,) float64 axis-angle rotation, or ``None`` on failure.
    tvec_canonical : (3,) float64 translation in FLAME-CANONICAL space
        (i.e. the EPnP-recovered translation divided by ``flame_scale``
        so callers can store it parameter-equivalent to FLAME's
        ``translation_params``). Or ``None`` on failure.
    ok : bool — True iff cv2.solvePnP succeeded.
    """
    import cv2

    if image_points_2d.shape[0] < 4:
        return None, None, False
    if image_points_2d.shape[0] != object_points_3d.shape[0]:
        raise ValueError(
            f'2D/3D point count mismatch: '
            f'{image_points_2d.shape[0]} vs {object_points_3d.shape[0]}')

    obj_scaled = (object_points_3d * float(flame_scale)).astype(np.float64)
    img_pts = np.ascontiguousarray(image_points_2d, dtype=np.float64).reshape(-1, 1, 2)
    obj_pts = np.ascontiguousarray(obj_scaled, dtype=np.float64).reshape(-1, 1, 3)
    K = np.ascontiguousarray(K, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None, None, False

    rvec_out = rvec.flatten().astype(np.float64)
    tvec_canonical = tvec.flatten() / float(flame_scale)
    if camera_convention == 'hravatar':
        rvec_out, tvec_canonical = convert_pose_opencv_to_hravatar(
            rvec_out, tvec_canonical,
        )
    elif camera_convention != 'opencv':
        raise ValueError(
            f'unknown camera_convention {camera_convention!r}; '
            f'expected "opencv" or "hravatar"')
    return rvec_out, tvec_canonical.astype(np.float64), True


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert an axis-angle (3,) vector to a (w, x, y, z) quaternion.

    Used by the bidirectional quaternion-flip detector (pseudo-online
    mode) to enforce sign continuity across frames. Causal mode does
    the same with the most recent frame's quaternion.
    """
    aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = aa / angle
    half = angle * 0.5
    s = float(np.sin(half))
    c = float(np.cos(half))
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Inverse of ``axis_angle_to_quat``."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    q = q / n
    w = q[0]
    sin_half = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if sin_half < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * float(np.arctan2(sin_half, abs(w)))
    if w < 0:
        angle = -angle
    axis = q[1:] / sin_half
    return (axis * angle).astype(np.float64)


def fix_quat_sign_continuity(quats: np.ndarray) -> np.ndarray:
    """Flip the sign of each (w, x, y, z) so consecutive samples form
    the shorter arc on the 4-sphere (q and -q represent the same
    rotation).

    Pseudo-online use: full-pass once the per-frame quats are collected.
    Online use: caller maintains a one-frame state and calls
    ``fix_quat_sign_continuity(np.stack([prev, cur]))[1]``.
    """
    out = np.array(quats, dtype=np.float64, copy=True)
    for i in range(1, out.shape[0]):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] = -out[i]
    return out
