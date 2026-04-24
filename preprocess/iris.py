# Iris landmark extraction using MediaPipe FaceLandmarker (Tasks API).
#
# This file used to depend on `face-detection-tflite` (fdlite), which calls
# `np.math.sqrt` at import time and is therefore broken under numpy 2.x.
# We replaced fdlite with the same MediaPipe FaceLandmarker model that
# scene/data_loader.py and utils/general_utils.py already use, so the iris
# centres are consistent across the rest of the pipeline.
#
# Output format (unchanged, consumed by preprocess/submodules/DECA/optimize.py):
#   {frame_basename: [x_first, y_first, x_second, y_second]}  in pixel units
# where (first, second) follows the original fdlite ordering after `[::-1]`,
# i.e. right-iris first, left-iris second in face-relative coordinates.
import argparse
import json
import os
import re
import sys

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# MediaPipe FaceLandmarker outputs 478 landmarks; 468..472 are the right-face
# iris ring (centre = 468), 473..477 are the left-face iris ring (centre = 473).
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# Match the path scene/data_loader.py loads.
DEFAULT_TASK_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'assets', 'smirk', 'face_landmarker.task'
)


def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings


def _build_detector(task_path):
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def annotate_iris_landmarks(image_path, savefolder, task_path=DEFAULT_TASK_PATH):
    """Annotate each frame with 2 iris landmarks (right then left)."""
    if not os.path.exists(task_path):
        raise FileNotFoundError(
            f"face_landmarker.task not found at {task_path}. "
            "Run download_assets.sh first."
        )

    frames = os.listdir(image_path)
    frames.sort(key=natural_sort_key)
    landmarks = {}

    with _build_detector(task_path) as detector:
        for frame in tqdm(frames):
            img = Image.open(os.path.join(image_path, frame)).convert('RGB')
            width, height = img.size
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.asarray(img, dtype=np.uint8),
            )

            result = detector.detect(mp_image)

            lmks = []
            if not result.face_landmarks:
                print("Empty iris landmarks")
            else:
                face_lm = result.face_landmarks[0]
                if len(face_lm) <= LEFT_IRIS_CENTER:
                    print("Empty iris landmarks")
                else:
                    right = face_lm[RIGHT_IRIS_CENTER]
                    left = face_lm[LEFT_IRIS_CENTER]
                    lmks = [
                        right.x * width, right.y * height,
                        left.x * width, left.y * height,
                    ]

            landmarks[frame] = lmks

    with open(os.path.join(savefolder, 'iris.json'), 'w') as f:
        json.dump(landmarks, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to images and deca and landmark jsons')
    args = parser.parse_args()
    image_path = os.path.join(args.path, 'image')
    if not os.path.exists(image_path):
        image_path = os.path.join(args.path, 'images')
    print(image_path)
    annotate_iris_landmarks(image_path=image_path, savefolder=args.path)
