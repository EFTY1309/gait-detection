"""
pose_extractor.py
-----------------
Extracts and normalizes body landmarks from a person crop using the
MediaPipe Tasks API (mp.tasks.vision.PoseLandmarker).

Used in MediaPipe >= 0.10.x where the old mp.solutions.pose was removed.

Landmark indices used later by feature_builder:
  11  left shoulder     12  right shoulder
  13  left elbow        14  right elbow
  23  left hip          24  right hip
  25  left knee         26  right knee
  27  left ankle        28  right ankle
"""

import os
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

NUM_LANDMARKS = 33

# Model file — downloaded automatically on first run
_MODEL_PATH = "pose_landmarker_full.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def _ensure_model() -> None:
    if not os.path.isfile(_MODEL_PATH):
        print(f"  Downloading MediaPipe PoseLandmarker model → {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("  Model download complete.")


class PoseExtractor:
    """
    Runs MediaPipe PoseLandmarker (Tasks API) on a single BGR crop and
    returns normalized landmarks.

    Returns
    -------
    np.ndarray of shape (33, 3) — [x, y, z] per landmark, normalized
    relative to the hip center, or None if no pose is detected.
    """

    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self, min_detection_confidence: float = 0.5):
        _ensure_model()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, raw_landmarks) -> np.ndarray:
        """Convert MediaPipe NormalizedLandmark list → normalized (33,3) array."""
        coords = np.array(
            [[p.x, p.y, p.z] for p in raw_landmarks], dtype=np.float32
        )  # (33, 3)
        hip_center = (coords[self.LEFT_HIP] + coords[self.RIGHT_HIP]) / 2.0
        coords -= hip_center
        hip_width = np.linalg.norm(coords[self.LEFT_HIP] - coords[self.RIGHT_HIP])
        if hip_width > 1e-6:
            coords /= hip_width
        return coords

    def _run(self, bgr_crop: np.ndarray):
        """Run the landmarker; return raw landmarks list or None."""
        if bgr_crop.size == 0:
            return None
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]   # first (and only) person in the crop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns normalized landmarks array shape (33, 3), or None.
        """
        raw = self._run(bgr_crop)
        if raw is None:
            return None
        return self._normalize(raw)

    def extract_with_landmarks_object(self, bgr_crop: np.ndarray):
        """
        Returns (normalized_array, raw_landmark_list) or (None, None).
        The raw_landmark_list is passed to visualizer.py for skeleton drawing.
        """
        raw = self._run(bgr_crop)
        if raw is None:
            return None, None
        return self._normalize(raw), raw

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
