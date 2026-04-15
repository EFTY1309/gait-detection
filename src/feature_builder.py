"""
feature_builder.py
------------------
Converts a sequence of normalized pose landmarks into a fixed-length
50-dimensional gait feature vector.

Pipeline
--------
1.  Receive landmarks shape (T, 33, 3)   (T frames collected for one person)
2.  Compute 10 joint angles per frame  →  shape (T, 10)
3.  Compute robust temporal summaries  →  shape (50,)

The 10 angles
-------------
  0  left hip angle         (left_shoulder  → left_hip   → left_knee)
  1  right hip angle        (right_shoulder → right_hip  → right_knee)
  2  left knee angle        (left_hip       → left_knee  → left_ankle)
  3  right knee angle       (right_hip      → right_knee → right_ankle)
  4  left ankle angle       (left_knee      → left_ankle → left_heel)
  5  right ankle angle      (right_knee     → right_ankle→ right_heel)
  6  left trunk tilt        (left_shoulder  → left_hip   → right_hip)
  7  right trunk tilt       (right_shoulder → right_hip  → left_hip)
  8  left arm swing         (left_shoulder  → left_elbow → left_wrist)
  9  right arm swing        (right_shoulder → right_elbow→ right_wrist)

Per-angle robust summaries
--------------------------
    mean, std, vel_std, iqr
    → 10 angles × 4 stats = 40 dims

Left-right asymmetry summaries
------------------------------
    mean and std of (left - right) for 5 paired joints
    → 5 pairs × 2 stats = 10 dims

Total feature size = 50 dims
"""

from typing import List, Optional

import numpy as np

# MediaPipe Pose landmark indices
LM = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_heel":      29,
    "right_heel":     30,
}

# Each triplet: (A, B, C) → angle computed at joint B
ANGLE_TRIPLETS = [
    ("left_shoulder",  "left_hip",    "left_knee"),    # 0 left hip angle
    ("right_shoulder", "right_hip",   "right_knee"),   # 1 right hip angle
    ("left_hip",       "left_knee",   "left_ankle"),   # 2 left knee angle
    ("right_hip",      "right_knee",  "right_ankle"),  # 3 right knee angle
    ("left_knee",      "left_ankle",  "left_heel"),    # 4 left ankle angle
    ("right_knee",     "right_ankle", "right_heel"),   # 5 right ankle angle
    ("left_shoulder",  "left_hip",    "right_hip"),    # 6 left trunk tilt
    ("right_shoulder", "right_hip",   "left_hip"),     # 7 right trunk tilt
    ("left_shoulder",  "left_elbow",  "left_wrist"),   # 8 left arm swing
    ("right_shoulder", "right_elbow", "right_wrist"),  # 9 right arm swing
]

# Symmetric left-right angle pairs. These capture personal asymmetry, which is
# often more discriminative than the absolute joint angles alone.
LEFT_RIGHT_ANGLE_PAIRS = [
    (0, 1),   # hip flexion asymmetry
    (2, 3),   # knee asymmetry
    (4, 5),   # ankle asymmetry
    (6, 7),   # trunk lean asymmetry
    (8, 9),   # arm swing asymmetry
]

NUM_ANGLES  = len(ANGLE_TRIPLETS)   # 10
FEATURE_VERSION = 2
FEATURE_DIM = NUM_ANGLES * 4 + len(LEFT_RIGHT_ANGLE_PAIRS) * 2


def _angle_at_b(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle (degrees) at joint B given three 3-D points A, B, C.
    Uses the dot-product formula:  angle = arccos(BA·BC / |BA||BC|)
    Returns 0.0 if vectors are degenerate (zero length).
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0
    cos_theta = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)   # guard against numerical drift
    return float(np.degrees(np.arccos(cos_theta)))


def compute_frame_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute all 10 joint angles for a single frame.

    Parameters
    ----------
    landmarks : np.ndarray shape (33, 3)

    Returns
    -------
    angles : np.ndarray shape (10,) in degrees
    """
    angles = np.zeros(NUM_ANGLES, dtype=np.float32)
    for i, (a_name, b_name, c_name) in enumerate(ANGLE_TRIPLETS):
        a = landmarks[LM[a_name]]
        b = landmarks[LM[b_name]]
        c = landmarks[LM[c_name]]
        angles[i] = _angle_at_b(a, b, c)
    return angles


def build_feature_vector(landmark_sequence: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Convert a list of per-frame landmark arrays into a single 50-dim feature vector.

    Parameters
    ----------
    landmark_sequence : list of np.ndarray, each shape (33, 3)
                        Must have at least 2 frames.

    Returns
    -------
        feature : np.ndarray shape (50,) or None if sequence is too short.

        Feature layout
        --------------
            0:10   mean        average joint posture
         10:20   std         swing amplitude
         20:30   vel_std     gait tempo / smoothness
         30:40   iqr         robust mid-range spread, less clip-length-sensitive
         40:45   asym_mean   average left-right bias per paired joint
         45:50   asym_std    variability of that left-right bias

        WHY THIS LAYOUT?
            vel_mean is usually near zero for everyone and added little discrimination.
            IQR is more robust than min/max/range and stabilizes quickly.
            Left-right asymmetry adds person-specific gait bias that often separates
            similar walkers earlier than absolute angles alone.
    """
    if len(landmark_sequence) < 2:
        return None

    # Stack → (T, 10)
    angle_matrix = np.array(
        [compute_frame_angles(lm) for lm in landmark_sequence],
        dtype=np.float32,
    )  # shape (T, 10)

    # Robust per-angle summaries. These converge quickly and remain stable as
    # clip length grows.
    mean = angle_matrix.mean(axis=0)                    # (10,) average posture
    std = angle_matrix.std(axis=0)                      # (10,) swing amplitude
    std = np.where(std < 1e-6, 1e-6, std)

    velocity = np.diff(angle_matrix, axis=0)           # (T-1, 10)
    vel_std = velocity.std(axis=0)                     # (10,) gait tempo

    q25 = np.percentile(angle_matrix, 25, axis=0)      # (10,)
    q75 = np.percentile(angle_matrix, 75, axis=0)      # (10,)
    iqr = q75 - q25                                    # (10,) robust spread

    asymmetry = np.stack(
        [angle_matrix[:, left] - angle_matrix[:, right]
         for left, right in LEFT_RIGHT_ANGLE_PAIRS],
        axis=1,
    )                                                  # (T, 5)
    asym_mean = asymmetry.mean(axis=0)                # (5,) left-right bias
    asym_std = asymmetry.std(axis=0)                  # (5,) asymmetry variation

    feature = np.concatenate([
        mean,
        std,
        vel_std,
        iqr,
        asym_mean,
        asym_std,
    ]).astype(np.float32)
    return feature
