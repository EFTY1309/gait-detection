"""
visualizer.py
-------------
Drawing utilities for the recognition output video.

Draws per tracked person:
  - Bounding box            (color changes per identity)
  - Identity label + score  (top of the box)
  - MediaPipe skeleton      (drawn manually with OpenCV — no mp.solutions needed)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe 33-landmark body connections (landmark index pairs)
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),
]

# Palette of BGR colors — one per unique identity
_PALETTE = [
    (0,   200, 255),   # yellow-ish
    (0,   255, 100),   # green
    (255, 100,   0),   # blue
    (180,   0, 255),   # purple
    (0,   100, 255),   # orange
    (255,   0, 150),   # pink
    (100, 255, 255),   # light cyan
    (255, 255,   0),   # aqua
]

# Keep a stable mapping: identity string → color index
_identity_color_map: Dict[str, int] = {}


def _color_for(identity: str) -> Tuple[int, int, int]:
    if identity not in _identity_color_map:
        _identity_color_map[identity] = len(_identity_color_map) % len(_PALETTE)
    return _PALETTE[_identity_color_map[identity]]


def draw_detection(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    identity: str,
    score: float,
    pose_landmarks=None,
    crop_offset: Optional[Tuple[int, int]] = None,  # kept for API compat, unused
) -> np.ndarray:
    """
    Draw bounding box + label (+ optional skeleton) on a frame in-place.

    Parameters
    ----------
    frame         : full BGR frame (modified in-place)
    bbox          : (x1, y1, x2, y2) bounding box on the full frame
    identity      : person name / "Unknown"
    score         : cosine similarity score (0–1)
    pose_landmarks: raw MediaPipe pose_landmarks proto (optional)
    crop_offset   : (x1, y1) of the crop within the frame, needed to draw
                    the skeleton in the right position on the full frame

    Returns
    -------
    frame : same array, modified in-place (also returned for convenience)
    """
    x1, y1, x2, y2 = bbox
    color = _color_for(identity)
    thickness = 2

    # --- Bounding box ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # --- Draw skeleton on full frame (MediaPipe coordinates are relative to crop) ---
    if pose_landmarks is not None:
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w > 0 and crop_h > 0:
            # Convert normalized (0-1) landmark coords to pixel coords on the full frame
            pts = [
                (int(x1 + p.x * crop_w), int(y1 + p.y * crop_h))
                for p in pose_landmarks
            ]
            # Draw connections
            for a, b in _POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
            # Draw joints
            for pt in pts:
                cv2.circle(frame, pt, 2, (255, 255, 255), -1)

    # --- Label background + text ---
    label = f"{identity}  {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    label_y = max(y1 - 6, th + 4)
    cv2.rectangle(
        frame,
        (x1, label_y - th - baseline - 4),
        (x1 + tw + 4, label_y + baseline - 2),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 2, label_y - baseline),
        font,
        font_scale,
        (0, 0, 0),   # black text on colored background
        font_thickness,
        cv2.LINE_AA,
    )

    return frame


def draw_frame_info(frame: np.ndarray, frame_idx: int, total: int) -> np.ndarray:
    """Overlay frame counter (top-right corner) — useful for debugging."""
    text = f"Frame {frame_idx}/{total}"
    cv2.putText(
        frame, text, (frame.shape[1] - 180, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return frame
