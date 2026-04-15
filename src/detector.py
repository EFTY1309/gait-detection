"""
detector.py
-----------
Person detection and multi-object tracking using YOLOv8 + ByteTrack.

Provides:
  - PersonDetector  : wraps a YOLOv8 model and runs frame-by-frame tracking
  - Detection       : lightweight dataclass returned per detected person per frame
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    track_id: int           # stable ID across frames (assigned by ByteTrack)
    bbox: tuple             # (x1, y1, x2, y2) in pixel coords
    confidence: float       # YOLO detection confidence
    crop: np.ndarray        # BGR crop of the bounding-box region


class PersonDetector:
    """
    Wraps YOLOv8 + ByteTrack to detect and track people in a video stream.

    Usage
    -----
    detector = PersonDetector()
    for frame_detections in detector.process_video("video.mp4"):
        for det in frame_detections:
            # det.track_id, det.bbox, det.crop ...

    Or process one frame at a time:
        detections = detector.process_frame(bgr_frame)
    """

    PERSON_CLASS_ID = 0   # COCO class index for "person"

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection + tracking on a single BGR frame.
        Returns a list of Detection objects (one per tracked person).
        """
        results = self.model.track(
            frame,
            persist=True,             # keep ByteTrack state between calls
            tracker="bytetrack.yaml",
            classes=[self.PERSON_CLASS_ID],
            conf=self.conf_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return detections

        boxes = result.boxes
        h, w = frame.shape[:2]

        for box, track_id, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = map(int, box)
            # clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            detections.append(Detection(
                track_id=int(track_id),
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                crop=crop,
            ))

        return detections

    def process_video(self, video_path: str):
        """
        Generator: yields List[Detection] for each frame in the video.
        Also yields the frame itself as a second value.

        Usage:
            for frame, detections in detector.process_video("video.mp4"):
                ...
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = self.process_frame(frame)
                yield frame, detections
        finally:
            cap.release()

    @staticmethod
    def get_video_properties(video_path: str) -> dict:
        """Return fps, width, height, frame_count for a video file."""
        cap = cv2.VideoCapture(video_path)
        props = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return props
