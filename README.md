# Gait Detection & Person Recognition from Video

A skeleton-based gait recognition system that identifies people by **how they walk** — no face recognition, no cooperation required. Built with YOLOv8, MediaPipe, and Python.

For a detailed explanation of how the system works end-to-end, see [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

---

## Installation

**Requirements:** Python 3.9+, Windows / Linux / macOS

```bash
pip install ultralytics mediapipe opencv-python numpy scikit-learn
```

On first run the system automatically downloads:
- `yolov8n.pt` (~6 MB) — YOLO person detection model
- `pose_landmarker_full.task` (~26 MB) — MediaPipe pose model

---

## Quick Start

```bash
# 1. Enroll people
python enroll.py --video data/enrollment_videos/alice_walk.mp4 --person Alice
python enroll.py --video data/enrollment_videos/bob_walk.mp4   --person Bob

# 2. Recognize in a new video
python recognize.py --video data/query_videos/test.mp4
```

Output video is saved to `output/test_recognized.mp4`.
