"""
recognize.py
------------
CLI script to run gait recognition on a query video.

For each tracked person in the video, the script:
  1. Collects pose landmarks frame-by-frame
  2. Once MIN_FRAMES are collected, builds a gait feature vector
  3. Matches against the enrolled gallery via cosine similarity
  4. Overlays the identity label + skeleton on each frame
  5. Writes the annotated video to the output/ folder

Usage
-----
  python recognize.py --video path/to/query_video.mp4
  python recognize.py --video query.mp4 --gallery gallery/gallery.pkl --threshold 0.6
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detector import PersonDetector
from src.feature_builder import build_feature_vector
from src.gallery_manager import load_gallery
from src.matcher import DEFAULT_MARGIN, DEFAULT_THRESHOLD, match
from src.pose_extractor import PoseExtractor
from src.visualizer import draw_detection, draw_frame_info

DEFAULT_GALLERY = "gallery/gallery.pkl"
OUTPUT_DIR = "output"
MIN_FRAMES = 60   # frames before first recognition attempt (~2 gait cycles at 30fps)
REEVAL_EVERY = 60  # re-run matching every N frames to update the identity
EVAL_WINDOW = 120  # use the most recent N frames when recomputing identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize persons in a video by their gait."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the query video.",
    )
    parser.add_argument(
        "--gallery",
        default=DEFAULT_GALLERY,
        help=f"Path to gallery file (default: {DEFAULT_GALLERY}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO detection confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=(
            "Minimum gap between the best and second-best gallery score "
            f"(default: {DEFAULT_MARGIN})."
        ),
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=MIN_FRAMES,
        help=f"Minimum frames per track before recognition (default: {MIN_FRAMES}).",
    )
    parser.add_argument(
        "--eval-window",
        type=int,
        default=EVAL_WINDOW,
        help=(
            "Number of most recent frames to use for each recognition update "
            f"(default: {EVAL_WINDOW}, 0 uses all accumulated frames)."
        ),
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip writing the annotated output video (useful for quick testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load gallery ---
    gallery = load_gallery(args.gallery)
    if not gallery:
        print(
            f"[Recognize] WARNING: Gallery is empty or not found at '{args.gallery}'.\n"
            "            Run enroll.py first, or pass --gallery with the correct path."
        )

    print(f"\n[Recognize] Video     : {args.video}")
    print(f"[Recognize] Gallery   : {args.gallery}  ({len(gallery)} persons enrolled)")
    print(f"[Recognize] Threshold : {args.threshold}\n")
    print(f"[Recognize] Margin    : {args.margin}\n")
    print(f"[Recognize] EvalWindow: {args.eval_window}\n")

    # --- Video properties for output writer ---
    props = PersonDetector.get_video_properties(args.video)
    fps    = max(props["fps"], 1.0)
    width  = props["width"]
    height = props["height"]

    # --- Output path ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.video))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_recognized.mp4")
    writer: Optional[cv2.VideoWriter] = None
    if not args.no_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # --- Per-track state ---
    # Accumulated pose landmarks while we gather MIN_FRAMES
    track_landmarks: Dict[int, List[np.ndarray]] = defaultdict(list)
    # Locked-in identity once recognition fires
    track_identity: Dict[int, Tuple[str, float]] = {}

    detector  = PersonDetector(conf_threshold=args.conf)
    extractor = PoseExtractor()

    frame_idx = 0
    total = props["frame_count"]

    try:
        for frame, detections in detector.process_video(args.video):
            frame_idx += 1

            for det in detections:
                tid = det.track_id

                # Extract skeleton from crop
                lm, landmark_proto = extractor.extract_with_landmarks_object(det.crop)

                # Accumulate landmarks
                if lm is not None:
                    track_landmarks[tid].append(lm)

                # Determine identity to display
                n = len(track_landmarks[tid])
                should_eval = (n >= args.min_frames) and (n % REEVAL_EVERY == 0 or tid not in track_identity)

                if should_eval:
                    # (Re-)compute feature from the most recent gait window.
                    # Using a bounded window prevents early ambiguous frames from
                    # dominating all later identity updates.
                    if args.eval_window > 0:
                        eval_landmarks = track_landmarks[tid][-args.eval_window:]
                    else:
                        eval_landmarks = track_landmarks[tid]

                    feature = build_feature_vector(eval_landmarks)
                    if feature is not None:
                        identity, score = match(
                            feature,
                            gallery,
                            threshold=args.threshold,
                            margin=args.margin,
                        )
                    else:
                        identity, score = "Unknown", 0.0
                    track_identity[tid] = (identity, score)
                    print(
                        f"  Track {tid:>3} → {identity:<20} score={score:.3f}  "
                        f"(frames used: {len(eval_landmarks)}/{n} total)"
                    )
                elif tid in track_identity:
                    identity, score = track_identity[tid]
                else:
                    # Still accumulating
                    identity = f"... ({n}/{args.min_frames})"
                    score = 0.0

                draw_detection(
                    frame,
                    det.bbox,
                    identity,
                    score,
                    pose_landmarks=landmark_proto,
                    crop_offset=(det.bbox[0], det.bbox[1]),
                )

            draw_frame_info(frame, frame_idx, total)

            if writer is not None:
                writer.write(frame)

    finally:
        extractor.close()
        if writer is not None:
            writer.release()

    print(f"\n[Recognize] Done.  Processed {frame_idx} frames.")
    if not args.no_output:
        print(f"[Recognize] Output written to: {out_path}")


if __name__ == "__main__":
    main()
