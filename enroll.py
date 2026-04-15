"""
enroll.py
---------
CLI script to enroll a person into the gallery.

Usage
-----
  python enroll.py --video path/to/walking_video.mp4 --person alice
  python enroll.py --video path/to/clip.mp4 --person bob --gallery gallery/gallery.pkl

The script extracts the dominant gait feature vector from the video and
appends it to the persistent gallery file.  Run it once per clip per person.
Multiple clips per person will be averaged over during matching.
"""

import argparse
import os
import sys

from src.gallery_manager import (
    add_to_gallery,
    build_features_from_video,
    gallery_summary,
    load_gallery,
    preview_enrollment_tracks,
    save_gallery,
)

DEFAULT_GALLERY = "gallery/gallery.pkl"
PREVIEW_DIR = "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enroll a person into the gait gallery."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the enrollment video (one person walking).",
    )
    parser.add_argument(
        "--person",
        required=True,
        help="Identity label for this person (e.g. 'alice').",
    )
    parser.add_argument(
        "--gallery",
        default=DEFAULT_GALLERY,
        help=f"Path to gallery file (default: {DEFAULT_GALLERY}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO detection confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=30,
        help="Minimum valid pose frames required to build a feature (default: 30).",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help=(
            "Track ID to enroll when the video contains multiple tracked people. "
            "If omitted, enrollment auto-selects the dominant track and now "
            "refuses ambiguous clips."
        ),
    )
    parser.add_argument(
        "--segment-window",
        type=int,
        default=120,
        help=(
            "Frames per enrollment segment. If the clip is longer than this, "
            "multiple feature vectors are extracted from one video (default: 120)."
        ),
    )
    parser.add_argument(
        "--segment-stride",
        type=int,
        default=60,
        help="Stride between enrollment segments (default: 60).",
    )
    parser.add_argument(
        "--inspect-tracks",
        action="store_true",
        help=(
            "Do not enroll. Instead, write a preview video with tracker IDs so "
            "you can choose the correct --track-id for a multi-person clip."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n[Enroll] Person  : {args.person}")
    print(f"[Enroll] Video   : {args.video}")
    print(f"[Enroll] Gallery : {args.gallery}\n")

    if args.inspect_tracks:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        preview_path = os.path.join(PREVIEW_DIR, f"{base_name}_track_preview.mp4")
        counts = preview_enrollment_tracks(
            video_path=args.video,
            output_path=preview_path,
            conf_threshold=args.conf,
        )
        print("[Enroll] Track preview created.")
        print(f"[Enroll] Preview video: {preview_path}")
        if counts:
            print("[Enroll] Track counts:")
            for tid, count in counts.items():
                print(f"  track {tid}: {count} detected frames")
        else:
            print("[Enroll] No tracks were detected in the preview video.")
        print("[Enroll] Re-run with --track-id <id> once you know the correct track.")
        return

    # Extract gait feature from the video
    try:
        feature = build_features_from_video(
            video_path=args.video,
            conf_threshold=args.conf,
            min_frames=args.min_frames,
            track_id=args.track_id,
            segment_window=args.segment_window,
            segment_stride=args.segment_stride,
            verbose=True,
        )
    except ValueError as exc:
        print(f"\n[Enroll] FAILED: {exc}")
        print(
            "[Enroll] Tip: run with --inspect-tracks to generate a preview video, "
            "then re-run with --track-id <id>."
        )
        sys.exit(1)

    if not feature:
        print("\n[Enroll] FAILED: could not extract a valid feature vector.")
        sys.exit(1)

    # Load existing gallery (or empty dict), append all segment features, save
    gallery = load_gallery(args.gallery)
    for vector in feature:
        add_to_gallery(gallery, args.person, vector)
    save_gallery(gallery, args.gallery)

    print(f"\n[Enroll] SUCCESS: '{args.person}' enrolled with {len(feature)} feature vector(s).")
    print(gallery_summary(gallery))


if __name__ == "__main__":
    main()
