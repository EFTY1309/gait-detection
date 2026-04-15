"""
gallery_manager.py
------------------
Manages the enrollment gallery: build feature vectors from a labeled video,
persist them to disk, and load them back for recognition.

Gallery format (in-memory)
--------------------------
A plain Python dict:
    {
        "alice": [ np.ndarray(FEATURE_DIM,), ... ],
        "bob":   [ np.ndarray(FEATURE_DIM,), ... ],
        ...
    }

Gallery format (on-disk)
------------------------
Pickle payload with metadata:
    {
        "__meta__": {"feature_dim": ..., "feature_version": ...},
        "gallery": {...}
    }

Older plain-dict galleries are still readable. Incompatible feature vectors are
dropped so old and new feature layouts never mix silently.
"""

import difflib
import os
import pickle
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.detector import PersonDetector
from src.feature_builder import FEATURE_DIM, FEATURE_VERSION, build_feature_vector
from src.pose_extractor import PoseExtractor

# Minimum number of valid pose frames before we try to build a feature vector.
# 30 frames ≈ 1 complete gait cycle at 30 fps.
MIN_FRAMES = 30
AMBIGUOUS_TRACK_RATIO = 0.85
SEGMENT_WINDOW = 120
SEGMENT_STRIDE = 60

Gallery = Dict[str, List[np.ndarray]]


def _extract_gallery_and_meta(payload) -> tuple[Gallery, Optional[dict]]:
    """Support both legacy plain-dict galleries and metadata-wrapped payloads."""
    if isinstance(payload, dict) and "gallery" in payload and "__meta__" in payload:
        return payload.get("gallery", {}), payload.get("__meta__", {})
    return payload, None


def _filter_incompatible_vectors(gallery: Gallery) -> Gallery:
    """Keep only 1-D feature vectors matching the current feature dimension."""
    filtered: Gallery = {}
    dropped = 0
    for person_id, vectors in gallery.items():
        keep: List[np.ndarray] = []
        for vector in vectors:
            array = np.asarray(vector, dtype=np.float32)
            if array.ndim == 1 and array.shape[0] == FEATURE_DIM:
                keep.append(array)
            else:
                dropped += 1
        if keep:
            filtered[person_id] = keep

    if dropped:
        print(
            f"[Gallery] WARNING: dropped {dropped} incompatible feature vector(s). "
            f"Expected dim={FEATURE_DIM}, feature_version={FEATURE_VERSION}. "
            "Re-enroll affected people with the current pipeline."
        )
    return filtered


def _resolve_video_path(video_path: str) -> str:
    """Resolve a video path, allowing a unique near-match in the same folder."""
    if os.path.isfile(video_path):
        return video_path

    directory = os.path.dirname(video_path) or "."
    requested_name = os.path.basename(video_path)
    requested_stem, requested_ext = os.path.splitext(requested_name)

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Video not found: {video_path}")

    files = [
        name for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
    ]
    lower_name = requested_name.lower()
    lower_stem = requested_stem.lower()
    lower_ext = requested_ext.lower()

    exact_casefold = [name for name in files if name.lower() == lower_name]
    if len(exact_casefold) == 1:
        return os.path.join(directory, exact_casefold[0])

    compatible_ext = [
        name for name in files
        if not lower_ext or os.path.splitext(name)[1].lower() == lower_ext
    ]

    prefix_matches = [
        name for name in compatible_ext
        if os.path.splitext(name)[0].lower().startswith(lower_stem)
    ]
    if len(prefix_matches) == 1:
        resolved = os.path.join(directory, prefix_matches[0])
        print(f"[Enroll] NOTE: using closest file match: {resolved}")
        return resolved

    suggestions = difflib.get_close_matches(requested_name, files, n=5, cutoff=0.4)
    if not suggestions and lower_stem:
        suggestions = difflib.get_close_matches(requested_stem, files, n=5, cutoff=0.4)

    hint = ""
    if suggestions:
        hint = "\nClosest matches:\n  " + "\n  ".join(
            os.path.join(directory, name) for name in suggestions
        )

    raise FileNotFoundError(f"Video not found: {video_path}{hint}")


def build_features_from_video(
    video_path: str,
    conf_threshold: float = 0.5,
    min_frames: int = MIN_FRAMES,
    track_id: Optional[int] = None,
    ambiguous_track_ratio: float = AMBIGUOUS_TRACK_RATIO,
    segment_window: int = SEGMENT_WINDOW,
    segment_stride: int = SEGMENT_STRIDE,
    verbose: bool = True,
) -> Optional[List[np.ndarray]]:
    """
    Process a *single-person* enrollment video and return one feature vector.

    The function picks the track_id that appears in the most frames
    (assumed to be the enrolled person). Crops from all other tracker IDs
    are ignored.

    Parameters
    ----------
    video_path    : path to the enrollment video
    conf_threshold: YOLO detection confidence threshold
    min_frames    : minimum valid pose frames required to compute a feature
    track_id      : if provided, use this tracker ID instead of auto-selecting
    ambiguous_track_ratio:
                  if auto-selecting, fail when the runner-up track has at least
                  this fraction of the best track's valid-frame count and also
                  meets min_frames. This avoids silently enrolling the wrong
                  person from a multi-person clip.
    segment_window:
                  number of frames per enrollment segment. If <= 0, or if the
                  selected track is shorter than this, the full clip becomes one
                  feature vector.
    segment_stride:
                  stride between neighboring enrollment segments.
    verbose       : print progress info

    Returns
    -------
    list of np.ndarray shape (FEATURE_DIM,) or None if not enough valid frames were found.
    """
    video_path = _resolve_video_path(video_path)

    detector = PersonDetector(conf_threshold=conf_threshold)
    extractor = PoseExtractor()

    # Accumulate landmarks per track_id   { track_id: [lm_array, ...] }
    track_landmarks: Dict[int, List[np.ndarray]] = {}
    total_frames = 0

    try:
        for frame, detections in detector.process_video(video_path):
            total_frames += 1
            for det in detections:
                lm = extractor.extract(det.crop)
                if lm is None:
                    continue
                if det.track_id not in track_landmarks:
                    track_landmarks[det.track_id] = []
                track_landmarks[det.track_id].append(lm)
    finally:
        extractor.close()

    if verbose:
        print(f"  Processed {total_frames} frames from: {os.path.basename(video_path)}")
        for tid, lms in track_landmarks.items():
            print(f"  Track {tid}: {len(lms)} valid pose frames")

    if not track_landmarks:
        print("  WARNING: No persons detected in video.")
        return None

    # Select which track to enroll.
    if track_id is not None:
        if track_id not in track_landmarks:
            available = ", ".join(str(tid) for tid in sorted(track_landmarks))
            raise ValueError(
                f"Requested track_id={track_id} was not found. "
                f"Available track IDs: {available}"
            )
        best_tid = track_id
    else:
        ranked_tracks = sorted(
            track_landmarks.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        best_tid, best_lms = ranked_tracks[0]

        if len(ranked_tracks) >= 2:
            second_tid, second_lms = ranked_tracks[1]
            best_count = len(best_lms)
            second_count = len(second_lms)
            if (
                best_count > 0
                and second_count >= min_frames
                and (second_count / best_count) >= ambiguous_track_ratio
            ):
                raise ValueError(
                    "Enrollment video is ambiguous: multiple tracks have similar "
                    f"support (track {best_tid}: {best_count} frames, "
                    f"track {second_tid}: {second_count} frames). "
                    "Use --track-id to select the intended person explicitly, "
                    "or use a single-person enrollment video."
                )

    best_lms = track_landmarks[best_tid]

    if len(best_lms) < min_frames:
        print(
            f"  WARNING: Only {len(best_lms)} valid pose frames "
            f"(need {min_frames}). Skipping."
        )
        return None

    features: List[np.ndarray] = []

    if segment_window <= 0 or len(best_lms) < segment_window:
        feature = build_feature_vector(best_lms)
        if feature is not None:
            features.append(feature)
    else:
        start_indices = list(range(0, len(best_lms) - segment_window + 1, segment_stride))
        last_start = len(best_lms) - segment_window
        if start_indices[-1] != last_start:
            start_indices.append(last_start)

        for start in start_indices:
            segment = best_lms[start:start + segment_window]
            feature = build_feature_vector(segment)
            if feature is not None:
                features.append(feature)

    if verbose and features:
        print(
            f"  Built {len(features)} feature vector(s): "
            f"shape={features[0].shape}, from track {best_tid}"
        )
        if len(features) > 1:
            print(
                f"  Enrollment segmentation: window={segment_window}, "
                f"stride={segment_stride}, track_frames={len(best_lms)}"
            )

    return features or None


def preview_enrollment_tracks(
    video_path: str,
    output_path: str,
    conf_threshold: float = 0.5,
) -> Dict[int, int]:
    """Render a preview video with tracker IDs overlaid for enrollment review."""
    video_path = _resolve_video_path(video_path)
    detector = PersonDetector(conf_threshold=conf_threshold)
    props = PersonDetector.get_video_properties(video_path)
    fps = max(props["fps"], 1.0)
    width = props["width"]
    height = props["height"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create preview video: {output_path}")

    track_counts: Dict[int, int] = {}
    frame_idx = 0

    try:
        for frame, detections in detector.process_video(video_path):
            frame_idx += 1
            for det in detections:
                tid = det.track_id
                track_counts[tid] = track_counts.get(tid, 0) + 1
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                label = f"track {tid}  frames={track_counts[tid]}"
                (tw, th), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    1,
                )
                label_y = max(y1 - 6, th + 4)
                cv2.rectangle(
                    frame,
                    (x1, label_y - th - baseline - 4),
                    (x1 + tw + 4, label_y + baseline - 2),
                    (0, 220, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            header = f"Enrollment track preview  frame {frame_idx}/{props['frame_count']}"
            cv2.putText(
                frame,
                header,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    return dict(sorted(track_counts.items()))


# ---------------------------------------------------------------------------
# Gallery persistence
# ---------------------------------------------------------------------------

def load_gallery(gallery_path: str) -> Gallery:
    """Load the gallery from disk. Returns an empty dict if file doesn't exist."""
    if not os.path.isfile(gallery_path):
        return {}
    with open(gallery_path, "rb") as f:
        payload = pickle.load(f)
    gallery, meta = _extract_gallery_and_meta(payload)
    if meta and meta.get("feature_version") != FEATURE_VERSION:
        print(
            f"[Gallery] WARNING: gallery feature_version={meta.get('feature_version')} "
            f"but current pipeline expects version={FEATURE_VERSION}."
        )
    return _filter_incompatible_vectors(gallery)


def save_gallery(gallery: Gallery, gallery_path: str) -> None:
    """Persist the gallery dict to disk using pickle."""
    gallery_dir = os.path.dirname(gallery_path)
    if gallery_dir:
        os.makedirs(gallery_dir, exist_ok=True)
    payload = {
        "__meta__": {
            "feature_dim": FEATURE_DIM,
            "feature_version": FEATURE_VERSION,
        },
        "gallery": gallery,
    }
    with open(gallery_path, "wb") as f:
        pickle.dump(payload, f)


def add_to_gallery(
    gallery: Gallery,
    person_id: str,
    feature: np.ndarray,
) -> None:
    """Add one feature vector to the in-memory gallery (modifies in-place)."""
    if person_id not in gallery:
        gallery[person_id] = []
    gallery[person_id].append(feature)


def gallery_summary(gallery: Gallery) -> str:
    """Return a human-readable summary of the gallery contents."""
    if not gallery:
        return "Gallery is empty."
    lines = ["Gallery contents:"]
    for person_id, vectors in gallery.items():
        lines.append(f"  {person_id}: {len(vectors)} feature vector(s)")
    return "\n".join(lines)
