# System Design: Gait-Based Person Recognition

*Current architecture, data flow, internal decision logic, and engineering trade-offs — updated to match the live implementation.*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Enrollment Workflow](#3-enrollment-workflow)
4. [Recognition Workflow](#4-recognition-workflow)
5. [Component: PersonDetector](#5-component-persondetector)
6. [Component: PoseExtractor](#6-component-poseextractor)
7. [Component: FeatureBuilder](#7-component-featurebuilder)
8. [Component: GalleryManager](#8-component-gallerymanager)
9. [Component: Matcher](#9-component-matcher)
10. [Component: Visualizer](#10-component-visualizer)
11. [Data Shapes](#11-data-shapes)
12. [Recognition State Machine](#12-recognition-state-machine)
13. [Feature Vector Design](#13-feature-vector-design)
14. [Matcher Design](#14-matcher-design)
15. [Design Decisions](#15-design-decisions)
16. [Failure Modes and Safeguards](#16-failure-modes-and-safeguards)
17. [Current Limitations](#17-current-limitations)

---

## 1. System Overview

The system identifies people by gait: their recurrent walking biomechanics rather than facial appearance, clothing, or colour. It uses a tracked person crop, estimates a normalized body skeleton, converts the skeleton sequence into a compact gait descriptor, and compares that descriptor against an enrolled gallery of known identities.

The current implementation has two major runtime modes:

```
Enrollment  →  extract one or more gait descriptors  →  append to gallery
Recognition →  extract recent gait descriptor        →  score against gallery → label track
```

The implementation is CPU-oriented. YOLOv8n handles detection and tracking, MediaPipe PoseLandmarker handles pose extraction, and the remaining logic is NumPy-based feature engineering and nearest-neighbour style matching.

The biggest change from earlier versions is that enrollment is no longer a single-vector operation by default. A single walking clip is segmented into multiple gait windows so one video contributes multiple feature vectors to the gallery. This improves robustness when two identities are close in feature space.

---

## 2. High-Level Architecture

### Module Dependency Map

```
enroll.py ─────────────────────────────────────────────────────────────┐
                                                                        │
recognize.py ──────────────────────────────────────────────────────────┤
                                                                        │
          ┌───────────────────────────────────────────────────────────┐ │
          │                         src/                              │ │
          │                                                           │ │
          │  detector.py              pose_extractor.py               │ │
          │  ┌──────────────┐         ┌───────────────────┐          │ │
          │  │PersonDetector│         │  PoseExtractor    │          │ │
          │  │YOLOv8n       │         │  MediaPipe Tasks  │          │ │
          │  │ByteTrack     │         │  PoseLandmarker   │          │ │
          │  └──────┬───────┘         └────────┬──────────┘          │ │
          │         │ Detection[]              │ ndarray (33,3)       │ │
          │         └──────────────┬───────────┘                      │ │
          │                        ▼                                  │ │
          │               feature_builder.py                          │ │
          │               ┌─────────────────────┐                     │ │
          │               │ build_feature_vector │                    │ │
          │               │ compute_frame_angles │                    │ │
          │               └──────────┬──────────┘                     │ │
          │                          │ ndarray (50,)                  │ │
          │          ┌───────────────┴───────────────┐                │ │
          │          ▼                               ▼                │ │
          │  gallery_manager.py                 matcher.py             │ │
          │  ┌───────────────────────┐          ┌──────────────────┐  │ │
          │  │build_features_from_   │          │ match()          │  │ │
          │  │video()                │          │ match_all()      │  │ │
          │  │preview_enrollment_    │          │ _score_gallery() │  │ │
          │  │tracks()               │          └────────┬─────────┘  │ │
          │  │load/save/add gallery  │                   │             │ │
          │  └──────────┬────────────┘                   ▼             │ │
          │             │ metadata-wrapped gallery       visualizer.py │ │
          │             ▼                               ┌────────────┐ │ │
          │      gallery/gallery.pkl                   │draw_...()   │ │ │
          │                                            └────────────┘ │ │
          └───────────────────────────────────────────────────────────┘ │
                                                                        │
          External model files                                           │
          ├── yolov8n.pt                 detector                         │
          └── pose_landmarker_full.task  pose extraction                  │
```

### Runtime Storage

The gallery is stored as a pickle payload containing both metadata and vectors:

```
{
    "__meta__": {
        "feature_dim": 50,
        "feature_version": 2,
    },
    "gallery": {
        "Jewel":  [ndarray(50,), ndarray(50,), ...],
        "Swel":   [ndarray(50,), ...],
        ...
    }
}
```

This versioned payload prevents old feature layouts from being mixed silently with the current one.

---

## 3. Enrollment Workflow

The enrollment pipeline is no longer “one video in, one feature out”. In the current system, the default behavior is “one video in, multiple segment features out”.

### CLI Flow

```
python enroll.py \
  --video PATH \
  --person NAME \
  [--track-id N] \
  [--segment-window 120] \
  [--segment-stride 60] \
  [--inspect-tracks]
```

### Operational Modes

#### Mode A — Inspect Tracks

When `--inspect-tracks` is passed, the system does not enroll anything. It runs detection and tracking over the video and writes a preview video with stable `track_id` labels drawn on each person. This is used for multi-person enrollment clips where the intended subject is not obvious.

Output:

```
output/<video_name>_track_preview.mp4
```

The console also prints the number of detected frames per track, which helps identify ambiguous cases.

#### Mode B — Standard Enrollment

When `--inspect-tracks` is not used, the pipeline is:

```
1. Resolve the input path.
2. Detect and track every person in the video.
3. Extract normalized pose landmarks per track.
4. Select one track for enrollment.
5. Convert that track's landmark sequence into one or more feature vectors.
6. Append all resulting vectors under the requested person ID.
7. Save the metadata-wrapped gallery.
```

### Track Selection Logic

There are two selection modes:

#### Explicit Selection

If the user passes `--track-id N`, that track is used directly.

#### Automatic Selection

If `--track-id` is omitted, the system ranks tracks by the number of valid pose frames and chooses the dominant track — but only if the clip is not ambiguous.

Ambiguity rule:

```
If second_best_track_count >= min_frames
and second_best_track_count / best_track_count >= 0.85
→ fail enrollment with an explicit error
```

This prevents silent mis-enrollment from multi-person clips.

### Segmented Enrollment

Once the chosen track is identified, its landmark sequence is split into overlapping windows.

Defaults:

```
segment_window = 120 frames
segment_stride = 60 frames
```

If the selected track has fewer than 120 frames, the full sequence produces one feature vector. Otherwise the system builds multiple overlapping windows and extracts one feature vector per window.

For a 324-frame track, the default segmentation produces 5 vectors.

### Enrollment Pipeline Diagram

```
video
  │
  ▼
PersonDetector.process_video()
  │
  ▼
per-frame detections  →  PoseExtractor.extract(crop)
  │
  ▼
track_landmarks: {track_id -> [ndarray(33,3), ...]}
  │
  ├── if --inspect-tracks: render preview and stop
  │
  └── else:
        select track
          │
          ├── explicit: --track-id
          └── auto: dominant track, unless ambiguous
                │
                ▼
        selected landmark sequence
                │
                ▼
        segment into windows (default 120, stride 60)
                │
                ▼
        build_feature_vector(window) for each segment
                │
                ▼
        add each vector to gallery[person_id]
                │
                ▼
        save_gallery() with metadata
```

---

## 4. Recognition Workflow

Recognition is track-based and uses recent evidence rather than full-history accumulation for scoring.

### Key Runtime Constants

```
MIN_FRAMES   = 60
REEVAL_EVERY = 60
EVAL_WINDOW  = 120
```

### Recognition State

Per video run, the system maintains:

```
track_landmarks : {track_id -> List[ndarray(33,3)]}
track_identity  : {track_id -> (identity, score)}
```

### Core Loop

For each frame:

```
1. Detect and track persons.
2. Extract pose landmarks per detected track.
3. Append landmarks into track_landmarks[track_id].
4. If enough frames have accumulated:
     - take the most recent eval window
     - build one feature vector from that recent slice
     - score it against the gallery
     - update track_identity[track_id]
5. Draw bbox, skeleton, label, and frame counter.
6. Optionally write output video.
```

### Why Recent-Window Matching Exists

Earlier versions matched on all accumulated frames for a track. That caused a failure mode where ambiguous early gait observations permanently polluted later identity estimates. The current implementation instead evaluates on the most recent `EVAL_WINDOW` frames, so later, cleaner gait evidence can overwrite a weak earlier guess.

### Recognition Pipeline Diagram

```
frame_t
  │
  ▼
PersonDetector.process_frame()
  │
  ▼
Detection(track_id, bbox, crop)
  │
  ▼
PoseExtractor.extract_with_landmarks_object(crop)
  │
  ├── no pose → skip landmark accumulation
  └── pose → append normalized landmarks to track_landmarks[track_id]
          │
          ▼
    n = len(track_landmarks[track_id])
          │
          ├── n < MIN_FRAMES → show collecting label
          └── evaluate every 60 frames or first eligibility
                   │
                   ▼
             recent_window = last 120 frames (default)
                   │
                   ▼
             build_feature_vector(recent_window)
                   │
                   ▼
             matcher.match(feature, gallery)
                   │
                   ▼
             draw_detection(..., identity, score)
```

---

## 5. Component: PersonDetector

### Responsibilities
- Run YOLOv8n person detection.
- Maintain ByteTrack state across frames.
- Emit stable `track_id` values for each tracked person.
- Return clipped crops for downstream pose extraction.

### Interface

```
PersonDetector(model_name="yolov8n.pt", conf_threshold=0.5)
  .process_frame(frame) -> List[Detection]
  .process_video(video_path) -> yields (frame, detections)
  .get_video_properties(video_path) -> fps, width, height, frame_count
```

### Detection Object

```
Detection:
  track_id    int
  bbox        (x1, y1, x2, y2)
  confidence  float
  crop        ndarray (BGR)
```

### Internal Notes

ByteTrack is kept alive across successive calls through `persist=True`, which is what makes `track_id` stable within a video.

---

## 6. Component: PoseExtractor

### Responsibilities
- Run MediaPipe Tasks PoseLandmarker on a single crop.
- Return 33 landmarks with x, y, z values.
- Normalize them around the hip center and hip width.

### Public API

```
extract(crop) -> ndarray(33,3) or None
extract_with_landmarks_object(crop) -> (ndarray(33,3), raw_landmark_list) or (None, None)
```

### Normalization

```
hip_center = (left_hip + right_hip) / 2
coords -= hip_center

hip_width = ||left_hip - right_hip||
coords /= hip_width   if hip_width > epsilon
```

This makes the representation translation- and scale-invariant in the crop frame.

---

## 7. Component: FeatureBuilder

### Responsibilities
- Convert one landmark sequence into one gait descriptor.
- Use joint-angle geometry rather than raw pixel or raw landmark coordinates.
- Produce a robust 50-dimensional feature vector.

### Inputs and Outputs

```
build_feature_vector(List[ndarray(33,3)]) -> ndarray(50,) or None
compute_frame_angles(ndarray(33,3)) -> ndarray(10,)
```

### Current 50-Dimensional Layout

The current feature vector is:

```
10 dims  mean       average posture per joint angle
10 dims  std        oscillation amplitude
10 dims  vel_std    gait tempo / angular smoothness
10 dims  iqr        robust interquartile spread
 5 dims  asym_mean  left-right gait bias
 5 dims  asym_std   variation in that left-right bias
-----------------------------------------------
50 dims  total
```

### Why This Replaced the Older 40-Dim Variant

The older variant used `vel_mean`, which carried little identity information because for cyclic gait it tends to be near zero for almost everyone. The new layout replaces that low-signal block with more discriminative, more stable descriptors:

- `iqr` instead of extreme-value statistics
- explicit left-right asymmetry features

This improved early separation without reintroducing the clip-length instability that came from `min`, `max`, and `range`.

---

## 8. Component: GalleryManager

### Responsibilities
- Resolve video paths safely.
- Load and save versioned galleries.
- Detect ambiguous enrollment clips.
- Build segmented enrollment features.
- Render track-preview videos.

### Key Behaviors

#### Versioned Gallery Persistence

`save_gallery()` writes both metadata and the gallery itself. `load_gallery()` filters out incompatible feature vectors if the saved dimension or feature version does not match the current code.

#### Path Resolution

If the requested enrollment file does not exist exactly, `_resolve_video_path()` tries a unique case-insensitive or prefix-based near-match in the same folder and prints a note when it does so.

#### Ambiguous Multi-Person Protection

Enrollment refuses to auto-pick a track when multiple tracks have similar support. This prevents hidden contamination of the gallery.

#### Track Preview Generation

`preview_enrollment_tracks()` writes an annotated preview video with `track_id` labels so a user can determine which person to enroll from a multi-person clip.

---

## 9. Component: Matcher

### Responsibilities
- Score a query feature against every identity in the gallery.
- Apply both threshold gating and ambiguity gating.
- Use gallery standardization only when the gallery is large enough.

### Current Matching Logic

The current implementation computes two score sets:

#### Raw Scores

Raw cosine similarity against the stored feature vectors.

This is used for absolute known-vs-unknown thresholding.

#### Standardized Scores

When there are at least `MIN_VECTORS_FOR_STANDARDIZATION = 4` stored vectors in total, the matcher z-scores the gallery and query feature dimension-wise before ranking identities.

This improves separation when multiple people are enrolled and the raw cosine space is too compressed.

### Decision Logic

```
1. Compute raw scores for thresholding.
2. Compute standardized scores for ranking when gallery size allows.
3. Let best identity come from ranked scores.
4. Let best raw score for that identity be the acceptance score.
5. If best raw score < threshold -> Unknown.
6. If ranked top1 - ranked top2 < margin -> Unknown.
7. Else return best identity.
```

### Default Parameters

```
DEFAULT_THRESHOLD = 0.85
DEFAULT_MARGIN    = 0.005
```

The smaller margin is important because close identities often produce cosine gaps on the order of `0.001–0.01`, especially when only a few people are enrolled.

---

## 10. Component: Visualizer

### Responsibilities
- Draw per-person bounding boxes.
- Render the raw MediaPipe skeleton on the original frame.
- Draw identity text and score.
- Draw frame counters.

### Rendering Logic

The skeleton is drawn manually with OpenCV using the raw landmark object returned by `extract_with_landmarks_object()`. This avoids dependence on the removed `mp.solutions.drawing_utils` path.

Identity strings are mapped to stable colours through an internal palette.

---

## 11. Data Shapes

```
frame                 ndarray uint8      (H, W, 3)
crop                  ndarray uint8      (h, w, 3)
raw_landmarks         MediaPipe proto    33 landmarks
normalized_landmarks  ndarray float32    (33, 3)
landmark_sequence     List[ndarray]      T × (33, 3)
angle_matrix          ndarray float32    (T, 10)
velocity_matrix       ndarray float32    (T-1, 10)
asymmetry_matrix      ndarray float32    (T, 5)
feature_vector        ndarray float32    (50,)
gallery               Dict[str, List[np.ndarray(50,)]]
scores                List[(str, float)]
identity_result       (str, float)
```

---

## 12. Recognition State Machine

```
New track
   │
   ▼
COLLECTING
  n < MIN_FRAMES
   │
   ├── no label yet except progress
   └── once n >= MIN_FRAMES
            │
            ▼
EVALUATING
  build feature from recent eval window
  run match()
            │
            ├── threshold fail → UNKNOWN
            ├── margin fail    → UNKNOWN
            └── accepted       → IDENTIFIED(name)
                    │
                    ▼
RE-EVALUATE every REEVAL_EVERY frames
using only the most recent EVAL_WINDOW frames
```

This differs from earlier “accumulate forever” versions by allowing later clean evidence to override earlier ambiguous evidence.

---

## 13. Feature Vector Design

### The 10 Joint Angles

The feature builder still uses 10 joint angles:

1. left hip
2. right hip
3. left knee
4. right knee
5. left ankle
6. right ankle
7. left trunk tilt
8. right trunk tilt
9. left arm swing
10. right arm swing

### Per-Angle Robust Statistics

The feature vector intentionally avoids clip-length-sensitive extremes. `iqr` is used instead of `min/max/range` because it captures spread without exploding as the clip grows longer.

### Asymmetry Features

The most important addition in the current implementation is explicit left-right asymmetry:

```
hip_left  - hip_right
knee_left - knee_right
ankle_left - ankle_right
trunk_left - trunk_right
arm_left - arm_right
```

Many people are distinguishable not by their mean posture alone but by how imbalanced their gait is across left and right limbs.

### Segmented Enrollment Interaction

The feature vector itself is still computed per window. Robustness now comes from the gallery containing several windows per person rather than forcing a single clip summary to represent all possible phases of that person’s gait.

---

## 14. Matcher Design

### Why One Vector Per Clip Was Not Enough

For pairs like `Jewel` and `Swel`, a single full-clip enrollment vector was not enough. Query windows from a mixed video would sometimes sit between the two classes, leading to unstable scores and frequent `Unknown` outputs.

Segmented enrollment reduces this problem by giving each identity a local cluster of windows rather than one global average. Matching then averages the query similarity over multiple within-person windows.

### Why Standardization Is Conditional

Standardization is powerful only when the gallery has enough diversity. With too few vectors, gallery-level mean and variance estimates are unstable. That is why the matcher only standardizes when there are at least four stored vectors in total.

---

## 15. Design Decisions

### Why Enrollment Now Segments by Default

Because one long clip contains multiple phases of gait. A single average vector can blur those phases together. Multiple overlapping windows preserve within-person variation and improve query-window matching.

### Why Recognition Uses a Recent Window

Because identity should be driven by the most recent stable walking evidence, not by the entire history of the track. Earlier ambiguous frames are often less useful than later clean frames.

### Why Ambiguous Enrollment Now Fails Hard

Because silently picking the longest track from a multi-person clip polluted the gallery in practice. Once that contamination happens, the matcher has no chance to recover correctly.

### Why the Margin Is Small

Because close identities often produce cosine gaps on the order of `0.001–0.01`, especially when only a few people are enrolled. A larger margin would convert many correct matches into `Unknown`.

---

## 16. Failure Modes and Safeguards

### Failure: Wrong Person Enrolled From a Multi-Person Clip

Safeguard:
- ambiguous-track ratio check
- `--inspect-tracks`
- `--track-id`

### Failure: Old Gallery Vectors Incompatible With New Feature Layout

Safeguard:
- metadata-wrapped gallery payload
- dimensionality filtering on load
- feature version warning

### Failure: Early Recognition Is Wrong But Later Frames Are Better

Safeguard:
- `EVAL_WINDOW`
- `REEVAL_EVERY`
- recent-window re-matching

### Failure: Similar People Collapse Into Unknown

Safeguard:
- segmented enrollment
- standardized ranking in matcher
- lower margin threshold

### Failure: User Misspells Enrollment File Name

Safeguard:
- near-match path resolution in `_resolve_video_path()`

---

## 17. Current Limitations

### Very Similar Walkers Still Need More Data

If two people are highly similar and only one clip each is enrolled, the system may still need 120–240 frames before the separation becomes reliable. Segmented enrollment helps, but it does not fully solve low-data identity overlap.

### Multi-Person Enrollment Still Requires Human Choice

The system can detect ambiguity and render track previews, but it cannot infer which track is the intended identity. That choice still has to come from the user.

### View Invariance Remains Limited

The system still uses 2-D pose-derived angles from a single camera. It is not truly view-invariant.

### Temporal Modeling Is Still Hand-Engineered

The system uses robust summary statistics rather than learned temporal embeddings. That keeps it simple and explainable, but there is still ceiling room for a learned sequence model.

### Open-Set Behaviour Depends on the Current Gallery

Because ranking quality depends on what identities are present in the gallery, the same query may behave differently as more people are enrolled. This is partly mitigated by standardization and margin gating, but not eliminated.
