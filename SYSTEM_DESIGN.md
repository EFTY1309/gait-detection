# System Design: Gait-Based Person Recognition

*Architecture, data flow, internal component design, and engineering decisions — from raw video pixels to a labeled identity output.*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Enrollment Pipeline — Full Flow](#3-enrollment-pipeline--full-flow)
4. [Recognition Pipeline — Full Flow](#4-recognition-pipeline--full-flow)
5. [Component: PersonDetector (detector.py)](#5-component-persondetector-detectorpy)
6. [Component: PoseExtractor (pose_extractor.py)](#6-component-poseextractor-pose_extractorpy)
7. [Component: FeatureBuilder (feature_builder.py)](#7-component-featurebuilder-feature_builderpy)
8. [Component: GalleryManager (gallery_manager.py)](#8-component-gallerymanager-gallery_managerpy)
9. [Component: Matcher (matcher.py)](#9-component-matcher-matcherpy)
10. [Component: Visualizer (visualizer.py)](#10-component-visualizer-visualizerpy)
11. [Data Shape Reference](#11-data-shape-reference)
12. [Recognition State Machine](#12-recognition-state-machine)
13. [The Feature Vector in Depth](#13-the-feature-vector-in-depth)
14. [Matching Algorithm in Depth](#14-matching-algorithm-in-depth)
15. [Design Decisions and Trade-offs](#15-design-decisions-and-trade-offs)
16. [Failure Modes and Mitigations](#16-failure-modes-and-mitigations)
17. [Limitations and Extension Points](#17-limitations-and-extension-points)

---

## 1. System Overview

The system identifies people in video by the way they walk — their gait. It does not use facial features, colour, or clothing. It builds a numerical fingerprint of each person''s walking biomechanics during a one-time enrollment step, then matches that fingerprint against the walking patterns it observes at recognition time.

The two runtime modes share the same four processing stages but differ in what they do with the final output:

```
Enrollment  →  extract gait fingerprint  →  store in gallery on disk
Recognition →  extract gait fingerprint  →  compare against gallery  →  label in output video
```

The system is intentionally CPU-friendly — no GPU is required. YOLOv8n, MediaPipe PoseLandmarker, and the numerical matching routines all run on CPU in real time on a modern laptop.

---

## 2. High-Level Architecture

### Module Dependency Map

```
enroll.py ─────────────────────────────────────────────────────────────┐
                                                                        │
recognize.py ──────────────────────────────────────────────────────────┤
                                                                        │
          ┌───────────────────────────────────────────────────────────┐ │
          │                     src/                                  │ │
          │                                                           │ │
          │   detector.py          pose_extractor.py                  │ │
          │   ┌──────────────┐     ┌───────────────────┐             │ │
          │   │ PersonDetector│     │  PoseExtractor    │             │ │
          │   │              │     │                   │             │ │
          │   │  YOLOv8n     │     │  MediaPipe Tasks  │             │ │
          │   │  ByteTrack   │     │  PoseLandmarker   │             │ │
          │   └──────┬───────┘     └────────┬──────────┘             │ │
          │          │ Detection[]           │ ndarray (33,3)         │ │
          │          └──────────┬────────────┘                        │ │
          │                     ▼                                      │ │
          │             feature_builder.py                             │ │
          │             ┌─────────────────┐                           │ │
          │             │  build_feature_ │                           │ │
          │             │  vector()       │                           │ │
          │             └────────┬────────┘                           │ │
          │                      │ ndarray (40,)                      │ │
          │          ┌───────────┴──────────────┐                     │ │
          │          ▼                           ▼                     │ │
          │  gallery_manager.py            matcher.py                  │ │
          │  ┌──────────────────┐          ┌───────────────┐          │ │
          │  │ add_to_gallery() │          │  match()      │          │ │
          │  │ save_gallery()   │          │               │          │ │
          │  │ load_gallery()   │          └───────┬───────┘          │ │
          │  └──────────────────┘                  │ (name, score)    │ │
          │         ▲  │ gallery.pkl               ▼                  │ │
          │         │  └──────────────────► visualizer.py             │ │
          │         │                       ┌────────────────┐        │ │
          │         │  gallery/             │ draw_detection │        │ │
          │         └── gallery.pkl ◄───────│ ()             │        │ │
          │                                 └────────────────┘        │ │
          └───────────────────────────────────────────────────────────┘ │
                                                                        │
          ┌─────────────────────────────────────────────────────────────┘
          │  External Models (auto-downloaded on first run)
          │  ├── yolov8n.pt                 (~6 MB,  COCO-trained detector)
          │  └── pose_landmarker_full.task  (~26 MB, MediaPipe model file)
          └─────────────────────────────────────────────────────────────
```

---

## 3. Enrollment Pipeline — Full Flow

```
INPUT: video file path + person name string
OUTPUT: updated gallery.pkl on disk

┌──────────────────────────────────────────────────────────────────────┐
│  enroll.py                                                           │
│                                                                      │
│  1. Parse CLI args                                                   │
│     --video  path   (required)                                       │
│     --person name   (required)                                       │
│     --gallery path  (default: gallery/gallery.pkl)                   │
│     --conf   float  (default: 0.50)                                  │
│     --min-frames int (default: 30)                                   │
│                                                                      │
│  2. Call gallery_manager.build_features_from_video()                 │
│     └── Returns: np.ndarray shape (40,) or None                      │
│                                                                      │
│  3. If result is not None:                                           │
│     a. gallery = load_gallery(path)                                  │
│     b. gallery_manager.add_to_gallery(gallery, person, vector)       │
│     c. save_gallery(gallery, path)                                   │
└──────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  gallery_manager.build_features_from_video()                         │
│                                                                      │
│  Instantiates:  PersonDetector(conf=args.conf)                       │
│                 PoseExtractor()                                       │
│                                                                      │
│  Per-frame loop:                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │  frame (H×W×3 BGR)                                             │  │
│  │     │                                                          │  │
│  │     ▼                                                          │  │
│  │  PersonDetector.process_frame(frame)                           │  │
│  │     → List[Detection]  (track_id, bbox, confidence, crop)      │  │
│  │     │                                                          │  │
│  │  For each Detection:                                           │  │
│  │     ▼                                                          │  │
│  │  PoseExtractor.extract(det.crop)                               │  │
│  │     → ndarray (33,3) normalized landmarks, or None             │  │
│  │     │                                                          │  │
│  │  Append to: track_landmarks[track_id].append(landmarks)        │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  After loop:                                                         │
│  1. Select dominant_track_id = track with most valid-pose frames     │
│  2. landmark_seq = track_landmarks[dominant_track_id]                │
│  3. If len(landmark_seq) < min_frames → return None                  │
│  4. feature_vector = feature_builder.build_feature_vector(           │
│         landmark_seq)          → ndarray (40,)                       │
│  5. Return feature_vector                                            │
└──────────────────────────────────────────────────────────────────────┘
```

The dominant track selection exists because enrollment videos are assumed to contain one primary subject. If YOLO happens to pick up a background person briefly, that track will have far fewer frames and is discarded automatically.

---

## 4. Recognition Pipeline — Full Flow

```
INPUT: video file path + loaded gallery dict
OUTPUT: annotated video written to output/

┌──────────────────────────────────────────────────────────────────────┐
│  recognize.py                                                        │
│                                                                      │
│  Constants (hardcoded at top of file):                               │
│    MIN_FRAMES   = 60    (accumulate this many frames before first    │
│                          recognition attempt per track)              │
│    REEVAL_EVERY = 60    (re-run recognition every N additional frames)│
│                                                                      │
│  State per track_id (Python dicts, reset each video run):           │
│    track_landmarks  : {track_id → List[ndarray(33,3)]}              │
│    track_identity   : {track_id → (name_str, score_float)}          │
│                                                                      │
│  Per-frame loop:                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │  frame  →  PersonDetector.process_frame(frame)                  │  │
│  │               → List[Detection]                                │  │
│  │                                                                │  │
│  │  For each Detection(track_id, bbox, conf, crop):               │  │
│  │                                                                │  │
│  │  1. PoseExtractor.extract_with_landmarks_object(crop)          │  │
│  │        → (ndarray(33,3), raw_landmark_list) or (None, None)    │  │
│  │                                                                │  │
│  │  2. If landmarks not None:                                     │  │
│  │        track_landmarks[track_id].append(normalized_array)      │  │
│  │                                                                │  │
│  │  3. n = len(track_landmarks[track_id])                         │  │
│  │     should_eval = (n >= MIN_FRAMES) and                        │  │
│  │                   (n % REEVAL_EVERY == 0 or                    │  │
│  │                    track_id not in track_identity)             │  │
│  │                                                                │  │
│  │  4. If should_eval:                                            │  │
│  │        fv = feature_builder.build_feature_vector(              │  │
│  │                 track_landmarks[track_id])   → ndarray(40,)    │  │
│  │        if fv is not None:                                      │  │
│  │           name, score = matcher.match(fv, gallery)             │  │
│  │           track_identity[track_id] = (name, score)             │  │
│  │                                                                │  │
│  │  5. identity = track_identity.get(track_id, None)              │  │
│  │     label = f"{name}  {score:.3f}  ({n}/{MIN_FRAMES})"         │  │
│  │             or "(collecting...  N/60)" if not yet evaluated    │  │
│  │                                                                │  │
│  │  6. Visualizer.draw_detection(frame, det, landmarks,           │  │
│  │                                label, identity_color)          │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Write annotated frame to output video (cv2.VideoWriter)             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component: PersonDetector (detector.py)

### Responsibilities
- Run YOLOv8n inference on every frame to produce bounding boxes
- Maintain ByteTrack state across frames to assign stable track IDs
- Clip bounding boxes to frame bounds and extract BGR crops

### Interface

```
PersonDetector(model_name="yolov8n.pt", conf_threshold=0.5)
  .process_frame(frame: ndarray H×W×3) → List[Detection]

Detection:
  .track_id    int           — stable ID from ByteTrack, persists per video
  .bbox        (x1,y1,x2,y2) int pixels, clamped to frame dimensions
  .confidence  float          — YOLO detection confidence [0,1]
  .crop        ndarray        — BGR sub-image at (x1:x2, y1:y2)
```

### Internal Flow

```
frame (H,W,3 BGR)
    │
    ▼
model.track(frame,
    persist=True,         ← keeps ByteTrack kalman state between calls
    tracker="bytetrack.yaml",
    classes=[0],          ← class 0 = person in COCO
    conf=threshold,
    verbose=False)
    │
    ▼
result.boxes.xyxy   → (N,4) float32 pixel coords
result.boxes.id     → (N,)  int     ByteTrack IDs
result.boxes.conf   → (N,)  float32 detection confidences
    │
    ▼
For each box:
  → clamp to [0,W] × [0,H]
  → crop = frame[y1:y2, x1:x2]
  → yield Detection(track_id, bbox, conf, crop)
```

### ByteTrack internals (relevant parts)

ByteTrack runs a Kalman filter per active track to predict each person's position in the next frame. When new detections arrive, it solves a bipartite matching problem (Hungarian algorithm) using IoU between predicted boxes and detected boxes as the cost function. High-confidence detections are matched first; low-confidence detections are retained in a "lost" buffer and re-matched in a second pass. A track is only deleted after it has been in the lost state for more than a configurable number of frames (`track_buffer` in bytetrack.yaml, default 30).

The `persist=True` flag tells ultralytics to use the same ByteTrack instance across successive `model.track()` calls on the same video stream, which is what allows track IDs to be stable.

---

## 6. Component: PoseExtractor (pose_extractor.py)

### Responsibilities
- Accept a BGR crop of a single person
- Run MediaPipe PoseLandmarker to get 33 body keypoints
- Normalize the keypoints to be position- and scale-invariant
- Return a (33,3) float32 array ready for feature extraction

### Interface

```
PoseExtractor(min_detection_confidence=0.5)
  .extract(bgr_crop: ndarray) → ndarray(33,3) or None
  .extract_with_landmarks_object(bgr_crop) → (ndarray(33,3), landmark_list) or (None, None)
  .close()

Context manager supported: with PoseExtractor() as pe: ...
```

### Why the Tasks API (not mp.solutions.pose)

MediaPipe removed `mp.solutions.pose` in 0.10.x. The replacement is the Tasks API: `mp.tasks.vision.PoseLandmarker`, which accepts a downloaded `.task` model file rather than bundling the model inside the Python package. The model is automatically downloaded from Google''s storage bucket on first run.

### Normalization Math

```
raw_coords shape (33, 3), each value in [0, 1] (MediaPipe outputs normalized coords)

Step 1 — translate:
    hip_center = (coords[23] + coords[24]) / 2      ← mean of left_hip, right_hip
    coords     = coords - hip_center                 ← hip sits at (0,0,0)

Step 2 — scale:
    hip_width  = ||coords[23] - coords[24]||₂        ← Euclidean distance
    if hip_width > 1e-6:
        coords = coords / hip_width

Result: position of every joint relative to hip center, expressed in units of hip widths.
A knee angle will be identical whether the person is 1m or 5m from camera.
```

Landmark index reference (relevant subset):

```
Index  Joint
  0    Nose
 11    Left shoulder
 12    Right shoulder
 13    Left elbow
 14    Right elbow
 15    Left wrist
 16    Right wrist
 23    Left hip          ← used for normalization origin
 24    Right hip         ← used for normalization scale
 25    Left knee
 26    Right knee
 27    Left ankle
 28    Right ankle
 29    Left heel
 30    Right heel
```

---

## 7. Component: FeatureBuilder (feature_builder.py)

### Responsibilities
- Accept a variable-length list of (33,3) landmark arrays
- Compute 10 joint angles per frame
- Summarize the angle time series with 4 statistics per angle
- Return a fixed-length 40-dimensional feature vector

### Interface

```
build_feature_vector(landmark_sequence: List[ndarray(33,3)]) → ndarray(40,) or None

compute_frame_angles(landmarks: ndarray(33,3)) → ndarray(10,) in degrees
```

### Joint Angle Computation

For each of the 10 joint triplets, the angle at the middle joint B is computed from three 3-D landmark positions:

```
Given points A, B, C (each (x,y,z)):
    BA = A - B
    BC = C - B
    cos(θ) = (BA · BC) / (||BA|| × ||BC||)
    θ = arccos(clip(cos(θ), -1, 1)) × (180/π)
```

The clip guards against floating-point values just outside [-1,1] that would produce NaN from arccos.

The 10 defined triplets:

```
Index  Name             A                  B (vertex)      C
  0    Left hip         left_shoulder      left_hip        left_knee
  1    Right hip        right_shoulder     right_hip       right_knee
  2    Left knee        left_hip           left_knee       left_ankle
  3    Right knee       right_hip          right_knee      right_ankle
  4    Left ankle       left_knee          left_ankle      left_heel
  5    Right ankle      right_knee         right_ankle     right_heel
  6    Left trunk tilt  left_shoulder      left_hip        right_hip
  7    Right trunk tilt right_shoulder     right_hip       left_hip
  8    Left arm swing   left_shoulder      left_elbow      left_wrist
  9    Right arm swing  right_shoulder     right_elbow     right_wrist
```

### Building the Feature Vector

```
Input:  landmark_sequence  List of T arrays, each (33,3)
        T must be >= 2 (need at least 2 frames for velocity)

Step 1 — Compute angle matrix:
        angles = np.array([compute_frame_angles(lm) for lm in sequence])
        shape: (T, 10),  values in degrees [0, 180]

Step 2 — Compute velocity:
        velocity = np.diff(angles, axis=0)
        shape: (T-1, 10)

Step 3 — Compute 4 statistics per angle:
               mean_a = np.mean(angles,   axis=0)   shape (10,)
                std_a = np.std(angles,    axis=0)   shape (10,)
              vel_mean = np.mean(velocity, axis=0)  shape (10,)
               vel_std = np.std(velocity,  axis=0)  shape (10,)

Step 4 — Concatenate:
        feature_vector = np.concatenate([mean_a, std_a, vel_mean, vel_std])
        shape: (40,)

Output: ndarray(40,)  dtype float32
```

Vector layout:

```
Indices  [0:10]   mean of each angle across T frames
Indices [10:20]   std  of each angle
Indices [20:30]   mean of frame-to-frame angle velocity
Indices [30:40]   std  of frame-to-frame angle velocity
```

### Why These Four Statistics

| Statistic | What it encodes | Typical range |
|---|---|---|
| mean | Average joint position during walking — posture habit | 90–170° for knees, varies per joint |
| std | Oscillation amplitude — how much each joint swings per step | 5–40° depending on joint |
| vel_mean | Net drift in joint angle over the clip — near 0 for cyclic gait, nonzero for asymmetries | ±2° per frame |
| vel_std | Walking tempo / cadence irregularity | 3–15° per frame |

---

## 8. Component: GalleryManager (gallery_manager.py)

### Responsibilities
- Orchestrate the full enrollment flow for a video file
- Load and save the gallery to disk as a pickle file
- Add new feature vectors to the gallery under a given name

### Gallery Data Structure

```
gallery: Dict[str, List[np.ndarray]]

Example (3 people enrolled, Alice with 2 clips):
{
    "Alice": [ndarray(40,), ndarray(40,)],
    "Bob":   [ndarray(40,)],
    "Carol": [ndarray(40,), ndarray(40,), ndarray(40,)]
}

Serialisation: pickle.dump(gallery, file, protocol=4)
File location: gallery/gallery.pkl  (default, configurable)
```

### add_to_gallery

Appends the new feature vector to the list for the given person, creating the list if it does not exist. Does not validate dimensionality at this layer — the caller (enroll.py) is responsible for passing a correctly shaped vector.

### build_features_from_video (dominant-track selection)

```
track_landmarks: Dict[int, List[ndarray(33,3)]]

After processing all frames:
    dominant_track = max(track_landmarks,
                         key=lambda tid: len(track_landmarks[tid]))

Rationale: in an enrollment video there is usually one intended subject.
Background pedestrians picked up by YOLO will have shorter tracks.
The longest track is overwhelmingly likely to be the enrolled person.
```

---

## 9. Component: Matcher (matcher.py)

### Responsibilities
- Compute cosine similarity between a query vector and every gallery entry
- Apply a two-stage decision (threshold + margin) to return a name or "Unknown"

### Interface

```
match(query_feature: ndarray(40,),
      gallery: Dict[str, List[ndarray(40,)]],
      threshold: float = 0.85,
      margin:    float = 0.03)
    → (identity: str, score: float)
```

### Cosine Similarity

```
cosine_similarity(a, b) = dot(a, b) / (||a||₂ × ||b||₂)

Range: [-1, 1]
  1.0  = identical direction (same person, ideal conditions)
  0.0  = orthogonal (unrelated features)
 -1.0  = opposite direction (not meaningful for this use case)
```

For a person with well-enrolled, stable features, same-person scores
consistently fall in the range **0.90–0.99**. Cross-person scores typically
stay below 0.80. The threshold of 0.85 sits in the middle of this gap.

Per-person score aggregation:

```
For each enrolled person P with k feature vectors [v₁, v₂, ..., vₖ]:
    person_score(P) = mean([cosine_similarity(query, vᵢ) for i in 1..k])
```

Averaging over multiple enrollment clips gives a more stable representative score than any single vector.

### Two-Stage Decision

```
all_scores = sorted list of (person_id, score) descending by score

Stage 1 — Absolute threshold:
    if all_scores[0].score < threshold:
        return ("Unknown", best_score)

Stage 2 — Margin check (only when 2+ people enrolled):
    if len(gallery) >= 2:
        margin_gap = all_scores[0].score - all_scores[1].score
        if margin_gap < margin:
            return ("Unknown", best_score)

Stage 3 — Accept:
    return (all_scores[0].person_id, best_score)
```

The margin check addresses the "generic walker" failure mode: someone not in the gallery who happens to walk vaguely like multiple enrolled people will score similarly to all of them. The gap between rank-1 and rank-2 will be small, triggering Unknown correctly.

Sensitivity analysis:

```
Scenario                              Best score  Gap    Decision
─────────────────────────────────────────────────────────────────
Enrolled person, clean conditions     0.95        0.12   ✓ Correct match
Unknown person, scores spread evenly  0.78        0.03   ✓ Unknown (threshold)
Unknown person, high-scoring spread   0.88        0.01   ✓ Unknown (margin)
Two similar people enrolled           0.86        0.08   ✓ Correct match
Two similar people, ambiguous query   0.84        0.02   ✓ Unknown (margin)
```

---

## 10. Component: Visualizer (visualizer.py)

### Responsibilities
- Draw a bounding box around each tracked person
- Overlay the skeleton (35 bone connections) on the frame
- Apply the identity label with confidence score and frame counter

### Why Manual Drawing

MediaPipe 0.10.x removed `mp.solutions.drawing_utils`. All skeleton rendering is done with OpenCV primitives using the list of 35 landmark-pair connections (`_POSE_CONNECTIONS`) defined in the module.

### Landmark-to-Pixel Conversion

The raw MediaPipe landmarks have coordinates in [0,1] relative to the crop. Converting back to absolute frame coordinates:

```
For landmark p (p.x, p.y in [0,1] relative to the person crop):

    crop region: x1,y1,x2,y2  (pixel coords in the full frame)
    crop_w = x2 - x1
    crop_h = y2 - y1

    pixel_x = int(x1 + p.x * crop_w)
    pixel_y = int(y1 + p.y * crop_h)
```

We use the raw landmark list (not the normalized array — those have been transformed) from `extract_with_landmarks_object()`. The raw list preserves the original [0,1] values relative to the crop, which are correctly projected back to full-frame pixel coordinates by the formula above.

---

## 11. Data Shape Reference

```
Name                  Type (dtype)         Shape       Where produced
───────────────────────────────────────────────────────────────────────────
frame                 ndarray (uint8)      (H,W,3)     cv2.VideoCapture
crop                  ndarray (uint8)      (h,w,3)     PersonDetector
raw_landmarks         MediaPipe list       33 items    PoseLandmarker
normalized_landmarks  ndarray (float32)   (33,3)      PoseExtractor
landmark_sequence     List[ndarray]        T×(33,3)    accumulator in enroll/recognize
angle_matrix          ndarray (float32)   (T,10)      compute_frame_angles × T
velocity_matrix       ndarray (float32)   (T-1,10)    np.diff(angle_matrix)
feature_vector        ndarray (float32)   (40,)       build_feature_vector
gallery               Dict[str,List]       —           load_gallery / add_to_gallery
cosine_scores         List[(str,float)]   N_people    match()
(identity, score)     (str, float)         —           match()
```

---

## 12. Recognition State Machine

Each track_id in a recognition run passes through four states:

```
                  ┌────────────────┐
    New track     │                │
    ─────────────►│  COLLECTING    │
                  │  n < 60 frames │
                  │  label: "..."  │
                  └───────┬────────┘
                          │ n == 60
                          ▼
                  ┌────────────────┐
                  │  EVALUATING    │◄──────────────────┐
                  │  build fv      │                   │
                  │  match()       │                   │
                  └───────┬────────┘                   │
                          │                            │
              ┌───────────┴─────────────┐              │
              │ score >= threshold      │ score < th   │
              │ gap >= margin           │ or gap < mg  │
              ▼                         ▼              │
    ┌──────────────────┐    ┌─────────────────────┐   │
    │  IDENTIFIED      │    │  UNKNOWN            │   │
    │  label: Name+sc  │    │  label: Unknown     │   │
    └────────┬─────────┘    └──────────┬──────────┘   │
             │                         │               │
             │  every 60 more frames   │               │
             └─────────────────────────┴───────────────┘
                      RE-EVALUATE (loop back)
```

The re-evaluation loop is the key difference from a simpler "lock-and-forget" design. Identity is never permanently committed — it is a best estimate that updates as evidence grows.

---

## 13. The Feature Vector in Depth

### Stability Requirement

A feature vector is useful only if the same person measured at different times produces nearly identical vectors. This convergence property must hold across:
- Different clip lengths (30-frame recognition window vs 200-frame enrollment clip)
- Minor lighting changes
- Slight camera angle differences

The statistics mean, std, vel_mean, vel_std all converge quickly: as T grows, they approach an asymptote determined by the person''s underlying gait pattern, not by how many frames were collected. Given a minimum of 60 frames (approximately 2 full walking cycles at 30fps), these values are stable.

### Why min/max/range Were Removed

During testing, an alternative 50-dim vector included per-angle minimum, maximum, and value range. These statistics are clip-length dependent: given more frames, YOLO/MediaPipe samples more extreme poses, so min decreases, max increases, and range grows. A 200-frame enrollment vector and a 60-frame recognition vector for the same person produced cosine similarities as low as 0.45 — worse than random — making the system unable to match a person to their own enrollment.

Removing these three statistics and using only the four stable ones (40 dims total) brought same-person cosine similarity to a reliable 0.90–0.99 range with as few as 60 accumulated frames.

### Cosine Similarity as the Right Metric

Euclidean distance in 40-dimensional space is sensitive to absolute vector magnitude. If one feature vector has slightly higher mean values across all angles (because the person was walking faster during enrollment than recognition), euclidean distance would be large even though the relative structure of the vector is identical. Cosine similarity measures only the direction of the vector, ignoring magnitude, which makes it robust to this class of variation.

---

## 14. Matching Algorithm in Depth

### Per-Person Score Aggregation

When a person has been enrolled multiple times, each clip produced one feature vector. Rather than choosing a single "representative" vector, the matcher averages cosine similarity over all enrollment vectors:

```
person_score = mean( cosine_similarity(query, vᵢ) for each stored vᵢ )
```

This is equivalent to measuring how close the query is to the centroid of the enrollment cluster in 40-dimensional space, but without computing the centroid explicitly. The benefit is that individual enrollment outliers (a clip where the person was carrying something) have reduced influence.

### The Margin Parameter

The margin of 0.03 is a hyperparameter derived from observed score distributions during testing. The typical inter-person cosine similarity gap for clearly different people was 0.08–0.15. The gap for genuinely ambiguous queries (unknown person resembling multiple enrolled people) was 0.00–0.02. A margin of 0.03 sits cleanly in the gap between these regimes.

If you enroll more similar-looking or similar-walking people, the margin may need to decrease to avoid false unknowns. If you need to minimize false positives in a security-sensitive context, increasing the margin to 0.05–0.08 adds conservatism at the cost of more Unknown labels.

---

## 15. Design Decisions and Trade-offs

### YOLOv8n vs Larger Models

YOLOv8n (nano) was chosen for speed. It runs at near-real-time on CPU at 720p. The gait recognition accuracy is limited more by the quality of the pose extraction and feature representation than by the precision of the bounding box — a slightly loose box around a person does not meaningfully degrade the skeleton quality. If operating in a crowded scene with heavily overlapping bounding boxes, switching to YOLOv8s or YOLOv8m would improve detection precision at a latency cost.

### MediaPipe PoseLandmarker vs Other Pose Estimators

MediaPipe was chosen because it is fast, ships with a downloadable model file, and requires no separate install beyond `pip install mediapipe`. The Tasks API (0.10+) is the current stable interface. Alternatives such as OpenPose or ViTPose would produce higher-accuracy landmarks (especially in challenging poses) but require significantly more setup and GPU resources.

### Skeleton-Based Features vs Silhouette-Based

Silhouette-based gait (Gait Energy Images, Gait Entropy Images) treats the body outline over time as the signal and is the dominant approach in academic literature. It requires clean background subtraction, which is difficult in moving-camera or crowded scenes. Skeleton-based features are more robust to scene complexity and produce an interpretable, compact representation, but require an accurate pose estimator — which MediaPipe provides.

### Nearest-Neighbour Matching vs Trained Classifier

A trained SVM or softmax classifier would likely outperform cosine nearest-neighbour at scale. However, classifiers require a fixed gallery size at training time — you cannot add a new person without retraining. The nearest-neighbour approach supports open-set recognition naturally: any person not in the gallery is returned as Unknown, and adding a new person requires only running enrollment again. For a 2–5 person gallery this trade-off clearly favours the simpler approach.

### Re-evaluation Every 60 Frames vs Continuous Streaming Updates

Computing the feature vector and running matching every single frame would be more responsive but computationally wasteful. Since the feature vector is an aggregate statistic over all accumulated frames, it changes very little from frame N to frame N+1 once N is large. A 60-frame stride gives a good compromise between responsiveness and efficiency.

---

## 16. Failure Modes and Mitigations

### Unknown Person Scores High for One Enrolled Person

If only one person is enrolled, the margin check is disabled and any walk that vaguely resembles the enrolled person''s features may exceed the 0.85 threshold. Mitigation: enroll a "dummy" second person, or lower the threshold. The more enrolled people there are, the more the margin check protects against false positives.

### Track ID Fragmentation

ByteTrack assigns a new ID when it loses a track for more than `track_buffer` frames (default 30 frames = 1 second at 30fps). If a person steps behind an obstruction for longer than this, they re-enter as a new track and the accumulated feature buffer is discarded. The system will spend another 60 frames collecting before re-identifying them. Mitigation: increase `track_buffer` in bytetrack.yaml, or implement a re-identification module.

### Enrollment Video Contains Multiple People

The dominant-track selection picks the track with the most valid-pose frames. If the intended subject is not on screen for the majority of the enrollment video, a background person could be enrolled instead. Mitigation: use a clean enrollment video with only the target person clearly visible.

### PoseExtractor Returns None Frequently

If MediaPipe is failing to detect poses, the landmark accumulator grows slowly. Common causes: the crop is too small (person very far away), the person is heavily occluded, or the detection confidence threshold is too high. Mitigation: lower `--conf` to 0.3, or ensure the person fills enough of the frame.

### Feature Vector Converges to Wrong Value Early

The first 60 frames may capture the person at an unusual angle (entering the frame, turning). The REEVAL_EVERY=60 mechanism allows the estimate to correct itself once more representative walking frames are available.

---

## 17. Limitations and Extension Points

### View Invariance

The current features are not fully view-invariant. A person walking directly toward the camera produces systematically different hip and knee angles than the same person walking across the frame. Enrolling from multiple angles partially mitigates this. A proper solution would involve a 3-D pose lifter (e.g., MotionBERT or VideoPose3D) to reconstruct true anatomical angles from 2-D projections.

### Temporal Structure

The four statistics (mean, std, vel_mean, vel_std) discard the exact gait cycle structure. A person''s gait can be further characterized by the shape of each joint''s periodic oscillation — its frequency, phase, and waveform. This information can be captured with Fourier descriptors on the angle sequences, or with sequence models (LSTM, temporal convolutional networks) that process the raw (T,10) angle matrix directly.

### Large-Scale Matching

At 2–5 people the nearest-neighbour search is trivially fast. At 1000+ people it would require approximate nearest-neighbour structures (e.g., FAISS) and a discriminative feature space trained specifically to separate identities. Replacing the hand-engineered 40-dim vector with a learned embedding (triplet loss or ArcFace on a sequence encoder) would produce significantly better discrimination at scale.

### Temporal Continuity Across Videos

The current system has no memory across video files. Each recognition run starts with empty track buffers. A persistent identity registry with appearance descriptors per known track would allow the system to recall people across multiple sessions without re-enrollment.

### Multi-View and Occlusion Handling

In a multi-camera setup, the same person''s skeleton can be reconstructed in 3-D by triangulating matched keypoints from two or more camera views. This would eliminate view-sensitivity and improve accuracy under partial occlusion. The feature extraction and matching pipeline would remain identical; only the pose extraction stage would need to be upgraded.
