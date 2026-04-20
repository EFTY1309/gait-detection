# Technology Overview

This file explains which library is used for which part of the gait-recognition system, what the gallery file stores, and how recognition works on a video.

## 1. Which Technology Is Used For What

### Person Detection and Tracking

- Library: `ultralytics`
- Model: `YOLOv8n`
- Tracking: `ByteTrack`
- Code: `src/detector.py`

Work:
- Detects people in each video frame.
- Assigns a stable `track_id` to each person across frames.
- Crops each detected person from the frame.

So this part answers:
- Where is the person?
- Is this the same person as in the previous frame?

### Pose / Skeleton Extraction

- Library: `mediapipe`
- Model: `PoseLandmarker`
- Code: `src/pose_extractor.py`

Work:
- Takes one person crop.
- Finds 33 body landmarks like shoulders, hips, knees, and ankles.
- Normalizes the landmarks around the hip center and hip width.

So this part answers:
- What is the body posture of this person?
- How is the body moving while walking?

### Gait Feature Generation

- Library: `numpy`
- Code: `src/feature_builder.py`

Work:
- Converts landmark sequences into joint-angle sequences.
- Builds one gait feature vector from a walking segment.
- Current feature size: `50` dimensions.

This feature vector summarizes walking style using:
- mean joint angles
- angle variation
- angular velocity variation
- robust spread (`iqr`)
- left-right asymmetry

So this part answers:
- What compact numeric signature represents this person's gait?

### Gallery Creation and Management

- Libraries: `pickle`, `numpy`, `opencv-python`
- Code: `src/gallery_manager.py`

Work:
- Loads and saves the enrolled gallery.
- Builds features from enrollment videos.
- Splits longer enrollment clips into multiple windows.
- Prevents ambiguous multi-person enrollment unless a `--track-id` is chosen.
- Can generate a preview video with track IDs.

So this part answers:
- How do we store known people?
- How do we add a new person to the system?

### Recognition / Matching

- Library: `numpy`
- Method: cosine similarity
- Code: `src/matcher.py`

Work:
- Compares the query gait feature against every enrolled identity.
- Computes similarity scores.
- Applies threshold and margin checks.
- Returns the best identity or `Unknown`.

So this part answers:
- Which enrolled person is most similar to this walking pattern?

### Video Reading and Drawing Output

- Library: `opencv-python`
- Code: `recognize.py`, `src/visualizer.py`, `src/detector.py`

Work:
- Reads the input video frame by frame.
- Writes output video.
- Draws bounding boxes, skeletons, names, and scores.

## 2. What Is Stored In The Gallery File

The gallery is stored in:

- `gallery/gallery.pkl`

This is a Python pickle file.

`.pkl` means:
- a serialized Python object saved to disk
- not an image, not a model, not a database
- it stores Python data structures directly

In this project, the file stores:

```python
{
    "__meta__": {
        "feature_dim": 50,
        "feature_version": 2,
    },
    "gallery": {
        "Jewel": [array(...), array(...), ...],
        "Swel": [array(...), array(...), ...],
    }
}
```

Meaning:
- `__meta__`: information about the current feature format
- `gallery`: all enrolled people
- each person name maps to a list of gait feature vectors

Why multiple vectors per person?

Because one enrollment video is split into several walking segments, and each segment produces one feature vector. That makes recognition more robust.

## 3. How Enrollment Works

Command example:

```bash
python enroll.py --video data\enrollment_videos\alice.mp4 --person Alice
```

Flow:

1. YOLOv8 + ByteTrack detects all people and gives track IDs.
2. MediaPipe extracts skeleton landmarks for each detected person.
3. The system chooses one track for enrollment.
4. Landmark frames from that track are grouped into segments.
5. Each segment becomes a 50-dim gait feature vector.
6. Those vectors are saved under the person's name in `gallery/gallery.pkl`.

If the video contains multiple similar tracks, the system may stop and ask you to use:

```bash
--inspect-tracks
```

and then:

```bash
--track-id N
```

## 4. How Recognition Works From A Video

Command example:

```bash
python recognize.py --video data\query_videos\test.mp4
```

Flow:

1. OpenCV reads the video frame by frame.
2. YOLOv8 + ByteTrack detects and tracks each person.
3. MediaPipe extracts pose landmarks from each person crop.
4. Landmarks are collected over time for each `track_id`.
5. After enough frames are collected, the system builds a gait feature vector.
6. The matcher compares that feature against all people in the gallery.
7. The best match is returned if it passes:
   - similarity threshold
   - margin check against the second-best identity
8. The name and score are drawn on the output video.

## 5. Simple Summary

- `ultralytics` + `YOLOv8n` + `ByteTrack`: find and track people
- `mediapipe`: extract body skeleton landmarks
- `numpy`: build gait features and compute similarity
- `pickle`: save enrolled gallery data into `.pkl`
- `opencv-python`: read videos, draw overlays, save result videos

## 6. Important Files

- `enroll.py`: enroll a person into the gallery
- `recognize.py`: recognize people from a query video
- `src/detector.py`: person detection and tracking
- `src/pose_extractor.py`: pose landmark extraction
- `src/feature_builder.py`: gait feature generation
- `src/gallery_manager.py`: gallery build/load/save logic
- `src/matcher.py`: identity matching logic
- `gallery/gallery.pkl`: saved enrolled identities and feature vectors