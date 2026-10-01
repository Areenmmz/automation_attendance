# Automated Attendance System

Camera → **motion** → **person** → **face detection** → **face recognition** → `database/database.csv`

Each stage only runs if the previous one found something, so an empty room costs almost no CPU.

| Stage | Method | File |
|---|---|---|
| 1. Motion | OpenCV MOG2 background subtraction | `modules/motion_detector.py` |
| 2. Person | YOLOv8n (falls back to OpenCV HOG if ultralytics is missing) | `modules/person_detector.py` |
| 3. Face detection | OpenCV YuNet (only faces inside a person box are kept) | `modules/face_detector.py` |
| 4. Face recognition | OpenCV SFace, cosine similarity against `dataset/` | `modules/face_recognizer.py` |
| 5. Logging | CSV + face snapshot, per-person cooldown | `modules/attendance_logger.py` |

## Folder layout

```
attendance_system/
├── config.py               # every setting (camera, thresholds, cooldown ...)
├── main.py                 # run attendance
├── register_user.py        # capture face images for a new person
├── build_embeddings.py     # dataset images -> models/embeddings.pkl
├── requirements.txt
├── modules/
├── dataset/                # one folder per person: dataset/<Name>/*.jpg
├── models/                 # ONNX models + embeddings.pkl (auto-downloaded/created)
├── database/database.csv   # attendance records
└── snapshots/<date>/       # face crop for each CSV row
```

## Setup

```
pip install -r requirements.txt
```

The YuNet and SFace models download into `models/` on first run. YOLOv8n downloads automatically through ultralytics.

## Use

1. Register people (one at a time):
   ```
   python register_user.py --name "Meraj" --count 20 --auto
   ```
   Or copy photos into `dataset/Meraj/` yourself.

2. Run attendance:
   ```
   python main.py
   python main.py --source 1
   python main.py --source "http://<camera-ip>/axis-cgi/mjpg/video.cgi"
   python main.py --source test_video.mp4
   ```
   Press **Q** to stop. Embeddings are rebuilt automatically whenever `dataset/` changes.

## CSV columns

`Date, Time, Frame_Number, Name, Similarity, Camera, Snapshot`

`Frame_Number` counts from the start of the current run. A person is logged once per `LOG_COOLDOWN_SECONDS` (default 60); set it to 0 to log every frame.

## Tuning

- Wrong names → raise `MATCH_THRESHOLD` (0.45–0.50).
- Known people shown as Unknown → lower it (0.36) or add more varied photos.
- Motion triggered by noise/lighting → raise `MOTION_MIN_AREA`.
- Slow CPU → `PERSON_EVERY_N_FRAMES = 2` or `3`, or lower `FRAME_WIDTH/HEIGHT`.
