"""
Automated attendance.

Pipeline for every frame:
    1. Motion?           no  -> skip the rest
    2. Person?           no  -> skip the rest
    3. Face (inside a person box)?  no -> skip the rest
    4. Recognise the face (who is it?)
    5. Save Date, Time, Frame_Number, Name ... to database/database.csv

    python main.py                 # uses CAMERA_SOURCE from config.py
    python main.py --source 1      # another USB camera
    python main.py --source "http://192.168.1.50/axis-cgi/mjpg/video.cgi"
    python main.py --source video.mp4

Press Q in the preview window to stop.
"""
import argparse
import os
import subprocess
import sys
import time

import cv2

import config
from modules.attendance_logger import AttendanceLogger
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.motion_detector import MotionDetector
from modules.person_detector import PersonDetector
from modules.utils import download_model, ensure_dirs, open_camera

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--source", default=None, help="Camera index, stream URL or video file")
parser.add_argument("--no-window", action="store_true", help="Run without the preview window")
args = parser.parse_args()

source = config.CAMERA_SOURCE
if args.source is not None:
    source = int(args.source) if args.source.isdigit() else args.source
show_window = config.SHOW_WINDOW and not args.no_window

# ---------------------------------------------------------------------------
# Setup: folders, models, embeddings
# ---------------------------------------------------------------------------
ensure_dirs(config.DATASET_DIR, config.MODELS_DIR, config.DATABASE_DIR, config.SNAPSHOTS_DIR)
download_model(config.YUNET_URL, config.YUNET_PATH)
download_model(config.SFACE_URL, config.SFACE_PATH)


def latest_mtime(folder):
    newest = os.path.getmtime(folder)
    for root, dirs, files in os.walk(folder):
        for name in dirs + files:
            newest = max(newest, os.path.getmtime(os.path.join(root, name)))
    return newest


needs_build = (not os.path.exists(config.EMBEDDINGS_FILE)
               or latest_mtime(config.DATASET_DIR) > os.path.getmtime(config.EMBEDDINGS_FILE))
if needs_build:
    print("[INFO] Dataset changed or no embeddings yet -> building embeddings")
    subprocess.run([sys.executable, os.path.join(config.BASE_DIR, "build_embeddings.py")], check=True)

motion_detector = MotionDetector(config.MOTION_RESIZE_WIDTH, config.MOTION_MIN_AREA,
                                 config.MOTION_HOLD_FRAMES)
person_detector = PersonDetector(config.PERSON_DETECTOR, config.YOLO_MODEL,
                                 config.PERSON_CONFIDENCE)
face_detector = FaceDetector(config.YUNET_PATH, config.FACE_SCORE_THRESHOLD,
                             config.FACE_NMS_THRESHOLD, config.MIN_FACE_SIZE)
recognizer = FaceRecognizer(config.SFACE_PATH, config.MATCH_THRESHOLD)
recognizer.load(config.EMBEDDINGS_FILE)
logger = AttendanceLogger(config.DATABASE_CSV, config.SNAPSHOTS_DIR,
                          config.LOG_COOLDOWN_SECONDS, config.LOG_UNKNOWN,
                          config.SAVE_SNAPSHOTS, config.CAMERA_NAME)

if len(recognizer.names) == 0:
    print("[WARN] No registered people. Everyone will be 'Unknown'. "
          "Add images to dataset/<name>/ or run register_user.py")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
cap = open_camera(source, config.FRAME_WIDTH, config.FRAME_HEIGHT)
frame_number = 0
person_boxes = []
fps, t_prev = 0.0, time.time()
print("[INFO] Running. Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[INFO] No more frames / camera disconnected")
        break
    frame_number += 1
    display = frame.copy() if show_window else None
    status = "No motion"
    results = []          # (face_row, name, similarity) for drawing

    # 1. Motion
    if motion_detector.detect(frame):
        status = "Motion - no person"

        # 2. Person
        if frame_number % config.PERSON_EVERY_N_FRAMES == 0 or not person_boxes:
            person_boxes = person_detector.detect(frame)

        if person_boxes:
            status = "Person - no face"

            # 3. Face (only faces that belong to a detected person)
            faces = [f for f in face_detector.detect(frame)
                     if FaceDetector.face_inside_persons(f, person_boxes)]

            if faces:
                status = f"{len(faces)} face(s)"

                # 4. Recognition  +  5. Logging
                for face in faces:
                    name, sim = recognizer.identify(frame, face)
                    logger.log(name, sim, frame_number, frame, face[:4])
                    results.append((face, name, sim))
    else:
        person_boxes = []

    # -----------------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------------
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
    t_prev = now

    if show_window:
        for (x1, y1, x2, y2, _) in person_boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 160, 0), 2)
        for face, name, sim in results:
            x, y, w, h = map(int, face[:4])
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, f"{name} {sim:.2f}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, f"Frame {frame_number} | {status} | {fps:.1f} FPS",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Attendance", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Attendance saved in {config.DATABASE_CSV}")
