"""
All settings for the attendance system live here.
Edit this file instead of touching the code in modules/.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Folders and files
# ----------------------------------------------------------------------------
DATASET_DIR = os.path.join(BASE_DIR, "dataset")          # dataset/<person_name>/*.jpg
MODELS_DIR = os.path.join(BASE_DIR, "models")            # ONNX models + embeddings
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_CSV = os.path.join(DATABASE_DIR, "database.csv")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")      # face crops of each logged event
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.pkl")

# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
# 0, 1, 2 ... for USB/laptop cameras, or a string URL for an IP camera, e.g.
# "http://192.168.1.50/axis-cgi/mjpg/video.cgi" or "rtsp://user:pass@ip:554/stream"
CAMERA_SOURCE = 0
CAMERA_NAME = "CAM_1"          # written to the CSV so you know which camera saw the person
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SHOW_WINDOW = True             # False = run headless (no preview window)

# ----------------------------------------------------------------------------
# Stage 1: Motion detection (background subtraction)
# ----------------------------------------------------------------------------
MOTION_RESIZE_WIDTH = 320      # motion is checked on a small copy of the frame (fast)
MOTION_MIN_AREA = 500          # min contour area (pixels at the resized scale) to count as motion
MOTION_HOLD_FRAMES = 30        # keep running the pipeline this many frames after motion stops
                               # (a person standing still in front of the camera stops "moving")

# ----------------------------------------------------------------------------
# Stage 2: Person detection
# ----------------------------------------------------------------------------
# "yolo" uses ultralytics YOLOv8n (better). If ultralytics/torch is not installed
# or blocked, the system falls back to "hog" automatically (OpenCV only, weaker).
PERSON_DETECTOR = "yolo"
YOLO_MODEL = "yolov8n.pt"      # downloaded automatically by ultralytics on first run
PERSON_CONFIDENCE = 0.45
PERSON_EVERY_N_FRAMES = 1      # set 2 or 3 on a slow CPU to skip frames for this stage

# ----------------------------------------------------------------------------
# Stage 3: Face detection (OpenCV YuNet)
# ----------------------------------------------------------------------------
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
FACE_SCORE_THRESHOLD = 0.85
FACE_NMS_THRESHOLD = 0.3
MIN_FACE_SIZE = 40             # ignore faces smaller than this (pixels); too small to recognise

# ----------------------------------------------------------------------------
# Stage 4: Face recognition (OpenCV SFace)
# ----------------------------------------------------------------------------
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
MATCH_THRESHOLD = 0.40         # cosine similarity; SFace's reference value is 0.363.
                               # Raise it (0.45-0.5) if you get wrong names, lower it if
                               # known people show as Unknown.
USE_FLIP_AUGMENTATION = True   # also store embeddings of horizontally flipped training images

# ----------------------------------------------------------------------------
# Stage 5: Logging to the CSV "database"
# ----------------------------------------------------------------------------
LOG_COOLDOWN_SECONDS = 60      # same person is logged at most once per this many seconds
                               # (0 = log every frame the person is recognised)
LOG_UNKNOWN = False            # True = also write rows for unrecognised faces
SAVE_SNAPSHOTS = True          # save a face crop for every row written to the CSV
