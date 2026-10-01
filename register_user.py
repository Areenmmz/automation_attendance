"""
Register a new person: captures face images from the camera into dataset/<name>/.

    python register_user.py --name "Meraj"
    python register_user.py --name "Meraj" --count 30 --auto

Keys:  SPACE = capture one image   A = toggle auto-capture   Q = quit
Only frames with exactly one face are saved. Move your head a little between
shots (left, right, up, down, with/without glasses) so the model sees variety.
You can also skip this script and copy photos into dataset/<name>/ by hand.
"""
import argparse
import os
import time

import cv2

import config
from modules.face_detector import FaceDetector
from modules.utils import download_model, ensure_dirs, open_camera, safe_name

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Person's name (becomes the folder name)")
parser.add_argument("--count", type=int, default=20, help="How many images to capture")
parser.add_argument("--auto", action="store_true", help="Start with auto-capture on")
parser.add_argument("--interval", type=float, default=0.7, help="Seconds between auto captures")
args = parser.parse_args()

person = safe_name(args.name)
person_dir = os.path.join(config.DATASET_DIR, person)
ensure_dirs(person_dir, config.MODELS_DIR)

download_model(config.YUNET_URL, config.YUNET_PATH)
face_detector = FaceDetector(config.YUNET_PATH, config.FACE_SCORE_THRESHOLD,
                             config.FACE_NMS_THRESHOLD, config.MIN_FACE_SIZE)

existing = len([f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))])
saved = 0
auto = args.auto
last_shot = 0.0

cap = open_camera(config.CAMERA_SOURCE, config.FRAME_WIDTH, config.FRAME_HEIGHT)
print(f"[INFO] Saving images to {person_dir} ({existing} already there)")

while saved < args.count:
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Camera read failed")
        break

    faces = face_detector.detect(frame)
    display = frame.copy()
    for f in faces:
        x, y, w, h = map(int, f[:4])
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    one_face = len(faces) == 1
    msg = f"{person}: {saved}/{args.count}  auto={'ON' if auto else 'OFF'}"
    if not one_face:
        msg += "  (need exactly 1 face)"
    cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if one_face else (0, 0, 255), 2)
    cv2.imshow("Register user", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("a"):
        auto = not auto

    take = (key == ord(" ")) or (auto and time.time() - last_shot >= args.interval)
    if take and one_face:
        idx = existing + saved + 1
        path = os.path.join(person_dir, f"{person}_{idx:03d}.jpg")
        cv2.imwrite(path, frame)      # save the full frame; the face is found again when building embeddings
        saved += 1
        last_shot = time.time()
        print(f"[INFO] Saved {path}")

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] {saved} new images for {person}. "
      f"Run 'python build_embeddings.py' (main.py also rebuilds automatically).")
