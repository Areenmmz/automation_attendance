import csv
import os
import time
from datetime import datetime

import cv2

COLUMNS = ["Date", "Time", "Frame_Number", "Name", "Similarity", "Camera", "Snapshot"]


class AttendanceLogger:
    """
    Stage 5. Appends rows to database/database.csv.
    A per-person cooldown stops the same person being written on every frame.
    """

    def __init__(self, csv_path, snapshots_dir, cooldown_seconds=60,
                 log_unknown=False, save_snapshots=True, camera_name="CAM_1"):
        self.csv_path = csv_path
        self.snapshots_dir = snapshots_dir
        self.cooldown = cooldown_seconds
        self.log_unknown = log_unknown
        self.save_snapshots = save_snapshots
        self.camera_name = camera_name
        self.last_logged = {}          # name -> unix time of last row

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(snapshots_dir, exist_ok=True)
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(COLUMNS)

    def should_log(self, name):
        if name == "Unknown" and not self.log_unknown:
            return False
        last = self.last_logged.get(name)
        return last is None or (time.time() - last) >= self.cooldown

    def log(self, name, similarity, frame_number, frame=None, face_box=None):
        """Write one row. Returns True if a row was written."""
        if not self.should_log(name):
            return False

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        snapshot_path = ""
        if self.save_snapshots and frame is not None and face_box is not None:
            x, y, w, h = [int(v) for v in face_box]
            pad = int(0.25 * max(w, h))
            H, W = frame.shape[:2]
            crop = frame[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
            if crop.size > 0:
                day_dir = os.path.join(self.snapshots_dir, date_str)
                os.makedirs(day_dir, exist_ok=True)
                fname = f"{now.strftime('%H-%M-%S')}_f{frame_number}_{name}.jpg"
                snapshot_path = os.path.join(day_dir, fname)
                cv2.imwrite(snapshot_path, crop)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([date_str, time_str, frame_number, name,
                                    f"{similarity:.3f}", self.camera_name,
                                    os.path.relpath(snapshot_path, os.path.dirname(self.snapshots_dir)) if snapshot_path else ""])

        self.last_logged[name] = time.time()
        print(f"[LOG] {date_str} {time_str} | frame {frame_number} | {name} ({similarity:.2f})")
        return True
