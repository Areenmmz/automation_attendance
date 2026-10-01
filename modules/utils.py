import os
import sys
import urllib.request

import cv2


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def download_model(url, path):
    """Download an ONNX model once; skip if it already exists."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[INFO] Downloading {os.path.basename(path)} ...")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    print(f"[INFO] Saved to {path}")
    return path


def open_camera(source, width=None, height=None):
    """Open a USB camera index or a stream URL. Uses DirectShow on Windows for indexes."""
    if isinstance(source, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {source}")
    return cap


def safe_name(name):
    """Turn a person's name into a folder-safe string."""
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in name.strip())
    return keep.replace(" ", "_")
