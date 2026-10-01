import cv2


class PersonDetector:
    """
    Stage 2. Finds people in the frame and returns their boxes as (x1, y1, x2, y2, conf).
    Uses YOLOv8n through ultralytics when available; otherwise OpenCV's HOG detector.
    """

    def __init__(self, backend="yolo", yolo_model="yolov8n.pt", confidence=0.45):
        self.confidence = confidence
        self.backend = backend
        self.model = None

        if backend == "yolo":
            try:
                from ultralytics import YOLO
                self.model = YOLO(yolo_model)
                print("[INFO] Person detector: YOLO")
            except Exception as e:
                print(f"[WARN] YOLO not available ({e}). Falling back to HOG.")
                self.backend = "hog"

        if self.backend == "hog":
            self.model = cv2.HOGDescriptor()
            self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("[INFO] Person detector: HOG")

    def detect(self, frame):
        if self.backend == "yolo":
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(self, frame):
        results = self.model.predict(frame, classes=[0], conf=self.confidence, verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        return boxes

    def _detect_hog(self, frame):
        # HOG is slow at full resolution; run it on a 640px-wide copy
        h, w = frame.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
        rects, weights = self.model.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        boxes = []
        for (x, y, bw, bh), wgt in zip(rects, weights):
            conf = float(wgt)
            if conf < 0.3:
                continue
            boxes.append((int(x / scale), int(y / scale),
                          int((x + bw) / scale), int((y + bh) / scale), conf))
        return boxes
