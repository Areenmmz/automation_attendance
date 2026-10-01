import cv2


class FaceDetector:
    """
    Stage 3. OpenCV YuNet face detector.
    Each detection is a row of 15 numbers: x, y, w, h, 5 landmark (x, y) pairs, score.
    SFace needs that full row to align the face, so we keep it as-is.
    """

    def __init__(self, model_path, score_threshold=0.85, nms_threshold=0.3, min_face_size=40):
        self.min_face_size = min_face_size
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold, nms_threshold, 5000
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        return [f for f in faces if min(f[2], f[3]) >= self.min_face_size]

    @staticmethod
    def face_inside_persons(face, person_boxes):
        """True if the face centre lies inside any person box."""
        cx = face[0] + face[2] / 2.0
        cy = face[1] + face[3] / 2.0
        for (x1, y1, x2, y2, _) in person_boxes:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False
