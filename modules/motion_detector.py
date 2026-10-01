import cv2


class MotionDetector:
    """
    Stage 1. Background subtraction (MOG2) on a small grayscale copy of the frame.
    Returns True when moving regions bigger than MOTION_MIN_AREA are found.
    A short "hold" keeps the pipeline active for a while after motion stops,
    so a person who stands still in front of the camera is still recognised.
    """

    def __init__(self, resize_width=320, min_area=500, hold_frames=30):
        self.resize_width = resize_width
        self.min_area = min_area
        self.hold_frames = hold_frames
        self.hold_counter = 0
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = self.resize_width / float(w)
        small = cv2.resize(frame, (self.resize_width, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        mask = self.subtractor.apply(gray)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moving = any(cv2.contourArea(c) >= self.min_area for c in contours)

        if moving:
            self.hold_counter = self.hold_frames
            return True
        if self.hold_counter > 0:
            self.hold_counter -= 1
            return True
        return False
