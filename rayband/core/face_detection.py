"""
Face detection and recognition for RayBand voice camera.
"""

import cv2
import dlib
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Load dlib frontal face detector
detector = dlib.get_frontal_face_detector()


class FaceDetector:
    """Handles face detection using dlib."""
    
    def __init__(self):
        self.detector = detector
    
    def detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """Returns list of face rectangles (x, y, w, h)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        faces_list = [(face.left(), face.top(), face.width(), face.height()) for face in faces]
        return faces_list

    def draw_faces(self, frame, faces: List[Tuple[int, int, int, int]]) -> None:
        """Draw rectangles around detected faces."""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


class FaceRecognizer:
    """Handles face recognition - DISABLED by default."""
    
    def __init__(self):
        self.face_recognition = None
        self.known_face_names: List[str] = []
        self.known_face_encodings = []
    
    def load_known_faces(self, known_dir: str = "known_faces") -> int:
        """Disabled - just return 0."""
        logger.info("ℹ️  Face recognition disabled - using basic detection only")
        return 0

    def recognize_faces(self, frame, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int, str]]:
        """Return basic faces without recognition."""
        return [(x, y, w, h, "Face") for (x, y, w, h) in faces]

    def draw_recognized(self, frame, recognized: List[Tuple[int, int, int, int, str]]) -> None:
        """Draw faces."""
        for (x, y, w, h, name) in recognized:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)