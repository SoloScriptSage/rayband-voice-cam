"""
Face detection and recognition for RayBand voice camera.
Uses OpenCV Haar Cascade (no dlib required).
"""

import cv2
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Try to import dlib, but use OpenCV as fallback
try:
    import dlib
    DLIB_AVAILABLE = True
    detector = dlib.get_frontal_face_detector()
    logger.info("✓ Using dlib for face detection")
except ImportError:
    DLIB_AVAILABLE = False
    detector = None
    logger.info("✓ Using OpenCV Haar Cascade for face detection (dlib not available)")


class FaceDetector:
    """Handles face detection using dlib or OpenCV."""
    
    def __init__(self):
        if DLIB_AVAILABLE:
            self.detector = detector
            self.use_dlib = True
        else:
            # Use OpenCV's Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.use_dlib = False
            
            if self.detector.empty():
                logger.error("Failed to load Haar Cascade classifier")
                self.detector = None
    
    def detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """Returns list of face rectangles (x, y, w, h)."""
        if self.detector is None:
            return []
        
        try:
            if self.use_dlib:
                # dlib detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                faces_list = [(face.left(), face.top(), face.width(), face.height()) for face in faces]
                return faces_list
            else:
                # OpenCV Haar Cascade detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                # Convert to (x, y, w, h) tuples
                return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

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