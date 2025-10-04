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

# Global cache for known faces
_known_face_names: List[str] = []
_known_face_encodings = []


class FaceDetector:
    """Handles face detection using dlib."""
    
    def __init__(self):
        self.detector = detector
    
    def detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """Returns list of face rectangles (x, y, w, h)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)  # dlib rectangles
        # Convert dlib rectangles to (x, y, w, h)
        faces_list = [(face.left(), face.top(), face.width(), face.height()) for face in faces]
        return faces_list

    def draw_faces(self, frame, faces: List[Tuple[int, int, int, int]]) -> None:
        """Draw rectangles around detected faces."""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


class FaceRecognizer:
    """Handles face recognition using face_recognition library."""
    
    def __init__(self):
        self.face_recognition = self._import_face_recognition()
        self.known_face_names: List[str] = []
        self.known_face_encodings = []
    
    def _import_face_recognition(self):
        """Import face_recognition library if available."""
        try:
            import face_recognition  # type: ignore
            return face_recognition
        except ImportError:
            logger.warning("face_recognition library not available. Only basic detection will work.")
            return None

    def load_known_faces(self, known_dir: str = "known_faces") -> int:
        """Load known faces from directory structure known_faces/<Name>/*.jpg.
        Returns number of identities loaded.
        """
        if self.face_recognition is None:
            return 0
        
        if not os.path.isdir(known_dir):
            return 0
        
        self.known_face_names.clear()
        self.known_face_encodings.clear()
        total = 0
        
        for name in os.listdir(known_dir):
            person_dir = os.path.join(known_dir, name)
            if not os.path.isdir(person_dir):
                continue
            
            for fname in os.listdir(person_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                
                path = os.path.join(person_dir, fname)
                try:
                    img = self.face_recognition.load_image_file(path)
                    encs = self.face_recognition.face_encodings(img)
                    if encs:
                        self.known_face_names.append(name)
                        self.known_face_encodings.append(encs[0])
                        total += 1
                except Exception as e:
                    logger.warning(f"Failed loading {path}: {e}")
        
        if total > 0:
            logger.info(f"✓ Loaded {total} known face images ({len(set(self.known_face_names))} identities)")
        
        return total

    def recognize_faces(self, frame, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int, str]]:
        """Return list of (x,y,w,h,name) using recognition if available, else name='Face'."""
        results = []
        
        if self.face_recognition is not None and self.known_face_encodings:
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert (x,y,w,h) to face_recognition (top,right,bottom,left)
            boxes = [(y, x+w, y+h, x) for (x,y,w,h) in faces]
            encodings = self.face_recognition.face_encodings(rgb, boxes)
            
            for (x, y, w, h), enc in zip(faces, encodings):
                name = "Unknown"
                try:
                    matches = self.face_recognition.compare_faces(self.known_face_encodings, enc, tolerance=0.5)
                    if True in matches:
                        name = self.known_face_names[matches.index(True)]
                except Exception:
                    pass
                results.append((x, y, w, h, name))
        else:
            for (x, y, w, h) in faces:
                results.append((x, y, w, h, "Face"))
        
        return results

    def draw_recognized(self, frame, recognized: List[Tuple[int, int, int, int, str]]) -> None:
        """Draw recognized faces with names."""
        for (x, y, w, h, name) in recognized:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = name or "Face"
            cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
