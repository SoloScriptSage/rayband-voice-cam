import cv2
import dlib
import os
from typing import List, Tuple, Optional

# Load dlib frontal face detector
detector = dlib.get_frontal_face_detector()

# Global cache for known faces
_known_face_names: List[str] = []
_known_face_encodings = []


def _import_face_recognition():
    try:
        import face_recognition  # type: ignore
        return face_recognition
    except Exception:
        return None

def detect_faces(frame):
    """Returns list of face rectangles (x, y, w, h)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)  # dlib rectangles
    # Convert dlib rectangles to (x, y, w, h)
    faces_list = [(face.left(), face.top(), face.width(), face.height()) for face in faces]
    return faces_list

def draw_faces(frame, faces):
    """Draw rectangles around detected faces"""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)


def load_known_faces(known_dir: str = None) -> int:
    """Load known faces from directory structure known_faces/<Name>/*.jpg.
    Returns number of identities loaded.
    """
    fr = _import_face_recognition()
    if fr is None:
        # Silent fallback to basic detection
        return 0
    if known_dir is None:
        known_dir = os.path.join(os.path.dirname(__file__), 'known_faces')
    if not os.path.isdir(known_dir):
        # No known faces folder; skip
        return 0
    _known_face_names.clear()
    _known_face_encodings.clear()
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
                img = fr.load_image_file(path)
                encs = fr.face_encodings(img)
                if encs:
                    _known_face_names.append(name)
                    _known_face_encodings.append(encs[0])
                    total += 1
            except Exception as e:
                print(f"⚠️  Failed loading {path}: {e}")
    if total > 0:
        print(f"✓ Loaded {total} known face images ({len(set(_known_face_names))} identities)")
    return total


def recognize_faces(frame, faces: List[Tuple[int,int,int,int]]):
    """Return list of (x,y,w,h,name) using recognition if available, else name='Face'."""
    results = []
    fr = _import_face_recognition()
    if fr is not None and _known_face_encodings:
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert (x,y,w,h) to face_recognition (top,right,bottom,left)
        boxes = [(y, x+w, y+h, x) for (x,y,w,h) in faces]
        encodings = fr.face_encodings(rgb, boxes)
        for (x, y, w, h), enc in zip(faces, encodings):
            name = "Unknown"
            try:
                matches = fr.compare_faces(_known_face_encodings, enc, tolerance=0.5)
                if True in matches:
                    # Pick first match or the best distance
                    name = _known_face_names[matches.index(True)]
            except Exception:
                pass
            results.append((x, y, w, h, name))
    else:
        for (x, y, w, h) in faces:
            results.append((x, y, w, h, "Face"))
    return results


def draw_recognized(frame, recognized):
    for (x, y, w, h, name) in recognized:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label = (name or "Face")
        cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
