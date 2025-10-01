import cv2
import os
from typing import List, Tuple


_HANDS = None


def _get_hands():
    global _HANDS
    if _HANDS is not None:
        return _HANDS
    try:
        import mediapipe as mp  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "mediapipe is not installed. Install it with: pip install mediapipe"
        ) from e

    mp_hands = mp.solutions.hands
    _HANDS = {
        "hands": mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
        "drawing": mp.solutions.drawing_utils,
        "styles": mp.solutions.drawing_styles,
        "mp_hands": mp_hands,
    }
    return _HANDS


def detect_fingers(frame) -> dict:
    """
    Detect hands/fingers on the given BGR frame.

    Returns a dict with keys:
      - results: MediaPipe Hands results or None
      - landmarks: list of 21 (x,y) pixel coords per hand
      - handedness: list of 'Left'/'Right' labels per hand
    """
    hands_bundle = _get_hands()
    hands = hands_bundle["hands"]
    mp_hands = hands_bundle["mp_hands"]

    # Convert BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    landmarks_px: List[List[Tuple[int, int]]] = []
    handedness: List[str] = []

    if res.multi_hand_landmarks:
        h, w = frame.shape[:2]
        for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
            pts: List[Tuple[int, int]] = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pts.append((x, y))
            landmarks_px.append(pts)

        if res.multi_handedness:
            for hand_label in res.multi_handedness:
                handedness.append(hand_label.classification[0].label)

    return {
        "results": res,
        "landmarks": landmarks_px,
        "handedness": handedness,
    }


def draw_fingers(frame, detection: dict) -> None:
    """Draw hand landmarks and connections on frame in-place."""
    if not detection or not detection.get("results"):
        return
    hands_bundle = _get_hands()
    drawing = hands_bundle["drawing"]
    styles = hands_bundle["styles"]
    mp_hands = hands_bundle["mp_hands"]

    for hand_landmarks in detection["results"].multi_hand_landmarks:
        drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            styles.get_default_hand_landmarks_style(),
            styles.get_default_hand_connections_style(),
        )


