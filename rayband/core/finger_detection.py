"""
Hand and finger detection for RayBand voice camera using MediaPipe.
"""

import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FingerDetector:
    """Handles hand and finger detection using MediaPipe."""
    
    def __init__(self):
        self.mp_hands = None
        self.hands = None
        self.mp_drawing = None
        self._initialize_mediapipe()
    
    def _initialize_mediapipe(self) -> bool:
        """Initialize MediaPipe hands detection."""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            logger.info("✓ MediaPipe hands initialized")
            return True
        except ImportError:
            logger.warning("MediaPipe not available. Hand detection disabled.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return False
    
    def detect_fingers(self, frame) -> Optional[object]:
        """Detect hands and fingers in the frame."""
        if self.hands is None:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            return results
        except Exception as e:
            logger.error(f"Error detecting fingers: {e}")
            return None
    
    def draw_fingers(self, frame, hand_results) -> None:
        """Draw hand landmarks on the frame."""
        if hand_results is None or self.mp_drawing is None:
            return
        
        try:
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
        except Exception as e:
            logger.error(f"Error drawing fingers: {e}")
    
    def is_available(self) -> bool:
        """Check if MediaPipe is available."""
        return self.hands is not None
