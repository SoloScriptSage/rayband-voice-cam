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
        self.mp_drawing_styles = None
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
            self.mp_drawing_styles = mp.solutions.drawing_styles
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
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Get hand label (Left/Right)
                    if hand_results.multi_handedness:
                        handedness = hand_results.multi_handedness[hand_idx]
                        hand_label = handedness.classification[0].label
                        
                        # Draw label near wrist
                        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        h, w, _ = frame.shape
                        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                        
                        cv2.putText(
                            frame, 
                            hand_label, 
                            (wrist_x - 30, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2, 
                            cv2.LINE_AA
                        )
                    
        except Exception as e:
            logger.error(f"Error drawing fingers: {e}")
    
    def count_raised_fingers(self, hand_results) -> dict:
        """Count raised fingers on each hand."""
        if not hand_results or not hand_results.multi_hand_landmarks:
            return {"left": 0, "right": 0}
        
        finger_count = {"left": 0, "right": 0}
        
        try:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Get hand label
                handedness = hand_results.multi_handedness[hand_idx]
                hand_label = handedness.classification[0].label.lower()
                
                landmarks = hand_landmarks.landmark
                
                # Thumb: check x-coordinate
                thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
                if hand_label == "right":
                    if thumb_tip.x < thumb_ip.x:
                        finger_count[hand_label] += 1
                else:
                    if thumb_tip.x > thumb_ip.x:
                        finger_count[hand_label] += 1
                
                # Other fingers: check if tip is above PIP joint
                finger_tips = [
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    self.mp_hands.HandLandmark.RING_FINGER_TIP,
                    self.mp_hands.HandLandmark.PINKY_TIP,
                ]
                finger_pips = [
                    self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                    self.mp_hands.HandLandmark.RING_FINGER_PIP,
                    self.mp_hands.HandLandmark.PINKY_PIP,
                ]
                
                for tip, pip in zip(finger_tips, finger_pips):
                    if landmarks[tip].y < landmarks[pip].y:
                        finger_count[hand_label] += 1
                        
        except Exception as e:
            logger.error(f"Error counting fingers: {e}")
        
        return finger_count
    
    def is_available(self) -> bool:
        """Check if MediaPipe is available."""
        return self.hands is not None
    
    def close(self):
        """Clean up MediaPipe resources."""
        if self.hands:
            self.hands.close()