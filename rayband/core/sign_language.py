"""
Sign Language Detection and Interpretation for RayBand.
Detects ASL gestures and translates them to text/speech.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from collections import deque
import time

logger = logging.getLogger(__name__)


class SignLanguageDetector:
    """Detects and interprets sign language gestures."""
    
    def __init__(self):
        self.current_sign = None
        self.sign_confidence = 0.0
        self.sign_history = deque(maxlen=10)  # Track last 10 detections
        self.last_spoken_sign = None
        self.last_spoken_time = 0
        self.speak_cooldown = 2.0  # Seconds between speaking same sign
        
        # Sign language database
        self.signs = {
            "thumbs_up": "👍 Good / Yes / Approve",
            "peace": "✌️ Peace / Victory / Two",
            "ok": "👌 OK / Perfect",
            "fist": "✊ Stop / Solidarity",
            "open_hand": "🖐️ Hello / Five / Stop",
            "pointing": "☝️ One / Attention",
            "love": "🤟 I Love You",
            "call_me": "🤙 Call Me / Hang Loose",
            "rock": "🤘 Rock On",
            "three": "🤚 Three",
            "four": "🖖 Four / Live Long and Prosper",
        }
        
        logger.info("✓ Sign Language Detector initialized")
    
    def detect_sign(self, hand_results) -> Tuple[Optional[str], float]:
        """
        Detect what sign is being shown.
        
        Returns:
            (sign_name, confidence) or (None, 0.0)
        """
        if not hand_results or not hand_results.multi_hand_landmarks:
            return None, 0.0
        
        try:
            # Get first hand (for now)
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            
            # Get finger states
            fingers = self._get_finger_states(landmarks, hand_results, 0)
            
            # Match to known signs
            sign, confidence = self._match_sign(fingers, landmarks)
            
            # Update history
            if sign:
                self.sign_history.append(sign)
                # Get most common sign in history (reduces flicker)
                if len(self.sign_history) >= 5:
                    from collections import Counter
                    most_common = Counter(self.sign_history).most_common(1)[0][0]
                    self.current_sign = most_common
                    self.sign_confidence = confidence
                else:
                    self.current_sign = sign
                    self.sign_confidence = confidence
            
            return self.current_sign, self.sign_confidence
            
        except Exception as e:
            logger.error(f"Error detecting sign: {e}")
            return None, 0.0
    
    def _get_finger_states(self, landmarks, hand_results, hand_idx) -> dict:
        """
        Determine which fingers are extended.
        
        Returns:
            {
                "thumb": bool,
                "index": bool, 
                "middle": bool,
                "ring": bool,
                "pinky": bool
            }
        """
        # Get hand label (left/right)
        handedness = hand_results.multi_handedness[hand_idx]
        hand_label = handedness.classification[0].label.lower()
        
        fingers = {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False
        }
        
        # Thumb (special case - check x-coordinate)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        if hand_label == "right":
            fingers["thumb"] = thumb_tip.x < thumb_ip.x
        else:
            fingers["thumb"] = thumb_tip.x > thumb_ip.x
        
        # Other fingers (check if tip is above PIP)
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]
        finger_names = ["index", "middle", "ring", "pinky"]
        
        for tip_idx, pip_idx, name in zip(finger_tips, finger_pips, finger_names):
            fingers[name] = landmarks[tip_idx].y < landmarks[pip_idx].y
        
        return fingers
    
    def _match_sign(self, fingers: dict, landmarks) -> Tuple[Optional[str], float]:
        """Match finger pattern to known signs."""
        
        # Count extended fingers
        extended_count = sum(fingers.values())
        
        # Get hand orientation/position features
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        index_tip = landmarks[8]
        
        # Hand angle (is it upright, sideways, etc.)
        hand_vertical = abs(middle_mcp.y - wrist.y) > abs(middle_mcp.x - wrist.x)
        
        # Pattern matching
        confidence = 0.8  # Base confidence
        
        # Thumbs Up 👍
        if fingers["thumb"] and extended_count == 1:
            return "thumbs_up", confidence
        
        # Peace / Victory ✌️
        if (fingers["index"] and fingers["middle"] and 
            not fingers["ring"] and not fingers["pinky"] and
            extended_count == 2):
            return "peace", confidence
        
        # OK Sign 👌 (thumb and index forming circle)
        if (fingers["thumb"] and not fingers["index"] and
            fingers["middle"] and fingers["ring"] and fingers["pinky"]):
            # Check if thumb and index are close
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + 
                              (thumb_tip.y - index_tip.y)**2)
            if distance < 0.05:
                return "ok", 0.9
        
        # Fist ✊
        if extended_count == 0:
            return "fist", confidence
        
        # Open Hand 🖐️
        if extended_count == 5:
            return "open_hand", confidence
        
        # Pointing ☝️
        if fingers["index"] and extended_count == 1 and not fingers["thumb"]:
            return "pointing", confidence
        
        # I Love You 🤟 (thumb, index, pinky)
        if (fingers["thumb"] and fingers["index"] and fingers["pinky"] and
            not fingers["middle"] and not fingers["ring"]):
            return "love", confidence
        
        # Call Me 🤙 (thumb and pinky)
        if (fingers["thumb"] and fingers["pinky"] and
            not fingers["index"] and not fingers["middle"] and not fingers["ring"]):
            return "call_me", confidence
        
        # Rock On 🤘 (index and pinky)
        if (fingers["index"] and fingers["pinky"] and
            not fingers["thumb"] and not fingers["middle"] and not fingers["ring"]):
            return "rock", confidence
        
        # Three 🤚
        if extended_count == 3 and not fingers["thumb"] and not fingers["pinky"]:
            return "three", confidence
        
        # Four 🖖
        if extended_count == 4 and not fingers["thumb"]:
            return "four", confidence
        
        return None, 0.0
    
    def draw_sign_info(self, frame, x: int = 10, y: int = 100):
        """Draw detected sign information on frame."""
        if not self.current_sign:
            # Show "waiting for sign" message
            cv2.putText(frame, "Show a sign...", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            return
        
        # Get sign description
        description = self.signs.get(self.current_sign, "Unknown")
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 35), (x + 400, y + 15), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw sign name and description
        text = f"Sign: {description}"
        cv2.putText(frame, text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw confidence bar
        bar_width = int(200 * self.sign_confidence)
        cv2.rectangle(frame, (x, y + 10), (x + 200, y + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y + 10), (x + bar_width, y + 20), (0, 255, 0), -1)
    
    def should_speak_sign(self) -> bool:
        """Check if we should speak the current sign (cooldown)."""
        if not self.current_sign:
            return False
        
        current_time = time.time()
        
        # Different sign or enough time passed
        if (self.current_sign != self.last_spoken_sign or 
            current_time - self.last_spoken_time > self.speak_cooldown):
            return True
        
        return False
    
    def mark_spoken(self):
        """Mark that we've spoken the current sign."""
        self.last_spoken_sign = self.current_sign
        self.last_spoken_time = time.time()
    
    def get_sign_description(self, sign_name: str) -> str:
        """Get human-readable description of a sign."""
        return self.signs.get(sign_name, "Unknown sign")
    
    def get_all_signs(self) -> dict:
        """Get dictionary of all known signs."""
        return self.signs.copy()


class SignLanguageHelper:
    """Helper utilities for sign language features."""
    
    @staticmethod
    def text_to_speech(text: str):
        """Convert text to speech (uses system TTS)."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.warning(f"Text-to-speech not available: {e}")
    
    @staticmethod
    def draw_help_menu(frame, signs: dict):
        """Draw help menu showing all available signs."""
        y_offset = 50
        cv2.putText(frame, "Available Signs:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        for sign_name, description in list(signs.items())[:5]:  # Show first 5
            cv2.putText(frame, description, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25