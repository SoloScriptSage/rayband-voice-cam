"""
Core functionality for RayBand voice camera.

This module contains the main processing components:
- Audio processing and speech recognition
- Camera handling and video processing
- Face detection and recognition
- Hand/finger tracking
"""

from .audio import AudioProcessor, SpeechRecognizer
from .camera import CameraController
from .face_detection import FaceDetector, FaceRecognizer
from .finger_detection import FingerDetector

__all__ = [
    "AudioProcessor", 
    "SpeechRecognizer",
    "CameraController", 
    "FaceDetector", 
    "FaceRecognizer",
    "FingerDetector"
]
