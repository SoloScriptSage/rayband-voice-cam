"""
Configuration management for RayBand voice camera.
"""

import os
from typing import Optional


class Config:
    """Configuration settings for the RayBand voice camera."""
    
    def __init__(self):
        # Paths
        self.MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "./model")
        
        # Audio settings
        self.AUDIO_DEVICE_ID = 1
        self.MIC_SAMPLERATE = 44100
        self.VOSK_SAMPLERATE = 16000
        self.BLOCKSIZE = 11025
        
        # Camera settings  
        self.CAMERA_BACKEND = os.getenv("RAYCAM_BACKEND", "DSHOW")
        self.CAMERA_INDEX = int(os.getenv("RAYCAM_INDEX", "0"))
        
        # Command cooldowns (seconds)
        self.PICTURE_COOLDOWN = 2.0
        self.RECORDING_COOLDOWN = 2.0
        
        # File paths
        self.CAPTURES_DIR = "captures"
        self.VIDEOS_DIR = "videos"
        self.KNOWN_FACES_DIR = "known_faces"
    
    def get_model_path(self) -> str:
        """Get the path to the Vosk model directory."""
        return self.MODEL_PATH
    
    def get_audio_device_id(self) -> int:
        """Get the audio device ID."""
        return self.AUDIO_DEVICE_ID
    
    def get_camera_settings(self) -> tuple:
        """Get camera backend and index settings."""
        return self.CAMERA_BACKEND, self.CAMERA_INDEX


# Global config instance
config = Config()
