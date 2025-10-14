"""
Configuration management for RayBand voice camera.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration settings for the RayBand voice camera."""
    
    def __init__(self):

        self._project_root = Path(__file__).parent.parent.parent.resolve()

        # Language models
        models_dir = self._project_root / "models"
        self.MODELS = {
            "english": str(models_dir / "vosk-model-en-us-0.22"),
            "ukrainian": str(models_dir / "vosk-model-uk-v3"),
        }

        # Default language
        self.current_language = "english"
        self.MODEL_PATH = self.MODELS[self.current_language]
        
        if not os.path.exists(self.MODEL_PATH):
            print(f"⚠️  WARNING: Model not found at {self.MODEL_PATH}")
            print(f"📁 Project root: {self._project_root}")
            print(f"📁 Models dir: {models_dir}")
            print(f"📁 Looking for: {os.listdir(models_dir) if models_dir.exists() else 'DIR NOT FOUND'}")
            
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
        self.LANGUAGE_SWITCH_COOLDOWN = 3.0
        
        # File paths
        self.CAPTURES_DIR = "captures"
        self.VIDEOS_DIR = "videos"
        self.KNOWN_FACES_DIR = "known_faces"
    
    def switch_language(self, language: str) -> bool:
        """Switch to a different lanuage model."""
        if language in self.MODELS:
            self.current_language = language
            self.MODEL_PATH = self.MODELS[language]
            return True
        return False
    
    def get_available_languages(self) -> list:
        """Get list of available languages."""
        return list(self.MODELS.keys())
    
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
