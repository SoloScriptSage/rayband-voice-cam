"""
File management utilities for RayBand voice camera.
"""

import os
import time
import wave
import shutil
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for captures, videos, and audio."""
    
    def __init__(self, captures_dir: str = "captures", videos_dir: str = "videos"):
        self.captures_dir = captures_dir
        self.videos_dir = videos_dir
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure capture and video directories exist."""
        for directory in [self.captures_dir, self.videos_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def save_snapshot(self, frame, directory: Optional[str] = None) -> str:
        """Save a camera frame as a JPEG snapshot."""
        if directory is None:
            directory = self.captures_dir
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        path = os.path.join(directory, filename)
        
        import cv2
        cv2.imwrite(path, frame)
        logger.info(f"📸 Saved snapshot: {path}")
        return path
    
    def get_video_path(self, directory: Optional[str] = None) -> tuple[str, str]:
        """Get paths for video and audio files."""
        if directory is None:
            directory = self.videos_dir
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"video_{timestamp}.mp4"
        audio_filename = f"video_{timestamp}.wav"
        
        video_path = os.path.join(directory, video_filename)
        audio_path = os.path.join(directory, audio_filename)
        
        return video_path, audio_path
    
    def try_mux_audio(self, video_path: str, audio_wav_path: str) -> str:
        """Try to mux audio and video using ffmpeg."""
        # Check ffmpeg availability
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            logger.info("ℹ️  ffmpeg not found in PATH; leaving WAV alongside video.")
            return ""
        
        # Create output with same name replacing extension
        base, _ = os.path.splitext(video_path)
        output_path = base + "_with_audio.mp4"
        
        try:
            cmd = [
                ffmpeg, '-y',
                '-i', video_path,
                '-i', audio_wav_path,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"❌ ffmpeg mux failed:\n {result.stderr}")
                return ""
            logger.info(f"✅ Muxed audio+video → {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ Error running ffmpeg: {e}")
            return ""
