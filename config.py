"""
Configuration management with environment variable support
"""
import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    model_path: Path
    captures_dir: Path
    videos_dir: Path
    cache_file: Path
    
    # Audio settings
    mic_device_id: int
    mic_sample_rate: int
    vosk_sample_rate: int
    audio_block_size: int
    
    # Camera settings
    camera_init_timeout: int
    camera_buffer_size: int
    
    # Command settings
    picture_cooldown: float
    recording_cooldown: float
    
    # UI settings
    window_name: str
    font_scale: float
    transcript_lines: int
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables with sensible defaults"""
        project_root = Path(__file__).parent.parent
        
        return cls(
            # Paths - use env vars or defaults
            model_path=Path(os.getenv('VOSK_MODEL_PATH', project_root / 'model')),
            captures_dir=Path(os.getenv('CAPTURES_DIR', project_root / 'captures')),
            videos_dir=Path(os.getenv('VIDEOS_DIR', project_root / 'videos')),
            cache_file=project_root / '.camera_cache.json',
            
            # Audio - tuned for responsiveness
            mic_device_id=int(os.getenv('MIC_DEVICE_ID', '1')),
            mic_sample_rate=44100,
            vosk_sample_rate=16000,
            audio_block_size=8000,  # ~0.18s latency at 44100Hz
            
            # Camera
            camera_init_timeout=5,
            camera_buffer_size=1,
            
            # Commands
            picture_cooldown=2.0,
            recording_cooldown=2.0,
            
            # UI
            window_name="RayBand Voice Cam",
            font_scale=0.7,
            transcript_lines=3,
        )