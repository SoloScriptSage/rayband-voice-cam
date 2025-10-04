"""
Audio processing and speech recognition for RayBand voice camera.
"""

import sounddevice as sd
import queue
import vosk
import json
import logging
import numpy as np
import threading
from scipy import signal
from typing import Optional

from ..utils.config import config

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio input processing and speech recognition."""
    
    def __init__(self):
        self.q = queue.Queue()
        self._audio_lock = threading.Lock()
        self.audio_state = {
            "text": "Say something...",
            "is_listening": False
        }
        self.recognizer = None
        self.model = None

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.info(f"Audio status: {status}")
        self.q.put(indata.copy())

    def get_last_text(self) -> str:
        """Get the current recognized text."""
        with self._audio_lock:
            return self.audio_state["text"]

    def start_audio_recognition(self, model_path: Optional[str] = None) -> bool:
        """Start audio recognition with Vosk model."""
        if model_path is None:
            model_path = config.get_model_path()
        
        # Verify device
        try:
            device_info = sd.query_devices(config.get_audio_device_id(), 'input')
            logger.info(f"✓ Using device {config.get_audio_device_id()}: {device_info['name']}")
            logger.info(f"  Sample rate: {config.MIC_SAMPLERATE} Hz → {config.VOSK_SAMPLERATE} Hz (resampled)")
        except Exception as e:
            logger.error(f"✗ Device error: {e}", exc_info=True)
            return False

        # Load Vosk model
        logger.info(f"📂 Loading Vosk model...")
        try:
            self.model = vosk.Model(model_path)
            logger.info("✓ Model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}", exc_info=True)
            return False

        self.recognizer = vosk.KaldiRecognizer(self.model, config.VOSK_SAMPLERATE)
        self.recognizer.SetWords(True)

        # Calculate resampling ratio
        resample_ratio = config.VOSK_SAMPLERATE / config.MIC_SAMPLERATE

        # Start audio stream
        logger.info("🎙️  Starting audio recognition...")
        try:
            with sd.InputStream(
                samplerate=config.MIC_SAMPLERATE,
                blocksize=config.BLOCKSIZE,
                dtype='float32',
                channels=1,
                device=config.get_audio_device_id(),
                callback=self.audio_callback
            ):
                logger.info("✅ Audio recognition active! Speak now...")
                with self._audio_lock:
                    self.audio_state["is_listening"] = True
                
                while True:
                    data = self.q.get()
                    
                    # Resample audio from MIC_SAMPLERATE to VOSK_SAMPLERATE
                    audio_data = data[:, 0]
                    
                    num_output_samples = int(len(audio_data) * resample_ratio)
                    resampled = signal.resample(audio_data, num_output_samples)
                    
                    # Scale properly to int16 range
                    resampled_scaled = np.clip(resampled * 32767, -32768, 32767)
                    resampled_int16 = np.int16(resampled_scaled)
                    
                    # Process with Vosk
                    if self.recognizer.AcceptWaveform(resampled_int16.tobytes()):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "")
                        if text:
                            with self._audio_lock:
                                self.audio_state["text"] = text
                            logger.info(f"🗣️  Recognized: {text}")
                    else:
                        # Partial results
                        partial = json.loads(self.recognizer.PartialResult())
                        partial_text = partial.get("partial", "")
                        if partial_text:
                            with self._audio_lock:
                                self.audio_state["text"] = partial_text
                            
        except Exception as e:
            logger.error(f"✗ Audio stream error: {e}", exc_info=True)
            with self._audio_lock:
                self.audio_state["is_listening"] = False
            import traceback
            traceback.print_exc()
        
        return True


class SpeechRecognizer:
    """High-level speech recognition interface."""
    
    def __init__(self):
        self.processor = AudioProcessor()
        self.thread = None
    
    def start(self, model_path: Optional[str] = None) -> None:
        """Start speech recognition in a background thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(
                target=self.processor.start_audio_recognition, 
                args=(model_path,), 
                daemon=True
            )
            self.thread.start()
    
    def get_text(self) -> str:
        """Get the current recognized text."""
        return self.processor.get_last_text()
    
    def is_listening(self) -> bool:
        """Check if speech recognition is active."""
        return self.processor.audio_state["is_listening"]
