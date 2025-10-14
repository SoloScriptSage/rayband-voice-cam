"""
Audio processing and speech recognition for RayBand voice camera.
Fixed to properly reload models on language switch.
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
    """Handles audio input and processing."""
    
    def __init__(self):
        self.q = queue.Queue()
        self._audio_lock = threading.Lock()
        self.audio_state = {
            "text": "Say something...",
            "is_listening": False
        }
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.debug(f"Audio status: {status}")
        self.q.put(indata.copy())
    
    def get_last_text(self) -> str:
        """Get the current recognized text."""
        with self._audio_lock:
            return self.audio_state["text"]


class SpeechRecognizer:
    """Handles speech recognition using Vosk."""
    
    def __init__(self, model_path: str, audio_processor: AudioProcessor):
        self.model_path = model_path
        self.audio_processor = audio_processor
        self.model: Optional[vosk.Model] = None
        self.recognizer: Optional[vosk.KaldiRecognizer] = None
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._should_reload = False
        self._reload_lock = threading.Lock()
        self._stream = None
    
    def reload_model(self, new_model_path: str) -> bool:
        """Request model reload with new path."""
        with self._reload_lock:
            self.model_path = new_model_path
            self._should_reload = True
        logger.info(f"🔄 Language switch requested: {new_model_path}")
        return True
    
    def start(self) -> bool:
        """Start speech recognition in background thread."""
        self._thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._thread.start()
        return True
    
    def _load_model(self):
        """Load or reload the Vosk model."""
        logger.info(f"📂 Loading Vosk model from {self.model_path}...")
        try:
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, config.VOSK_SAMPLERATE)
            self.recognizer.SetWords(True)
            logger.info("✓ Model loaded")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}", exc_info=True)
            return False
    
    def _recognition_loop(self):
        """Main recognition loop running in separate thread."""
        # Verify device
        try:
            device_info = sd.query_devices(config.AUDIO_DEVICE_ID, 'input')
            logger.info(f"✓ Using device {config.AUDIO_DEVICE_ID}: {device_info['name']}")
            logger.info(f"  Sample rate: {config.MIC_SAMPLERATE} Hz → {config.VOSK_SAMPLERATE} Hz (resampled)")
        except Exception as e:
            logger.error(f"✗ Device error: {e}", exc_info=True)
            return
        
        # Load initial model
        if not self._load_model():
            return
        
        resample_ratio = config.VOSK_SAMPLERATE / config.MIC_SAMPLERATE
        
        logger.info("🎙️  Starting audio recognition...")
        try:
            self._stream = sd.InputStream(
                samplerate=config.MIC_SAMPLERATE,
                blocksize=config.BLOCKSIZE,
                dtype='float32',
                channels=1,
                device=config.AUDIO_DEVICE_ID,
                callback=self.audio_processor.audio_callback
            )
            self._stream.start()
            
            logger.info("✅ Audio recognition active! Speak now...")
            with self.audio_processor._audio_lock:
                self.audio_processor.audio_state["is_listening"] = True
            
            self.is_running = True
            
            while self.is_running:
                # Check if we need to reload the model
                should_reload = False
                with self._reload_lock:
                    if self._should_reload:
                        should_reload = True
                        self._should_reload = False
                
                if should_reload:
                    logger.info("🔄 Reloading model...")
                    # Clear the queue
                    while not self.audio_processor.q.empty():
                        try:
                            self.audio_processor.q.get_nowait()
                        except queue.Empty:
                            break
                    
                    # Reload model
                    if self._load_model():
                        logger.info("✅ Model reloaded successfully!")
                        with self.audio_processor._audio_lock:
                            self.audio_processor.audio_state["text"] = "Model reloaded - speak now!"
                    else:
                        logger.error("❌ Failed to reload model")
                        continue
                
                # Get audio data
                try:
                    data = self.audio_processor.q.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Resample audio
                audio_data = data[:, 0]
                num_output_samples = int(len(audio_data) * resample_ratio)
                resampled = signal.resample(audio_data, num_output_samples)
                
                # Scale to int16 range
                resampled_scaled = np.clip(resampled * 32767, -32768, 32767)
                resampled_int16 = np.int16(resampled_scaled)
                
                # Process with Vosk
                if self.recognizer.AcceptWaveform(resampled_int16.tobytes()):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        with self.audio_processor._audio_lock:
                            self.audio_processor.audio_state["text"] = text
                        logger.info(f"🗣️  Recognized: {text}")
                else:
                    # Partial results
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        with self.audio_processor._audio_lock:
                            self.audio_processor.audio_state["text"] = partial_text
                            
        except Exception as e:
            logger.error(f"✗ Audio stream error: {e}", exc_info=True)
            with self.audio_processor._audio_lock:
                self.audio_processor.audio_state["is_listening"] = False
            import traceback
            traceback.print_exc()
        finally:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
            self.is_running = False
    
    def stop(self):
        """Stop speech recognition."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)