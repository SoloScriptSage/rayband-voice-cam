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

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio input and processing."""
    
    def __init__(self):
        self.queue = queue.Queue()
        self._audio_lock = threading.Lock()
        self.audio_state = {
            "text": "Say something...",
            "is_listening": False
        }
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.debug(f"Audio status: {status}")
        self.queue.put(indata.copy())
    
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
    
    def start(self) -> bool:
        """Start speech recognition."""
        # Verify device
        try:
            device_info = sd.query_devices(self.audio_processor.device_id, 'input')
            logger.info(f"✓ Using device {self.audio_processor.device_id}: {device_info['name']}")
            logger.info(f"  Sample rate: {self.audio_processor.mic_samplerate} Hz → "
                       f"{self.audio_processor.vosk_samplerate} Hz (resampled)")
        except Exception as e:
            logger.error(f"✗ Device error: {e}", exc_info=True)
            return False
        
        # Load Vosk model
        logger.info(f"📂 Loading Vosk model from {self.model_path}...")
        try:
            self.model = vosk.Model(self.model_path)
            logger.info("✓ Model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}", exc_info=True)
            return False
        
        self.recognizer = vosk.KaldiRecognizer(self.model, self.audio_processor.vosk_samplerate)
        self.recognizer.SetWords(True)
        
        # Start recognition thread
        self._thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def _recognition_loop(self):
        """Main recognition loop running in separate thread."""
        resample_ratio = self.audio_processor.vosk_samplerate / self.audio_processor.mic_samplerate
        
        logger.info("🎙️  Starting audio recognition...")
        try:
            with sd.InputStream(
                samplerate=self.audio_processor.mic_samplerate,
                blocksize=self.audio_processor.blocksize,
                dtype='float32',
                channels=1,
                device=self.audio_processor.device_id,
                callback=self.audio_processor.audio_callback
            ):
                logger.info("✅ Audio recognition active! Speak now...")
                with self.audio_processor._audio_lock:
                    self.audio_processor.audio_state["is_listening"] = True
                
                self.is_running = True
                
                while self.is_running:
                    try:
                        data = self.audio_processor.queue.get(timeout=0.1)
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
            self.is_running = False
    
    def stop(self):
        """Stop speech recognition."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)