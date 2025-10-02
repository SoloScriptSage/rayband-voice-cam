import sounddevice as sd
import queue
import vosk
import json
import logging
import numpy as np
import threading

from config import AUDIO_DEVICE_ID, MIC_SAMPLERATE, VOSK_SAMPLERATE, BLOCKSIZE
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

q = queue.Queue()

_audio_lock = threading.Lock()
# Use a dictionary so it's mutable and shared properly
audio_state = {
    "text": "Say something...",
    "is_listening": False
}

def audio_callback(indata, frames, time, status):
    if status:
        logger.info(f"Audio status: {status}")
    q.put(indata.copy())

def get_last_text():
    """Get the current recognized text"""
    with _audio_lock:
        return audio_state["text"]

def start_audio_recognition(model_path):
    # Verify device - NO LOCK NEEDED
    try:
        device_info = sd.query_devices(AUDIO_DEVICE_ID, 'input')
        logger.info(f"✓ Using device {AUDIO_DEVICE_ID}: {device_info['name']}")
        logger.info(f"  Sample rate: {MIC_SAMPLERATE} Hz → {VOSK_SAMPLERATE} Hz (resampled)")
    except Exception as e:
        logger.error(f"✗ Device error: {e}", exc_info=True)
        return

    # Load Vosk model - NO LOCK NEEDED
    logger.info(f"📂 Loading Vosk model...")
    try:
        model = vosk.Model(model_path)
        logger.info("✓ Model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}", exc_info=True)
        return

    recognizer = vosk.KaldiRecognizer(model, VOSK_SAMPLERATE)
    recognizer.SetWords(True)

    # Calculate resampling ratio
    resample_ratio = VOSK_SAMPLERATE / MIC_SAMPLERATE

    # Start audio stream
    logger.info("🎙️  Starting audio recognition...")
    try:
        with sd.InputStream(
            samplerate=MIC_SAMPLERATE,
            blocksize=BLOCKSIZE,
            dtype='float32',
            channels=1,
            device=AUDIO_DEVICE_ID,
            callback=audio_callback
        ):
            logger.info("✅ Audio recognition active! Speak now...")
            with _audio_lock:  # LOCK ONLY FOR WRITING STATE
                audio_state["is_listening"] = True
            
            while True:
                data = q.get()
                
                # Resample audio from MIC_SAMPLERATE to VOSK_SAMPLERATE
                audio_data = data[:, 0]
                
                num_output_samples = int(len(audio_data) * resample_ratio)
                resampled = signal.resample(audio_data, num_output_samples)
                
                # Scale properly to int16 range
                resampled_scaled = np.clip(resampled * 32767, -32768, 32767)
                resampled_int16 = np.int16(resampled_scaled)
                
                # Process with Vosk
                if recognizer.AcceptWaveform(resampled_int16.tobytes()):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        with _audio_lock:  # LOCK ONLY FOR WRITING
                            audio_state["text"] = text
                        logger.info(f"🗣️  Recognized: {text}")
                else:
                    # Partial results
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        with _audio_lock:  # LOCK ONLY FOR WRITING
                            audio_state["text"] = partial_text
                            
    except Exception as e:
        logger.error(f"✗ Audio stream error: {e}", exc_info=True)
        with _audio_lock:
            audio_state["is_listening"] = False
        import traceback
        traceback.print_exc()