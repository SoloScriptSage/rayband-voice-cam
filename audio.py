import sounddevice as sd
import queue
import vosk
import json
import numpy as np
from scipy import signal

q = queue.Queue()

# Use a dictionary so it's mutable and shared properly
audio_state = {
    "text": "Say something...",
    "is_listening": False
}

# Audio parameters
DEVICE_ID = 1
MIC_SAMPLERATE = 44100  # Your USB mic's native sample rate
VOSK_SAMPLERATE = 16000  # What Vosk expects
BLOCKSIZE = 11025  # Smaller = faster response (0.25 seconds)

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    q.put(indata.copy())

def get_last_text():
    """Get the current recognized text"""
    return audio_state["text"]

def start_audio_recognition(model_path):
    # Verify device
    try:
        device_info = sd.query_devices(DEVICE_ID, 'input')
        print(f"✓ Using device {DEVICE_ID}: {device_info['name']}")
        print(f"  Sample rate: {MIC_SAMPLERATE} Hz → {VOSK_SAMPLERATE} Hz (resampled)")
    except Exception as e:
        print(f"❌ Device error: {e}")
        return

    # Load Vosk model
    print(f"📂 Loading Vosk model...")
    try:
        model = vosk.Model(model_path)
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    recognizer = vosk.KaldiRecognizer(model, VOSK_SAMPLERATE)
    recognizer.SetWords(True)

    # Calculate resampling ratio
    resample_ratio = VOSK_SAMPLERATE / MIC_SAMPLERATE

    # Start audio stream
    print("🎙️  Starting audio recognition...")
    try:
        with sd.InputStream(
            samplerate=MIC_SAMPLERATE,
            blocksize=BLOCKSIZE,
            dtype='float32',
            channels=1,
            device=DEVICE_ID,
            callback=audio_callback
        ):
            print("✅ Audio recognition active! Speak now...")
            audio_state["is_listening"] = True
            
            while True:
                data = q.get()
                
                # Resample audio from MIC_SAMPLERATE to VOSK_SAMPLERATE
                audio_data = data[:, 0]  # Get mono channel
                
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
                        audio_state["text"] = text
                        print(f"🗣️  Recognized: {text}")
                else:
                    # Partial results (shows live transcription)
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        # Update with partial results IMMEDIATELY for live display
                        audio_state["text"] = partial_text
                        # print(f"... {partial_text}                    ", end='\r')
                            
    except Exception as e:
        print(f"❌ Audio stream error: {e}")
        audio_state["is_listening"] = False
        import traceback
        traceback.print_exc()