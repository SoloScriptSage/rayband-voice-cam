import sounddevice as sd
import vosk
import sys
import json
import numpy as np
from scipy import signal

MODEL_PATH = "D:/Projects/rayband-facebook/model"
DEVICE_ID = 1
MIC_SAMPLERATE = 44100  # Your mic's native sample rate
VOSK_SAMPLERATE = 16000  # What Vosk expects
BLOCKSIZE = 22050  # Larger blocksize for 44100 Hz (0.5 seconds)

print("Loading model...")
model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, VOSK_SAMPLERATE)

print(f"\n✅ Model loaded. Using device {DEVICE_ID}")
print(f"Device info: {sd.query_devices(DEVICE_ID, 'input')}")
print(f"Mic sample rate: {MIC_SAMPLERATE} Hz → Resampling to {VOSK_SAMPLERATE} Hz")
print("\n🎤 Recording... SPEAK NOW! (Press Ctrl+C to stop)\n")

# Buffer for resampling
resample_ratio = VOSK_SAMPLERATE / MIC_SAMPLERATE

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    
    # Resample from 44100 to 16000 Hz
    audio_data = indata[:, 0]  # Get mono channel
    
    # Debug: Check audio level
    audio_level = np.abs(audio_data).max()
    if audio_level > 0.01:  # Threshold for detecting audio
        print(f"🔊 Audio detected! Level: {audio_level:.3f}", end='\r')
    
    num_output_samples = int(len(audio_data) * resample_ratio)
    resampled = signal.resample(audio_data, num_output_samples)
    
    # Scale properly to int16 range (-32768 to 32767)
    resampled_scaled = np.clip(resampled * 32767, -32768, 32767)
    resampled_int16 = np.int16(resampled_scaled)
    
    if recognizer.AcceptWaveform(resampled_int16.tobytes()):
        result = json.loads(recognizer.Result())
        text = result.get('text', '')
        if text:
            print(f"\n✓ You said: \"{text}\"")
        else:
            print(f"\n⚠️  Vosk processed audio but detected no text")
    else:
        # Show partial results
        partial = json.loads(recognizer.PartialResult())
        partial_text = partial.get('partial', '')
        if partial_text:
            print(f"... {partial_text}                    ", end='\r')

try:
    with sd.InputStream(samplerate=MIC_SAMPLERATE, blocksize=BLOCKSIZE, device=DEVICE_ID, 
                        channels=1, callback=callback):
        print("Recording started... Speak clearly!")
        input()  # Keep running until Enter is pressed
except KeyboardInterrupt:
    print("\n\n✓ Stopped recording")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()