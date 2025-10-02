import os
import threading
from audio import start_audio_recognition
from video import start_camera
from config import MODEL_PATH

# Start audio recognition in a separate thread
audio_thread = threading.Thread(target=start_audio_recognition, args=(MODEL_PATH,), daemon=True)
audio_thread.start()

# Start camera feed with overlay
start_camera()
