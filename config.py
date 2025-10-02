import os

# Paths
MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "./model")

# Audio settings
AUDIO_DEVICE_ID = 1
MIC_SAMPLERATE = 44100
VOSK_SAMPLERATE = 16000
BLOCKSIZE = 11025

# Camera settings  
CAMERA_BACKEND = os.getenv("RAYCAM_BACKEND", "DSHOW")
CAMERA_INDEX = int(os.getenv("RAYCAM_INDEX", "0"))

# Command cooldowns (seconds)
PICTURE_COOLDOWN = 2.0
RECORDING_COOLDOWN = 2.0