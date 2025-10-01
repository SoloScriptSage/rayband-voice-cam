RayBand Voice Cam
=================

Voice-controlled camera HUD for Windows using Vosk (speech-to-text) and OpenCV. Shows live face boxes, optional hand/finger landmarks, on-screen transcript, and supports voice commands like taking photos and recording videos with microphone audio.

Features
--------
- Live camera preview with overlay and transcript
- Speech recognition via Vosk (offline)
- Voice commands:
  - "take a picture" → save JPEG to `captures/`
  - "start recording" / "stop recording" → save MP4 to `videos/` and record mic audio (WAV); optional ffmpeg mux into a single MP4
- Dynamic transcript overlay with wrapping
- Face detection (dlib)
- Optional hand/finger detection (MediaPipe Hands)
- Startup camera backend/index auto-discovery with caching for faster subsequent launches

Requirements
------------
- Windows 10/11
- Python 3.9+
- A microphone and a camera

Installation
------------
1. Create/activate a virtual environment (recommended)
   
   ```powershell
   python -m venv .venv
   . .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies
   
   ```powershell
   pip install opencv-python sounddevice scipy numpy vosk dlib
   # Optional (for hand detection)
   pip install mediapipe
   # Optional (for name recognition)
   pip install face_recognition
   # If prompted for models for face_recognition:
   pip install git+https://github.com/ageitgey/face_recognition_models
   ```

3. Download a Vosk model and set the path in `main.py`:
   - Place model into `model/` and update `MODEL_PATH` if needed (default: `D:/Projects/rayband-facebook/model`).

Running
-------
```powershell
py .\main.py
```
- The microphone recognition starts in a background thread.
- The camera window opens; press `q` to quit.

Voice Commands
--------------
- "take a picture" → saves `captures/capture_YYYYMMDD_HHMMSS.jpg`
- "start recording" → starts video to `videos/video_YYYYMMDD_HHMMSS.mp4` and mic audio to WAV; if `ffmpeg` is on PATH, a muxed `*_with_audio.mp4` is created on stop
- "stop recording" → stops and finalizes files

Notes:
- Commands are debounced to avoid rapid repeats.
- The green transcript at the bottom shows live and final recognized text.

Outputs
-------
- Photos → `captures/`
- Videos → `videos/`
- Audio WAV (if not muxed) → `videos/`

Face Detection and Recognition
------------------------------
- Face detection is always on (dlib).
- Optional name recognition (if `face_recognition` installed):
  - Create folders like:
    - `known_faces/Alex/1.jpg`
    - `known_faces/Alex/2.jpg`
  - Clear, frontal images work best. Restart app to reload.

Hand/Finger Detection (Optional)
--------------------------------
- Implemented in `finger_detect.py` using MediaPipe Hands.
- To enable overlay in the preview, import and call in `video.py`:
  
  ```python
  from finger_detect import detect_fingers, draw_fingers
  # after grabbing `frame`:
  hand_det = detect_fingers(frame)
  draw_fingers(frame, hand_det)
  ```
- Install: `pip install mediapipe`

Configuration
-------------
- `main.py`: update `MODEL_PATH` to your Vosk model directory.
- `audio.py`: `DEVICE_ID`, sample rates, and block sizes can be tuned for your mic.
- `video.py`:
  - Camera backend/index auto-detection with caching to `.camera_cache.json`
  - Change overlays (font scale, margins) in the transcript section
  - Cooldowns for commands via `CommandRouter`

Troubleshooting
---------------
- Camera won’t open / slow to open:
  - Close other apps (Teams/Zoom/OBS).
  - Windows Settings → Privacy & Security → Camera: allow desktop apps.
  - First run may probe multiple backends/indices; subsequent runs are faster due to caching.
- Transcript shows "?":
  - We render ASCII only; Unicode ellipses are replaced with `...`.
- No audio in video:
  - Ensure mic works; WAV is recorded separately. If `ffmpeg` is installed, it muxes into `*_with_audio.mp4`.
  - Install `ffmpeg` and ensure it’s on PATH.
- Name recognition warnings:
  - Shown only when recognition is used. Install `face_recognition` and `face_recognition_models` if you want labels.

Git & Repo
----------
- Suggested repo name: `rayband-voice-cam`
- `.gitignore` already excludes large folders like `captures/`, `videos/`, and `model/`.

License
-------
MIT


