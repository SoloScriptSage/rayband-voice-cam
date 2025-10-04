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

### Quick Start
```powershell
# 1. Clone the repository
git clone https://github.com/SoloScriptSage/rayband-voice-cam.git
cd rayband-voice-cam

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -e .

# 4. Download speech recognition model
python scripts/setup_model.py

# 5. Test hardware
python scripts/test_hardware.py

# 6. Run the application
python -m rayband.cli.main
```

### Manual Installation
```powershell
# Install dependencies manually
pip install opencv-python sounddevice scipy numpy vosk dlib
pip install mediapipe face_recognition  # Optional features

# Download Vosk model manually
# Visit: https://alphacephei.com/vosk/models
# Extract to model/ directory
```

Running
-------

### New Package Structure (Recommended)
```powershell
# Run with new structure
python -m rayband.cli.main

# Or use the console script (after pip install -e .)
rayband
```

### Legacy Mode (Backward Compatible)
```powershell
# Run with legacy structure
python main.py
```

### Hardware Testing
```powershell
# Test all hardware components
python scripts/test_hardware.py

# Or use the console script
rayband-test
```

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

Project Structure
-----------------
```
rayband-voice-cam/
├── rayband/                    # Main package
│   ├── core/                   # Core functionality
│   │   ├── audio.py           # Audio processing & speech recognition
│   │   ├── camera.py          # Camera handling & video processing
│   │   ├── face_detection.py  # Face detection & recognition
│   │   └── finger_detection.py # Hand/finger tracking
│   ├── utils/                  # Utilities
│   │   ├── config.py          # Configuration management
│   │   ├── command_router.py  # Voice command handling
│   │   └── file_utils.py      # File operations
│   ├── hardware/              # Hardware components
│   │   └── components/        # KiCad components
│   └── cli/                   # Command line interface
│       └── main.py
├── tests/                     # Test files
├── docs/                      # Documentation
├── schematics/                # Hardware schematics
├── scripts/                   # Utility scripts
│   ├── setup_model.py        # Model download script
│   └── test_hardware.py      # Hardware testing
├── requirements.txt
├── setup.py                   # Package installation
├── pyproject.toml            # Modern Python packaging
└── README.md
```

Development
-----------
```powershell
# Install in development mode
pip install -e .[dev]

# Run tests
python -m pytest

# Run linting
black rayband/
flake8 rayband/
mypy rayband/

# Build package
python -m build
```

Git & Repo
----------
- Repository: `rayband-voice-cam`
- `.gitignore` excludes large folders like `captures/`, `videos/`, and `model/`.
- Git LFS handles large schematic files.

License
-------
MIT


