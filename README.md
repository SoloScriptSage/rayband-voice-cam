# Rayband Voice Cam

A voice-controlled camera application with real-time speech recognition, face detection, hand tracking, and sign language recognition capabilities. Built with Python, OpenCV, and various ML frameworks.

## Features

- **Voice-Controlled Camera Interface** with real-time HUD overlay
- **Offline Speech Recognition** powered by Vosk
- **Voice Commands** for hands-free photo and video capture
- **Face Detection and Recognition** using dlib and face_recognition
- **Hand/Finger Detection** via MediaPipe
- **Sign Language Recognition** for gesture-based interaction
- **Hardware Integration** support
- **Smart Camera Detection** with caching for faster startup

## Project Structure

```
rayband-voice-cam/
├── rayband/
│   ├── cli/                    # Command-line interface modules
│   └── core/                   # Core functionality
│       ├── audio.py            # Audio capture and speech recognition
│       ├── camera.py           # Camera handling and video recording
│       ├── face_detection.py  # Face detection and recognition
│       ├── finger_detection.py # Hand/finger tracking
│       └── sign_language.py    # Sign language recognition
├── hardware/                   # Hardware integration modules
├── utils/                      # Utility functions
├── schemas/                    # Data schemas and models
├── scripts/                    # Helper scripts
├── tests/                      # Test suite
├── captures/                   # Photo output directory
├── videos/                     # Video output directory
├── known_faces/                # Face recognition training data
├── models/                     # ML models directory
├── docs/                       # Documentation
├── setup.py                    # Package installation script
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                   # This file
```

## Requirements

### System
- Windows 10/11
- Python 3.9+
- Microphone and webcam

### Dependencies
See `requirements.txt` for the complete list. Core dependencies include:
- opencv-python
- sounddevice
- scipy
- numpy
- vosk
- dlib
- mediapipe
- face_recognition

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SoloScriptSage/rayband-voice-cam.git
   cd rayband-voice-cam
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv311
   venv311\Scripts\activate  # Windows
   # or
   source venv311/bin/activate  # Linux/Mac
   ```

3. **Install the package**
   ```bash
   pip install -e .
   # or install from requirements
   pip install -r requirements.txt
   ```

4. **Download Vosk model**
   - Download a Vosk speech recognition model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
   - Extract to the `models/` directory
   - Update the model path in your configuration if needed

## Usage

### Running the Application

```bash
# Run the main application
python rayband/cli/main.py

# Or if installed as package
python -m rayband
```

### Voice Commands

| Command | Action |
|---------|--------|
| "take a picture" | Captures a timestamped photo to `captures/` |
| "start recording" | Begins video recording with audio |
| "stop recording" | Stops recording and saves to `videos/` |

### Controls
- **Voice Commands**: Speak naturally to control the camera
- **Keyboard**: Press `q` to quit the application
- **Sign Language**: Use configured gestures for interaction (if enabled)

## Configuration

Configuration files and settings can be found in the respective module files:

- **Audio settings**: `rayband/core/audio.py`
- **Camera settings**: `rayband/core/camera.py`
- **Face recognition**: `rayband/core/face_detection.py`
- **Hand tracking**: `rayband/core/finger_detection.py`
- **Sign language**: `rayband/core/sign_language.py`

Camera configuration is cached in `.camera_cache.json` for faster subsequent launches.

## Features in Detail

### Face Recognition

1. Create directory structure for known faces:
   ```
   known_faces/
   ├── PersonName1/
   │   ├── photo1.jpg
   │   └── photo2.jpg
   └── PersonName2/
       └── photo1.jpg
   ```

2. Use clear, frontal face images for best results
3. Restart the application to reload face database

### Hand and Finger Detection

The application uses MediaPipe Hands for real-time hand tracking and gesture recognition. Hand landmarks and finger positions are detected and can be used for gesture-based controls.

### Sign Language Recognition

Implements gesture recognition for sign language interpretation. The system can recognize and respond to configured sign language gestures in real-time.

### Hardware Integration

The `hardware/` module provides interfaces for additional hardware components and sensors that can be integrated with the camera system.

## Output Files

- **Photos**: `captures/capture_YYYYMMDD_HHMMSS.jpg`
- **Videos**: `videos/video_YYYYMMDD_HHMMSS.mp4`
- **Audio**: `videos/video_YYYYMMDD_HHMMSS.wav` (if not muxed)
- **Muxed Video**: `videos/video_YYYYMMDD_HHMMSS_with_audio.mp4` (with ffmpeg)

## Development

### Running Tests

```bash
pytest tests/
```

### Project Setup

The project uses `setup.py` and `pyproject.toml` for package configuration. To install in development mode:

```bash
pip install -e .
```

## Troubleshooting

### Camera Issues
- Close other applications using the camera (Teams, Zoom, OBS)
- Check Windows Privacy Settings → Camera permissions
- First run may take longer while detecting camera backends
- Check `.camera_cache.json` if camera detection fails

### Audio Issues
- Verify microphone permissions and functionality
- Install `ffmpeg` for audio/video muxing
- Check audio device selection in `audio.py`

### Recognition Issues
- Ensure sufficient lighting for face/hand detection
- Use clear, high-quality images for face recognition training
- Check that ML models are properly downloaded and accessible

### Performance
- Close unnecessary background applications
- Reduce camera resolution if experiencing lag
- Check CPU/GPU usage during operation

## Documentation

Additional documentation can be found in the `docs/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Vosk](https://alphacephei.com/vosk/) - Offline speech recognition
- [dlib](http://dlib.net/) - Face detection
- [MediaPipe](https://mediapipe.dev/) - Hand tracking
- [face_recognition](https://github.com/ageitgey/face_recognition) - Facial recognition
- [OpenCV](https://opencv.org/) - Computer vision capabilities

## Contact

Project Link: [https://github.com/SoloScriptSage/rayband-voice-cam](https://github.com/SoloScriptSage/rayband-voice-cam)