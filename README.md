# 🕶️ RayBand Voice Camera

A real-time voice-controlled camera system for smart glasses featuring computer vision, speech recognition, and gesture detection with multilingual support.



## ✨ Features

### 🎤 Voice Control
- **Hands-free operation** with natural language commands
- **Multilingual support**: English and Ukrainian with dynamic language switching
- **Offline speech recognition** using Vosk models
- **Full Unicode/Cyrillic text rendering** via PyQt5

### 📸 Media Capture
- **Voice-activated photo capture**: "Take a picture"
- **Video recording with audio**: "Start recording" / "Stop recording"
- **Automatic audio/video muxing** using FFmpeg
- **Organized file management** (separate folders for photos/videos)

### 👁️ Computer Vision
- **Real-time face detection** using OpenCV Haar Cascades or dlib
- **Hand tracking** with MediaPipe Hands
- **Sign language recognition** (11+ ASL gestures including thumbs up, peace, OK, etc.)
- **30+ FPS performance** with optimized video processing

### 🖥️ User Interface
- **Modern PyQt5 GUI** with real-time video display
- **Live transcription overlay** with automatic text wrapping
- **Status indicators**: FPS counter, recording status, current language
- **Responsive design** with proper Unicode support for international text

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Webcam
- Microphone
- FFmpeg (optional, for audio/video muxing)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rayband-voice-camera.git
cd rayband-voice-camera
```

2. **Create virtual environment**
```bash
python -m venv venv311
venv311\Scripts\activate  # Windows
# source venv311/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Vosk models**

Download the speech recognition models and place them in the `models/` directory:
- [English (US) model](https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip)
- [Ukrainian model](https://alphacephei.com/vosk/models/vosk-model-uk-v3.zip)

Extract them to:
```
models/
├── vosk-model-en-us-0.22/
└── vosk-model-uk-v3/
```

5. **Run the application**
```bash
python