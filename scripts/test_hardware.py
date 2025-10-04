#!/usr/bin/env python3
"""
Hardware testing script for RayBand voice camera components.
"""

import sys
import cv2
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rayband.core.camera import CameraController
from rayband.core.audio import SpeechRecognizer
from rayband.utils.config import config


def test_camera():
    """Test camera functionality."""
    print("📹 Testing camera...")
    
    try:
        controller = CameraController()
        cap, backend, index = controller._try_open_camera()
        
        if cap is None:
            print("❌ Camera test failed - no camera found")
            return False
        
        print(f"✅ Camera opened successfully")
        print(f"   Backend: {backend}")
        print(f"   Index: {index}")
        
        # Test frame capture
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame capture successful ({frame.shape[1]}x{frame.shape[0]})")
            cap.release()
            return True
        else:
            print("❌ Frame capture failed")
            cap.release()
            return False
            
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        return False


def test_audio():
    """Test audio functionality."""
    print("🎙️ Testing audio...")
    
    try:
        import sounddevice as sd
        
        # List audio devices
        devices = sd.query_devices()
        print(f"✅ Found {len(devices)} audio devices")
        
        # Test default device
        device_id = config.get_audio_device_id()
        device_info = sd.query_devices(device_id, 'input')
        print(f"✅ Using device {device_id}: {device_info['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio test error: {e}")
        return False


def test_speech_recognition():
    """Test speech recognition (without starting full system)."""
    print("🗣️ Testing speech recognition...")
    
    try:
        model_path = config.get_model_path()
        if not os.path.exists(model_path):
            print(f"❌ Speech recognition test failed - model not found at {model_path}")
            print("   Run: python scripts/setup_model.py")
            return False
        
        print(f"✅ Model found at {model_path}")
        
        # Test model loading
        import vosk
        model = vosk.Model(model_path)
        print("✅ Vosk model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Speech recognition test error: {e}")
        return False


def test_face_detection():
    """Test face detection."""
    print("👤 Testing face detection...")
    
    try:
        from rayband.core.face_detection import FaceDetector
        
        detector = FaceDetector()
        print("✅ Face detector initialized")
        
        # Test with a simple frame
        test_frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
        faces = detector.detect_faces(test_frame)
        print(f"✅ Face detection working (found {len(faces)} faces in test frame)")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detection test error: {e}")
        return False


def main():
    """Run all hardware tests."""
    print("🔧 RayBand Voice Camera - Hardware Test")
    print("=" * 50)
    
    tests = [
        ("Camera", test_camera),
        ("Audio", test_audio),
        ("Speech Recognition", test_speech_recognition),
        ("Face Detection", test_face_detection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your hardware is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
