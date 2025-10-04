"""
Camera controller for RayBand voice camera.
"""

import cv2
import os
import time
import wave
import json
import logging
import subprocess
import shutil
from typing import Optional, Tuple, Callable

from ..utils.config import config
from ..utils.command_router import CommandRouter
from ..utils.file_utils import FileManager
from .face_detection import FaceDetector, FaceRecognizer
from .finger_detection import FingerDetector
from .audio import SpeechRecognizer

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Handles audio recording during video capture."""
    
    def __init__(self, device_id: int, sample_rate: int = 44100, channels: int = 1):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.stream = None
        self.wave_file = None
        self.is_active = False

    def _callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        if status:
            logger.info(f"AudioRecorder status: {status}")
        if self.wave_file is not None:
            self.wave_file.writeframes(indata.tobytes())

    def start(self, wav_path: str) -> bool:
        """Start audio recording."""
        if self.is_active:
            return True
        
        try:
            import sounddevice as sd
            
            # Prepare wave file
            self.wave_file = wave.open(wav_path, 'wb')
            self.wave_file.setnchannels(self.channels)
            self.wave_file.setsampwidth(2)  # 16-bit
            self.wave_file.setframerate(self.sample_rate)

            # Open mic stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=1024,
                dtype='int16',
                channels=self.channels,
                device=self.device_id,
                callback=self._callback,
            )
            self.stream.start()
            self.is_active = True
            logger.info(f"🎙️  Mic recording started → {wav_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start mic recording: {e}", exc_info=True)
            self._cleanup()
            return False

    def stop(self) -> str:
        """Stop audio recording."""
        if not self.is_active:
            return ""
        
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
        
        try:
            if self.wave_file is not None:
                self.wave_file.close()
        finally:
            self.wave_file = None
            self.is_active = False
        
        logger.info("🎙️  Mic recording stopped")
        return ""
    
    def _cleanup(self):
        """Clean up partial state."""
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        try:
            if self.wave_file is not None:
                self.wave_file.close()
        except Exception:
            pass
        self.stream = None
        self.wave_file = None
        self.is_active = False


class CameraController:
    """Main camera controller for the RayBand voice camera."""
    
    def __init__(self):
        self.file_manager = FileManager()
        self.command_router = CommandRouter()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.finger_detector = FingerDetector()
        self.speech_recognizer = SpeechRecognizer()
        
        # Set up cooldowns
        self.command_router.set_cooldown("take_picture", config.PICTURE_COOLDOWN)
        self.command_router.set_cooldown("start_recording", config.RECORDING_COOLDOWN)
        self.command_router.set_cooldown("stop_recording", config.RECORDING_COOLDOWN)
        
        # Camera state
        self.cap = None
        self.is_recording = False
        self.video_writer = None
        self.video_path = None
        self.audio_recorder = None
        
        # Load known faces
        self.face_recognizer.load_known_faces(config.KNOWN_FACES_DIR)
    
    def _wrap_text_to_width(self, text: str, font, font_scale: float, thickness: int, 
                           max_width: int, max_lines: int = 3) -> list:
        """Wrap text to fit within specified width."""
        if not text:
            return []
        
        # Sanitize to ASCII; OpenCV Hershey fonts are ASCII-only
        text_ascii = text.encode('ascii', 'ignore').decode()
        words = text_ascii.split()
        lines = []
        current = ""
        
        for word in words:
            candidate = word if current == "" else current + " " + word
            (w, h), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
            if w <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
                if len(lines) >= max_lines:
                    break
        
        if current and len(lines) < max_lines:
            lines.append(current)

        # Ellipsize last line if overflow remains
        remaining_words = words[len(" ".join(lines).split()):]
        if remaining_words and len(lines) > 0:
            last = lines[-1]
            while last and cv2.getTextSize(last + "...", font, font_scale, thickness)[0][0] > max_width:
                last = last[:-1].rstrip()
            if last:
                lines[-1] = last + "..."
        
        return lines
    
    def _try_open_camera(self) -> Tuple[Optional[cv2.VideoCapture], Optional[str], Optional[int]]:
        """Try to open a camera with various backends and indices."""
        backend_options = [
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_ANY, "ANY"),
        ]
        indices = list(range(0, 3))

        # Try cached camera first
        cached_index, cached_backend = self._load_cached_camera()
        if cached_index >= 0 and cached_backend >= 0:
            logger.info(f"… Trying cached camera backend={cached_backend}, index={cached_index}")
            cap = cv2.VideoCapture(cached_index, cached_backend)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                ok = False
                for attempt in range(2):
                    ok, test_frame = cap.read()
                    if ok:
                        break
                    time.sleep(0.1)
                if ok:
                    logger.info(f"✅ Using cached camera backend={cached_backend}, index={cached_index}")
                    return cap, str(cached_backend), cached_index
                cap.release()
                self._clear_cached_camera()

        for backend, name in backend_options:
            for idx in indices:
                logger.info(f"… Trying camera backend={name}, index={idx}")
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ok = False
                    for attempt in range(2):
                        ok, test_frame = cap.read()
                        if ok:
                            break
                        time.sleep(0.1)
                    if ok:
                        logger.info(f"Camera opened using backend={name}, index={idx}")
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        self._save_cached_camera(idx, backend)
                        return cap, name, idx
                    logger.info(f"   ↳ Opened but read failed on backend={name}, index={idx}; retrying others…")
                    cap.release()

        return None, None, None
    
    def _load_cached_camera(self) -> Tuple[int, int]:
        """Load cached camera settings."""
        cache_path = os.path.join(os.path.dirname(__file__), ".camera_cache.json")
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return int(data.get('index', -1)), int(data.get('backend', -1))
        except Exception:
            return -1, -1

    def _save_cached_camera(self, index: int, backend: int) -> None:
        """Save camera settings to cache."""
        cache_path = os.path.join(os.path.dirname(__file__), ".camera_cache.json")
        try:
            with open(cache_path, 'w') as f:
                json.dump({'index': index, 'backend': backend}, f)
        except Exception:
            pass

    def _clear_cached_camera(self) -> None:
        """Clear cached camera settings."""
        cache_path = os.path.join(os.path.dirname(__file__), ".camera_cache.json")
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception:
            pass
    
    def start(self) -> None:
        """Start the camera system."""
        # Start speech recognition
        self.speech_recognizer.start()
        
        # Open camera
        self.cap, backend_name, cam_index = self._try_open_camera()
        if self.cap is None:
            logger.error("❌ Cannot access camera: tried MSMF/DSHOW/ANY on indices 0-2")
            logger.info("   Tips: close other apps using the camera, check Windows Privacy settings,")
            return

        logger.info("✅ Camera started. Press 'q' to quit.")

        # Create the window
        try:
            cv2.namedWindow("RayBan Prototype", cv2.WINDOW_NORMAL)
        except Exception:
            pass

        recovery_attempted = False
        last_handled_text = ""

        while True:
            ret, frame = self.cap.read()
            if not ret:
                if not recovery_attempted:
                    logger.error("⚠️  Camera read failed; attempting to recover...")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self._clear_cached_camera()
                    new_cap, backend_name, cam_index = self._try_open_camera()
                    if new_cap is not None:
                        self.cap = new_cap
                        recovery_attempted = True
                        continue
                break

            # Detect and draw faces
            faces = self.face_detector.detect_faces(frame)
            recognized_faces = self.face_recognizer.recognize_faces(frame, faces)
            self.face_recognizer.draw_recognized(frame, recognized_faces)

            # Detect fingers if available
            if self.finger_detector.is_available():
                hand_results = self.finger_detector.detect_fingers(frame)
                self.finger_detector.draw_fingers(frame, hand_results)

            # Get current audio text
            current_text = self.speech_recognizer.get_text()

            # Display the audio text with wrapping
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            margin_left = 20
            margin_bottom = 20
            available_width = frame.shape[1] - margin_left - 20
            lines = self._wrap_text_to_width(current_text, font, font_scale, thickness, available_width, max_lines=3)

            # Compute dynamic overlay height
            (line_w, line_h), _ = cv2.getTextSize("Ag", font, font_scale, thickness)
            total_height = len(lines) * (line_h + 6) + 14

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, frame.shape[0] - total_height),
                (frame.shape[1], frame.shape[0]),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            base_y = frame.shape[0] - margin_bottom - (len(lines) - 1) * (line_h + 6)
            for i, line in enumerate(lines):
                y = base_y + i * (line_h + 6)
                cv2.putText(
                    frame,
                    line,
                    (margin_left, y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                    cv2.LINE_AA
                )

            # Draw REC indicator if recording
            if self.is_recording:
                cv2.circle(frame, (20, 30), 8, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    "REC",
                    (40, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # Write frame to video if recording
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)

            # Command handling
            if current_text and current_text != last_handled_text:
                command = self.command_router.process_command(current_text)
                
                if command == "take_picture" and self.command_router.should_run("take_picture"):
                    self.file_manager.save_snapshot(frame)
                    self.command_router.mark_ran("take_picture")
                    last_handled_text = current_text

                elif command == "start_recording" and not self.is_recording and self.command_router.should_run("start_recording"):
                    self._start_recording(frame)
                    self.command_router.mark_ran("start_recording")
                    last_handled_text = current_text

                elif command == "stop_recording" and self.is_recording and self.command_router.should_run("stop_recording"):
                    self._stop_recording()
                    self.command_router.mark_ran("stop_recording")
                    last_handled_text = current_text

            cv2.imshow("RayBan Prototype", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()
    
    def _start_recording(self, frame) -> None:
        """Start video recording with audio."""
        self.video_path, audio_wav_path = self.file_manager.get_video_path()

        # Figure out frame size and FPS
        height, width = frame.shape[:2]
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            logger.error("❌ Failed to start video writer")
            return

        # Start mic recorder
        self.audio_recorder = AudioRecorder(device_id=config.get_audio_device_id())
        try:
            self.audio_recorder.start(audio_wav_path)
        except Exception:
            logger.error("⚠️  Failed to start mic; continuing video-only recording")

        self.is_recording = True
        logger.info(f"⏺️ Recording started: {self.video_path}")
    
    def _stop_recording(self) -> None:
        """Stop video recording and process files."""
        self.is_recording = False
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        if self.audio_recorder is not None:
            self.audio_recorder.stop()
            self.audio_recorder = None
        
        # Try to mux audio and video
        if self.video_path:
            audio_wav_path = self.video_path.replace('.mp4', '.wav')
            muxed_path = self.file_manager.try_mux_audio(self.video_path, audio_wav_path)
            
            if muxed_path:
                logger.info(f"⏹️ Recording saved (with audio): {muxed_path}")
            else:
                logger.info(f"⏹️ Recording saved: {self.video_path}")
                if os.path.exists(audio_wav_path):
                    logger.info(f"🎵 Audio WAV saved: {audio_wav_path}")
    
    def stop(self) -> None:
        """Stop the camera system."""
        if self.cap is not None:
            self.cap.release()
        
        if self.video_writer is not None:
            self.video_writer.release()
        
        if self.audio_recorder is not None:
            self.audio_recorder.stop()
        
        cv2.destroyAllWindows()
        logger.info("✅ Camera stopped.")
