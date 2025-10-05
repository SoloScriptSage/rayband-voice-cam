"""
Camera handling and video processing for RayBand voice camera.
"""

import cv2
import os
import time
import json
import logging
import wave
import shutil
import subprocess
from typing import Optional, Tuple

from ..utils.config import config
from ..utils.command_router import CommandRouter
from ..utils.file_utils import FileManager
from .audio import AudioProcessor, SpeechRecognizer
from .face_detection import FaceDetector

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
    """Main controller for camera operations and video processing."""
    
    def __init__(self):
        self.config = config
        self.file_manager = FileManager(
            config.CAPTURES_DIR, 
            config.VIDEOS_DIR
        )
        self.command_router = CommandRouter()
        self.face_detector = FaceDetector()
        
        # Setup command cooldowns
        self.command_router.set_cooldown("take_picture", config.PICTURE_COOLDOWN)
        self.command_router.set_cooldown("start_recording", config.RECORDING_COOLDOWN)
        self.command_router.set_cooldown("stop_recording", config.RECORDING_COOLDOWN)
        
        # Audio setup
        self.audio_processor = AudioProcessor()
        self.speech_recognizer = SpeechRecognizer(
            config.MODEL_PATH,
            self.audio_processor
        )
        
        # Camera state
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_path: Optional[str] = None
        self.audio_wav_path: Optional[str] = None
        self.audio_recorder: Optional[AudioRecorder] = None
        
        # Command handling state
        self.last_handled_text = ""
        
        # Cache file
        self._cache_path = ".camera_cache.json"
    
    def _load_cached_camera(self) -> Tuple[int, int]:
        """Load cached camera settings."""
        try:
            with open(self._cache_path, 'r') as f:
                data = json.load(f)
                return int(data.get('index', -1)), int(data.get('backend', -1))
        except Exception:
            return -1, -1
    
    def _save_cached_camera(self, index: int, backend: int) -> None:
        """Save camera settings to cache."""
        try:
            with open(self._cache_path, 'w') as f:
                json.dump({'index': index, 'backend': backend}, f)
            logger.info(f"💾 Cached camera settings")
        except Exception:
            pass
    
    def _clear_cached_camera(self) -> None:
        """Clear camera cache."""
        try:
            if os.path.exists(self._cache_path):
                os.remove(self._cache_path)
                logger.info("🗑️  Cleared camera cache")
        except Exception:
            pass
    
    def _try_open_camera(self) -> Tuple[Optional[cv2.VideoCapture], str, int]:
        """Try to open camera with various backends and indices."""
        backend_options = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "ANY"),
        ]
        indices = list(range(0, 3))
        
        # Try cached camera first
        cached_index, cached_backend = self._load_cached_camera()
        if cached_index >= 0 and cached_backend >= 0:
            logger.info(f"📦 Trying cached camera backend={cached_backend}, index={cached_index}")
            cap = cv2.VideoCapture(cached_index, cached_backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    logger.info(f"✅ Using cached camera (fast open!)")
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    return cap, str(cached_backend), cached_index
                cap.release()
            logger.info("⚠️  Cached camera failed, trying other options...")
            self._clear_cached_camera()
        
        # Try all combinations
        for backend, name in backend_options:
            for idx in indices:
                logger.info(f"… Trying camera backend={name}, index={idx}")
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        logger.info(f"✅ Camera opened using backend={name}, index={idx}")
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        self._save_cached_camera(idx, backend)
                        return cap, name, idx
                    cap.release()
        
        return None, "", -1
    
    def _wrap_text(self, text: str, max_width: int, max_lines: int = 3):
        """Wrap text to fit within width."""
        if not text:
            return []
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
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
        
        # Add ellipsis if needed
        if len(words) > len(" ".join(lines).split()) and lines:
            lines[-1] = lines[-1] + "..."
        
        return lines
    
    def _handle_commands(self, current_text: str, frame):
        """Handle voice commands."""
        if not current_text or current_text == self.last_handled_text:
            return
        
        # Take picture
        if self.command_router.should_take_picture(current_text):
            if self.command_router.should_run("take_picture"):
                path = self.file_manager.save_snapshot(frame)
                self.command_router.mark_ran("take_picture")
                self.last_handled_text = current_text
                logger.info(f"📸 Picture saved: {path}")
        
        # Start recording
        elif self.command_router.should_start_recording(current_text):
            if not self.is_recording and self.command_router.should_run("start_recording"):
                self._start_recording(frame)
                self.command_router.mark_ran("start_recording")
                self.last_handled_text = current_text
        
        # Stop recording
        elif self.command_router.should_stop_recording(current_text):
            if self.is_recording and self.command_router.should_run("stop_recording"):
                self._stop_recording()
                self.command_router.mark_ran("stop_recording")
                self.last_handled_text = current_text
    
    def _start_recording(self, frame):
        """Start video recording."""
        self.video_path, self.audio_wav_path = self.file_manager.get_video_path()
        
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
        self.audio_recorder = AudioRecorder(device_id=config.AUDIO_DEVICE_ID)
        try:
            self.audio_recorder.start(self.audio_wav_path)
        except Exception:
            logger.error("⚠️  Failed to start mic; continuing video-only recording")
        
        self.is_recording = True
        logger.info(f"⏺️ Recording started: {self.video_path}")
    
    def _stop_recording(self):
        """Stop video recording."""
        self.is_recording = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        if self.audio_recorder:
            self.audio_recorder.stop()
            self.audio_recorder = None
        
        # Try to mux audio and video
        if self.video_path and self.audio_wav_path and os.path.exists(self.audio_wav_path):
            muxed_path = self.file_manager.try_mux_audio(self.video_path, self.audio_wav_path)
            
            if muxed_path:
                logger.info(f"⏹️ Recording saved (with audio): {muxed_path}")
            else:
                logger.info(f"⏹️ Recording saved: {self.video_path}")
                logger.info(f"🎵 Audio WAV saved: {self.audio_wav_path}")
        elif self.video_path:
            logger.info(f"⏹️ Recording saved: {self.video_path}")
        
        self.video_path = None
        self.audio_wav_path = None
    
    def start(self):
        """Start the camera system."""
        # Start speech recognition
        logger.info("🎤 Starting speech recognition...")
        self.speech_recognizer.start()
        
        # Open camera
        logger.info("📹 Opening camera...")
        self.cap, backend_name, cam_index = self._try_open_camera()
        
        if self.cap is None:
            logger.error("❌ Cannot access camera")
            return
        
        logger.info("✅ Camera started. Press 'q' to quit.")
        
        # Create window
        try:
            cv2.namedWindow("RayBand Voice Camera", cv2.WINDOW_NORMAL)
        except Exception:
            pass
        
        # Main loop
        recovery_attempted = False
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
                logger.error("❌ Camera failed")
                break
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            self.face_detector.draw_faces(frame, faces)
            
            # Get current text
            current_text = self.audio_processor.get_last_text()
            
            # Handle commands
            self._handle_commands(current_text, frame)
            
            # Draw transcript overlay
            margin_left = 20
            margin_bottom = 20
            available_width = frame.shape[1] - margin_left - 20
            lines = self._wrap_text(current_text, available_width)
            
            if lines:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (_, line_h), _ = cv2.getTextSize("Ag", font, font_scale, thickness)
                total_height = len(lines) * (line_h + 6) + 14
                
                # Draw semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (0, frame.shape[0] - total_height),
                    (frame.shape[1], frame.shape[0]),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Draw text
                base_y = frame.shape[0] - margin_bottom - (len(lines) - 1) * (line_h + 6)
                for i, line in enumerate(lines):
                    y = base_y + i * (line_h + 6)
                    cv2.putText(
                        frame, line, (margin_left, y),
                        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA
                    )
            
            # Draw REC indicator
            if self.is_recording:
                cv2.circle(frame, (20, 30), 8, (0, 0, 255), -1)
                cv2.putText(
                    frame, "REC", (40, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                )
            
            # Write frame if recording
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)
            
            # Display
            cv2.imshow("RayBand Voice Camera", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.is_recording:
            self._stop_recording()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.speech_recognizer.stop()
        logger.info("✅ Camera stopped.")