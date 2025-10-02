import cv2
import os
import time
import wave
import shutil
import subprocess
import logging
from typing import Callable, Dict
import sounddevice as sd
import json
from face_detect import detect_faces, draw_faces
from audio import get_last_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def _ensure_captures_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _save_snapshot(frame, directory: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    path = os.path.join(directory, filename)
    cv2.imwrite(path, frame)
    return path


def _should_take_picture(recognized_text: str) -> bool:
    if not recognized_text:
        return False
    text = recognized_text.lower()
    triggers = [
        "take a picture",
        "take picture",
        "take photo",
        "take a photo",
        "snap a photo",
        "snap picture",
        "capture photo",
        "capture picture",
        "screenshot",
    ]
    return any(trigger in text for trigger in triggers)


def _should_start_recording(recognized_text: str) -> bool:
    if not recognized_text:
        return False
    text = recognized_text.lower()
    triggers = [
        "start recording",
        "start video",
        "begin recording",
        "record video",
    ]
    return any(trigger in text for trigger in triggers)


def _should_stop_recording(recognized_text: str) -> bool:
    if not recognized_text:
        return False
    text = recognized_text.lower()
    triggers = [
        "stop recording",
        "end recording",
        "stop video",
        "finish recording",
    ]
    return any(trigger in text for trigger in triggers)


class CommandRouter:
    def __init__(self):
        self._last_command_text = ""
        self._last_run_at: Dict[str, float] = {}
        self._cooldowns: Dict[str, float] = {}

    def set_cooldown(self, command_name: str, seconds: float) -> None:
        self._cooldowns[command_name] = seconds

    def should_run(self, command_name: str) -> bool:
        cooldown = self._cooldowns.get(command_name, 0.0)
        last = self._last_run_at.get(command_name, 0.0)
        return (time.time() - last) >= cooldown

    def mark_ran(self, command_name: str) -> None:
        self._last_run_at[command_name] = time.time()


class AudioRecorder:
    def __init__(self, device_id: int, sample_rate: int = 44100, channels: int = 1):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.stream = None
        self.wave_file = None
        self.is_active = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Non-fatal; just log
            logger.info(f"AudioRecorder status: {status}")
        if self.wave_file is not None:
            # indata is int16 because we request dtype='int16'
            self.wave_file.writeframes(indata.tobytes())

    def start(self, wav_path: str) -> bool:
        if self.is_active:
            return True
        try:
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
            # Cleanup partial state
            try:
                if self.stream is not None:
                    self.stream.stop(); self.stream.close()
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
            return False

    def stop(self) -> str:
        if not self.is_active:
            return ""
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
        wav_path = ""
        try:
            if self.wave_file is not None:
                # wave doesn't expose the filename; track externally if needed
                # We can close and rely on caller to know the path
                self.wave_file.close()
        finally:
            self.wave_file = None
            self.is_active = False
        logger.info("🎙️  Mic recording stopped")
        return wav_path


def _try_mux_audio(video_path: str, audio_wav_path: str) -> str:
    # Check ffmpeg availability
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.info("ℹ️  ffmpeg not found in PATH; leaving WAV alongside video.")
        return ""
    # Create output with same name replacing extension
    base, _ = os.path.splitext(video_path)
    output_path = base + "_with_audio.mp4"
    try:
        # -shortest to stop at shortest stream
        cmd = [
            ffmpeg, '-y',
            '-i', video_path,
            '-i', audio_wav_path,
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"❌ ffmpeg mux failed:\n {result.stderr}", exc_info=True)
            return ""
        logger.info(f"✅ Muxed audio+video → {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Error running ffmpeg: {e}", exc_info=True)
        return ""


def _wrap_text_to_width(text: str, font, font_scale: float, thickness: int, max_width: int, max_lines: int = 3):
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

    # Ellipsize last line if overflow remains (ASCII "...")
    remaining_words = words[len(" ".join(lines).split()):]
    if remaining_words and len(lines) > 0:
        last = lines[-1]
        while last and cv2.getTextSize(last + "...", font, font_scale, thickness)[0][0] > max_width:
            last = last[:-1].rstrip()
        if last:
            lines[-1] = last + "..."
    return lines


_CAMERA_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".camera_cache.json")


def _load_cached_camera():
    try:
        with open(_CAMERA_CACHE_PATH, 'r') as f:
            data = json.load(f)
            return int(data.get('index', -1)), int(data.get('backend', -1))
    except Exception:
        return -1, -1


def _save_cached_camera(index: int, backend: int) -> None:
    try:
        with open(_CAMERA_CACHE_PATH, 'w') as f:
            json.dump({'index': index, 'backend': backend}, f)
    except Exception:
        pass


def _clear_cached_camera() -> None:
    try:
        if os.path.exists(_CAMERA_CACHE_PATH):
            os.remove(_CAMERA_CACHE_PATH)
    except Exception:
        pass


def _env_force_backend() -> int:
    """Return an OpenCV backend flag from env var RAYCAM_BACKEND if set."""
    value = os.environ.get("RAYCAM_BACKEND", "").strip().upper()
    if not value:
        return -1
    # Accept names or integer codes
    name_to_flag = {
        "MSMF": getattr(cv2, "CAP_MSMF", -1),
        "DSHOW": getattr(cv2, "CAP_DSHOW", -1),
        "ANY": getattr(cv2, "CAP_ANY", -1),
    }
    if value in name_to_flag and name_to_flag[value] != -1:
        return name_to_flag[value]
    try:
        return int(value)
    except Exception:
        return -1


def _env_force_index() -> int:
    """Return a forced camera index from env var RAYCAM_INDEX if set."""
    value = os.environ.get("RAYCAM_INDEX", "").strip()
    if value == "":
        return -1
    try:
        return int(value)
    except Exception:
        return -1


def _env_force_dshow_name() -> str:
    value = os.environ.get("RAYCAM_DSHOW_NAME", "").strip()
    return value


def _enumerate_dshow_devices() -> list:
    """Return a list of DirectShow video device names via ffmpeg if available."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return []
    try:
        # ffmpeg prints device list to stderr
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        stderr = result.stderr
        names = []
        in_video_section = False
        for line in stderr.splitlines():
            line = line.strip()
            if "DirectShow video devices" in line:
                in_video_section = True
                continue
            if "DirectShow audio devices" in line:
                in_video_section = False
            if in_video_section:
                # Typical line: "  \"Integrated Camera\""
                if line.startswith("\"") and line.endswith("\""):
                    try:
                        names.append(line.strip('"'))
                    except Exception:
                        pass
                # Or: "  "HD Pro Webcam C920" (Alternative name: ...)"
                elif line.startswith('"'):
                    try:
                        # take the first quoted segment
                        first = line.split('"')[1]
                        if first:
                            names.append(first)
                    except Exception:
                        pass
        return list(dict.fromkeys(names))
    except Exception:
        return []


def start_camera():
    def _try_open_camera() -> tuple:
        # Try multiple backends and indices to find a working camera
        backend_options = [
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_ANY, "ANY"),
        ]
        indices = list(range(0, 3))

        forced_backend = _env_force_backend()
        forced_index = _env_force_index()
        forced_name = _env_force_dshow_name()
        if forced_backend != -1:
            # Place forced backend first
            readable = "FORCED"
            backend_options = [(forced_backend, readable)] + [
                (b, n) for (b, n) in backend_options if b != forced_backend
            ]
        if forced_index != -1:
            indices = [forced_index] + [i for i in indices if i != forced_index]

        # Try forced DSHOW name first if provided
        if forced_name:
            for backend_try in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                logger.info(f"… Trying DSHOW by name backend={backend_try}, name=video={forced_name}")
                cap = cv2.VideoCapture(f"video={forced_name}", backend_try)
                if cap.isOpened():
                    ok = False
                    for attempt in range(2):
                        ok, test_frame = cap.read()
                        if ok:
                            break
                        time.sleep(0.1)
                    if ok:
                        logger.info(f"✅ Camera opened by name using backend={backend_try}")
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        # Do not cache name-based opens; cache resolves to index/backend only
                        return cap, "DSHOW-NAME", -1
                    cap.release()

        # Try cached camera first
        cached_index, cached_backend = _load_cached_camera()
        if cached_index >= 0 and cached_backend >= 0 and forced_index == -1 and forced_backend == -1:
            logger.info(f"… Trying cached camera backend={cached_backend}, index={cached_index}")
            cap = cv2.VideoCapture(cached_index, cached_backend)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                ok = False
                # Retry a few times in case device is momentarily busy
                for attempt in range(2):
                    ok, test_frame = cap.read()
                    if ok:
                        break
                    time.sleep(0.1)
                if ok:
                    logger.info(f"✅ Using cached camera backend={cached_backend}, index={cached_index}")
                    return cap, str(cached_backend), cached_index
                # Cached entry invalid; clear and continue to probe
                cap.release()
                _clear_cached_camera()
        for backend, name in backend_options:
            for idx in indices:
                logger.info(f"… Trying camera backend={name}, index={idx}")
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    # Validate we can read a frame
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
                        _save_cached_camera(idx, backend)
                        return cap, name, idx
                    logger.info(f"   ↳ Opened but read failed on backend={name}, index={idx}; retrying others…")
                    cap.release()

        # As a last resort, try DSHOW by enumerated names
        dshow_names = _enumerate_dshow_devices()
        for dev_name in dshow_names:
            logger.info(f"… Trying DSHOW by enumerated name: video={dev_name}")
            cap = cv2.VideoCapture(f"video={dev_name}", cv2.CAP_DSHOW)
            if cap.isOpened():
                ok = False
                for attempt in range(2):
                    ok, test_frame = cap.read()
                    if ok:
                        break
                    time.sleep(0.1)
                if ok:
                    logger.info(f"Camera opened by name (DSHOW): {dev_name}")
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    return cap, "DSHOW-NAME", -1
                cap.release()
        return None, None, None

    cap, backend_name, cam_index = _try_open_camera()
    if cap is None:
        tried_range = "0-9" if _env_force_index() == -1 else str(_env_force_index())
        logger.error(f"❌ Cannot access camera: tried MSMF/DSHOW/ANY on indices {tried_range}", exc_info=True)
        logger.info("   Tips: close other apps using the camera, check Windows Privacy settings,"
              " try a different USB port, or update drivers.")
        return

    logger.info("✅ Camera started. Press 'q' to quit.")

    captures_dir = os.path.join(os.path.dirname(__file__), "captures")
    _ensure_captures_dir(captures_dir)

    videos_dir = os.path.join(os.path.dirname(__file__), "videos")
    _ensure_captures_dir(videos_dir)

    # Video writer state
    is_recording = False
    video_writer = None
    video_path = None
    audio_wav_path = None
    audio_recorder = None

    # Command routing
    router = CommandRouter()
    router.set_cooldown("take_picture", 2.0)
    router.set_cooldown("start_recording", 2.0)
    router.set_cooldown("stop_recording", 2.0)

    last_handled_text = ""
    last_snapshot_time = 0.0
    debounce_seconds = 2.0

    # Create the window explicitly (helps some Windows setups)
    try:
        cv2.namedWindow("RayBan Prototype", cv2.WINDOW_NORMAL)
    except Exception:
        pass

    recovery_attempted = False
    while True:
        ret, frame = cap.read()
        if not ret:
            # If the stream fails mid-run (e.g., other app grabbed device), try to recover once
            if not recovery_attempted:
                logger.error("⚠️  Camera read failed; attempting to recover with different backend/index…", exc_info=True)
                try:
                    cap.release()
                except Exception:
                    pass
                _clear_cached_camera()
                new_cap, backend_name, cam_index = _try_open_camera()
                if new_cap is not None:
                    cap = new_cap
                    recovery_attempted = True
                    continue
            break

        # Detect and draw faces
        faces = detect_faces(frame)
        draw_faces(frame, faces)

        # Get current audio text
        current_text = get_last_text()

        # Display the audio text with wrapping
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        margin_left = 20
        margin_bottom = 20
        available_width = frame.shape[1] - margin_left - 20
        lines = _wrap_text_to_width(current_text, font, font_scale, thickness, available_width, max_lines=3)

        # Compute dynamic overlay height based on number of lines
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
        if is_recording:
            rec_text = "REC"
            cv2.circle(frame, (20, 30), 8, (0, 0, 255), -1)
            cv2.putText(
                frame,
                rec_text,
                (40, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        # Write frame to video if recording
        if is_recording and video_writer is not None:
            video_writer.write(frame)

        # Command handling: take a picture and recording controls
        now = time.time()
        if current_text and current_text != last_handled_text:
            # Take picture
            if _should_take_picture(current_text) and router.should_run("take_picture"):
                saved_path = _save_snapshot(frame, captures_dir)
                last_snapshot_time = now
                last_handled_text = current_text
                router.mark_ran("take_picture")
                logger.info(f"📸 Saved snapshot: {saved_path}")

            # Start recording
            elif _should_start_recording(current_text) and not is_recording and router.should_run("start_recording"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_filename = f"video_{timestamp}.mp4"
                video_path = os.path.join(videos_dir, video_filename)
                audio_wav_path = os.path.join(videos_dir, f"video_{timestamp}.wav")

                # Figure out frame size and FPS
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 30.0

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                if not video_writer.isOpened():
                    logger.error("❌ Failed to start video writer", exc_info=True)
                else:
                    # Start mic recorder
                    try:
                        from audio import DEVICE_ID as MIC_DEVICE_ID
                    except Exception:
                        MIC_DEVICE_ID = 1
                    audio_recorder = AudioRecorder(device_id=MIC_DEVICE_ID, sample_rate=44100, channels=1)
                    try:
                        audio_recorder.start(audio_wav_path)
                    except Exception as _:
                        logger.error("⚠️  Failed to start mic; continuing video-only recording", exc_info=True)

                    is_recording = True
                    last_handled_text = current_text
                    router.mark_ran("start_recording")
                    logger.info(f"⏺️ Recording started: {video_path}")

            # Stop recording
            elif _should_stop_recording(current_text) and is_recording and router.should_run("stop_recording"):
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                # Stop mic and try to mux
                if audio_recorder is not None:
                    try:
                        audio_recorder.stop()
                    except Exception:
                        pass
                    audio_recorder = None
                muxed_path = ""
                if video_path and audio_wav_path and os.path.exists(audio_wav_path):
                    muxed_path = _try_mux_audio(video_path, audio_wav_path)
                last_handled_text = current_text
                router.mark_ran("stop_recording")
                if video_path:
                    if muxed_path:
                        logger.info(f"⏹️ Recording saved (with audio): {muxed_path}")
                    else:
                        logger.info(f"⏹️ Recording saved: {video_path}")
                        if audio_wav_path and os.path.exists(audio_wav_path):
                            logger.info(f"🎵 Audio WAV saved: {audio_wav_path}")

        cv2.imshow("RayBan Prototype", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # Ensure writer is closed on exit
    try:
        if 'video_writer' in locals() and video_writer is not None:
            video_writer.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    logger.info("✅ Camera stopped.")