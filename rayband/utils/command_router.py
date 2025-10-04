"""
Voice command routing and processing for RayBand voice camera.
"""

import time
from typing import Dict, List


class CommandRouter:
    """Handles voice command recognition and cooldown management."""
    
    def __init__(self):
        self._last_command_text = ""
        self._last_run_at: Dict[str, float] = {}
        self._cooldowns: Dict[str, float] = {}
        
        # Default command triggers
        self._picture_triggers = [
            "take a picture", "take picture", "take photo", "take a photo",
            "snap a photo", "snap picture", "capture photo", "capture picture",
            "screenshot"
        ]
        
        self._start_recording_triggers = [
            "start recording", "start video", "begin recording", "record video"
        ]
        
        self._stop_recording_triggers = [
            "stop recording", "end recording", "stop video", "finish recording"
        ]

    def set_cooldown(self, command_name: str, seconds: float) -> None:
        """Set cooldown period for a command."""
        self._cooldowns[command_name] = seconds

    def should_run(self, command_name: str) -> bool:
        """Check if a command can run (not in cooldown)."""
        cooldown = self._cooldowns.get(command_name, 0.0)
        last = self._last_run_at.get(command_name, 0.0)
        return (time.time() - last) >= cooldown

    def mark_ran(self, command_name: str) -> None:
        """Mark that a command was executed."""
        self._last_run_at[command_name] = time.time()

    def should_take_picture(self, recognized_text: str) -> bool:
        """Check if the text contains picture-taking commands."""
        if not recognized_text:
            return False
        text = recognized_text.lower()
        return any(trigger in text for trigger in self._picture_triggers)

    def should_start_recording(self, recognized_text: str) -> bool:
        """Check if the text contains start recording commands."""
        if not recognized_text:
            return False
        text = recognized_text.lower()
        return any(trigger in text for trigger in self._start_recording_triggers)

    def should_stop_recording(self, recognized_text: str) -> bool:
        """Check if the text contains stop recording commands."""
        if not recognized_text:
            return False
        text = recognized_text.lower()
        return any(trigger in text for trigger in self._stop_recording_triggers)
    
    def process_command(self, recognized_text: str) -> str:
        """Process recognized text and return command type."""
        if not recognized_text or recognized_text == self._last_command_text:
            return ""
        
        if self.should_take_picture(recognized_text):
            return "take_picture"
        elif self.should_start_recording(recognized_text):
            return "start_recording"
        elif self.should_stop_recording(recognized_text):
            return "stop_recording"
        
        self._last_command_text = recognized_text
        return ""
