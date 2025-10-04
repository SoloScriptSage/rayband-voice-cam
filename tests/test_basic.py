"""
Basic tests for RayBand voice camera.
"""

import unittest
from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rayband.utils.config import Config
from rayband.utils.command_router import CommandRouter


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test that config can be created."""
        config = Config()
        self.assertIsInstance(config.MODEL_PATH, str)
        self.assertIsInstance(config.AUDIO_DEVICE_ID, int)
        self.assertIsInstance(config.MIC_SAMPLERATE, int)


class TestCommandRouter(unittest.TestCase):
    """Test command routing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = CommandRouter()
    
    def test_picture_commands(self):
        """Test picture-taking command recognition."""
        test_commands = [
            "take a picture",
            "take picture", 
            "snap a photo",
            "capture photo",
            "screenshot"
        ]
        
        for command in test_commands:
            with self.subTest(command=command):
                self.assertTrue(self.router.should_take_picture(command))
    
    def test_recording_commands(self):
        """Test recording command recognition."""
        start_commands = ["start recording", "begin recording", "record video"]
        stop_commands = ["stop recording", "end recording", "finish recording"]
        
        for command in start_commands:
            with self.subTest(command=command):
                self.assertTrue(self.router.should_start_recording(command))
        
        for command in stop_commands:
            with self.subTest(command=command):
                self.assertTrue(self.router.should_stop_recording(command))
    
    def test_cooldown_management(self):
        """Test command cooldown functionality."""
        self.router.set_cooldown("test_command", 1.0)
        
        # Should be able to run initially
        self.assertTrue(self.router.should_run("test_command"))
        
        # Mark as run
        self.router.mark_ran("test_command")
        
        # Should not be able to run immediately after
        self.assertFalse(self.router.should_run("test_command"))


if __name__ == "__main__":
    unittest.main()
