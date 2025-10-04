"""
RayBand Voice Camera - Smart glasses with voice control and computer vision.

A voice-controlled camera system for smart glasses featuring:
- Real-time speech recognition
- Face detection and recognition  
- Voice commands for photo/video capture
- Computer vision overlays
"""

__version__ = "1.0.0"
__author__ = "RayBand Team"
__email__ = "team@rayband.dev"

# Core modules
from . import core
from . import utils
from . import hardware

__all__ = ["core", "utils", "hardware"]
