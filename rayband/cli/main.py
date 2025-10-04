"""
Main entry point for RayBand voice camera.
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import rayband
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rayband.core.camera import CameraController
from rayband.utils.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the RayBand voice camera application."""
    logger.info("🚀 Starting RayBand Voice Camera...")
    
    try:
        # Create and start camera controller
        controller = CameraController()
        controller.start()
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
