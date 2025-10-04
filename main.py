#!/usr/bin/env python3
"""
Legacy main entry point for RayBand voice camera.
This file maintains backward compatibility with the old structure.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main entry point - redirects to new package structure."""
    print("🔄 RayBand Voice Camera - Legacy Mode")
    print("   Redirecting to new package structure...")
    print("   For better organization, use: python -m rayband.cli.main")
    print()
    
    try:
        # Import and run the new main
        from rayband.cli.main import main as new_main
        new_main()
    except ImportError as e:
        print(f"❌ Error importing new structure: {e}")
        print("   Falling back to legacy imports...")
        
        # Fallback to old structure
        try:
            import threading
            from audio import start_audio_recognition
            from video import start_camera
            from config import MODEL_PATH
            
            print("⚠️  Using legacy code structure")
            print("   Consider upgrading to the new package structure")
            
            # Start audio recognition in a separate thread
            audio_thread = threading.Thread(target=start_audio_recognition, args=(MODEL_PATH,), daemon=True)
            audio_thread.start()
            
            # Start camera feed with overlay
            start_camera()
            
        except ImportError as legacy_e:
            print(f"❌ Legacy import also failed: {legacy_e}")
            print("   Please check your installation and dependencies")
            sys.exit(1)


if __name__ == "__main__":
    main()