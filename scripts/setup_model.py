#!/usr/bin/env python3
"""
Setup script for downloading and configuring Vosk speech recognition models.
"""

import os
import sys
import urllib.request
import zipfile
import argparse
from pathlib import Path


def download_model(model_name: str = "vosk-model-en-us-0.22", model_dir: str = "model") -> bool:
    """Download a Vosk model from the official repository."""
    model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    zip_path = f"{model_name}.zip"
    
    print(f"📥 Downloading {model_name}...")
    print(f"   URL: {model_url}")
    
    try:
        # Download the model
        urllib.request.urlretrieve(model_url, zip_path)
        print(f"✅ Downloaded {zip_path}")
        
        # Extract the model
        print(f"📦 Extracting to {model_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        print(f"🗑️  Removed {zip_path}")
        
        # Verify extraction
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            print(f"✅ Model successfully extracted to {model_path}")
            return True
        else:
            print(f"❌ Model extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False


def main():
    """Main function for model setup."""
    parser = argparse.ArgumentParser(description="Setup Vosk speech recognition models")
    parser.add_argument("--model", default="vosk-model-en-us-0.22", 
                       help="Model name to download (default: vosk-model-en-us-0.22)")
    parser.add_argument("--dir", default="model", 
                       help="Directory to extract model to (default: model)")
    
    args = parser.parse_args()
    
    print("🎤 RayBand Voice Camera - Model Setup")
    print("=" * 50)
    
    # Create model directory if it doesn't exist
    os.makedirs(args.dir, exist_ok=True)
    
    # Download the model
    success = download_model(args.model, args.dir)
    
    if success:
        print("\n🎉 Model setup complete!")
        print(f"   Model path: {os.path.join(args.dir, args.model)}")
        print("\nYou can now run the RayBand voice camera:")
        print("   python -m rayband.cli.main")
    else:
        print("\n❌ Model setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
