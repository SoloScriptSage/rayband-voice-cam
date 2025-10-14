"""
Diagnostic tool to check Vosk model integrity and test loading.
Run this to diagnose model loading issues.
"""

import os
import sys
from pathlib import Path

def check_directory_structure(model_path):
    """Check if model has correct directory structure."""
    print(f"\n📁 Checking: {model_path}")
    
    if not os.path.exists(model_path):
        print("   ❌ Directory does not exist!")
        return False
    
    print("   ✅ Directory exists")
    
    # Check for required Vosk files/directories
    required = {
        'am': 'Acoustic Model directory',
        'conf': 'Configuration directory', 
        'graph': 'Graph directory',
        'ivector': 'I-vector directory (optional)'
    }
    
    all_good = True
    for item, description in required.items():
        item_path = os.path.join(model_path, item)
        if os.path.exists(item_path):
            print(f"   ✅ {item}/ - {description}")
        else:
            if item == 'ivector':
                print(f"   ⚠️  {item}/ - {description} (missing, but optional)")
            else:
                print(f"   ❌ {item}/ - {description} (MISSING!)")
                all_good = False
    
    # List actual contents
    try:
        contents = os.listdir(model_path)
        print(f"\n   📋 Actual contents: {', '.join(contents)}")
    except Exception as e:
        print(f"   ❌ Cannot list contents: {e}")
        return False
    
    return all_good

def test_vosk_import():
    """Test if Vosk can be imported."""
    print("\n🔍 Testing Vosk import...")
    try:
        import vosk
        print(f"   ✅ Vosk imported successfully (version may vary)")
        return True
    except ImportError as e:
        print(f"   ❌ Cannot import Vosk: {e}")
        print("   💡 Install with: pip install vosk")
        return False

def test_model_loading(model_path):
    """Try to actually load the model."""
    print(f"\n🧪 Testing model loading: {model_path}")
    
    try:
        import vosk
        print("   ⏳ Loading model (this may take a moment)...")
        model = vosk.Model(model_path)
        print("   ✅ Model loaded successfully!")
        del model  # Clean up
        return True
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        print(f"   📝 Error type: {type(e).__name__}")
        return False

def main():
    print("=" * 60)
    print("🔧 RayBand Model Diagnostic Tool")
    print("=" * 60)
    
    # Get project root
    script_path = Path(__file__).resolve()
    # If script is in project root, models are in ./models
    # If script is elsewhere, try to find models
    project_root = script_path.parent
    
    # Try to find models directory
    possible_locations = [
        project_root / "models",
        project_root.parent / "models",
        Path.cwd() / "models",
    ]
    
    models_dir = None
    for loc in possible_locations:
        if loc.exists():
            models_dir = loc
            break
    
    if models_dir is None:
        print("\n❌ Cannot find models directory!")
        print("   Tried:")
        for loc in possible_locations:
            print(f"     - {loc}")
        return 1
    
    print(f"\n✅ Found models directory: {models_dir}")
    
    # Test Vosk import
    if not test_vosk_import():
        return 1
    
    # Check each model
    models = {
        "English": models_dir / "vosk-model-en-us-0.22",
        "Ukrainian": models_dir / "vosk-model-uk-v3"
    }
    
    results = {}
    for name, path in models.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {name} Model")
        print(f"{'=' * 60}")
        
        # Check structure
        structure_ok = check_directory_structure(str(path))
        
        # Try loading if structure is OK
        if structure_ok:
            loading_ok = test_model_loading(str(path))
            results[name] = loading_ok
        else:
            results[name] = False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 Summary")
    print(f"{'=' * 60}")
    
    all_ok = True
    for name, ok in results.items():
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"   {name}: {status}")
        if not ok:
            all_ok = False
    
    if all_ok:
        print("\n🎉 All models are working correctly!")
        print("   You should be able to run the camera application.")
        return 0
    else:
        print("\n⚠️  Some models have issues")
        print("\n💡 Possible fixes:")
        print("   1. Re-download the model from https://alphacephei.com/vosk/models")
        print("   2. Make sure you extracted the FULL zip file")
        print("   3. Check that you have enough disk space")
        print("   4. Verify the model files aren't corrupted")
        return 1

if __name__ == "__main__":
    sys.exit(main())