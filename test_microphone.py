import sounddevice as sd

# List all audio devices
print("Available audio devices:")
print(sd.query_devices())

print("\n" + "="*50)
print("Testing each INPUT device for 3 seconds...")
print("Speak into your microphone!")
print("="*50 + "\n")

# Test each input device
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:  # Only input devices
        print(f"\nTesting Device {i}: {device['name']}")
        try:
            # Record 3 seconds
            print("  Recording... speak now!")
            recording = sd.rec(
                int(3 * device['default_samplerate']),
                samplerate=int(device['default_samplerate']),
                channels=1,
                device=i,
                dtype='int16'
            )
            sd.wait()
            
            # Check if we got audio data
            max_volume = abs(recording).max()
            print(f"  ✓ Max volume detected: {max_volume}")
            
            if max_volume > 100:  # Threshold for actual audio
                print(f"  ✅ GOOD AUDIO DETECTED on device {i}!")
            else:
                print(f"  ⚠️  Very low volume - might not be the right device")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

print("\n" + "="*50)
print("Use the device number with GOOD AUDIO in your audio.py")