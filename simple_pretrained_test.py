#!/usr/bin/env python3
"""
Simple test for pre-trained GAN implementation.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🧪 SIMPLE PRE-TRAINED GAN TEST")
    print("=" * 50)
    
    try:
        # Test basic import
        print("1. Testing imports...")
        from nude_generator.core.pretrained_gan import load_pretrained_nude_gan
        print("✅ Imports successful")
        
        # Test model loading
        print("2. Loading model...")
        gan = load_pretrained_nude_gan(model_type="dcgan_nude", device="cpu")
        print("✅ Model loaded")
        
        # Test model info
        print("3. Getting model info...")
        info = gan.get_model_info()
        print(f"✅ Model: {info['model_type']}")
        print(f"   Device: {info['device']}")
        print(f"   Parameters: {info['generator_params']:,}")
        
        # Test random generation
        print("4. Generating random image...")
        random_image = gan.generate_nude(num_images=1, seed=42)
        print(f"✅ Generated image: {random_image.size}")
        
        # Save image
        os.makedirs("data/output", exist_ok=True)
        random_image.save("data/output/simple_test_output.png")
        print("✅ Image saved to data/output/simple_test_output.png")
        
        # Test with input image if available
        input_path = "data/input/sample_input.png"
        if os.path.exists(input_path):
            print("5. Testing with input image...")
            nude_version = gan.generate_nude(input_image=input_path, seed=42)
            nude_version.save("data/output/simple_test_nude.png")
            print("✅ Nude version saved to data/output/simple_test_nude.png")
        else:
            print("5. Input image not found, skipping")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The pre-trained GAN implementation is working correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

