#!/usr/bin/env python3
"""
Test script for the real nude generator using Hugging Face NSFW model.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🔥 TESTING REAL NUDE GENERATOR (HUGGING FACE NSFW MODEL)")
    print("=" * 70)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from nude_generator.core.real_nude_generator import load_real_nude_generator
        print("✅ Imports successful")
        
        # Load the real model
        print("2. Loading real NSFW Stable Diffusion model...")
        print("   This may take a few minutes to download the model...")
        generator = load_real_nude_generator(device="cpu")
        print("✅ Real NSFW model loaded successfully")
        
        # Get model info
        print("3. Getting model information...")
        info = generator.get_model_info()
        print(f"✅ Model: {info['model_id']}")
        print(f"   Device: {info['device']}")
        print(f"   Capabilities: {', '.join(info['capabilities'])}")
        
        # Create output directory
        os.makedirs("data/output", exist_ok=True)
        
        # Test 1: Text-to-nude generation
        print("4. Testing text-to-nude generation...")
        nude_image = generator.generate_nude_from_text(
            prompt="beautiful woman, artistic nude, studio lighting, realistic",
            seed=42,
            num_inference_steps=20  # Reduced for faster testing
        )
        nude_image.save("data/output/real_text_to_nude.png")
        print("✅ Text-to-nude image saved to data/output/real_text_to_nude.png")
        
        # Test 2: Clothing removal from input image
        input_path = "data/input/sample_input.png"
        if os.path.exists(input_path):
            print("5. Testing clothing removal from input image...")
            nude_version = generator.remove_clothes_from_image(
                input_image=input_path,
                prompt="nude, naked, beautiful skin, realistic body",
                seed=42,
                num_inference_steps=20
            )
            nude_version.save("data/output/real_nude_removal.png")
            print("✅ Clothing removal result saved to data/output/real_nude_removal.png")
            
            # Create comparison
            create_comparison_image(
                input_path,
                "data/output/real_nude_removal.png",
                "data/output/real_comparison.png"
            )
            print("✅ Comparison saved to data/output/real_comparison.png")
        else:
            print("5. Input image not found, skipping clothing removal test")
        
        # Test 3: Generate variations
        print("6. Testing nude variations generation...")
        variations = generator.generate_nude_variations(
            base_image=input_path if os.path.exists(input_path) else None,
            num_variations=2,
            seed=42
        )
        
        for i, variation in enumerate(variations):
            variation.save(f"data/output/real_variation_{i+1}.png")
            print(f"   ✅ Variation {i+1} saved")
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("🔍 Now examining the generated images...")
        
        # Examine generated images
        examine_generated_images()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def create_comparison_image(input_path, output_path, comparison_path):
    """Create a side-by-side comparison image."""
    try:
        from PIL import Image
        
        # Load images
        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')
        
        # Resize to same height
        height = 512
        input_img = input_img.resize((512, height))
        output_img = output_img.resize((512, height))
        
        # Create comparison
        total_width = 1024 + 20  # 20px gap
        comparison = Image.new('RGB', (total_width, height + 60), 'white')
        
        # Paste images
        comparison.paste(input_img, (0, 30))
        comparison.paste(output_img, (532, 30))
        
        comparison.save(comparison_path)
        
    except Exception as e:
        print(f"Warning: Could not create comparison image: {e}")


def examine_generated_images():
    """Examine and describe the generated images."""
    print("\n🔍 EXAMINING GENERATED IMAGES")
    print("=" * 50)
    
    output_dir = Path("data/output")
    
    # Check each generated image
    for image_file in ["real_text_to_nude.png", "real_nude_removal.png"]:
        image_path = output_dir / image_file
        if image_path.exists():
            print(f"\n📸 Examining {image_file}:")
            
            try:
                from PIL import Image
                img = Image.open(image_path)
                
                print(f"   Size: {img.size}")
                print(f"   Mode: {img.mode}")
                print(f"   File size: {image_path.stat().st_size / 1024:.1f} KB")
                
                # Basic image analysis
                img_array = np.array(img) if 'np' in globals() else None
                if img_array is not None:
                    avg_color = img_array.mean(axis=(0, 1))
                    print(f"   Average RGB: ({avg_color[0]:.1f}, {avg_color[1]:.1f}, {avg_color[2]:.1f})")
                    
                    # Check if it's not just a solid color
                    std_dev = img_array.std()
                    print(f"   Color variation (std dev): {std_dev:.1f}")
                    
                    if std_dev < 10:
                        print("   ⚠️  WARNING: Low color variation - might be solid color")
                    else:
                        print("   ✅ Good color variation - likely a real image")
                
            except Exception as e:
                print(f"   ❌ Error examining image: {e}")
        else:
            print(f"\n❌ {image_file} not found")
    
    print("\n📋 SUMMARY:")
    print("Check the generated images in data/output/ to verify quality")
    print("Real NSFW Stable Diffusion should produce high-quality nude images")


if __name__ == "__main__":
    # Import numpy for image analysis
    try:
        import numpy as np
    except ImportError:
        print("Warning: numpy not available for image analysis")
    
    main()

