#!/usr/bin/env python3
"""
Lightweight test for real nude generator with memory optimization.
"""

import os
import sys
from pathlib import Path
import gc
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🔥 LIGHTWEIGHT REAL NUDE GENERATOR TEST")
    print("=" * 50)
    
    try:
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("1. Testing imports...")
        from nude_generator.core.real_nude_generator import RealNudeGenerator
        print("✅ Imports successful")
        
        # Create output directory
        os.makedirs("data/output", exist_ok=True)
        
        print("2. Loading NSFW model with memory optimization...")
        
        # Initialize with memory optimizations
        generator = RealNudeGenerator(device="cpu")
        
        # Enable memory efficient settings
        if hasattr(generator.pipeline, "enable_attention_slicing"):
            generator.pipeline.enable_attention_slicing()
        # Skip CPU offload as it requires accelerator
        
        print("✅ Model loaded with optimizations")
        
        print("3. Testing lightweight text-to-nude generation...")
        
        # Generate with minimal settings for memory efficiency
        nude_image = generator.generate_nude_from_text(
            prompt="woman, nude, simple",
            num_inference_steps=10,  # Minimal steps
            width=256,  # Smaller size
            height=256,
            seed=42
        )
        
        output_path = "data/output/lightweight_nude.png"
        nude_image.save(output_path)
        print(f"✅ Generated image saved to {output_path}")
        
        # Examine the result
        examine_image(output_path)
        
        # Clear memory
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n🎉 LIGHTWEIGHT TEST COMPLETED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def examine_image(image_path):
    """Examine the generated image and provide detailed analysis."""
    print(f"\n🔍 EXAMINING: {image_path}")
    
    try:
        from PIL import Image
        import numpy as np
        
        if not os.path.exists(image_path):
            print("❌ Image file not found")
            return
        
        img = Image.open(image_path)
        img_array = np.array(img)
        
        print(f"   📏 Size: {img.size}")
        print(f"   🎨 Mode: {img.mode}")
        print(f"   💾 File size: {os.path.getsize(image_path) / 1024:.1f} KB")
        
        # Analyze image content
        if len(img_array.shape) == 3:
            avg_color = img_array.mean(axis=(0, 1))
            print(f"   🌈 Average RGB: ({avg_color[0]:.1f}, {avg_color[1]:.1f}, {avg_color[2]:.1f})")
            
            # Check color variation
            std_dev = img_array.std()
            print(f"   📊 Color variation: {std_dev:.1f}")
            
            # Analyze if it's a real image
            if std_dev < 5:
                print("   ⚠️  WARNING: Very low variation - likely solid color or noise")
                quality = "FAILED"
            elif std_dev < 15:
                print("   ⚠️  WARNING: Low variation - might be simple pattern")
                quality = "POOR"
            elif std_dev < 30:
                print("   ✅ Moderate variation - likely has some content")
                quality = "FAIR"
            else:
                print("   ✅ Good variation - likely realistic image")
                quality = "GOOD"
            
            # Check for skin tones (nude images should have skin-like colors)
            skin_like = check_skin_tones(avg_color)
            print(f"   👤 Skin-like colors: {'✅ Yes' if skin_like else '❌ No'}")
            
            # Overall assessment
            print(f"\n📋 QUALITY ASSESSMENT: {quality}")
            
            if quality in ["GOOD", "FAIR"] and skin_like:
                print("🎉 SUCCESS: This appears to be a real nude image!")
            elif quality in ["GOOD", "FAIR"]:
                print("⚠️  PARTIAL: Real image but may not be nude")
            else:
                print("❌ FAILED: Not a realistic image")
        
    except Exception as e:
        print(f"   ❌ Error analyzing image: {e}")


def check_skin_tones(avg_color):
    """Check if the average color resembles skin tones."""
    r, g, b = avg_color
    
    # Skin tone ranges (approximate)
    skin_ranges = [
        # Light skin
        (200, 255, 180, 255, 150, 220),
        # Medium skin  
        (150, 200, 120, 180, 80, 150),
        # Dark skin
        (80, 150, 60, 120, 40, 100)
    ]
    
    for r_min, r_max, g_min, g_max, b_min, b_max in skin_ranges:
        if r_min <= r <= r_max and g_min <= g <= g_max and b_min <= b <= b_max:
            return True
    
    return False


if __name__ == "__main__":
    main()

