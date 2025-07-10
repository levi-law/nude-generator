#!/usr/bin/env python3
"""
Minimal test for real nude generator - text-to-nude only.
"""

import os
import sys
from pathlib import Path
import gc
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🔥 MINIMAL REAL NUDE GENERATOR TEST")
    print("=" * 50)
    
    try:
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("1. Loading NSFW Stable Diffusion model...")
        
        # Import and load only the main pipeline
        from diffusers import StableDiffusionPipeline
        
        model_id = "Kernel/sd-nsfw"
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for CPU
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline = pipeline.to("cpu")
        
        # Enable memory optimization
        pipeline.enable_attention_slicing()
        
        print("✅ NSFW model loaded successfully")
        
        # Create output directory
        os.makedirs("data/output", exist_ok=True)
        
        print("2. Generating nude image...")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Generate with minimal settings
        prompt = "beautiful woman, nude, naked, artistic, realistic skin"
        negative_prompt = "clothes, clothing, dressed, shirt, pants"
        
        print(f"   Prompt: {prompt}")
        
        # Generate image
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=8,  # Very minimal steps
                width=256,
                height=256,
                guidance_scale=7.5
            )
        
        # Save immediately
        image = result.images[0]
        output_path = "data/output/minimal_real_nude.png"
        image.save(output_path)
        
        print(f"✅ Image saved to {output_path}")
        
        # Quick analysis
        analyze_image(output_path)
        
        # Clean up
        del pipeline
        del result
        del image
        gc.collect()
        
        print("\n🎉 MINIMAL TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def analyze_image(image_path):
    """Quick image analysis."""
    print(f"\n🔍 ANALYZING: {image_path}")
    
    try:
        from PIL import Image
        import numpy as np
        
        if not os.path.exists(image_path):
            print("❌ Image not found")
            return
        
        img = Image.open(image_path)
        img_array = np.array(img)
        
        print(f"   Size: {img.size}")
        print(f"   File size: {os.path.getsize(image_path) / 1024:.1f} KB")
        
        if len(img_array.shape) == 3:
            avg_color = img_array.mean(axis=(0, 1))
            std_dev = img_array.std()
            
            print(f"   Average RGB: ({avg_color[0]:.1f}, {avg_color[1]:.1f}, {avg_color[2]:.1f})")
            print(f"   Color variation: {std_dev:.1f}")
            
            # Assessment
            if std_dev < 5:
                print("   ❌ FAILED: Solid color or noise")
            elif std_dev < 15:
                print("   ⚠️  POOR: Very low variation")
            elif std_dev < 30:
                print("   ✅ FAIR: Some content visible")
            else:
                print("   ✅ GOOD: Rich content")
            
            # Check for skin-like colors
            r, g, b = avg_color
            if 80 <= r <= 255 and 60 <= g <= 200 and 40 <= b <= 180:
                print("   👤 ✅ Skin-like colors detected")
            else:
                print("   👤 ❌ No skin-like colors")
        
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    main()

