#!/usr/bin/env python3
"""
Test script to generate nude version of the provided sample image.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Import only the basic generator to avoid mediapipe dependency
    import sys
    import os
    from pathlib import Path
    from PIL import Image, ImageDraw
    import torch
    import numpy as np
    from diffusers import StableDiffusionInpaintPipeline
    
    print("🚀 Starting nude generation test with real image...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Paths
    input_path = "data/input/sample_input.png"
    output_path = "data/output/nude_output.png"
    mask_path = "data/output/mask_output.png"
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        sys.exit(1)
    
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    # Initialize generator with CPU for testing (to avoid CUDA memory issues)
    print("🔧 Initializing generator...")
    generator = NudeGenerator(
        model_id="runwayml/stable-diffusion-inpainting",  # Smaller model for testing
        device="cpu"  # Use CPU to avoid potential CUDA issues
    )
    
    print("🎨 Generating nude version...")
    print("⚠️  Note: Using CPU mode for compatibility - this will be slower")
    
    # Generate with fast settings for testing
    result = generator.generate_nude(
        image=input_path,
        prompt="nude body, realistic skin, natural lighting",
        negative_prompt="clothing, clothes, fabric, low quality, blurry",
        num_inference_steps=20,  # Reduced for faster testing
        guidance_scale=6.0,
        seed=42
    )
    
    # Create output directory if it doesn't exist
    os.makedirs("data/output", exist_ok=True)
    
    # Save result
    result.save(output_path)
    print(f"✅ Generated image saved: {output_path}")
    
    # Also create and save the mask for reference
    original_image = Image.open(input_path)
    mask = generator.create_clothing_mask(original_image)
    mask.save(mask_path)
    print(f"✅ Mask saved: {mask_path}")
    
    print("🎉 Test completed successfully!")
    print(f"📊 Input size: {original_image.size}")
    print(f"📊 Output size: {result.size}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error during generation: {e}")
    import traceback
    traceback.print_exc()

