#!/usr/bin/env python3
"""
Simple test script to generate nude version using diffusers directly.
"""

import os
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

def create_simple_mask(image):
    """Create a simple mask for clothing areas."""
    width, height = image.size
    mask = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(mask)
    
    # Create a mask for the torso and lower body areas
    # Torso area
    torso_x1 = int(width * 0.25)
    torso_y1 = int(height * 0.3)
    torso_x2 = int(width * 0.75)
    torso_y2 = int(height * 0.7)
    draw.rectangle([torso_x1, torso_y1, torso_x2, torso_y2], fill='white')
    
    # Lower body area
    lower_x1 = int(width * 0.3)
    lower_y1 = int(height * 0.6)
    lower_x2 = int(width * 0.7)
    lower_y2 = int(height * 0.9)
    draw.rectangle([lower_x1, lower_y1, lower_x2, lower_y2], fill='white')
    
    return mask

def main():
    print("🚀 Starting simple nude generation test...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Paths
    input_path = "data/input/sample_input.png"
    output_path = "data/output/nude_output.png"
    mask_path = "data/output/mask_output.png"
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        return
    
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    try:
        # Load the image
        image = Image.open(input_path).convert('RGB')
        print(f"📊 Original image size: {image.size}")
        
        # Resize to 512x512 for processing
        image = image.resize((512, 512))
        print(f"📊 Resized to: {image.size}")
        
        # Create mask
        mask = create_simple_mask(image)
        
        # Create output directory
        os.makedirs("data/output", exist_ok=True)
        
        # Save mask for reference
        mask.save(mask_path)
        print(f"✅ Mask saved: {mask_path}")
        
        # Initialize pipeline with CPU
        print("🔧 Loading Stable Diffusion inpainting pipeline...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float32,  # Use float32 for CPU
            safety_checker=None,
            requires_safety_checker=False
        )
        
        print("⚠️  Using CPU mode - this will take several minutes...")
        
        # Generate
        print("🎨 Generating nude version...")
        result = pipe(
            prompt="nude body, realistic skin, natural lighting",
            image=image,
            mask_image=mask,
            negative_prompt="clothing, clothes, fabric, low quality, blurry",
            num_inference_steps=20,  # Reduced for faster testing
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(42)
        ).images[0]
        
        # Save result
        result.save(output_path)
        print(f"✅ Generated image saved: {output_path}")
        print("🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

