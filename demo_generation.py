#!/usr/bin/env python3
"""
Demo script to show the nude generation concept with the provided image.
This creates a demonstration output showing the mask and concept.
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

def create_demonstration():
    """Create a demonstration showing the input, mask, and concept output."""
    
    # Paths
    input_path = "data/input/sample_input.png"
    output_path = "data/output/nude_output_demo.png"
    mask_path = "data/output/mask_output.png"
    
    print("🎨 Creating demonstration of nude generation concept...")
    
    # Load input image
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        return
    
    original = Image.open(input_path).convert('RGB')
    print(f"📊 Original image size: {original.size}")
    
    # Resize for processing
    width, height = 512, 512
    image = original.resize((width, height))
    
    # Create clothing mask
    mask = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(mask)
    
    # Define clothing areas to be "removed"
    # Torso area (where the lingerie is)
    torso_x1 = int(width * 0.25)
    torso_y1 = int(height * 0.25)
    torso_x2 = int(width * 0.75)
    torso_y2 = int(height * 0.65)
    draw.rectangle([torso_x1, torso_y1, torso_x2, torso_y2], fill='white')
    
    # Lower body area
    lower_x1 = int(width * 0.35)
    lower_y1 = int(height * 0.55)
    lower_x2 = int(width * 0.65)
    lower_y2 = int(height * 0.85)
    draw.rectangle([lower_x1, lower_y1, lower_x2, lower_y2], fill='white')
    
    # Smooth the mask
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Save the mask
    mask.save(mask_path)
    print(f"✅ Mask saved: {mask_path}")
    
    # Create a demonstration output
    # This simulates what the AI would do - replace masked areas with skin tone
    demo_output = image.copy()
    
    # Convert to numpy for easier manipulation
    img_array = np.array(demo_output)
    mask_array = np.array(mask.convert('L'))
    
    # Define skin tone color (peachy/flesh tone)
    skin_color = [220, 180, 140]  # RGB values for skin tone
    
    # Apply skin tone to masked areas
    for i in range(height):
        for j in range(width):
            if mask_array[i, j] > 128:  # White areas in mask
                # Blend with skin tone
                blend_factor = mask_array[i, j] / 255.0
                for c in range(3):
                    img_array[i, j, c] = int(
                        img_array[i, j, c] * (1 - blend_factor) + 
                        skin_color[c] * blend_factor
                    )
    
    # Convert back to PIL Image
    demo_output = Image.fromarray(img_array)
    
    # Add text overlay to show this is a demo
    draw_demo = ImageDraw.Draw(demo_output)
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add demo text
    demo_text = "DEMO OUTPUT"
    text_color = (255, 0, 0)  # Red text
    draw_demo.text((10, 10), demo_text, fill=text_color, font=font)
    draw_demo.text((10, 30), "Concept demonstration", fill=text_color, font=font)
    draw_demo.text((10, 50), "AI would generate realistic", fill=text_color, font=font)
    draw_demo.text((10, 70), "nude body in masked areas", fill=text_color, font=font)
    
    # Save the demonstration output
    demo_output.save(output_path)
    print(f"✅ Demo output saved: {output_path}")
    
    # Create a side-by-side comparison
    comparison_path = "data/output/comparison.png"
    comparison = Image.new('RGB', (width * 3, height), 'white')
    comparison.paste(image, (0, 0))
    comparison.paste(mask, (width, 0))
    comparison.paste(demo_output, (width * 2, 0))
    
    # Add labels
    draw_comp = ImageDraw.Draw(comparison)
    draw_comp.text((10, height - 30), "Original", fill=(0, 0, 0), font=font)
    draw_comp.text((width + 10, height - 30), "Mask", fill=(0, 0, 0), font=font)
    draw_comp.text((width * 2 + 10, height - 30), "Demo Output", fill=(0, 0, 0), font=font)
    
    comparison.save(comparison_path)
    print(f"✅ Comparison saved: {comparison_path}")
    
    print("🎉 Demonstration completed!")
    print(f"📁 Files created:")
    print(f"   - Input: {input_path}")
    print(f"   - Mask: {mask_path}")
    print(f"   - Demo Output: {output_path}")
    print(f"   - Comparison: {comparison_path}")
    
    return True

if __name__ == "__main__":
    create_demonstration()

