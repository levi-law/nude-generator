#!/usr/bin/env python3
"""
Example usage of the Nude Generator.

This script demonstrates how to use the nude generator
with different configurations and options.
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from nude_generator import NudeGenerator
from config import get_generation_params


def create_sample_image():
    """Create a sample clothed person image for demonstration."""
    # Create a simple figure with clothes
    image = Image.new('RGB', (512, 512), 'lightblue')
    draw = ImageDraw.Draw(image)
    
    # Head
    draw.ellipse([206, 50, 306, 150], fill='peachpuff')
    
    # Torso with shirt
    draw.rectangle([180, 150, 332, 300], fill='red')  # Red shirt
    draw.text((220, 200), "SHIRT", fill='white')
    
    # Arms
    draw.rectangle([150, 170, 180, 280], fill='peachpuff')  # Left arm
    draw.rectangle([332, 170, 362, 280], fill='peachpuff')  # Right arm
    
    # Legs with pants
    draw.rectangle([190, 300, 240, 450], fill='blue')  # Left leg
    draw.rectangle([272, 300, 322, 450], fill='blue')  # Right leg
    draw.text((220, 350), "PANTS", fill='white')
    
    return image


def example_basic_usage():
    """Example 1: Basic nude generation."""
    print("Example 1: Basic Nude Generation")
    print("-" * 40)
    
    # Create sample image
    sample_image = create_sample_image()
    sample_image.save("sample_clothed.png")
    print("✅ Sample clothed image created: sample_clothed.png")
    
    # Initialize generator
    generator = NudeGenerator()
    print("✅ Nude generator initialized")
    
    # Generate nude version
    print("🔄 Generating nude version...")
    result = generator.generate_nude(
        image=sample_image,
        prompt="nude body, realistic skin, natural lighting",
        negative_prompt="clothing, clothes, shirt, pants, low quality",
        num_inference_steps=30,
        seed=42
    )
    
    # Save result
    result.save("example_basic_nude.png")
    print("✅ Basic nude generation completed: example_basic_nude.png")
    print()


def example_custom_mask():
    """Example 2: Custom mask regions."""
    print("Example 2: Custom Mask Regions")
    print("-" * 40)
    
    sample_image = create_sample_image()
    generator = NudeGenerator()
    
    # Define custom regions to modify (only torso)
    custom_regions = [(180, 150, 332, 300)]  # Only the shirt area
    
    print("🔄 Generating with custom mask (torso only)...")
    result = generator.generate_nude(
        image=sample_image,
        mask_regions=custom_regions,
        prompt="nude torso, realistic skin",
        negative_prompt="shirt, clothing, low quality",
        num_inference_steps=25,
        seed=123
    )
    
    result.save("example_custom_mask.png")
    print("✅ Custom mask generation completed: example_custom_mask.png")
    print()


def example_different_qualities():
    """Example 3: Different quality settings."""
    print("Example 3: Different Quality Settings")
    print("-" * 40)
    
    sample_image = create_sample_image()
    generator = NudeGenerator()
    
    qualities = ["fast", "default", "high"]
    
    for quality in qualities:
        print(f"🔄 Generating with {quality} quality...")
        
        # Get parameters for this quality
        params = get_generation_params(quality)
        
        result = generator.generate_nude(
            image=sample_image,
            **params,
            seed=456
        )
        
        result.save(f"example_{quality}_quality.png")
        print(f"✅ {quality.capitalize()} quality completed: example_{quality}_quality.png")
    
    print()


def example_batch_processing():
    """Example 4: Batch processing."""
    print("Example 4: Batch Processing")
    print("-" * 40)
    
    # Create multiple sample images
    input_dir = Path("batch_input")
    input_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        sample = create_sample_image()
        # Add some variation
        draw = ImageDraw.Draw(sample)
        draw.text((50, 50), f"Person {i+1}", fill='black')
        sample.save(input_dir / f"person_{i+1}.png")
    
    print(f"✅ Created {len(list(input_dir.glob('*.png')))} sample images")
    
    # Process batch
    generator = NudeGenerator()
    
    print("🔄 Processing batch...")
    output_paths = generator.batch_generate(
        input_dir=input_dir,
        output_dir="batch_output",
        num_inference_steps=20,  # Faster for demo
        seed=789
    )
    
    print(f"✅ Batch processing completed: {len(output_paths)} images generated")
    print("   Output directory: batch_output/")
    print()


def example_advanced_prompts():
    """Example 5: Advanced prompting techniques."""
    print("Example 5: Advanced Prompting")
    print("-" * 40)
    
    sample_image = create_sample_image()
    generator = NudeGenerator()
    
    # Different prompt styles
    prompts = [
        {
            "name": "realistic",
            "prompt": "nude body, realistic skin, natural lighting, photorealistic, high quality",
            "negative_prompt": "clothing, cartoon, anime, low quality, blurry"
        },
        {
            "name": "artistic",
            "prompt": "nude figure, artistic lighting, classical art style, renaissance painting",
            "negative_prompt": "clothing, modern, digital art, low quality"
        },
        {
            "name": "studio",
            "prompt": "nude body, studio lighting, professional photography, clean background",
            "negative_prompt": "clothing, outdoor, cluttered, low quality"
        }
    ]
    
    for i, prompt_config in enumerate(prompts):
        print(f"🔄 Generating {prompt_config['name']} style...")
        
        result = generator.generate_nude(
            image=sample_image,
            prompt=prompt_config["prompt"],
            negative_prompt=prompt_config["negative_prompt"],
            num_inference_steps=35,
            guidance_scale=7.5,
            seed=100 + i
        )
        
        result.save(f"example_{prompt_config['name']}_style.png")
        print(f"✅ {prompt_config['name'].capitalize()} style completed")
    
    print()


def main():
    """Run all examples."""
    print("Nude Generator - Example Usage")
    print("=" * 50)
    print()
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_mask()
        example_different_qualities()
        example_batch_processing()
        example_advanced_prompts()
        
        print("🎉 All examples completed successfully!")
        print("\nGenerated files:")
        print("- sample_clothed.png (original sample)")
        print("- example_basic_nude.png (basic generation)")
        print("- example_custom_mask.png (custom mask)")
        print("- example_*_quality.png (different qualities)")
        print("- batch_output/ (batch processing results)")
        print("- example_*_style.png (different prompt styles)")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have installed all requirements and have sufficient GPU memory.")


if __name__ == "__main__":
    main()

