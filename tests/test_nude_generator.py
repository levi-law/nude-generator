#!/usr/bin/env python3
"""
Test script for the Nude Generator implementation.

This script tests the basic functionality of the nude generator
with sample images and validates the implementation.
"""

import os
import sys
import logging
from pathlib import Path
import traceback

import torch
from PIL import Image, ImageDraw
import numpy as np

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from nude_generator import NudeGenerator
    from config import get_generation_params, get_clothing_region
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image(size=(512, 512), save_path="test_input.png"):
    """Create a simple test image for testing purposes."""
    # Create a simple figure silhouette for testing
    image = Image.new('RGB', size, 'lightblue')
    draw = ImageDraw.Draw(image)
    
    width, height = size
    
    # Draw a simple human figure silhouette
    # Head
    head_center = (width // 2, height // 4)
    head_radius = 40
    draw.ellipse([
        head_center[0] - head_radius,
        head_center[1] - head_radius,
        head_center[0] + head_radius,
        head_center[1] + head_radius
    ], fill='peachpuff')
    
    # Body (torso)
    torso_coords = get_clothing_region("torso", size)
    draw.rectangle(torso_coords, fill='lightcoral')
    
    # Arms
    arm_width = 30
    arm_length = 100
    # Left arm
    draw.rectangle([
        torso_coords[0] - arm_width,
        torso_coords[1] + 20,
        torso_coords[0],
        torso_coords[1] + 20 + arm_length
    ], fill='peachpuff')
    # Right arm
    draw.rectangle([
        torso_coords[2],
        torso_coords[1] + 20,
        torso_coords[2] + arm_width,
        torso_coords[1] + 20 + arm_length
    ], fill='peachpuff')
    
    # Legs
    leg_coords = get_clothing_region("lower_body", size)
    leg_width = (leg_coords[2] - leg_coords[0]) // 2 - 10
    # Left leg
    draw.rectangle([
        leg_coords[0],
        leg_coords[1],
        leg_coords[0] + leg_width,
        leg_coords[3]
    ], fill='darkblue')
    # Right leg
    draw.rectangle([
        leg_coords[2] - leg_width,
        leg_coords[1],
        leg_coords[2],
        leg_coords[3]
    ], fill='darkblue')
    
    # Add some clothing texture
    draw.text((width//2 - 30, height//2), "SHIRT", fill='white')
    draw.text((width//2 - 30, height//2 + 100), "PANTS", fill='white')
    
    image.save(save_path)
    logger.info(f"Test image created: {save_path}")
    return image


def test_basic_functionality():
    """Test basic nude generation functionality."""
    logger.info("Testing basic nude generation functionality...")
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if device == "cpu":
            logger.warning("CUDA not available. Test will be slow on CPU.")
            # Use a smaller model for CPU testing
            model_id = "runwayml/stable-diffusion-inpainting"
        else:
            model_id = "stabilityai/stable-diffusion-2-inpainting"
        
        # Initialize generator
        logger.info("Initializing nude generator...")
        generator = NudeGenerator(
            model_id=model_id,
            device=device
        )
        
        # Test mask creation
        logger.info("Testing mask creation...")
        mask = generator.create_clothing_mask(test_image)
        mask.save("test_mask.png")
        logger.info("Mask created successfully")
        
        # Test image preprocessing
        logger.info("Testing image preprocessing...")
        processed = generator.preprocess_image(test_image)
        processed.save("test_processed.png")
        logger.info("Image preprocessing successful")
        
        # Test generation with fast parameters for testing
        logger.info("Testing nude generation (this may take a while)...")
        result = generator.generate_nude(
            image=test_image,
            prompt="nude body, realistic skin",
            negative_prompt="clothing, clothes, low quality",
            num_inference_steps=20,  # Reduced for faster testing
            guidance_scale=6.0,
            seed=42
        )
        
        # Save result
        result.save("test_output.png")
        logger.info("Nude generation test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_mask_regions():
    """Test different mask region configurations."""
    logger.info("Testing different mask regions...")
    
    try:
        test_image = create_test_image()
        generator = NudeGenerator(device="cpu")  # Use CPU for quick testing
        
        # Test different regions
        regions = ["torso", "full_body", "lower_body", "upper_body"]
        
        for region in regions:
            logger.info(f"Testing {region} region...")
            coords = get_clothing_region(region, test_image.size)
            mask = generator.create_clothing_mask(test_image, mask_regions=[coords])
            mask.save(f"test_mask_{region}.png")
            
        logger.info("Mask region tests completed")
        return True
        
    except Exception as e:
        logger.error(f"Mask region test failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing functionality."""
    logger.info("Testing batch processing...")
    
    try:
        # Create test directory with multiple images
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        
        # Create multiple test images
        for i in range(3):
            test_image = create_test_image(save_path=test_dir / f"test_{i}.png")
        
        # Test batch processing
        generator = NudeGenerator(device="cpu")  # Use CPU for testing
        
        output_paths = generator.batch_generate(
            input_dir=test_dir,
            output_dir="test_output",
            num_inference_steps=10,  # Very fast for testing
            guidance_scale=5.0
        )
        
        logger.info(f"Batch processing completed: {len(output_paths)} images generated")
        return True
        
    except Exception as e:
        logger.error(f"Batch processing test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    logger.info("Starting comprehensive tests...")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Mask Regions", test_mask_regions),
        ("Batch Processing", test_batch_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print results summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST RESULTS SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("All tests passed! ✅")
        return True
    else:
        logger.warning("Some tests failed! ❌")
        return False


def main():
    """Main test function."""
    print("Nude Generator Test Suite")
    print("=" * 50)
    
    # Check dependencies
    try:
        import torch
        import diffusers
        from PIL import Image
        print("✅ All required dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA not available, tests will run on CPU (slower)")
    
    print("\nStarting tests...\n")
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("\nGenerated files:")
        print("- test_input.png (test image)")
        print("- test_mask.png (generated mask)")
        print("- test_output.png (nude generation result)")
        print("- test_mask_*.png (different region masks)")
        print("- test_output/ (batch processing results)")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    main()

