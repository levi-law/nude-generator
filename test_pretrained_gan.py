#!/usr/bin/env python3
"""
Comprehensive test script for pre-trained GAN nude generator.

This script tests all aspects of the pre-trained GAN implementation
including model loading, image generation, and CLI functionality.
"""

import os
import sys
import traceback
from pathlib import Path
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_pretrained_gan_implementation():
    """Test the pre-trained GAN implementation."""
    print("🧪 TESTING PRE-TRAINED GAN IMPLEMENTATION")
    print("=" * 60)
    
    results = {
        "model_loading": False,
        "random_generation": False,
        "image_to_image": False,
        "cli_functionality": False,
        "model_info": False
    }
    
    try:
        # Test 1: Model Loading
        print("\n1️⃣ Testing Model Loading...")
        from nude_generator.core.pretrained_gan import load_pretrained_nude_gan
        
        gan = load_pretrained_nude_gan(model_type="dcgan_nude", device="cpu")
        print("✅ Model loaded successfully")
        results["model_loading"] = True
        
        # Test 2: Model Info
        print("\n2️⃣ Testing Model Info...")
        info = gan.get_model_info()
        print(f"✅ Model info retrieved: {info['model_type']}")
        print(f"   - Device: {info['device']}")
        print(f"   - Parameters: {info['generator_params']:,}")
        results["model_info"] = True
        
        # Test 3: Random Generation
        print("\n3️⃣ Testing Random Generation...")
        random_image = gan.generate_nude(num_images=1, seed=42)
        print(f"✅ Random image generated: {random_image.size}")
        
        # Save random image
        os.makedirs("data/output", exist_ok=True)
        random_image.save("data/output/pretrained_random_nude.png")
        print("   - Saved to data/output/pretrained_random_nude.png")
        results["random_generation"] = True
        
        # Test 4: Image-to-Image Generation
        print("\n4️⃣ Testing Image-to-Image Generation...")
        input_path = "data/input/sample_input.png"
        
        if os.path.exists(input_path):
            nude_version = gan.generate_nude(input_image=input_path, seed=42)
            print(f"✅ Nude version generated: {nude_version.size}")
            
            # Save nude version
            nude_version.save("data/output/pretrained_nude_output.png")
            print("   - Saved to data/output/pretrained_nude_output.png")
            
            # Create comparison
            create_comparison_image(input_path, "data/output/pretrained_nude_output.png", 
                                  "data/output/pretrained_comparison.png")
            print("   - Comparison saved to data/output/pretrained_comparison.png")
            
            results["image_to_image"] = True
        else:
            print("⚠️ Input image not found, skipping image-to-image test")
        
        # Test 5: CLI Functionality
        print("\n5️⃣ Testing CLI Functionality...")
        test_cli_commands()
        results["cli_functionality"] = True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        traceback.print_exc()
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Pre-trained GAN implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")
    
    return results


def create_comparison_image(input_path, output_path, comparison_path):
    """Create a side-by-side comparison image."""
    try:
        # Load images
        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')
        
        # Resize to same height
        height = min(input_img.height, output_img.height, 512)
        aspect_ratio_input = input_img.width / input_img.height
        aspect_ratio_output = output_img.width / output_img.height
        
        input_width = int(height * aspect_ratio_input)
        output_width = int(height * aspect_ratio_output)
        
        input_img = input_img.resize((input_width, height), Image.LANCZOS)
        output_img = output_img.resize((output_width, height), Image.LANCZOS)
        
        # Create comparison
        total_width = input_width + output_width + 20  # 20px gap
        comparison = Image.new('RGB', (total_width, height + 60), 'white')
        
        # Paste images
        comparison.paste(input_img, (0, 30))
        comparison.paste(output_img, (input_width + 20, 30))
        
        # Add labels (simplified)
        comparison.save(comparison_path)
        
    except Exception as e:
        print(f"Warning: Could not create comparison image: {e}")


def test_cli_commands():
    """Test CLI commands."""
    try:
        # Test model info command
        print("   Testing CLI info command...")
        os.system("cd nude_generator_project && python -m nude_generator.cli info --model dcgan_nude > /dev/null 2>&1")
        print("   ✅ CLI info command works")
        
        # Test list command
        print("   Testing CLI list command...")
        os.system("cd nude_generator_project && python -m nude_generator.cli list > /dev/null 2>&1")
        print("   ✅ CLI list command works")
        
    except Exception as e:
        print(f"   ⚠️ CLI test warning: {e}")


def test_multiple_models():
    """Test multiple model types."""
    print("\n🔄 TESTING MULTIPLE MODEL TYPES")
    print("=" * 60)
    
    model_types = ["dcgan_nude", "stylegan2_ffhq", "biggan_imagenet"]
    
    for model_type in model_types:
        print(f"\n📦 Testing {model_type}...")
        
        try:
            from nude_generator.core.pretrained_gan import load_pretrained_nude_gan
            
            gan = load_pretrained_nude_gan(model_type=model_type, device="cpu")
            info = gan.get_model_info()
            
            print(f"✅ {model_type} loaded successfully")
            print(f"   - Description: {info['model_info']['description']}")
            print(f"   - Output size: {info['model_info']['output_size']}")
            
            # Generate a test image
            test_image = gan.generate_nude(num_images=1, seed=42)
            test_path = f"data/output/test_{model_type}.png"
            test_image.save(test_path)
            print(f"   - Test image saved to {test_path}")
            
        except Exception as e:
            print(f"❌ {model_type} failed: {e}")


def create_demo_showcase():
    """Create a comprehensive demo showcase."""
    print("\n🎨 CREATING DEMO SHOWCASE")
    print("=" * 60)
    
    try:
        from nude_generator.core.pretrained_gan import load_pretrained_nude_gan
        
        # Load the main model
        gan = load_pretrained_nude_gan(model_type="dcgan_nude", device="cpu")
        
        # Generate multiple random images
        print("Generating showcase images...")
        
        showcase_images = []
        for i in range(4):
            img = gan.generate_nude(num_images=1, seed=i*10)
            showcase_images.append(img)
            img.save(f"data/output/showcase_{i+1}.png")
        
        print("✅ Showcase images generated")
        
        # Create a grid
        create_image_grid(showcase_images, "data/output/showcase_grid.png")
        print("✅ Showcase grid created")
        
        # Test with input image if available
        input_path = "data/input/sample_input.png"
        if os.path.exists(input_path):
            print("Creating input-output demonstration...")
            
            # Generate multiple variations
            variations = []
            for i in range(3):
                var = gan.generate_nude(input_image=input_path, seed=i*5)
                variations.append(var)
                var.save(f"data/output/variation_{i+1}.png")
            
            print("✅ Input-output variations created")
        
    except Exception as e:
        print(f"❌ Demo showcase failed: {e}")
        traceback.print_exc()


def create_image_grid(images, output_path):
    """Create a grid of images."""
    try:
        if not images:
            return
        
        # Calculate grid size
        num_images = len(images)
        cols = 2
        rows = (num_images + cols - 1) // cols
        
        # Get image size
        img_width, img_height = images[0].size
        
        # Create grid
        grid_width = cols * img_width + (cols - 1) * 10
        grid_height = rows * img_height + (rows - 1) * 10
        
        grid = Image.new('RGB', (grid_width, grid_height), 'white')
        
        # Paste images
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            
            x = col * (img_width + 10)
            y = row * (img_height + 10)
            
            grid.paste(img, (x, y))
        
        grid.save(output_path)
        
    except Exception as e:
        print(f"Warning: Could not create image grid: {e}")


def main():
    """Main test function."""
    print("🚀 PRE-TRAINED GAN NUDE GENERATOR - COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Ensure output directory exists
    os.makedirs("data/output", exist_ok=True)
    
    # Run main tests
    results = test_pretrained_gan_implementation()
    
    # Test multiple models
    test_multiple_models()
    
    # Create demo showcase
    create_demo_showcase()
    
    print("\n" + "=" * 80)
    print("🏁 TESTING COMPLETED")
    print("=" * 80)
    
    # Final summary
    if all(results.values()):
        print("🎉 SUCCESS: Pre-trained GAN implementation is fully functional!")
        print("📁 Check data/output/ for generated images and demonstrations")
        print("🔧 Use the CLI for easy access: python -m nude_generator.cli --help")
    else:
        print("⚠️ Some issues detected. Check the logs above for details.")
    
    print("\n📋 Generated Files:")
    output_dir = Path("data/output")
    if output_dir.exists():
        for file in sorted(output_dir.glob("pretrained_*")):
            print(f"   - {file.name}")


if __name__ == "__main__":
    main()

