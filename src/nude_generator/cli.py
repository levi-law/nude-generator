#!/usr/bin/env python3
"""
Command-line interface for the Nude Generator using pre-trained GANs.

This CLI provides easy access to pre-trained GAN models for nude image generation
without requiring custom training or complex setup.
"""

import argparse
import sys
import os
from pathlib import Path
from PIL import Image
import logging

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pretrained_gan import PretrainedNudeGAN, load_pretrained_nude_gan

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Nude Generator - AI-powered nude image generation using pre-trained GANs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate nude version of an image
  python -m nude_generator.cli generate input.jpg -o output.png
  
  # Generate multiple random nude images
  python -m nude_generator.cli random -n 5 -o output_dir/
  
  # Use specific model type
  python -m nude_generator.cli generate input.jpg -o output.png --model stylegan2_ffhq
  
  # Batch process multiple images
  python -m nude_generator.cli batch input_dir/ -o output_dir/
  
  # Get model information
  python -m nude_generator.cli info --model dcgan_nude
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate nude version of input image')
    generate_parser.add_argument('input', help='Input image path')
    generate_parser.add_argument('-o', '--output', required=True, help='Output image path')
    generate_parser.add_argument('--model', default='dcgan_nude', 
                               choices=['dcgan_nude', 'stylegan2_ffhq', 'biggan_imagenet'],
                               help='Model type to use (default: dcgan_nude)')
    generate_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                               help='Device to use (default: auto)')
    generate_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Random generation command
    random_parser = subparsers.add_parser('random', help='Generate random nude images')
    random_parser.add_argument('-n', '--num-images', type=int, default=1, help='Number of images to generate')
    random_parser.add_argument('-o', '--output', required=True, help='Output directory or file path')
    random_parser.add_argument('--model', default='dcgan_nude',
                             choices=['dcgan_nude', 'stylegan2_ffhq', 'biggan_imagenet'],
                             help='Model type to use (default: dcgan_nude)')
    random_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                             help='Device to use (default: auto)')
    random_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple images')
    batch_parser.add_argument('input_dir', help='Input directory containing images')
    batch_parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    batch_parser.add_argument('--model', default='dcgan_nude',
                            choices=['dcgan_nude', 'stylegan2_ffhq', 'biggan_imagenet'],
                            help='Model type to use (default: dcgan_nude)')
    batch_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                            help='Device to use (default: auto)')
    batch_parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'],
                            help='Image file extensions to process')
    
    # Model info command
    info_parser = subparsers.add_parser('info', help='Get information about available models')
    info_parser.add_argument('--model', default='dcgan_nude',
                           choices=['dcgan_nude', 'stylegan2_ffhq', 'biggan_imagenet'],
                           help='Model to get info about (default: dcgan_nude)')
    info_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                           help='Device to use (default: auto)')
    
    # List models command
    list_parser = subparsers.add_parser('list', help='List available pre-trained models')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'generate':
            generate_nude_image(args)
        elif args.command == 'random':
            generate_random_images(args)
        elif args.command == 'batch':
            batch_process_images(args)
        elif args.command == 'info':
            show_model_info(args)
        elif args.command == 'list':
            list_available_models()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def generate_nude_image(args):
    """Generate nude version of input image."""
    logger.info(f"Generating nude version of {args.input}")
    
    # Validate input
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading {args.model} model...")
    gan = load_pretrained_nude_gan(
        model_type=args.model,
        device=args.device
    )
    
    # Generate nude image
    logger.info("Generating nude image...")
    nude_image = gan.generate_nude(
        input_image=args.input,
        seed=args.seed
    )
    
    # Save result
    nude_image.save(args.output)
    logger.info(f"Nude image saved to {args.output}")
    
    # Show model info
    info = gan.get_model_info()
    logger.info(f"Used model: {info['model_type']} on {info['device']}")


def generate_random_images(args):
    """Generate random nude images."""
    logger.info(f"Generating {args.num_images} random nude images")
    
    # Setup output
    output_path = Path(args.output)
    if args.num_images == 1 and not output_path.suffix:
        # Single image, but no extension provided
        output_path = output_path.with_suffix('.png')
    elif args.num_images > 1:
        # Multiple images, create directory
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # Single image with extension, create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading {args.model} model...")
    gan = load_pretrained_nude_gan(
        model_type=args.model,
        device=args.device
    )
    
    # Generate images
    logger.info("Generating images...")
    images = gan.generate_nude(
        num_images=args.num_images,
        seed=args.seed
    )
    
    # Save results
    if args.num_images == 1:
        images.save(output_path)
        logger.info(f"Image saved to {output_path}")
    else:
        for i, image in enumerate(images):
            image_path = output_path / f"random_nude_{i+1:03d}.png"
            image.save(image_path)
            logger.info(f"Image {i+1} saved to {image_path}")
    
    logger.info(f"Generated {args.num_images} images successfully")


def batch_process_images(args):
    """Batch process multiple images."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all image files
    image_files = []
    for ext in args.extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading {args.model} model...")
    gan = load_pretrained_nude_gan(
        model_type=args.model,
        device=args.device
    )
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        try:
            # Generate nude version
            nude_image = gan.generate_nude(input_image=str(image_file))
            
            # Save result
            output_file = output_dir / f"nude_{image_file.stem}.png"
            nude_image.save(output_file)
            
            logger.info(f"Saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            continue
    
    logger.info("Batch processing completed")


def show_model_info(args):
    """Show information about a model."""
    logger.info(f"Loading {args.model} model...")
    
    gan = load_pretrained_nude_gan(
        model_type=args.model,
        device=args.device
    )
    
    info = gan.get_model_info()
    
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Model Type: {info['model_type']}")
    print(f"Device: {info['device']}")
    print(f"Parameters: {info['generator_params']:,}")
    
    if info['model_info']:
        model_info = info['model_info']
        print(f"Description: {model_info.get('description', 'N/A')}")
        print(f"Input Size: {model_info.get('input_size', 'N/A')}")
        print(f"Output Size: {model_info.get('output_size', 'N/A')}")
    
    print("="*50)


def list_available_models():
    """List all available pre-trained models."""
    models = {
        'dcgan_nude': {
            'description': 'DCGAN trained on nude portraits (artistic style)',
            'resolution': '128x128',
            'style': 'Artistic/Classical paintings',
            'speed': 'Fast',
            'quality': 'Good'
        },
        'stylegan2_ffhq': {
            'description': 'StyleGAN2 trained on FFHQ faces',
            'resolution': '1024x1024',
            'style': 'Photorealistic faces',
            'speed': 'Medium',
            'quality': 'Excellent'
        },
        'biggan_imagenet': {
            'description': 'BigGAN trained on ImageNet',
            'resolution': '256x256',
            'style': 'Various objects/scenes',
            'speed': 'Medium',
            'quality': 'Very Good'
        }
    }
    
    print("\n" + "="*70)
    print("AVAILABLE PRE-TRAINED MODELS")
    print("="*70)
    
    for model_name, info in models.items():
        print(f"\n{model_name.upper()}")
        print("-" * len(model_name))
        print(f"Description: {info['description']}")
        print(f"Resolution:  {info['resolution']}")
        print(f"Style:       {info['style']}")
        print(f"Speed:       {info['speed']}")
        print(f"Quality:     {info['quality']}")
    
    print("\n" + "="*70)
    print("USAGE EXAMPLES:")
    print("="*70)
    print("# Use DCGAN for fast artistic nude generation:")
    print("python -m nude_generator.cli generate input.jpg -o output.png --model dcgan_nude")
    print("\n# Use StyleGAN2 for high-quality photorealistic results:")
    print("python -m nude_generator.cli generate input.jpg -o output.png --model stylegan2_ffhq")
    print("\n# Use BigGAN for diverse generation:")
    print("python -m nude_generator.cli generate input.jpg -o output.png --model biggan_imagenet")
    print("="*70)


if __name__ == '__main__':
    main()

