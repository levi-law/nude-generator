#!/usr/bin/env python3
"""
Command-line interface for GAN-based nude generator.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nude_generator.core.gan_generator import GANNudeGenerator
from nude_generator.training.train_gan import GANTrainer, create_synthetic_dataset


def generate_command(args):
    """Handle the generate command."""
    print(f"🎨 Generating nude version using GAN...")
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Check if model exists
    if args.model and not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("💡 Use 'nude-generator train' to train a model first")
        return 1
    
    try:
        # Initialize generator
        generator = GANNudeGenerator(
            device=args.device,
            model_path=args.model,
            img_height=args.size,
            img_width=args.size,
        )
        
        # Generate nude image
        result = generator.generate_nude(args.input, args.output)
        
        print(f"✅ Generated nude image saved to: {args.output}")
        
        # Also save mask if requested
        if args.save_mask:
            mask_path = args.output.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
            mask = generator.create_clothing_mask(result)
            mask.save(mask_path)
            print(f"✅ Clothing mask saved to: {mask_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        if args.model is None:
            print("💡 No model specified. Use --model to specify a trained model path")
            print("💡 Or use 'nude-generator train' to train a model first")
        return 1


def train_command(args):
    """Handle the train command."""
    print(f"🚀 Training GAN nude generator...")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🔢 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    
    # Create synthetic dataset if requested
    if args.create_synthetic:
        print("🎭 Creating synthetic dataset...")
        create_synthetic_dataset(args.data_dir, num_samples=args.synthetic_samples)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("💡 Use --create-synthetic to create a synthetic dataset for testing")
        return 1
    
    try:
        # Initialize trainer
        trainer = GANTrainer(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            img_height=args.size,
            img_width=args.size,
            lambda_pixel=args.lambda_pixel,
        )
        
        # Start training
        trainer.train(
            num_epochs=args.epochs,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
        )
        
        print("🎉 Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def batch_command(args):
    """Handle the batch command."""
    print(f"📦 Batch processing images...")
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Check directories
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    input_files = [
        f for f in Path(args.input_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not input_files:
        print(f"❌ No image files found in {args.input_dir}")
        return 1
    
    print(f"📊 Found {len(input_files)} images to process")
    
    try:
        # Initialize generator
        generator = GANNudeGenerator(
            device=args.device,
            model_path=args.model,
            img_height=args.size,
            img_width=args.size,
        )
        
        # Process each image
        for i, input_file in enumerate(input_files, 1):
            output_file = Path(args.output_dir) / f"nude_{input_file.name}"
            
            print(f"🎨 Processing {i}/{len(input_files)}: {input_file.name}")
            
            try:
                generator.generate_nude(str(input_file), str(output_file))
                print(f"✅ Saved: {output_file}")
            except Exception as e:
                print(f"❌ Failed to process {input_file.name}: {e}")
        
        print("🎉 Batch processing completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GAN-based Nude Image Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate nude version of an image
  nude-generator generate input.jpg -o output.png --model trained_model.pth
  
  # Train a new model
  nude-generator train --data-dir training_data --epochs 100
  
  # Create synthetic dataset and train
  nude-generator train --create-synthetic --epochs 50
  
  # Batch process multiple images
  nude-generator batch input_folder/ -o output_folder/ --model trained_model.pth
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate nude version of an image')
    generate_parser.add_argument('input', help='Input image path')
    generate_parser.add_argument('-o', '--output', required=True, help='Output image path')
    generate_parser.add_argument('--model', help='Path to trained model file')
    generate_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], 
                                help='Device to use for generation')
    generate_parser.add_argument('--size', type=int, default=256, 
                                help='Image size (height and width)')
    generate_parser.add_argument('--save-mask', action='store_true', 
                                help='Save clothing mask alongside result')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the GAN model')
    train_parser.add_argument('--data-dir', default='training_data', 
                             help='Directory containing training data')
    train_parser.add_argument('--epochs', type=int, default=100, 
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=4, 
                             help='Batch size for training')
    train_parser.add_argument('--size', type=int, default=256, 
                             help='Image size (height and width)')
    train_parser.add_argument('--lambda-pixel', type=float, default=100.0, 
                             help='Weight for pixel-wise loss')
    train_parser.add_argument('--output-dir', default='saved_models', 
                             help='Output directory for models and samples')
    train_parser.add_argument('--save-interval', type=int, default=10, 
                             help='Save model every N epochs')
    train_parser.add_argument('--create-synthetic', action='store_true', 
                             help='Create synthetic dataset for testing')
    train_parser.add_argument('--synthetic-samples', type=int, default=100, 
                             help='Number of synthetic samples to create')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple images')
    batch_parser.add_argument('input_dir', help='Input directory containing images')
    batch_parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    batch_parser.add_argument('--model', help='Path to trained model file')
    batch_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], 
                             help='Device to use for generation')
    batch_parser.add_argument('--size', type=int, default=256, 
                             help='Image size (height and width)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'generate':
        return generate_command(args)
    elif args.command == 'train':
        return train_command(args)
    elif args.command == 'batch':
        return batch_command(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

