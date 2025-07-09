#!/usr/bin/env python3
"""
Command-line interface for the Nude Generator.

This module provides a comprehensive CLI for the nude generator with
support for single image processing, batch processing, and various
configuration options.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from .core.generator import NudeGenerator
from .core.advanced_generator import AdvancedNudeGenerator
from .utils.config import get_generation_params


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_generate(args):
    """Handle single image generation command."""
    try:
        if args.advanced:
            generator = AdvancedNudeGenerator(
                model_id=args.model,
                device=args.device,
                enable_controlnet=args.enable_controlnet,
                enable_clothing_detection=args.enable_clothing_detection
            )
            
            result = generator.generate_nude_advanced(
                image=args.input,
                quality=args.quality,
                preserve_face=args.preserve_face,
                preserve_hands=args.preserve_hands,
                seed=args.seed
            )
            
            # Save main result
            result["image"].save(args.output)
            
            # Save mask if requested
            if args.save_mask:
                mask_path = Path(args.output).with_suffix('.mask.png')
                result["mask"].save(mask_path)
                print(f"Mask saved to: {mask_path}")
                
        else:
            generator = NudeGenerator(
                model_id=args.model,
                device=args.device
            )
            
            # Get generation parameters
            gen_params = get_generation_params(args.quality)
            if args.prompt:
                gen_params["prompt"] = args.prompt
            if args.negative_prompt:
                gen_params["negative_prompt"] = args.negative_prompt
            if args.steps:
                gen_params["num_inference_steps"] = args.steps
            if args.guidance_scale:
                gen_params["guidance_scale"] = args.guidance_scale
                
            result = generator.generate_nude(
                image=args.input,
                seed=args.seed,
                **gen_params
            )
            
            result.save(args.output)
        
        print(f"Generated image saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_batch(args):
    """Handle batch processing command."""
    try:
        if args.advanced:
            generator = AdvancedNudeGenerator(
                model_id=args.model,
                device=args.device,
                enable_controlnet=args.enable_controlnet,
                enable_clothing_detection=args.enable_clothing_detection
            )
            
            results = generator.batch_process_advanced(
                input_dir=args.input,
                output_dir=args.output,
                quality=args.quality,
                preserve_face=args.preserve_face,
                preserve_hands=args.preserve_hands,
                save_metadata=args.save_metadata,
                seed=args.seed
            )
            
        else:
            generator = NudeGenerator(
                model_id=args.model,
                device=args.device
            )
            
            # Get generation parameters
            gen_params = get_generation_params(args.quality)
            if args.prompt:
                gen_params["prompt"] = args.prompt
            if args.negative_prompt:
                gen_params["negative_prompt"] = args.negative_prompt
            if args.steps:
                gen_params["num_inference_steps"] = args.steps
            if args.guidance_scale:
                gen_params["guidance_scale"] = args.guidance_scale
                
            results = generator.batch_generate(
                input_dir=args.input,
                output_dir=args.output,
                seed=args.seed,
                **gen_params
            )
        
        successful = len([r for r in results if r.get("success", True)])
        total = len(results) if isinstance(results, list) else len(results)
        print(f"Batch processing completed: {successful}/{total} images processed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_info(args):
    """Display system and model information."""
    import torch
    
    print("Nude Generator - System Information")
    print("=" * 40)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    try:
        import diffusers
        print(f"Diffusers version: {diffusers.__version__}")
    except ImportError:
        print("Diffusers: Not installed")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Nude Generator - AI-powered nude image generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single image
  nude-generator generate input.jpg -o output.png
  
  # Batch processing
  nude-generator batch input_folder/ -o output_folder/
  
  # High quality with custom prompt
  nude-generator generate input.jpg -o output.png --quality high --prompt "custom prompt"
  
  # Advanced features
  nude-generator generate input.jpg -o output.png --advanced --preserve-face --preserve-hands
        """
    )
    
    # Global arguments
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate nude version of a single image")
    gen_parser.add_argument("input", help="Input image path")
    gen_parser.add_argument("-o", "--output", required=True, help="Output image path")
    gen_parser.add_argument("--model", default="stabilityai/stable-diffusion-2-inpainting",
                           help="Model ID to use")
    gen_parser.add_argument("--quality", choices=["fast", "default", "high"], default="default",
                           help="Generation quality")
    gen_parser.add_argument("--prompt", help="Custom generation prompt")
    gen_parser.add_argument("--negative-prompt", help="Custom negative prompt")
    gen_parser.add_argument("--steps", type=int, help="Number of inference steps")
    gen_parser.add_argument("--guidance-scale", type=float, help="Guidance scale")
    gen_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    gen_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                           help="Device to use")
    gen_parser.add_argument("--advanced", action="store_true", 
                           help="Use advanced generator with additional features")
    gen_parser.add_argument("--preserve-face", action="store_true", default=True,
                           help="Preserve face areas (advanced mode)")
    gen_parser.add_argument("--preserve-hands", action="store_true", default=True,
                           help="Preserve hand areas (advanced mode)")
    gen_parser.add_argument("--enable-controlnet", action="store_true",
                           help="Enable ControlNet for pose preservation (advanced mode)")
    gen_parser.add_argument("--enable-clothing-detection", action="store_true",
                           help="Enable automatic clothing detection (advanced mode)")
    gen_parser.add_argument("--save-mask", action="store_true",
                           help="Save the generated mask (advanced mode)")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process multiple images")
    batch_parser.add_argument("input", help="Input directory path")
    batch_parser.add_argument("-o", "--output", required=True, help="Output directory path")
    batch_parser.add_argument("--model", default="stabilityai/stable-diffusion-2-inpainting",
                             help="Model ID to use")
    batch_parser.add_argument("--quality", choices=["fast", "default", "high"], default="default",
                             help="Generation quality")
    batch_parser.add_argument("--prompt", help="Custom generation prompt")
    batch_parser.add_argument("--negative-prompt", help="Custom negative prompt")
    batch_parser.add_argument("--steps", type=int, help="Number of inference steps")
    batch_parser.add_argument("--guidance-scale", type=float, help="Guidance scale")
    batch_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    batch_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                             help="Device to use")
    batch_parser.add_argument("--advanced", action="store_true",
                             help="Use advanced generator with additional features")
    batch_parser.add_argument("--preserve-face", action="store_true", default=True,
                             help="Preserve face areas (advanced mode)")
    batch_parser.add_argument("--preserve-hands", action="store_true", default=True,
                             help="Preserve hand areas (advanced mode)")
    batch_parser.add_argument("--enable-controlnet", action="store_true",
                             help="Enable ControlNet for pose preservation (advanced mode)")
    batch_parser.add_argument("--enable-clothing-detection", action="store_true",
                             help="Enable automatic clothing detection (advanced mode)")
    batch_parser.add_argument("--save-metadata", action="store_true", default=True,
                             help="Save generation metadata (advanced mode)")
    batch_parser.set_defaults(func=cmd_batch)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display system information")
    info_parser.set_defaults(func=cmd_info)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Handle commands
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

