#!/usr/bin/env python3
"""
Real Nude Generator using Hugging Face NSFW Stable Diffusion model.

This implementation uses the actual Kernel/sd-nsfw model from Hugging Face
which is a fine-tuned Stable Diffusion model specifically for NSFW content.
"""

import torch
import logging
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from typing import Union, Optional, List
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
    from diffusers import DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    logger.warning("Diffusers not available. Install with: pip install diffusers")
    DIFFUSERS_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available. Install with: pip install opencv-python")
    CV2_AVAILABLE = False


class RealNudeGenerator:
    """
    Real nude generator using Hugging Face NSFW Stable Diffusion model.
    
    This class provides high-quality nude image generation using a pre-trained
    NSFW Stable Diffusion model from Hugging Face.
    """
    
    def __init__(self, device: str = "auto", model_id: str = "Kernel/sd-nsfw"):
        """
        Initialize the Real Nude Generator.
        
        Args:
            device: Device to use ('auto', 'cpu', 'cuda')
            model_id: Hugging Face model ID for NSFW Stable Diffusion
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library is required. Install with: pip install diffusers")
        
        self.device = self._setup_device(device)
        self.model_id = model_id
        self.pipeline = None
        self.inpaint_pipeline = None
        
        logger.info(f"Initializing Real Nude Generator on {self.device}")
        self._load_models()
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_models(self):
        """Load the NSFW Stable Diffusion models."""
        try:
            logger.info(f"Loading NSFW Stable Diffusion model: {self.model_id}")
            
            # Load main generation pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable safety checker for NSFW content
                requires_safety_checker=False
            )
            
            # Use DPM++ scheduler for better quality
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Load inpainting pipeline for clothing removal
            logger.info("Loading inpainting pipeline for clothing removal...")
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
                self.inpaint_pipeline.enable_attention_slicing()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def generate_nude_from_text(
        self,
        prompt: str,
        negative_prompt: str = "clothes, clothing, dressed, shirt, pants, underwear, bra, panties",
        num_images: int = 1,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate nude images from text prompts.
        
        Args:
            prompt: Text description of the desired nude image
            negative_prompt: What to avoid in the generation
            num_images: Number of images to generate
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            
        Returns:
            Generated nude image(s)
        """
        logger.info(f"Generating {num_images} nude image(s) from text prompt")
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Enhance prompt for nude generation
        enhanced_prompt = f"{prompt}, nude, naked, beautiful body, realistic skin, high quality, detailed"
        
        # Generate images
        with torch.autocast(self.device):
            result = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        
        images = result.images
        
        if num_images == 1:
            return images[0]
        return images
    
    def remove_clothes_from_image(
        self,
        input_image: Union[str, Image.Image],
        mask: Optional[Union[str, Image.Image]] = None,
        prompt: str = "nude, naked, beautiful skin, realistic body",
        negative_prompt: str = "clothes, clothing, dressed, deformed, blurry",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Remove clothes from an existing image using inpainting.
        
        Args:
            input_image: Input image path or PIL Image
            mask: Mask image or path (white = inpaint, black = keep)
            prompt: Description of what to generate in masked areas
            negative_prompt: What to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed
            
        Returns:
            Image with clothes removed
        """
        logger.info("Removing clothes from image using inpainting")
        
        # Load input image
        if isinstance(input_image, str):
            image = Image.open(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")
        
        # Create or load mask
        if mask is None:
            # Auto-generate mask for clothing areas
            mask = self._create_clothing_mask(image)
        elif isinstance(mask, str):
            mask = Image.open(mask).convert("L")
        else:
            mask = mask.convert("L")
        
        # Resize images to compatible size
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Perform inpainting
        with torch.autocast(self.device):
            result = self.inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        
        return result.images[0]
    
    def _create_clothing_mask(self, image: Image.Image) -> Image.Image:
        """
        Create a mask for clothing areas in the image.
        
        This is a simplified approach - in practice, you'd want to use
        a trained segmentation model for better accuracy.
        """
        logger.info("Creating clothing mask")
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Create a basic mask focusing on torso area
        height, width = img_array.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define torso region (approximate)
        torso_top = int(height * 0.25)
        torso_bottom = int(height * 0.75)
        torso_left = int(width * 0.25)
        torso_right = int(width * 0.75)
        
        # Create mask for torso area
        mask[torso_top:torso_bottom, torso_left:torso_right] = 255
        
        # Apply some smoothing if OpenCV is available
        if CV2_AVAILABLE:
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return Image.fromarray(mask)
    
    def generate_nude_variations(
        self,
        base_image: Union[str, Image.Image],
        num_variations: int = 3,
        prompt: str = "nude, beautiful body, realistic skin",
        strength: float = 0.7,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate nude variations of a base image.
        
        Args:
            base_image: Base image to create variations from
            num_variations: Number of variations to generate
            prompt: Description for the variations
            strength: How much to change from original (0.0-1.0)
            seed: Random seed
            
        Returns:
            List of nude variation images
        """
        logger.info(f"Generating {num_variations} nude variations")
        
        # Load base image
        if isinstance(base_image, str):
            image = Image.open(base_image).convert("RGB")
        else:
            image = base_image.convert("RGB")
        
        image = image.resize((512, 512))
        
        variations = []
        
        for i in range(num_variations):
            # Set different seed for each variation
            current_seed = seed + i if seed is not None else None
            if current_seed is not None:
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)
            
            # Create mask for variation
            mask = self._create_clothing_mask(image)
            
            # Generate variation
            with torch.autocast(self.device):
                result = self.inpaint_pipeline(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                    strength=strength,
                    num_inference_steps=25
                )
            
            variations.append(result.images[0])
        
        return variations
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "pipeline_loaded": self.pipeline is not None,
            "inpaint_pipeline_loaded": self.inpaint_pipeline is not None,
            "description": "Real NSFW Stable Diffusion model for nude generation",
            "capabilities": [
                "Text-to-nude generation",
                "Clothing removal via inpainting",
                "Nude variations generation",
                "High-quality realistic results"
            ]
        }


def load_real_nude_generator(device: str = "auto") -> RealNudeGenerator:
    """
    Load the real nude generator with NSFW Stable Diffusion model.
    
    Args:
        device: Device to use ('auto', 'cpu', 'cuda')
        
    Returns:
        Initialized RealNudeGenerator instance
    """
    return RealNudeGenerator(device=device)


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = load_real_nude_generator()
    
    # Generate nude from text
    nude_image = generator.generate_nude_from_text(
        prompt="beautiful woman, artistic nude, studio lighting",
        seed=42
    )
    nude_image.save("generated_nude.png")
    
    # Remove clothes from existing image
    if Path("input_image.jpg").exists():
        nude_version = generator.remove_clothes_from_image(
            "input_image.jpg",
            seed=42
        )
        nude_version.save("nude_version.png")
    
    print("Real nude generation completed!")

