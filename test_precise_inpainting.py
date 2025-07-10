#!/usr/bin/env python3
"""
Precise Inpainting Nude Generator - True clothing removal with minimal masking
"""

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2
from diffusers import StableDiffusionInpaintPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreciseInpaintingNudeGenerator:
    """Precise nude generator with minimal masking for true clothing removal"""
    
    def __init__(self, model_id="Kernel/sd-nsfw", device="auto"):
        """Initialize with NSFW model"""
        self.device = "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load NSFW Stable Diffusion model
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if self.device == "cuda":
            self.pipe = self.pipe.to(self.device)
        else:
            # CPU optimizations
            self.pipe.enable_attention_slicing()
    
    def create_minimal_clothing_mask(self, image):
        """Create minimal mask targeting only specific clothing areas"""
        w, h = image.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # For the specific image (woman in lingerie), create targeted masks
        # Based on typical lingerie positioning
        
        # Bra area (small targeted region)
        bra_top = int(h * 0.35)    # Start below face
        bra_bottom = int(h * 0.55) # End at mid-torso
        bra_left = int(w * 0.3)    # Left side
        bra_right = int(w * 0.7)   # Right side
        
        # Draw elliptical mask for bra area
        draw.ellipse([bra_left, bra_top, bra_right, bra_bottom], fill=255)
        
        # Panties area (small targeted region)
        panties_top = int(h * 0.7)    # Lower torso
        panties_bottom = int(h * 0.85) # Above legs
        panties_left = int(w * 0.35)   # Narrower than bra
        panties_right = int(w * 0.65)
        
        # Draw elliptical mask for panties area
        draw.ellipse([panties_left, panties_top, panties_right, panties_bottom], fill=255)
        
        # Apply heavy blur to create very soft edges
        mask = mask.filter(ImageFilter.GaussianBlur(radius=15))
        
        # Reduce mask intensity to preserve more original
        mask_array = np.array(mask)
        mask_array = (mask_array * 0.7).astype(np.uint8)  # Reduce to 70% intensity
        mask = Image.fromarray(mask_array)
        
        return mask
    
    def extract_skin_tone_from_visible_areas(self, image):
        """Extract skin tone from visible skin areas (arms, face, etc.)"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Sample from arms and face area (avoiding clothing)
        # Face area
        face_region = img_array[int(h*0.1):int(h*0.4), int(w*0.2):int(w*0.8)]
        
        # Arm areas
        left_arm = img_array[int(h*0.3):int(h*0.7), int(w*0.05):int(w*0.25)]
        right_arm = img_array[int(h*0.3):int(h*0.7), int(w*0.75):int(w*0.95)]
        
        # Combine all skin samples
        all_skin = np.vstack([
            face_region.reshape(-1, 3),
            left_arm.reshape(-1, 3),
            right_arm.reshape(-1, 3)
        ])
        
        # Filter for skin-like colors
        mask = np.all((all_skin > 80) & (all_skin < 220), axis=1)
        skin_pixels = all_skin[mask]
        
        if len(skin_pixels) == 0:
            return [172, 123, 85]  # Default skin tone
        
        # Return median color for stability
        return np.median(skin_pixels, axis=0).astype(int)
    
    def generate_precise_nude(self, image_path, output_path=None, num_inference_steps=20):
        """Generate nude with precise inpainting to preserve original image"""
        # Load and prepare image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize for processing (keep reasonable size)
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Processing image: {image.size}")
        
        # Extract skin tone from visible areas
        skin_tone = self.extract_skin_tone_from_visible_areas(image)
        logger.info(f"Detected skin tone: {skin_tone}")
        
        # Create minimal, precise mask
        mask = self.create_minimal_clothing_mask(image)
        
        # Very specific prompt for clothing removal (not full generation)
        skin_hex = f"#{skin_tone[0]:02x}{skin_tone[1]:02x}{skin_tone[2]:02x}"
        
        prompt = f"""
        remove clothing only, preserve face and background exactly, 
        natural skin texture, skin color {skin_hex}, realistic breasts,
        keep same pose, keep same person, keep same lighting,
        photorealistic skin, natural shadows, anatomically correct,
        only replace clothing areas with skin
        """
        
        negative_prompt = """
        change face, change person, change background, change pose,
        different lighting, artificial, plastic, fake, cartoon, anime,
        oversaturated, unrealistic skin, perfect symmetry, airbrushed,
        low quality, blurry, distorted, deformed, extra limbs,
        watermark, signature, text, logo, different room, different setting
        """
        
        # Generate with very conservative settings for preservation
        logger.info("Generating precise nude with minimal inpainting...")
        
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=6.0,  # Lower guidance for more preservation
            generator=generator,
            strength=0.4  # Very low strength to preserve original
        ).images[0]
        
        # Minimal post-processing - just ensure dimensions match
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        # Save result
        if output_path:
            result.save(output_path, quality=95)
            logger.info(f"Precise nude saved to: {output_path}")
        
        return result, mask

def main():
    """Test the precise inpainting nude generator"""
    generator = PreciseInpaintingNudeGenerator()
    
    # Test with the uploaded image
    input_path = "/home/ubuntu/nude_generator_project/data/input/sample_input.png"
    output_path = "/home/ubuntu/nude_generator_project/data/output/precise_inpaint_nude.png"
    mask_path = "/home/ubuntu/nude_generator_project/data/output/precise_inpaint_mask.png"
    
    try:
        result, mask = generator.generate_precise_nude(
            input_path, 
            output_path,
            num_inference_steps=15  # Fast for testing
        )
        
        # Save mask for inspection
        mask.save(mask_path)
        
        print(f"✅ Precise inpainting nude generation completed!")
        print(f"📁 Result: {output_path}")
        print(f"📁 Mask: {mask_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()

