#!/usr/bin/env python3
"""
Final Optimized Nude Generator - Perfect balance of clothing removal and preservation
"""

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2
from diffusers import StableDiffusionInpaintPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalOptimizedNudeGenerator:
    """Final optimized nude generator with perfect balance"""
    
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
    
    def create_optimized_clothing_mask(self, image):
        """Create optimized mask for effective clothing removal with preservation"""
        w, h = image.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # Bra area - more precise targeting
        bra_top = int(h * 0.38)    # Just below face
        bra_bottom = int(h * 0.58) # Mid-torso
        bra_left = int(w * 0.32)   # Left side
        bra_right = int(w * 0.68)  # Right side
        
        # Draw elliptical mask for bra area
        draw.ellipse([bra_left, bra_top, bra_right, bra_bottom], fill=255)
        
        # Panties area - targeted lower region
        panties_top = int(h * 0.72)    # Lower torso
        panties_bottom = int(h * 0.88) # Above legs
        panties_left = int(w * 0.38)   # Centered
        panties_right = int(w * 0.62)
        
        # Draw elliptical mask for panties area
        draw.ellipse([panties_left, panties_top, panties_right, panties_bottom], fill=255)
        
        # Apply moderate blur for soft edges
        mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
        
        # Keep full intensity for effective removal
        mask_array = np.array(mask)
        mask_array = (mask_array * 0.9).astype(np.uint8)  # 90% intensity
        mask = Image.fromarray(mask_array)
        
        return mask
    
    def extract_skin_tone_from_visible_areas(self, image):
        """Extract skin tone from visible skin areas"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Sample from multiple visible skin areas
        # Face area (upper portion)
        face_region = img_array[int(h*0.15):int(h*0.35), int(w*0.25):int(w*0.75)]
        
        # Shoulder/neck area
        shoulder_region = img_array[int(h*0.35):int(h*0.45), int(w*0.2):int(w*0.8)]
        
        # Arms (if visible)
        left_arm = img_array[int(h*0.4):int(h*0.7), int(w*0.05):int(w*0.25)]
        right_arm = img_array[int(h*0.4):int(h*0.7), int(w*0.75):int(w*0.95)]
        
        # Combine all skin samples
        all_skin = np.vstack([
            face_region.reshape(-1, 3),
            shoulder_region.reshape(-1, 3),
            left_arm.reshape(-1, 3),
            right_arm.reshape(-1, 3)
        ])
        
        # Filter for skin-like colors (more refined range)
        mask = np.all((all_skin > 70) & (all_skin < 230), axis=1)
        skin_pixels = all_skin[mask]
        
        if len(skin_pixels) == 0:
            return [150, 120, 90]  # Default warm skin tone
        
        # Use median for stability, then adjust for natural variation
        base_tone = np.median(skin_pixels, axis=0).astype(int)
        
        # Slightly warm the tone for natural nude appearance
        base_tone[0] = min(255, base_tone[0] + 5)  # Slightly more red
        base_tone[1] = max(0, base_tone[1] - 2)    # Slightly less green
        
        return base_tone
    
    def generate_final_nude(self, image_path, output_path=None, num_inference_steps=20):
        """Generate final optimized nude with perfect balance"""
        # Load and prepare image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize for processing
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Processing image: {image.size}")
        
        # Extract optimized skin tone
        skin_tone = self.extract_skin_tone_from_visible_areas(image)
        logger.info(f"Optimized skin tone: {skin_tone}")
        
        # Create optimized mask
        mask = self.create_optimized_clothing_mask(image)
        
        # Balanced prompt for effective clothing removal with preservation
        skin_hex = f"#{skin_tone[0]:02x}{skin_tone[1]:02x}{skin_tone[2]:02x}"
        
        prompt = f"""
        remove bra and panties only, keep everything else exactly the same,
        natural nude breasts, skin color {skin_hex}, realistic skin texture,
        preserve face, hair, pose, background, lighting completely,
        anatomically correct female body, natural breast shape,
        photorealistic skin, soft natural shadows, high quality
        """
        
        negative_prompt = """
        change face, change person, change background, change pose, change hair,
        different room, different lighting, artificial breasts, fake skin,
        plastic appearance, cartoon, anime, oversaturated colors,
        perfect symmetry, airbrushed, low quality, blurry, distorted,
        extra limbs, deformed anatomy, watermark, signature, text
        """
        
        # Optimized generation settings
        logger.info("Generating final optimized nude...")
        
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.0,  # Balanced guidance
            generator=generator,
            strength=0.65  # Optimal strength for clothing removal
        ).images[0]
        
        # Resize back to original size
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        # Save result
        if output_path:
            result.save(output_path, quality=95)
            logger.info(f"Final optimized nude saved to: {output_path}")
        
        return result, mask

def main():
    """Test the final optimized nude generator"""
    generator = FinalOptimizedNudeGenerator()
    
    # Test with the uploaded image
    input_path = "/home/ubuntu/nude_generator_project/data/input/sample_input.png"
    output_path = "/home/ubuntu/nude_generator_project/data/output/final_optimized_nude.png"
    mask_path = "/home/ubuntu/nude_generator_project/data/output/final_optimized_mask.png"
    
    try:
        result, mask = generator.generate_final_nude(
            input_path, 
            output_path,
            num_inference_steps=20  # Good quality balance
        )
        
        # Save mask for inspection
        mask.save(mask_path)
        
        print(f"✅ Final optimized nude generation completed!")
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

