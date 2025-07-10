#!/usr/bin/env python3
"""
Simplified test for perfect nude generator focusing on core improvements
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
from diffusers import StableDiffusionInpaintPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePerfectNudeGenerator:
    """Simplified perfect nude generator with core improvements"""
    
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
    
    def extract_skin_tone(self, image):
        """Extract dominant skin tone from image"""
        img_array = np.array(image)
        
        # Focus on center area (likely face/body)
        h, w = img_array.shape[:2]
        center_region = img_array[h//4:3*h//4, w//4:3*w//4]
        pixels = center_region.reshape(-1, 3)
        
        # Remove very dark/light pixels
        mask = np.all((pixels > 50) & (pixels < 200), axis=1)
        skin_pixels = pixels[mask]
        
        if len(skin_pixels) == 0:
            return [139, 119, 101]  # Default skin tone
        
        # Return median color (more stable than mean)
        return np.median(skin_pixels, axis=0).astype(int)
    
    def create_simple_clothing_mask(self, image):
        """Create simple mask for clothing areas"""
        # Convert to HSV for better color segmentation
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Create mask for non-skin areas (simplified approach)
        # This is a basic approach - in real implementation you'd use proper segmentation
        h, w = image.size
        mask = Image.new('L', (w, h), 0)
        
        # Simple rectangular mask for torso area (where clothing typically is)
        mask_array = np.array(mask)
        
        # Define torso region (adjust based on typical pose)
        start_y = int(h * 0.3)  # Below face
        end_y = int(h * 0.8)    # Above legs
        start_x = int(w * 0.25) # Left side
        end_x = int(w * 0.75)   # Right side
        
        # Create mask for torso area
        mask_array[start_y:end_y, start_x:end_x] = 255
        
        # Convert back to PIL and smooth
        mask = Image.fromarray(mask_array)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
        
        return mask
    
    def generate_perfect_nude(self, image_path, output_path=None, num_inference_steps=30):
        """Generate improved nude with better preservation"""
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
        
        # Extract skin tone
        skin_tone = self.extract_skin_tone(image)
        logger.info(f"Detected skin tone: {skin_tone}")
        
        # Create clothing mask
        mask = self.create_simple_clothing_mask(image)
        
        # Enhanced prompt with specific skin tone
        skin_hex = f"#{skin_tone[0]:02x}{skin_tone[1]:02x}{skin_tone[2]:02x}"
        
        prompt = f"""
        beautiful nude female body, realistic skin texture, natural breasts,
        skin color {skin_hex}, photorealistic, high quality, detailed skin,
        natural shadows, anatomically correct, soft lighting, professional photography,
        preserve face and background, only remove clothing, natural pose
        """
        
        negative_prompt = """
        artificial, plastic, fake, cartoon, anime, drawing, painting,
        oversaturated, unrealistic skin, perfect symmetry, airbrushed,
        low quality, blurry, distorted, deformed, extra limbs,
        watermark, signature, text, logo, change face, change background,
        different person, different pose
        """
        
        # Generate with conservative settings for better preservation
        logger.info("Generating improved nude image...")
        
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=generator,
            strength=0.6  # Lower strength to preserve more original
        ).images[0]
        
        # Simple post-processing
        result = self.simple_post_process(result, image, mask, skin_tone)
        
        # Resize back to original size
        if original_size != result.size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        # Save result
        if output_path:
            result.save(output_path, quality=95)
            logger.info(f"Improved nude saved to: {output_path}")
        
        return result, mask
    
    def simple_post_process(self, result, original, mask, skin_tone):
        """Simple post-processing for better results"""
        # Ensure all images have the same size
        if result.size != original.size:
            result = result.resize(original.size, Image.Resampling.LANCZOS)
        if mask.size != original.size:
            mask = mask.resize(original.size, Image.Resampling.LANCZOS)
        
        result_array = np.array(result)
        original_array = np.array(original)
        mask_array = np.array(mask) / 255.0
        
        # Ensure mask has correct dimensions
        if len(mask_array.shape) == 2:
            mask_array = mask_array[:, :, np.newaxis]
        
        # Blend edges more smoothly
        mask_blur = cv2.GaussianBlur(mask_array[:, :, 0], (21, 21), 0)
        mask_blur = mask_blur[:, :, np.newaxis]
        
        # Blend with very soft edges to preserve more original
        blended = result_array * mask_blur + original_array * (1 - mask_blur)
        
        return Image.fromarray(blended.astype(np.uint8))

def main():
    """Test the simplified perfect nude generator"""
    generator = SimplePerfectNudeGenerator()
    
    # Test with the uploaded image
    input_path = "/home/ubuntu/nude_generator_project/data/input/sample_input.png"
    output_path = "/home/ubuntu/nude_generator_project/data/output/simple_perfect_nude.png"
    mask_path = "/home/ubuntu/nude_generator_project/data/output/simple_perfect_mask.png"
    
    try:
        result, mask = generator.generate_perfect_nude(
            input_path, 
            output_path,
            num_inference_steps=25  # Faster for testing
        )
        
        # Save mask for inspection
        mask.save(mask_path)
        
        print(f"✅ Simple perfect nude generation completed!")
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

