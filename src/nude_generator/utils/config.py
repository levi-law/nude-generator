"""
Configuration settings for the Nude Generator.

This module contains default settings and configuration options
for the nude image generation pipeline.
"""

import os
from typing import Dict, Any, List, Tuple

# Model configurations
MODELS = {
    "stable_diffusion_2_inpainting": {
        "model_id": "stabilityai/stable-diffusion-2-inpainting",
        "description": "High-quality inpainting model, good balance of speed and quality",
        "recommended": True
    },
    "stable_diffusion_xl_inpainting": {
        "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "description": "Higher resolution inpainting, requires more VRAM",
        "recommended": False
    },
    "runwayml_inpainting": {
        "model_id": "runwayml/stable-diffusion-inpainting",
        "description": "Original SD 1.5 inpainting model, faster but lower quality",
        "recommended": False
    }
}

# Default generation parameters
DEFAULT_GENERATION_PARAMS = {
    "prompt": "nude body, realistic skin, natural lighting, high quality, detailed anatomy",
    "negative_prompt": (
        "clothing, fabric, clothes, shirt, pants, dress, underwear, bra, panties, "
        "low quality, blurry, distorted, deformed, ugly, bad anatomy, "
        "extra limbs, missing limbs, floating limbs, disconnected limbs, "
        "mutation, mutated, ugly, disgusting, blurry, amputation, "
        "watermark, text, signature"
    ),
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "strength": 1.0,
    "target_size": (512, 512)
}

# High-quality generation parameters
HIGH_QUALITY_PARAMS = {
    "prompt": (
        "nude body, realistic skin, natural lighting, high quality, detailed anatomy, "
        "photorealistic, 8k, masterpiece, professional photography"
    ),
    "negative_prompt": (
        "clothing, fabric, clothes, shirt, pants, dress, underwear, bra, panties, "
        "low quality, blurry, distorted, deformed, ugly, bad anatomy, "
        "extra limbs, missing limbs, floating limbs, disconnected limbs, "
        "mutation, mutated, ugly, disgusting, blurry, amputation, "
        "watermark, text, signature, cartoon, anime, 3d render, painting"
    ),
    "num_inference_steps": 75,
    "guidance_scale": 8.0,
    "strength": 1.0,
    "target_size": (768, 768)
}

# Fast generation parameters
FAST_PARAMS = {
    "prompt": "nude body, realistic skin",
    "negative_prompt": "clothing, clothes, low quality, blurry",
    "num_inference_steps": 25,
    "guidance_scale": 6.0,
    "strength": 0.9,
    "target_size": (512, 512)
}

# Predefined mask regions for common clothing areas
CLOTHING_REGIONS = {
    "torso": {
        "description": "Upper body/torso area",
        "relative_coords": (0.25, 0.3, 0.75, 0.7)  # (x1, y1, x2, y2) as fractions
    },
    "full_body": {
        "description": "Full body excluding head and hands",
        "relative_coords": (0.2, 0.2, 0.8, 0.9)
    },
    "lower_body": {
        "description": "Lower body/legs area",
        "relative_coords": (0.3, 0.6, 0.7, 0.95)
    },
    "upper_body": {
        "description": "Upper body/chest area",
        "relative_coords": (0.25, 0.25, 0.75, 0.6)
    }
}

# Device and memory settings
DEVICE_SETTINGS = {
    "auto_detect": True,
    "prefer_cuda": True,
    "enable_memory_efficient_attention": True,
    "enable_cpu_offload": True,
    "enable_attention_slicing": True
}

# File and directory settings
FILE_SETTINGS = {
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
    "output_format": "PNG",
    "output_quality": 95,
    "cache_dir": os.path.expanduser("~/.cache/nude_generator"),
    "log_level": "INFO"
}

# Safety and ethical settings
SAFETY_SETTINGS = {
    "disable_safety_checker": True,  # Required for nude content generation
    "age_verification_required": True,
    "consent_verification_required": True,
    "watermark_outputs": False,
    "log_usage": True
}

# Advanced features
ADVANCED_FEATURES = {
    "auto_clothing_detection": False,  # Requires additional models
    "pose_preservation": False,       # Requires ControlNet
    "face_preservation": True,        # Preserve face area by default
    "hand_preservation": True,        # Preserve hand areas by default
    "background_preservation": True   # Keep background unchanged
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODELS.get(model_name, MODELS["stable_diffusion_2_inpainting"])

def get_generation_params(quality: str = "default") -> Dict[str, Any]:
    """Get generation parameters based on quality setting."""
    if quality == "high":
        return HIGH_QUALITY_PARAMS.copy()
    elif quality == "fast":
        return FAST_PARAMS.copy()
    else:
        return DEFAULT_GENERATION_PARAMS.copy()

def get_clothing_region(region_name: str, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Convert relative coordinates to absolute coordinates for given image size."""
    if region_name not in CLOTHING_REGIONS:
        region_name = "full_body"
    
    rel_coords = CLOTHING_REGIONS[region_name]["relative_coords"]
    width, height = image_size
    
    x1 = int(rel_coords[0] * width)
    y1 = int(rel_coords[1] * height)
    x2 = int(rel_coords[2] * width)
    y2 = int(rel_coords[3] * height)
    
    return (x1, y1, x2, y2)

# Environment variable overrides
def load_env_config():
    """Load configuration from environment variables."""
    config_overrides = {}
    
    # Model selection
    if os.getenv("NUDE_GEN_MODEL"):
        config_overrides["model_id"] = os.getenv("NUDE_GEN_MODEL")
    
    # Device selection
    if os.getenv("NUDE_GEN_DEVICE"):
        config_overrides["device"] = os.getenv("NUDE_GEN_DEVICE")
    
    # Cache directory
    if os.getenv("NUDE_GEN_CACHE_DIR"):
        config_overrides["cache_dir"] = os.getenv("NUDE_GEN_CACHE_DIR")
    
    # Quality setting
    if os.getenv("NUDE_GEN_QUALITY"):
        config_overrides["quality"] = os.getenv("NUDE_GEN_QUALITY")
    
    return config_overrides

