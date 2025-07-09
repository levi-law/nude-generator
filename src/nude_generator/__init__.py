"""
Nude Generator - AI-powered nude image generation using Stable Diffusion.

This package provides production-ready tools for generating nude versions
of images using advanced AI inpainting techniques.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"
__description__ = "AI-powered nude image generator using Stable Diffusion inpainting"

from .core.generator import NudeGenerator
from .core.advanced_generator import AdvancedNudeGenerator

__all__ = [
    "NudeGenerator",
    "AdvancedNudeGenerator",
]

