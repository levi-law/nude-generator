"""
Nude Generator - AI-powered nude image generation using GANs.

This package provides GAN-based nude image generation capabilities
using Pix2Pix architecture for high-quality, realistic results.
"""

__version__ = "2.0.0"
__author__ = "Nude Generator Team"
__description__ = "GAN-based nude image generator"

# Import main classes
from .core.gan_generator import GANNudeGenerator

# Import training utilities
try:
    from .training.train_gan import GANTrainer, NudeDataset
except ImportError:
    # Training modules might not be available in all environments
    pass

__all__ = [
    'GANNudeGenerator',
    'GANTrainer', 
    'NudeDataset',
]