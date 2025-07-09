"""
Training module for GAN-based nude generator.
"""

from .train_gan import GANTrainer, NudeDataset, create_synthetic_dataset

__all__ = ['GANTrainer', 'NudeDataset', 'create_synthetic_dataset']

