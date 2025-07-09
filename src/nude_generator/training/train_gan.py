#!/usr/bin/env python3
"""
Training script for GAN-based nude generator.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nude_generator.core.gan_generator import GANNudeGenerator, GeneratorUNet, Discriminator


class NudeDataset(Dataset):
    """Dataset for nude generation training."""
    
    def __init__(self, root_dir: str, transform=None, img_height: int = 256, img_width: int = 256):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing 'clothed' and 'nude' subdirectories
            transform: Optional transform to be applied on a sample
            img_height: Height to resize images to
            img_width: Width to resize images to
        """
        self.root_dir = Path(root_dir)
        self.clothed_dir = self.root_dir / "clothed"
        self.nude_dir = self.root_dir / "nude"
        self.img_height = img_height
        self.img_width = img_width
        
        # Get list of image files
        self.clothed_images = sorted([f for f in self.clothed_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        self.nude_images = sorted([f for f in self.nude_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        # Ensure we have matching pairs
        assert len(self.clothed_images) == len(self.nude_images), "Number of clothed and nude images must match"
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    def __len__(self):
        return len(self.clothed_images)
    
    def __getitem__(self, idx):
        # Load images
        clothed_img = Image.open(self.clothed_images[idx]).convert('RGB')
        nude_img = Image.open(self.nude_images[idx]).convert('RGB')
        
        # Apply transforms
        clothed_tensor = self.transform(clothed_img)
        nude_tensor = self.transform(nude_img)
        
        return clothed_tensor, nude_tensor


class GANTrainer:
    """Trainer class for GAN nude generator."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        img_height: int = 256,
        img_width: int = 256,
        device: str = "auto",
        lambda_pixel: float = 100.0,
    ):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory containing training data
            batch_size: Batch size for training
            img_height: Height of images
            img_width: Width of images
            device: Device to train on
            lambda_pixel: Weight for pixel-wise loss
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.lambda_pixel = lambda_pixel
        
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.generator = GeneratorUNet(in_channels=3, out_channels=3).to(self.device)
        self.discriminator = Discriminator(in_channels=6).to(self.device)
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_pixelwise = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        
        # Dataset and dataloader
        self.dataset = NudeDataset(data_dir, img_height=img_height, img_width=img_width)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        
        print(f"Dataset size: {len(self.dataset)} image pairs")
        
        # Training history
        self.g_losses = []
        self.d_losses = []
    
    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        
        for i, (real_A, real_B) in enumerate(progress_bar):
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            
            # Adversarial ground truths
            batch_size = real_A.size(0)
            valid = torch.ones((batch_size, 1, 30, 30), device=self.device, requires_grad=False)
            fake = torch.zeros((batch_size, 1, 30, 30), device=self.device, requires_grad=False)
            
            # ------------------
            #  Train Generator
            # ------------------
            self.optimizer_G.zero_grad()
            
            # Generate fake images
            fake_B = self.generator(real_A)
            
            # GAN loss
            pred_fake = self.discriminator(fake_B, real_A)
            loss_GAN = self.criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = self.criterion_pixelwise(fake_B, real_B)
            
            # Total generator loss
            loss_G = loss_GAN + self.lambda_pixel * loss_pixel
            
            loss_G.backward()
            self.optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.optimizer_D.zero_grad()
            
            # Real loss
            pred_real = self.discriminator(real_B, real_A)
            loss_real = self.criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = self.discriminator(fake_B.detach(), real_A)
            loss_fake = self.criterion_GAN(pred_fake, fake)
            
            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake)
            
            loss_D.backward()
            self.optimizer_D.step()
            
            # Update progress
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            
            progress_bar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}'
            })
        
        avg_g_loss = epoch_g_loss / len(self.dataloader)
        avg_d_loss = epoch_d_loss / len(self.dataloader)
        
        return avg_g_loss, avg_d_loss
    
    def train(self, num_epochs: int, save_interval: int = 10, output_dir: str = "saved_models"):
        """
        Train the GAN.
        
        Args:
            num_epochs: Number of epochs to train
            save_interval: Save model every N epochs
            output_dir: Directory to save models and samples
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train for one epoch
            g_loss, d_loss = self.train_epoch(epoch)
            
            # Record losses
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f} - "
                  f"Time: {epoch_time:.2f}s")
            
            # Save model and generate samples
            if epoch % save_interval == 0:
                self.save_model(f"{output_dir}/model_epoch_{epoch}.pth")
                self.generate_samples(f"{output_dir}/samples_epoch_{epoch}.png")
                self.plot_losses(f"{output_dir}/losses.png")
        
        print("Training completed!")
        
        # Save final model
        self.save_model(f"{output_dir}/final_model.pth")
        self.generate_samples(f"{output_dir}/final_samples.png")
        self.plot_losses(f"{output_dir}/final_losses.png")
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
        }, path)
        print(f"Model saved to {path}")
    
    def generate_samples(self, path: str, num_samples: int = 4):
        """Generate sample images."""
        self.generator.eval()
        
        with torch.no_grad():
            # Get a batch of test images
            real_A, real_B = next(iter(self.dataloader))
            real_A = real_A[:num_samples].to(self.device)
            real_B = real_B[:num_samples].to(self.device)
            
            # Generate fake images
            fake_B = self.generator(real_A)
            
            # Create comparison grid
            fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
            
            for i in range(num_samples):
                # Original (clothed)
                img_A = (real_A[i].cpu() + 1) / 2  # Denormalize
                axes[0, i].imshow(img_A.permute(1, 2, 0))
                axes[0, i].set_title("Input (Clothed)")
                axes[0, i].axis('off')
                
                # Generated (nude)
                img_fake = (fake_B[i].cpu() + 1) / 2  # Denormalize
                axes[1, i].imshow(img_fake.permute(1, 2, 0))
                axes[1, i].set_title("Generated (Nude)")
                axes[1, i].axis('off')
                
                # Target (real nude)
                img_B = (real_B[i].cpu() + 1) / 2  # Denormalize
                axes[2, i].imshow(img_B.permute(1, 2, 0))
                axes[2, i].set_title("Target (Real Nude)")
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
        
        self.generator.train()
    
    def plot_losses(self, path: str):
        """Plot training losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()


def create_synthetic_dataset(output_dir: str, num_samples: int = 100):
    """
    Create a synthetic dataset for demonstration purposes.
    In practice, you would use real paired data.
    """
    os.makedirs(f"{output_dir}/clothed", exist_ok=True)
    os.makedirs(f"{output_dir}/nude", exist_ok=True)
    
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    for i in range(num_samples):
        # Create synthetic clothed image (with clothing patterns)
        clothed_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add clothing patterns (simplified)
        clothed_img[64:192, 64:192] = [200, 150, 100]  # Torso clothing
        clothed_img[128:224, 96:160] = [100, 100, 200]  # Lower clothing
        
        # Create corresponding nude image (skin tones)
        nude_img = clothed_img.copy()
        nude_img[64:192, 64:192] = [220, 180, 140]  # Skin tone for torso
        nude_img[128:224, 96:160] = [220, 180, 140]  # Skin tone for lower body
        
        # Save images
        Image.fromarray(clothed_img).save(f"{output_dir}/clothed/{i:04d}.png")
        Image.fromarray(nude_img).save(f"{output_dir}/nude/{i:04d}.png")
    
    print(f"Synthetic dataset created in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GAN nude generator")
    parser.add_argument("--data_dir", type=str, default="training_data", 
                       help="Directory containing training data")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size")
    parser.add_argument("--img_size", type=int, default=256, 
                       help="Image size (height and width)")
    parser.add_argument("--lambda_pixel", type=float, default=100.0, 
                       help="Weight for pixel-wise loss")
    parser.add_argument("--output_dir", type=str, default="saved_models", 
                       help="Output directory for models and samples")
    parser.add_argument("--create_synthetic", action="store_true", 
                       help="Create synthetic dataset for testing")
    
    args = parser.parse_args()
    
    # Create synthetic dataset if requested
    if args.create_synthetic:
        create_synthetic_dataset(args.data_dir, num_samples=50)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Data directory {args.data_dir} not found!")
        print("Use --create_synthetic to create a synthetic dataset for testing.")
        return
    
    # Initialize trainer
    trainer = GANTrainer(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_height=args.img_size,
        img_width=args.img_size,
        lambda_pixel=args.lambda_pixel,
    )
    
    # Start training
    trainer.train(
        num_epochs=args.epochs,
        save_interval=10,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

