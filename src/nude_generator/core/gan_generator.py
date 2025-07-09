#!/usr/bin/env python3
"""
GAN-based Nude Generator using Pix2Pix architecture.
This implementation uses conditional GANs for image-to-image translation
specifically designed for nude generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from typing import Optional, Tuple, Union


class UNetDown(nn.Module):
    """Downsampling block for U-Net generator."""
    
    def __init__(self, in_size: int, out_size: int, normalize: bool = True, dropout: float = 0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block for U-Net generator."""
    
    def __init__(self, in_size: int, out_size: int, dropout: float = 0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    """U-Net Generator for Pix2Pix nude generation."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super(GeneratorUNet, self).__init__()

        # Encoder (downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # Decoder (upsampling)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        # Final layer
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    """PatchGAN Discriminator for Pix2Pix."""
    
    def __init__(self, in_channels: int = 6):  # 3 for input + 3 for target
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class GANNudeGenerator:
    """Main GAN-based nude generator class."""
    
    def __init__(
        self,
        device: str = "auto",
        model_path: Optional[str] = None,
        img_height: int = 256,
        img_width: int = 256,
    ):
        """
        Initialize the GAN nude generator.
        
        Args:
            device: Device to run on ('cpu', 'cuda', or 'auto')
            model_path: Path to pre-trained model weights
            img_height: Height of input/output images
            img_width: Width of input/output images
        """
        self.device = self._setup_device(device)
        self.img_height = img_height
        self.img_width = img_width
        
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
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),
            transforms.ToPILImage(),
        ])
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def generate_nude(
        self,
        image: Union[str, Image.Image, np.ndarray],
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate nude version of input image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            save_path: Optional path to save the result
            
        Returns:
            Generated nude image as PIL Image
        """
        # Load and preprocess image
        if isinstance(image, str):
            input_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            input_image = Image.fromarray(image).convert('RGB')
        else:
            input_image = image.convert('RGB')
        
        # Transform image
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
        
        # Generate nude image
        self.generator.eval()
        with torch.no_grad():
            generated_tensor = self.generator(input_tensor)
        
        # Convert back to PIL Image
        generated_image = self.inverse_transform(generated_tensor.squeeze(0).cpu())
        
        # Save if path provided
        if save_path:
            generated_image.save(save_path)
        
        return generated_image
    
    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        lambda_pixel: float = 100.0,
    ) -> Tuple[float, float]:
        """
        Perform one training step.
        
        Args:
            real_A: Input images (clothed)
            real_B: Target images (nude)
            lambda_pixel: Weight for pixel-wise loss
            
        Returns:
            Tuple of (generator_loss, discriminator_loss)
        """
        # Adversarial ground truths
        valid = torch.ones((real_A.size(0), 1, 30, 30), device=self.device, requires_grad=False)
        fake = torch.zeros((real_A.size(0), 1, 30, 30), device=self.device, requires_grad=False)
        
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
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        
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
        
        return loss_G.item(), loss_D.item()
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    def create_clothing_mask(self, image: Image.Image) -> Image.Image:
        """
        Create a mask for clothing areas (simplified version).
        In a full implementation, this would use segmentation models.
        """
        # Convert to numpy for processing
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Create a simple mask for demonstration
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define clothing areas (torso and lower body)
        # Torso area
        torso_y1 = int(height * 0.25)
        torso_y2 = int(height * 0.65)
        torso_x1 = int(width * 0.25)
        torso_x2 = int(width * 0.75)
        mask[torso_y1:torso_y2, torso_x1:torso_x2] = 255
        
        # Lower body area
        lower_y1 = int(height * 0.55)
        lower_y2 = int(height * 0.85)
        lower_x1 = int(width * 0.35)
        lower_x2 = int(width * 0.65)
        mask[lower_y1:lower_y2, lower_x1:lower_x2] = 255
        
        return Image.fromarray(mask, mode='L')


def main():
    """Example usage of the GAN nude generator."""
    # Initialize generator
    generator = GANNudeGenerator(device="cpu", img_height=256, img_width=256)
    
    # Example generation (requires trained model)
    input_path = "data/input/sample_input.png"
    output_path = "data/output/gan_nude_output.png"
    
    if os.path.exists(input_path):
        try:
            result = generator.generate_nude(input_path, output_path)
            print(f"Generated nude image saved to: {output_path}")
        except Exception as e:
            print(f"Generation failed: {e}")
            print("Note: This requires a trained model. Use train_gan.py to train first.")
    else:
        print(f"Input image not found: {input_path}")


if __name__ == "__main__":
    main()

