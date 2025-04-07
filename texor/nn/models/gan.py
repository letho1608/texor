"""Generative Adversarial Network implementations"""
from typing import Dict
from ...core import Tensor
from ..layers import (
    Layer, Linear, Sequential, Conv2d, ConvTranspose2d,
    BatchNorm2d, Flatten, Reshape
)
from ..activations import ReLU, LeakyReLU, Tanh, Sigmoid
from ..loss import binary_cross_entropy
from ...optim.optimizers import Optimizer
from ...core.ops import ones_like, zeros_like

class GAN:
    """Generative Adversarial Network base class"""
    def __init__(self, generator: Layer, discriminator: Layer):
        self.generator = generator
        self.discriminator = discriminator
        
    def generator_loss(self, fake_output: Tensor) -> Tensor:
        """Compute generator loss"""
        return binary_cross_entropy(fake_output, ones_like(fake_output))
        
    def discriminator_loss(self, real_output: Tensor, fake_output: Tensor) -> Tensor:
        """Compute discriminator loss"""
        real_loss = binary_cross_entropy(real_output, ones_like(real_output))
        fake_loss = binary_cross_entropy(fake_output, zeros_like(fake_output))
        return real_loss + fake_loss
        
    def train_step(self, real_data: Tensor, noise: Tensor,
                  gen_optimizer: Optimizer, disc_optimizer: Optimizer) -> Dict[str, float]:
        """Single training step"""
        # Train discriminator
        disc_optimizer.zero_grad()
        
        fake_data = self.generator(noise)
        real_output = self.discriminator(real_data)
        fake_output = self.discriminator(fake_data)
        
        disc_loss = self.discriminator_loss(real_output, fake_output)
        disc_loss.backward()
        disc_optimizer.step()
        
        # Train generator
        gen_optimizer.zero_grad()
        
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data)
        
        gen_loss = self.generator_loss(fake_output)
        gen_loss.backward()
        gen_optimizer.step()
        
        return {
            'generator_loss': gen_loss.numpy(),
            'discriminator_loss': disc_loss.numpy()
        }
        
    def generate(self, noise: Tensor) -> Tensor:
        """Generate samples from noise"""
        return self.generator(noise)

class DCGAN(GAN):
    """Deep Convolutional GAN"""
    def __init__(self, generator: Layer, discriminator: Layer, 
                 latent_dim: int = 100):
        super().__init__(generator, discriminator)
        self.latent_dim = latent_dim
        
    @staticmethod
    def create_generator(latent_dim: int, channels: int = 3) -> Sequential:
        """Create default generator architecture"""
        return Sequential([
            Linear(latent_dim, 256 * 7 * 7),
            ReLU(),
            Reshape(-1, 256, 7, 7),
            ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(64),
            ReLU(),
            ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            Tanh()
        ])
        
    @staticmethod
    def create_discriminator(channels: int = 3) -> Sequential:
        """Create default discriminator architecture"""
        return Sequential([
            Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2),
            Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2),
            Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2),
            Flatten(),
            Linear(256 * 7 * 7, 1),
            Sigmoid()
        ])

def create_dcgan(latent_dim: int = 100, channels: int = 3) -> DCGAN:
    """Create DCGAN model with default architectures"""
    generator = DCGAN.create_generator(latent_dim, channels)
    discriminator = DCGAN.create_discriminator(channels)
    return DCGAN(generator, discriminator, latent_dim)