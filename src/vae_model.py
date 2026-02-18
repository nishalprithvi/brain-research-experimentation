
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=(100, 100), latent_dim=256):
        """
        Step 2: The VAE (The Compressor) for 100x100 ADNI Matrices.
        
        Input: 100x100 Matrix (1 channel)
        Latent: 16x16 Block (Spatial) -> 256 values.
        
        We use a Linear bottleneck to strict 256 size (16x16) to ensure exact shape
        for the diffusion model, regardless of convolution spatial dims.
        """
        super(VAE, self).__init__()
        
        # Encoder
        # 100x100 -> Conv -> ...
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 100 -> 50
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 50 -> 25
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 25 -> 13
            nn.ReLU(),
            nn.Flatten() # 128*13*13 = 21632
        )
        
        # Latent Space (Mu and LogVar)
        # We want latent to be reshaped to 1x16x16 = 256 later.
        self.fc_mu = nn.Linear(128 * 13 * 13, latent_dim)
        self.fc_logvar = nn.Linear(128 * 13 * 13, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 13 * 13)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1), # 13 -> 25 (with output_padding?)
            # Validating shapes:
            # (13-1)*2 - 2*1 + 3 = 24 + 1 = 25. Correct.
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 25 -> 50
            # (25-1)*2 - 2 + 3 + 1 = 48 + 2 = 50. Correct.
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 50 -> 100
            # (50-1)*2 - 2 + 3 + 1 = 98 + 2 = 100. Correct.
            nn.Sigmoid() # Normalize to [0, 1]
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # x: [Batch, 100, 100] -> [Batch, 1, 100, 100]
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        z = self.reparameterize(mu, logvar)
        
        # Decode
        z_projected = self.decoder_input(z)
        z_reshaped = z_projected.view(-1, 128, 13, 13)
        reconstruction = self.decoder(z_reshaped)
        
        # Return z reshaped as 1x16x16 for Diffusion compatibility
        # z is [Batch, 256] -> [Batch, 1, 16, 16]
        return reconstruction, mu, logvar, z.view(-1, 1, 16, 16)

    def decode(self, z):
        # z input: [Batch, 1, 16, 16] -> Flatten -> [Batch, 256]
        z_flat = z.view(-1, 256)
        z_projected = self.decoder_input(z_flat)
        z_reshaped = z_projected.view(-1, 128, 13, 13)
        reconstruction = self.decoder(z_reshaped)
        return reconstruction

if __name__ == "__main__":
    # Test
    model = VAE()
    x = torch.randn(10, 100, 100)
    recon, mu, logvar, z = model(x)
    print(f"Input: {x.shape}")
    print(f"Recon: {recon.shape}") # Should be [10, 1, 100, 100]
    print(f"Latent Spatial: {z.shape}") # Should be [10, 1, 16, 16]
