
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse

# Local imports
from src.data_loader import load_adni_data
from src.vae_model import VAE
from src.diffusion_model import DiffusionUNet

def train_vae(model, dataloader, epochs=50, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    print(f"\n[VAE] Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        total_batches = 0
        for x in dataloader:
            x = x[0].float().to(device) # x is (Batch, 100, 100)
            
            optimizer.zero_grad()
            recon, mu, logvar, z = model(x)
            
            # Loss = Reconstruction + KLD
            # MSE sum reduction common for generative tasks or mean? 
            # Sum per batch then divide by N is standard for VAE usually.
            recon_loss = F.mse_loss(recon, x.unsqueeze(1), reduction='sum')
            
            # KLD
            # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kld_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch+1) % 5 == 0:
            print(f"[VAE] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
    return model

def train_diffusion(unet, vae, dataloader, epochs=100, device='cpu'):
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    unet.train()
    vae.eval() # Freeze VAE
    
    criterion = torch.nn.MSELoss()
    
    print(f"\n[Diffusion] Starting Training for {epochs} epochs...")
    
    # Diffusion parameters (Linear Schedule)
    timesteps = 1000
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    # sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    for epoch in range(epochs):
        total_loss = 0
        for x in dataloader:
            x = x[0].float().to(device)
            
            # 1. Encode to latent z using VAE
            with torch.no_grad():
                _, _, _, z = vae(x) # z is (Batch, 1, 16, 16)
            
            # 2. Diffusion Steps
            # Sample random timesteps
            current_batch_size = z.shape[0]
            t = torch.randint(0, timesteps, (current_batch_size,), device=device).long()
            
            noise = torch.randn_like(z)
            
            # Reparameterization trick: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*epsilon
            sqrt_ac = torch.sqrt(alphas_cumprod)[t]
            sqrt_one_minus_ac = torch.sqrt(1 - alphas_cumprod)[t]
            
            # Reshape for broadcasting
            sqrt_ac = sqrt_ac.view(-1, 1, 1, 1)
            sqrt_one_minus_ac = sqrt_one_minus_ac.view(-1, 1, 1, 1)
            
            noisy_z = sqrt_ac * z + sqrt_one_minus_ac * noise
            
            optimizer.zero_grad()
            
            # Predict NOISE
            noise_pred = unet(noisy_z, t)
            
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch+1) % 10 == 0:
            print(f"[Diffusion] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

    return unet

def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading Data...")
    matrices, _ = load_adni_data(data_dir='./data') # Returns (matrices, labels)
    # Convert to TensorDataset
    tensor_x = torch.from_numpy(matrices)
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Data Loaded: {len(dataset)} samples.")
    
    # 2. Train VAE
    vae = VAE(input_dim=(100, 100), latent_dim=256).to(device)
    vae = train_vae(vae, dataloader, epochs=args.epochs_vae, device=device)
    torch.save(vae.state_dict(), 'vae_adni.pth')
    print("VAE Saved.")
    
    # 3. Train Diffusion
    unet = DiffusionUNet().to(device)
    unet = train_diffusion(unet, vae, dataloader, epochs=args.epochs_diff, device=device)
    torch.save(unet.state_dict(), 'diffusion_adni.pth')
    print("Diffusion Model Saved.")
    
    print("Pipeline Complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs_vae', type=int, default=50)
    parser.add_argument('--epochs_diff', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    run_training(args)

if __name__ == "__main__":
    main()
