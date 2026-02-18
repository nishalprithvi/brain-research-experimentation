
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from src.vae_model import VAE
from src.diffusion_model import DiffusionUNet
from src.data_loader import load_adni_data

def generate_synthetic_data(vae, unet, n_samples=10, device='cpu'):
    unet.eval()
    vae.eval()
    
    print(f"Generating {n_samples} synthetic brain networks...")
    
    # 1. Start from Pure Noise in Latent Space
    # Shape: (N, 1, 16, 16)
    shape = (n_samples, 1, 16, 16)
    x_t = torch.randn(shape).to(device)
    
    # 2. Reverse Diffusion Loop
    timesteps = 1000
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    # Simple sampling loop (ignoring DDIM/advanced samplers for "Hello World")
    # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps) + sigma * z
    
    for i in reversed(range(timesteps)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        
        with torch.no_grad():
            noise_pred = unet(x_t, t)
            
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_hat = alphas_cumprod[i]
        
        if i > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
            
        # Standard DDPM Sampler logic
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat)
        
        x_t = coef1 * (x_t - coef2 * noise_pred) + torch.sqrt(beta_t) * noise
        
    # 3. Decode Latent -> Brain Matrix
    with torch.no_grad():
        generated_matrices = vae.decode(x_t)
    
    return generated_matrices.cpu().squeeze().numpy()

def plot_comparisons(real_matrices, synthetic_matrices, save_dir='./results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("Plotting comparisons...")
    
    # 1. Heatmap Comparison
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(real_matrices[0], cmap='viridis', vmin=0, vmax=1)
    plt.title("Real Brain Network (ADNI)")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(synthetic_matrices[0], cmap='viridis', vmin=0, vmax=1)
    plt.title("Synthetic Brain Network (Diffusion)")
    
    plt.savefig(os.path.join(save_dir, 'heatmap_comparison.png'))
    plt.close()
    
    # 2. Degree Distribution Comparison
    # Degree = Sum of connections strength (Weighted Degree)
    real_degrees = real_matrices.sum(axis=1).flatten()
    syn_degrees = synthetic_matrices.sum(axis=1).flatten()
    
    plt.figure(figsize=(8, 6))
    sns.kdeplot(real_degrees, label='Real', fill=True)
    sns.kdeplot(syn_degrees, label='Synthetic', fill=True)
    plt.title("Degree Distribution Comparison")
    plt.xlabel("Weighted Degree (Node Strength)")
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'degree_distribution.png'))
    plt.close()
    
    print(f"Plots saved to {save_dir}")

def run_validation(n_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Real Data for comparison
    matrices, _ = load_adni_data(data_dir='./data')
    real_sample = matrices[:n_samples]
    
    # Load Models
    vae = VAE(input_dim=(100, 100), latent_dim=256).to(device)
    vae.load_state_dict(torch.load('vae_adni.pth', map_location=device))
    
    unet = DiffusionUNet().to(device)
    unet.load_state_dict(torch.load('diffusion_adni.pth', map_location=device))
    
    # Generate
    synthetic_sample = generate_synthetic_data(vae, unet, n_samples=n_samples, device=device)
    
    # Check values
    print(f"Synthetic Min: {synthetic_sample.min():.4f}, Max: {synthetic_sample.max():.4f}")
    
    # Plot
    plot_comparisons(real_sample, synthetic_sample)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10)
    args = parser.parse_args()
    
    run_validation(args.n_samples)

if __name__ == "__main__":
    main()
