import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
from sklearn.model_selection import StratifiedKFold

# Local imports
from src.data_loader_3class import load_adni_data
from src.vae_model import VAE
from src.diffusion_model import DiffusionUNet
from src.gcn_model import LatentDenseGCN

def train_vae(model, dataloader, epochs=50, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    print(f"\n[VAE] Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.float().to(device)
            optimizer.zero_grad()
            recon, mu, logvar, z = model(x)
            recon_loss = F.mse_loss(recon, x.unsqueeze(1), reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch+1) % 5 == 0:
            print(f"[VAE] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    return model

def train_diffusion(unet, vae, dataloader, epochs=100, device='cpu'):
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    unet.train()
    vae.eval()
    criterion = nn.MSELoss()
    print(f"\n[Diffusion] Starting Training for {epochs} epochs...")
    
    timesteps = 1000
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.float().to(device)
            with torch.no_grad():
                _, _, _, z = vae(x)
                
            t = torch.randint(0, timesteps, (z.shape[0],), device=device).long()
            noise = torch.randn_like(z)
            
            sqrt_ac = torch.sqrt(alphas_cumprod)[t].view(-1, 1, 1, 1)
            sqrt_one_minus_ac = torch.sqrt(1 - alphas_cumprod)[t].view(-1, 1, 1, 1)
            noisy_z = sqrt_ac * z + sqrt_one_minus_ac * noise
            
            optimizer.zero_grad()
            noise_pred = unet(noisy_z, t)
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch+1) % 10 == 0:
            print(f"[Diffusion] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
    return unet

def train_latent_gcn(gcn, vae, dataloader, epochs=100, device='cpu'):
    vae.eval()
    print(f"\n[Latent GCN] Extracting Latents for {len(dataloader.dataset)} samples...")
    latents, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            _, _, _, z = vae(x)
            latents.append(z.cpu())
            labels.append(y.cpu())
            
    X = torch.cat(latents, dim=0).to(device)
    y = torch.cat(labels, dim=0).to(device)
    
    # Class Distribution
    unique, counts = torch.unique(y, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"[Latent GCN] Class Distribution: {dist}")
    
    # Stratified Split
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(kf.split(X.cpu(), y.cpu()))
    
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    test_ds = TensorDataset(X[test_idx], y[test_idx])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # Adjust weights for 3 classes
    # Simple inverse frequency weighting
    weights = []
    for c in range(3):
        weights.append(1.0 / dist.get(c, 1.0))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    weights = weights / weights.min() # Normalize so smallest is 1.0
    print(f"[Latent GCN] Using class weights: {weights}")
    
    optimizer = optim.Adam(gcn.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    best_acc = 0.0
    print(f"[Latent GCN] Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        gcn.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = gcn(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            
        train_acc = 100 * correct / total
        
        gcn.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                outputs = gcn(xb)
                _, predicted = torch.max(outputs.data, 1)
                val_total += yb.size(0)
                val_correct += (predicted == yb).sum().item()
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(gcn.state_dict(), 'gcn_3class.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"[Latent GCN] Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
            
    # Load back the best
    gcn.load_state_dict(torch.load('gcn_3class.pth', weights_only=True))
    return gcn

def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    matrices, labels = load_adni_data(data_dir='./data')
    
    # Filter out invalid labels (-1)
    valid_mask = labels != -1
    num_invalid = len(labels) - valid_mask.sum()
    if num_invalid > 0:
        print(f"[DataLoader] Filtering out {num_invalid} samples with invalid label (-1).")
        matrices = matrices[valid_mask]
        labels = labels[valid_mask]
    
    tensor_x = torch.from_numpy(matrices)
    tensor_y = torch.from_numpy(labels).long()
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 1. Train VAE
    vae = VAE(input_dim=(100, 100), latent_dim=256).to(device)
    vae = train_vae(vae, dataloader, epochs=args.epochs_vae, device=device)
    torch.save(vae.state_dict(), 'vae_3class.pth')
    
    # 2. Train Diffusion
    unet = DiffusionUNet().to(device)
    unet = train_diffusion(unet, vae, dataloader, epochs=args.epochs_diff, device=device)
    torch.save(unet.state_dict(), 'diffusion_3class.pth')
    
    # 3. Train Latent Dense GCN (n_classes=3)
    gcn = LatentDenseGCN(num_nodes=16, in_features=16, hidden_dim=32, n_classes=3).to(device)
    gcn = train_latent_gcn(gcn, vae, dataloader, epochs=args.epochs_gcn, device=device)
    # Model saving is handled inside function to keep the best validation state
    print("Pipeline Complete Phase 1.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs_vae', type=int, default=50)
    parser.add_argument('--epochs_diff', type=int, default=100)
    parser.add_argument('--epochs_gcn', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    run_training(args)

if __name__ == "__main__":
    main()
