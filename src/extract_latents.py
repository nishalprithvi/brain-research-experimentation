
import torch
import numpy as np
import os
import argparse
from src.vae_model import VAE
from src.data_loader import load_adni_data

def extract_latents(data_dir='./data', model_path='vae_adni.pth', output_dir='./data'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading Data...")
    matrices, labels = load_adni_data(data_dir=data_dir)
    print(f"Total samples: {len(labels)}")
    
    # 2. Filter for AD/CN (0/1)
    mask = (labels == 0) | (labels == 1)
    filtered_matrices = matrices[mask]
    filtered_labels = labels[mask]
    
    # Check distribution
    unique, counts = np.unique(filtered_labels, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Filtered samples: {len(filtered_labels)}")
    print(f"Distribution: {dist}") # {0: N_CN, 1: N_AD}
    
    # 3. Load VAE
    vae = VAE(input_dim=(100, 100), latent_dim=256).to(device)
    if os.path.exists(model_path):
        vae.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded VAE from {model_path}")
    else:
        raise FileNotFoundError(f"VAE model not found at {model_path}. Please train VAE first.")
    
    vae.eval()
    
    # 4. Extract Latents
    tensor_x = torch.from_numpy(filtered_matrices).float().to(device)
    
    latents_list = []
    batch_size = 32
    
    print("Encoding matrices to latent space...")
    with torch.no_grad():
        for i in range(0, len(tensor_x), batch_size):
            batch = tensor_x[i:i+batch_size]
            _, mu, _, _ = vae(batch) # Use mu (clean mean) as the representation
            # mu shape: (B, 256) -> Reshape to (B, 1, 16, 16)
            mu = mu.view(-1, 1, 16, 16)
            latents_list.append(mu.cpu())
            
    latents = torch.cat(latents_list, dim=0)
    labels_tensor = torch.from_numpy(filtered_labels).long()
    
    print(f"Extracted Latents Shape: {latents.shape}")
    print(f"Labels Shape: {labels_tensor.shape}")
    
    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    save_path_x = os.path.join(output_dir, 'latents.pt')
    save_path_y = os.path.join(output_dir, 'labels.pt')
    
    torch.save(latents, save_path_x)
    torch.save(labels_tensor, save_path_y)
    print(f"Saved to {save_path_x} and {save_path_y}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    
    extract_latents(data_dir=args.data_dir)
