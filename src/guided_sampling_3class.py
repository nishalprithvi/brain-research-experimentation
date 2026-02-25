
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import dgl
from dgl.data.utils import save_graphs
from src.vae_model import VAE
from src.diffusion_model import DiffusionUNet
from src.gcn_model import LatentDenseGCN
from src.data_loader import load_adni_data

def calculate_uniqueness(synthetic_matrices, real_matrices_path='./data'):
    """
    Checks if synthetic matrices are just copies of real data.
    Metric: Max Pearson Correlation with any real sample.
    """
    print("Loading real data for uniqueness check...")
    try:
        real_matrices, _ = load_adni_data(data_dir=real_matrices_path)
        # Flatten for correlation: (N, 10000)
        real_flat = real_matrices.reshape(real_matrices.shape[0], -1)
        syn_flat = synthetic_matrices.reshape(synthetic_matrices.shape[0], -1)
        
        # Normalize
        real_norm = (real_flat - real_flat.mean(axis=1, keepdims=True)) / (real_flat.std(axis=1, keepdims=True) + 1e-8)
        syn_norm = (syn_flat - syn_flat.mean(axis=1, keepdims=True)) / (syn_flat.std(axis=1, keepdims=True) + 1e-8)
        
        # Compute Correlation pairwise
        # This might be heavy if N is large.
        # Let's do batch processing if needed. For 1300 real x 10 syn, it's fine.
        # shape: (10, 10000) @ (10000, 1300) -> (10, 1300)
        corr_matrix = np.dot(syn_norm, real_norm.T) / real_flat.shape[1]
        
        # Max correlation for each synthetic sample
        max_corrs = np.max(corr_matrix, axis=1)
        avg_max_corr = np.mean(max_corrs)
        
        print(f"Uniqueness Check: Avg Max Correlation with Real Data = {avg_max_corr:.4f}")
        return avg_max_corr
    except Exception as e:
        print(f"Warning: Could not check uniqueness. Error: {e}")
        return 0.0

def log_experiment(save_dir, log_data):
    log_path = os.path.join(save_dir, 'experiment_log.txt')
    with open(log_path, 'a') as f:
        f.write("-" * 50 + "\n")
        f.write(f"Experiment Run: {log_data.get('timestamp', '')}\n")
        for k, v in log_data.items():
            if k != 'timestamp':
                f.write(f"{k}: {v}\n")
        f.write("-" * 50 + "\n")
    print(f"Logged results to {log_path}")

def save_as_dgl(matrices, save_path):
    """
    Convert Adjacency Matrices to DGL Graphs and save as .bin
    """
    graphs = []
    for adj in matrices:
        # matrix is 100x100
        # Create graph from dense
        # adj is weighted. 
        # dgl.from_scipy wants sparse.
        # We can use torch convert.
        
        src, dst = np.nonzero(adj) # Get all non-zero edges?
        # If fully connected, this is dense.
        # Weighted graph:
        # Create edges for all > 0? Or just all?
        # Assuming fully connected weighted graph
        # Creating a graph with 10000 edges is fine.
        
        g = dgl.graph((src, dst), num_nodes=adj.shape[0])
        g.edata['feat'] = torch.tensor(adj[src, dst], dtype=torch.float32).unsqueeze(1)
        # Match 'E_features'?
        g.edata['E_features'] = g.edata['feat']
        graphs.append(g)
        
    save_graphs(save_path, graphs)
    print(f"Saved {len(graphs)} DGL graphs to {save_path}")

def guided_sampling(vae_path, unet_path, gcn_path, target_class=1, n_samples=10, guidance_scale=2.0, save_dir='./results_guidance_3class', save_name='synthetic.bin'):
    import datetime
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading models for Class {target_class} Generation...")
    vae = VAE(input_dim=(100, 100), latent_dim=256).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()
    
    unet = DiffusionUNet().to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()
    
    gcn = LatentDenseGCN(num_nodes=16, in_features=16, hidden_dim=32, n_classes=3).to(device)
    gcn.load_state_dict(torch.load(gcn_path, map_location=device, weights_only=True))
    gcn.eval()
    
    print(f"Generating {n_samples} samples with Guidance Scale {guidance_scale} for Class {target_class}...")
    
    batch_size = 500
    n_batches = (n_samples + batch_size - 1) // batch_size
    all_matrices = []
    
    timesteps = 1000
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    for b in range(n_batches):
        current_batch_size = min(batch_size, n_samples - b * batch_size)
        print(f"  Batch {b+1}/{n_batches} (Size: {current_batch_size})")
        
        shape = (current_batch_size, 1, 16, 16)
        x_t = torch.randn(shape).to(device)
        target = torch.full((current_batch_size,), target_class, device=device).long()
        
        for i in reversed(range(timesteps)):
            t = torch.full((current_batch_size,), i, device=device, dtype=torch.long)
            x_in = x_t.detach().requires_grad_(True)
            noise_pred = unet(x_in, t)
            
            beta_t = betas[i]
            alpha_t = alphas[i]
            alpha_hat = alphas_cumprod[i]
            sqrt_alpha_hat = torch.sqrt(alpha_hat)
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)
            
            x_0_hat = (x_in - sqrt_one_minus_alpha_hat * noise_pred) / sqrt_alpha_hat
            
            logits = gcn(x_0_hat)
            loss = F.cross_entropy(logits, target)
            grad = torch.autograd.grad(outputs=loss, inputs=x_in)[0]
            
            with torch.no_grad():
                gradient_scale = guidance_scale * sqrt_one_minus_alpha_hat
                noise_pred_guided = noise_pred + gradient_scale * grad
                
                if i > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                    
                coef1 = 1 / torch.sqrt(alpha_t)
                coef2 = (1 - alpha_t) / (sqrt_one_minus_alpha_hat)
                x_t = coef1 * (x_t - coef2 * noise_pred_guided) + torch.sqrt(beta_t) * noise
        
        with torch.no_grad():
            decoded_batch = vae.decode(x_t).cpu().squeeze().numpy()
            all_matrices.append(decoded_batch)
            
    matrices = np.concatenate(all_matrices, axis=0)
    print(f"Generated {len(matrices)} matrices total.")
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_as_dgl(matrices, os.path.join(save_dir, save_name))
    uniqueness_score = calculate_uniqueness(matrices)
    
    log_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'target_class': target_class,
        'guidance_scale': guidance_scale,
        'n_samples': n_samples,
        'uniqueness_avg_corr': f"{uniqueness_score:.4f}",
        'saved_file': save_name
    }
    log_experiment(save_dir, log_data)
    print(f"Finished generating Class {target_class}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=2.0)
    args = parser.parse_args()
    
    # Calculate required samples based on real dataset distribution
    from src.data_loader_3class import load_adni_data
    print("Evaluating real dataset distribution to balance classes...")
    _, labels = load_adni_data(data_dir='./data')
    
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Real Distribution: {dist}")
    
    n_cn = dist.get(0, 0)
    n_ad = dist.get(1, 0)
    n_mci = dist.get(2, 0)
    
    target_ad_samples = max(0, n_cn - n_ad)
    target_mci_samples = max(0, n_cn - n_mci)
    
    print(f"Targeting {target_ad_samples} Synthetic AD and {target_mci_samples} Synthetic MCI to match CN ({n_cn}).")
    
    if target_ad_samples > 0:
        guided_sampling(
            vae_path='vae_3class.pth',
            unet_path='diffusion_3class.pth',
            gcn_path='gcn_3class.pth',
            target_class=1,
            n_samples=target_ad_samples,
            guidance_scale=args.scale,
            save_name='synthetic_ad.bin'
        )
        
    if target_mci_samples > 0:
        guided_sampling(
            vae_path='vae_3class.pth',
            unet_path='diffusion_3class.pth',
            gcn_path='gcn_3class.pth',
            target_class=2,
            n_samples=target_mci_samples,
            guidance_scale=args.scale,
            save_name='synthetic_mci.bin'
        )

if __name__ == "__main__":
    main()
