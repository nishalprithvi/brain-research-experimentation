
import dgl
from dgl.data.utils import load_graphs, save_graphs
import numpy as np
import argparse
import os
import torch
from src.data_loader import load_adni_dgl_with_labels

def filter_synthetic_data(synthetic_path, real_data_dir='./data', output_path='./results_guidance/filtered_synthetic.bin', threshold_min=0.5, threshold_max=0.98):
    """
    Filters synthetic graphs to ensure they are:
    1. Realistic (High enough similarity to real brains)
    2. Unique (Not an exact copy of a real brain)
    """
    print(f"Loading synthetic data from {synthetic_path}...")
    try:
        syn_graphs, _ = load_graphs(synthetic_path)
    except Exception as e:
        print(f"Error loading synthetic data: {e}")
        return

    print(f"Loading real data from {real_data_dir}...")
    real_graphs, _ = load_adni_dgl_with_labels(data_dir=real_data_dir)
    
    # needed: simple way to get adjacency from DGL graphs effectively for correlation
    # We iterate and extract 'feat' or 'E_features' or rebuild form edges
    
    def get_adj_flat(glist):
        flat_list = []
        for g in glist:
            # Reconstruct weighted adjacency
            # Assuming 100 nodes
            # Edges:
            if 'feat' in g.edata:
                w = g.edata['feat'].squeeze().numpy()
            elif 'E_features' in g.edata:
                w = g.edata['E_features'].squeeze().numpy()
            else:
                w = np.ones(g.num_edges()) # binary
                
            # If graph is fully connected (100x100), edge order matters.
            # DGL edges correspond to src, dst arrays properly.
            # Since correlation is order-independent IF the order of edges is CONSISTENT?
            # Yes, if we always construct graphs same way (0->0, 0->1... 99->99).
            # But real graphs might be sparse or ordered differently?
            # Schaefer100 is fully connected usually.
            # Safer: convert to Dense Matrix (100x100) then Flatten.
            
            src, dst = g.edges()
            # If sparse, this is tricky. 
            # Let's assume Dense for correlation safety.
            adj = np.zeros((100, 100))
            adj[src, dst] = w
            
            # Use upper triangle? or full flat?
            # Full flat is fine.
            flat_list.append(adj.flatten())
            
        return np.array(flat_list)

    print("converting graphs to vectors for comparing...")
    real_flat = get_adj_flat(real_graphs)
    syn_flat = get_adj_flat(syn_graphs)
    
    print(f"Real Vectors: {real_flat.shape}")
    print(f"Syn Vectors: {syn_flat.shape}")
    
    # Normalize for Correlation (Pearson)
    # Norm: (x - mean) / std
    real_mean = real_flat.mean(axis=1, keepdims=True)
    real_std = real_flat.std(axis=1, keepdims=True) + 1e-8
    real_norm = (real_flat - real_mean) / real_std
    
    syn_mean = syn_flat.mean(axis=1, keepdims=True)
    syn_std = syn_flat.std(axis=1, keepdims=True) + 1e-8
    syn_norm = (syn_flat - syn_mean) / syn_std
    
    print("Computing Correlation Matrix (Process may take a moment)...")
    # (M, D) @ (D, N) -> (M, N) 
    # M synthetic, N real.
    # Dimensions: 500 x 10000 @ 10000 x 1327
    # 5M operations. Fast.
    corr_matrix = np.dot(syn_norm, real_norm.T) / real_flat.shape[1]
    
    valid_indices = []
    
    print(f"Filtering with thresholds: Min={threshold_min}, Max={threshold_max}")
    for i in range(len(syn_norm)):
        # Max Sim with any real brain
        max_sim = np.max(corr_matrix[i])
        
        # Check 1: Junk
        if max_sim < threshold_min:
            continue
            
        # Check 2: Clone
        if max_sim > threshold_max:
            continue
            
        valid_indices.append(i)
        
    print(f"Filter Complete. Kept {len(valid_indices)} / {len(syn_graphs)} unique & valid brains.")
    
    if len(valid_indices) > 0:
        # subset DGL list
        # We need to pick from generic list
        filtered_graphs = [syn_graphs[i] for i in valid_indices]
        
        save_graphs(output_path, filtered_graphs)
        print(f"Saved filtered data to {output_path}")
        
        # Log
        log_path = os.path.dirname(output_path)
        with open(os.path.join(log_path, 'experiment_log.txt'), 'a') as f:
            f.write(f"\n--- Uniqueness Filtering ---\n")
            f.write(f"Input Samples: {len(syn_graphs)}\n")
            f.write(f"Valid Samples: {len(valid_indices)}\n")
            f.write(f"Thresholds: {threshold_min} < r < {threshold_max}\n")
    else:
        print("No samples passed the filter! Try generating with lower guidance scale.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--syn_path', type=str, default='./results_guidance/synthetic_hard_negatives.bin')
    parser.add_argument('--threshold_min', type=float, default=0.5)
    parser.add_argument('--threshold_max', type=float, default=0.98)
    args = parser.parse_args()
    
    filter_synthetic_data(args.syn_path, threshold_min=args.threshold_min, threshold_max=args.threshold_max)
