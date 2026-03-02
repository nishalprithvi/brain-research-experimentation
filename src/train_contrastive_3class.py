
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np
import os
import argparse
from dgl.data.utils import load_graphs
from src.standard_gcn import StandardGCN
from src.data_loader_3class import load_adni_dgl_with_labels


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=64):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        
        # Projector: 64 -> 64
        self.projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, projection_dim)
        )
        
    def forward(self, g):
        # Get representation natively leveraging StandardGCN's internal learnable node_embedding matrix
        h = self.encoder(g, features=None)
        
        # Project
        z = self.projector(h)
        return z

def augment_graph(g, drop_edge_prob=0.2, mask_feat_prob=0.2):
    """
    Graph Augmentation:
    1. Drop Edges randomly.
    2. Mask Node Features (if we had them, here we assume randomness or just edge drop).
    """
    # Clone to avoid mutating original
    g_aug = g.clone()
    
    # 1. Drop Edges
    if drop_edge_prob > 0:
        num_edges = g_aug.num_edges()
        mask = torch.empty(num_edges).uniform_(0, 1) > drop_edge_prob
        # Select edges to keep using boolean mask? DGL remove_edges uses IDs.
        # It's faster to pick edges to REMOVE.
        remove_indices = torch.nonzero(~mask).squeeze()
        if len(remove_indices.shape) > 0 and len(remove_indices) > 0:
             g_aug.remove_edges(remove_indices)
             
    # 2. Mask Features (Optional but good)
    if mask_feat_prob > 0 and 'feat' in g_aug.ndata:
        feat = g_aug.ndata['feat']
        mask = torch.empty((feat.shape[0], feat.shape[1])).uniform_(0, 1) > mask_feat_prob
        g_aug.ndata['feat'] = feat * mask.float()
    
    return g_aug

def info_nce_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (B, D) projections of two views of the batch.
    """
    batch_size = z1.shape[0]
    device = z1.device
    
    # Normalize
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    
    # Similarity (B, B)
    # logits[i, j] = sim(z1[i], z2[j])
    # We want logits[i, i] to be high (positive pair)
    # And everything else to be low.
    
    # Concatenate representations: all_z = [z1, z2] (2B, D) to compare against ALL negatives?
    # SimCLR standard:
    # Positive pair: (z1[i], z2[i])
    # Negatives: z1[i] vs z1[j], z1[i] vs z2[j] (j!=i)
    
    # Let's implementation simple version: z1 vs z2
    logits = torch.matmul(z1, z2.T) / temperature
    
    # Target is diagonal
    labels = torch.arange(batch_size, device=device).long()
    
    # Symmetric loss?
    # Usually we compare [z1, z2] against [z1, z2].
    
    # Easy implementation:
    # Loss(z1, z2) + Loss(z2, z1)
    
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def collate_cl(graphs):
    # Collate for Contrastive Learning
    # We need to return TWO augmented views for each graph
    view1 = []
    view2 = []
    
    for g in graphs:
        v1 = augment_graph(g)
        v2 = augment_graph(g)
        view1.append(v1)
        view2.append(v2)
        
    batch1 = dgl.batch(view1)
    batch2 = dgl.batch(view2)
    return batch1, batch2

def train_contrastive(epochs=100, batch_size=32, syn_dir='./results_guidance_3class'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data (Real + Synthetic)
    print("Loading Real Data...")
    glist_real, _ = load_adni_dgl_with_labels(data_dir='./data')
    
    # Sanitize (Copied from retrain script logic)
    def sanitize(glist):
        for g in glist:
            # 1. Edge Weights
            weights = None
            if 'E_features' in g.edata: weights = g.edata['E_features']
            elif 'feat' in g.edata: weights = g.edata['feat']
            else: weights = torch.ones(g.num_edges())
            if weights.dim() == 1: weights = weights.view(-1, 1)
            
            # Clean
            keys = list(g.edata.keys())
            for k in keys: del g.edata[k]
            g.edata['feat'] = weights # Keep edge weights
            
            n_keys = list(g.ndata.keys())
            for k in n_keys: del g.ndata[k]
            
            # 2. Sparsification (Top 20%)
            # Reduce density from ~0.8 to ~0.2 to prevent oversmoothing
            if weights is not None:
                # Calculate threshold for Top 20%
                k = int(g.num_edges() * 0.2)
                if k > 0:
                    # Flatten weights
                    w_flat = weights.view(-1)
                    # Find top-k threshold
                    topk_val = torch.topk(w_flat, k).values[-1]
                    
                    # Identify edges to REMOVE (those below threshold)
                    # Use a small epsilon to handle ties or exact binary matches if any
                    mask = w_flat < topk_val
                    remove_indices = torch.nonzero(mask).squeeze()
                    
                    if len(remove_indices) > 0:
                        original_edges = g.num_edges()
                        g.remove_edges(remove_indices)
                        
        return glist
        
    glist_real = sanitize(glist_real)
    
    glist_syn = []
    
    # --- BASELINE VERIFICATION: DISABLE SYNTHETIC DATA ---
    USE_SYNTHETIC = True # Re-enabled after switching to LayerNorm
    
    if USE_SYNTHETIC:
        for cls_name in ['ad', 'mci']:
            syn_path = os.path.join(syn_dir, f'filtered_synthetic_{cls_name}.bin')
            if os.path.exists(syn_path):
                print(f"Loading Synthetic Data from {syn_path}...")
                glist_tmp, _ = load_graphs(syn_path)
                glist_syn.extend(sanitize(glist_tmp))
            else:
                print(f"Warning: Synthetic Data not found at {syn_path}")
    else:
        print("[CONFIG] Synthetic Data DISABLED for Pre-Training (Baseline Verification).")
        
    full_dataset = glist_real + glist_syn
    print(f"Pre-Training on {len(full_dataset)} graphs (Real + Syn).")
    
    # DataLoader
    loader = torch.utils.data.DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_cl
    )
    
    # 2. Model
    encoder = StandardGCN(num_nodes=100, hidden_dim=64, n_classes=64) # Output 64 embedding
    # Note: StandardGCN usually outputs n_classes (2). We want separate features.
    # Hack: Set n_classes=64 to act as feature extractor output.
    
    # CRITICAL FIX: Replace AvgPooling with SumPooling
    # AvgPooling washes out sparse identity features. SumPooling preserves signal.
    from dgl.nn import SumPooling
    encoder.pooling = SumPooling()
    print("Replaced Encoder Pooling with SumPooling.")
    
    model = SimCLR(encoder, projection_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    print("\n--- Starting Contrastive Pre-Training ---")
    
    # Sanity Check: Graph Density
    if len(loader) > 0:
        sample_g = next(iter(loader))[0] # Get a batch
        # Unbatch to check one graph
        one_g = dgl.unbatch(sample_g)[0]
        density = one_g.num_edges() / (one_g.num_nodes() ** 2)
        print(f"DEBUG: Sample Graph Density: {density:.4f} (Edges/Nodes^2)")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for b1, b2 in loader:
            b1 = b1.to(device)
            b2 = b2.to(device)
            
            z1 = model(b1)
            z2 = model(b2)
            
            loss = info_nce_loss(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Check (First batch only)
            if epoch == 0 and total_loss == 0:
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item()
                print(f"DEBUG: Initial Gradient Norm: {grad_norm:.6f}")
                
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Contrastive Loss: {avg_loss:.4f}")
            
    # Save Encoder Only
    torch.save(model.encoder.state_dict(), 'gcn_pretrained_3class.pth')
    print("Saved Pre-Trained Encoder to gcn_pretrained_3class.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    train_contrastive(epochs=args.epochs)
