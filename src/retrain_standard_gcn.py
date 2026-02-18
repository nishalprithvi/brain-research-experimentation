
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
import dgl
import numpy as np
import os
import argparse
from dgl.data.utils import load_graphs
from sklearn.model_selection import StratifiedKFold
from src.standard_gcn import StandardGCN
from src.data_loader import load_adni_dgl_with_labels

def collate(samples):
    # DGL GraphDataLoader needs custom collate if samples are (graph, label)
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def train_standard_gcn(real_data_dir='./data', syn_data_path='./results_guidance/synthetic_hard_negatives.bin', epochs=100, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Function to sanitize graphs
    def sanitize(glist):
        for g in glist:
            # Standardize Edge Features to 'feat' with shape (E, 1)
            # Remove others to avoid schema mismatch
            
            # 1. Get weights
            weights = None
            if 'E_features' in g.edata:
                weights = g.edata['E_features']
            elif 'feat' in g.edata:
                weights = g.edata['feat']
            else:
                weights = torch.ones(g.num_edges()) # Default 1s
                
            # 2. Ensure shape (E, 1)
            # If (E,), view as (E, 1)
            if weights.dim() == 1:
                weights = weights.view(-1, 1)
            
            # 3. Clean edata
            keys = list(g.edata.keys())
            for k in keys:
                del g.edata[k]
                
            # 4. Set standard
            g.edata['feat'] = weights
            # g.edata['E_features'] = weights # Optional, keep consistent if needed
            
            # Clean ndata too if needed (N_features)
            n_keys = list(g.ndata.keys())
            for k in n_keys:
                if k != 'feat': # Keep 'feat' if logic uses it? StandardGCN uses node_embedding, doesn't use ndata
                     del g.ndata[k]
        return glist

    # 1. Load Real Data
    print("Loading Real Data...")
    glist_real, labels_real = load_adni_dgl_with_labels(data_dir=real_data_dir)
    glist_real = sanitize(glist_real)
    
    # Filter AD/CN
    # Labels: 0=CN, 1=AD, -1=Unknown/Other
    mask = (labels_real == 0) | (labels_real == 1)
    
    # Apply mask to list... list doesn't support bool indexing directly easily
    # Comprehension
    glist_real_filtered = []
    labels_real_filtered = []
    for i, m in enumerate(mask):
        if m:
            glist_real_filtered.append(glist_real[i])
            labels_real_filtered.append(labels_real[i])
            
    labels_real_filtered = np.array(labels_real_filtered)
    print(f"Real Samples (CN/AD): {len(glist_real_filtered)}")
    unique, counts = np.unique(labels_real_filtered, return_counts=True)
    print(f"Real Distribution: {dict(zip(unique, counts))}")
    
    # 2. Load Synthetic Data
    if os.path.exists(syn_data_path):
        print(f"Loading Synthetic Data from {syn_data_path}...")
        glist_syn, _ = load_graphs(syn_data_path)
        # Sanitize Synthetic Data too
        glist_syn = sanitize(glist_syn)
        
        # Synthetic data is guided to be AD (Class 1)
        labels_syn = np.ones(len(glist_syn), dtype=int)
        print(f"Synthetic Samples: {len(glist_syn)}")
    else:
        print("No Synthetic Data found. Training on Real only.")
        glist_syn = []
        labels_syn = np.array([], dtype=int)
        
    # 3. Combine for Train, use Real only for Test?
    # Standard practice: Augment Train, Test on Real.
    # Split Real into Train/Test first.
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(kf.split(np.zeros(len(labels_real_filtered)), labels_real_filtered))
    
    g_real_train = [glist_real_filtered[i] for i in train_idx]
    l_real_train = labels_real_filtered[train_idx]
    
    g_real_test = [glist_real_filtered[i] for i in test_idx]
    l_real_test = labels_real_filtered[test_idx]
    
    # Augment Training Set
    g_train = g_real_train + glist_syn
    l_train = np.concatenate([l_real_train, labels_syn])
    
    print(f"Training Set: {len(g_train)} ({len(g_real_train)} Real + {len(glist_syn)} Syn)")
    print(f"Test Set (Real Only): {len(g_real_test)}")
    
    # Create DataLoaders
    train_data = list(zip(g_train, l_train))
    test_data = list(zip(g_real_test, l_real_test))
    
    train_loader = GraphDataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = GraphDataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    # 4. Model
    model = StandardGCN(num_nodes=100, hidden_dim=64, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Class Weights?
    # New distribution in Train might be balanced or imbalanced differently.
    unique_train, counts_train = np.unique(l_train, return_counts=True)
    dist_train = dict(zip(unique_train, counts_train))
    print(f"Train Distribution: {dist_train}")
    
    n_cn = dist_train.get(0, 0)
    n_ad = dist_train.get(1, 0)
    if n_ad > 0:
        weight = torch.tensor([1.0, n_cn / n_ad], dtype=torch.float32).to(device)
        print(f"Using class weights: {weight}")
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
        
    # 5. Training Loop
    print("\n--- Starting Training StandardGCN ---")
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for bg, labels in train_loader:
            bg = bg.to(device)
            labels = labels.to(device)
            
            # Forward
            # Use node embeddings, no external features
            logits = model(bg)
            # Ensure logits are float
            logits = logits.float()
            
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bg, labels in test_loader:
                bg = bg.to(device)
                labels = labels.to(device)
                logits = model(bg)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'standard_gcn_best.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    # Log to file
    log_path = './results_guidance/experiment_log.txt'
    if not os.path.exists('./results_guidance'):
        os.makedirs('./results_guidance')
    with open(log_path, 'a') as f:
        f.write(f"\n--- Retraining StandardGCN ---\n")
        f.write(f"Real Samples: {len(glist_real_filtered)}\n")
        f.write(f"Synthetic Samples: {len(glist_syn)}\n")
        f.write(f"Best Test Accuracy: {best_acc:.2f}%\n")
        f.write("-" * 30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--syn_path', type=str, default='./results_guidance/synthetic_hard_negatives.bin')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    train_standard_gcn(syn_data_path=args.syn_path, epochs=args.epochs)
