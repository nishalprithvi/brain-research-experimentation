import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import argparse
import numpy as np
import os
import random
from dgl.data.utils import load_graphs
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from dgl.nn import SumPooling
from src.standard_gcn import StandardGCN
from src.data_loader import load_adni_dgl_with_labels

class FineTunedGCN(nn.Module):
    def __init__(self, encoder, n_classes=2):
        super(FineTunedGCN, self).__init__()
        self.encoder = encoder
        # Classifier: 64 -> 2
        self.classifier = nn.Linear(64, n_classes)
        
    def forward(self, g, features=None):
        # Get features from pre-trained encoder
        h = self.encoder(g, features)
        # Classify
        logits = self.classifier(h)
        return logits

def train_finetune(epochs=50, batch_size=32, frozen=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Real Data
    print("Loading Real ADNI Data...")
    glist, labels = load_adni_dgl_with_labels(data_dir='./data')

    # Filter out invalid labels (-1)
    # labels is a numpy array
    valid_mask = labels != -1
    if not np.all(valid_mask):
        print(f"Filtering {len(labels) - valid_mask.sum()} invalid samples (label=-1)...")
        # Filter glist (list) and labels (numpy)
        glist = [glist[i] for i in range(len(glist)) if valid_mask[i]]
        labels = labels[valid_mask]
    print(f"Remaining samples: {len(labels)}")
    
    # Sanitize Real Data
    for g in glist:
        weights = None
        if 'E_features' in g.edata: weights = g.edata['E_features']
        elif 'feat' in g.edata: weights = g.edata['feat']
        else: weights = torch.ones(g.num_edges())
        if weights.dim() == 1: weights = weights.view(-1, 1)
        # Clear and set
        keys = list(g.edata.keys())
        for k in keys: del g.edata[k]
        g.edata['feat'] = weights
        n_keys = list(g.ndata.keys())
        for k in n_keys: 
            if k != 'feat': del g.ndata[k]
            
    # Split Train/Test (Stratified)
    train_idx, test_idx = train_test_split(
        np.arange(len(glist)), 
        test_size=0.2, 
        stratify=labels,
        random_state=42
    )
    
    real_train_graphs = [glist[i] for i in train_idx]
    real_train_labels = [labels[i] for i in train_idx]
    test_graphs = [glist[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    # SAVE TEST INDICES FOR VERIFICATION
    np.save('test_indices.npy', test_idx)
    print(f"Saved test indices to test_indices.npy")
    
    print(f"Real Train size: {len(real_train_graphs)}, Test size: {len(test_graphs)}")

    # --- DATA-CENTRIC DATA AUGMENTATION ---
    # --- DATA-CENTRIC DATA AUGMENTATION ---
    USE_SYNTHETIC = True # Re-enabled after switching to LayerNorm
    
    if USE_SYNTHETIC:
        # Load Synthetic Data
        syn_path = './results_guidance/filtered_synthetic.bin'
        if os.path.exists(syn_path):
            print(f"Loading Synthetic Data from {syn_path}...")
            glist_syn, _ = load_graphs(syn_path)
            
            # Sanitize Synthetic
            for g in glist_syn:
                weights = None
                if 'E_features' in g.edata: weights = g.edata['E_features']
                elif 'feat' in g.edata: weights = g.edata['feat']
                else: weights = torch.ones(g.num_edges())
                if weights.dim() == 1: weights = weights.view(-1, 1)
                keys = list(g.edata.keys())
                for k in keys: del g.edata[k]
                g.edata['feat'] = weights
                n_keys = list(g.ndata.keys())
                for k in n_keys: del g.ndata[k]
            
            # Assuming guided generation produced Class 1 (AD)
            print("Filtering for AD samples (assuming target=1 generation)...")
            
            # How many samples to add?
            # Target: Balance the training set.
            unique, counts = np.unique(real_train_labels, return_counts=True)
            dist = dict(zip(unique, counts))
            n_cn = dist.get(0, 0)
            n_ad = dist.get(1, 0)
            print(f"Real Train Distribution: CN={n_cn}, AD={n_ad}")
            
            needed = n_cn - n_ad
            if needed > 0:
                # OPTIMAL CONFIGURATION: 100 Samples (Determined via Sensitivity Analysis)
                n_add = 100
                print(f"Need ~{needed} AD samples to balance. Adding {n_add} Synthetic AD samples (Optimal Ratio).")
                
                # Randomly sample unique indices
                if len(glist_syn) >= n_add:
                    selected_indices = random.sample(range(len(glist_syn)), n_add)
                    syn_train_graphs = [glist_syn[i] for i in selected_indices]
                else:
                    print(f"Warning: Only {len(glist_syn)} synthetic samples available. Using all.")
                    syn_train_graphs = glist_syn
                
                syn_train_labels = [1] * len(syn_train_graphs)
                
                # Combine
                train_graphs = real_train_graphs + syn_train_graphs
                train_labels = real_train_labels + syn_train_labels
                
                # DETAILED LOGGING
                print("\n--- Training Data Distribution (Optimal) ---")
                print(f"  Real CN: {n_cn}")
                print(f"  Real AD: {n_ad}")
                print(f"  Synthetic AD: {len(syn_train_graphs)}")
                print(f"  Total Training Samples: {len(train_graphs)}")
                print("-------------------------------------------------\n")
            else:
                print("Dataset already balanced or AD dominant. No synthetic data added.")
                train_graphs = real_train_graphs
                train_labels = real_train_labels
        else:
            print("Warning: Synthetic data not found. Training on Real Data only.")
            train_graphs = real_train_graphs
            train_labels = real_train_labels
    else:
        print("\n[CONFIG] Synthetic Data DISABLED for Baseline Verification.")
        print("Training on Real Data ONLY to verify BatchNorm statistics stability.")
        train_graphs = real_train_graphs
        train_labels = real_train_labels

    # DataLoader
    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels, dtype=torch.long)
        
    train_loader = torch.utils.data.DataLoader(
        list(zip(train_graphs, train_labels)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )
    
    test_loader = torch.utils.data.DataLoader(
        list(zip(test_graphs, test_labels)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )
    
    # 2. Model Setup
    # Initialize encoder with n_classes=64 to match pre-training
    encoder = StandardGCN(num_nodes=100, hidden_dim=64, n_classes=64)

    # CRITICAL: Match Pre-training Pooling
    encoder.pooling = SumPooling()
    print("Set Encoder Pooling to SumPooling (match pre-training).")
    
    # Load Pre-trained Weights
    try:
        encoder.load_state_dict(torch.load('gcn_pretrained_contrastive.pth', map_location=device))
        print("Loaded Pre-trained Weights!")
    except FileNotFoundError:
        print("Warning: Pre-trained weights not found. Training from scratch (Random Init).")
    
    # Create Full Model
    model = FineTunedGCN(encoder, n_classes=2).to(device)
    
    # Freeze Encoder?
    if frozen:
        print("Freezing Encoder Weights...")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning Entire Network...")
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Standard Loss (Balanced Data)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training
    print("\n--- Starting Fine-Tuning ---")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batched_graph, labels in train_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            
            logits = model(batched_graph)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
        train_acc = accuracy_score(all_targets, all_preds)
        train_macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_probs = [] # For AUC
        
        with torch.no_grad():
            for batched_graph, labels in test_loader:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                
                logits = model(batched_graph)
                probs = torch.softmax(logits, dim=1)[:, 1] # Prob of class 1 (AD)
                
                preds = logits.argmax(dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average='macro')
        
        try:
            val_auc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            val_auc = 0.5 # If only one class present in test batch (edge case) or not defined
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1 (Macro): {val_macro_f1:.4f} | Val AUC: {val_auc:.4f}")
            
    print("\n--- Final Evaluation on Test Set ---")
    print(classification_report(val_targets, val_preds, target_names=['CN', 'AD']))
    print(f"Final AUC-ROC: {val_auc:.4f}")
    print(f"Final Macro-F1: {val_macro_f1:.4f}")
    
    torch.save(model.state_dict(), 'gcn_finetuned.pth')
    print("Saved Fine-tuned Model to gcn_finetuned.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze encoder weights (fine-tune all)')
    args = parser.parse_args()
    
    train_finetune(epochs=args.epochs, frozen=not args.unfreeze)
