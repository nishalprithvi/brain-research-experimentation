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
from src.data_loader_3class import load_adni_dgl_with_labels

class FineTunedGCN(nn.Module):
    def __init__(self, encoder, n_classes=3):
        super(FineTunedGCN, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(64, n_classes)
        
    def forward(self, g, features=None):
        h = self.encoder(g, features)
        logits = self.classifier(h)
        return logits

def train_finetune(epochs=50, batch_size=32, frozen=True, syn_dir='./results_guidance_3class'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Real Data
    print("Loading Real ADNI Data...")
    glist, labels = load_adni_dgl_with_labels(data_dir='./data')

    valid_mask = labels != -1
    if not np.all(valid_mask):
        print(f"Filtering {len(labels) - valid_mask.sum()} invalid samples (label=-1)...")
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
    
    np.save('test_indices_3class.npy', test_idx)
    
    print(f"Real Train size: {len(real_train_graphs)}, Test size: {len(test_graphs)}")

    # Data Balancing
    train_graphs = list(real_train_graphs)
    train_labels = list(real_train_labels)
    
    unique, counts = np.unique(real_train_labels, return_counts=True)
    dist = dict(zip(unique, counts))
    n_cn = dist.get(0, 0)
    n_ad = dist.get(1, 0)
    n_mci = dist.get(2, 0)
    
    print(f"Real Train Distribution: CN={n_cn}, AD={n_ad}, MCI={n_mci}")
    
    def inject_synthetic(cls_label, cls_name, cls_count, max_inject=100):
        needed = n_cn - cls_count
        if needed > 0:
            syn_path = os.path.join(syn_dir, f'filtered_synthetic_{cls_name}.bin')
            if os.path.exists(syn_path):
                print(f"Loading Synthetic {cls_name.upper()} Data from {syn_path}...")
                glist_syn, _ = load_graphs(syn_path)
                
                # Sanitize Synthetic
                for g in glist_syn:
                    w = g.edata.get('E_features', g.edata.get('feat', torch.ones(g.num_edges())))
                    if w.dim() == 1: w = w.view(-1, 1)
                    for k in list(g.edata.keys()): del g.edata[k]
                    g.edata['feat'] = w
                    for k in list(g.ndata.keys()): del g.ndata[k]
                
                # Apply hard cap to prevent overwhelming biological signal
                n_add_calculated = min(needed, len(glist_syn))
                n_add = min(n_add_calculated, max_inject)
                print(f"Calculated deficit: {needed}. Adding {n_add} Synthetic {cls_name.upper()} samples to balance (Capped at {max_inject}).")
                
                # Reseed python random manually to ensure reproducibility
                random.seed((torch.initial_seed() + cls_label) % (1<<31))
                selected_indices = random.sample(range(len(glist_syn)), n_add)
                syn_train_graphs = [glist_syn[i] for i in selected_indices]
                
                train_graphs.extend(syn_train_graphs)
                train_labels.extend([cls_label] * n_add)
            else:
                print(f"Warning: {syn_path} not found. Proceeding unbalanced for {cls_name.upper()}.")

    # Inject AD and MCI (Capped at 100 each, consistent with optimized 2-class setup)
    inject_synthetic(1, 'ad', n_ad, max_inject=100)
    inject_synthetic(2, 'mci', n_mci, max_inject=100)
    
    print(f"\nTotal Training Samples after Synthesis: {len(train_graphs)}")

    # DataLoader
    def collate(samples):
        graphs, lbls = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(lbls, dtype=torch.long)
        
    # Standard PyTorch Generator for Repro
    g_train = torch.Generator()
    g_train.manual_seed(torch.initial_seed() % (1<<31))
    
    train_loader = torch.utils.data.DataLoader(
        list(zip(train_graphs, train_labels)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=g_train
    )
    
    test_loader = torch.utils.data.DataLoader(
        list(zip(test_graphs, test_labels)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )
    
    # 2. Model Setup
    encoder = StandardGCN(num_nodes=100, hidden_dim=64, n_classes=64)
    encoder.pooling = SumPooling()
    
    try:
        encoder.load_state_dict(torch.load('gcn_pretrained_3class.pth', map_location=device))
        print("Loaded Pre-trained Weights!")
    except FileNotFoundError:
        print("Warning: Pre-trained weights not found. Training from scratch (Random Init).")
    
    model = FineTunedGCN(encoder, n_classes=3).to(device)
    
    if frozen:
        print("Freezing Encoder Weights...")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning Entire Network...")
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Checkpointing Logic
    best_val_auc = 0.0
    best_epoch = 0
    best_model_path = 'gcn_finetuned_3class_best.pth'
    
    print("\n--- Starting Fine-Tuning ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_targets = [], []
        
        for batched_graph, lbls in train_loader:
            batched_graph, lbls = batched_graph.to(device), lbls.to(device)
            
            logits = model(batched_graph)
            loss = criterion(logits, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.cpu().numpy())
            
        train_acc = accuracy_score(all_targets, all_preds)
        train_macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        
        with torch.no_grad():
            for batched_graph, lbls in test_loader:
                batched_graph, lbls = batched_graph.to(device), lbls.to(device)
                logits = model(batched_graph)
                probs = torch.softmax(logits, dim=1) # (B, 3)
                preds = logits.argmax(dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(lbls.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        
        try:
            val_auc = roc_auc_score(val_targets, val_probs, multi_class='ovr')
        except ValueError:
            val_auc = 0.5 
            
        # Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_macro_f1:.4f} | Val AUC: {val_auc:.4f}")
            
    # Load best model for Final Eval
    print(f"\n--- Loading Best Model from Epoch {best_epoch} (Val AUC: {best_val_auc:.4f}) ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        
    model.eval()
    val_preds, val_targets, val_probs = [], [], []
    with torch.no_grad():
        for batched_graph, lbls in test_loader:
            batched_graph, lbls = batched_graph.to(device), lbls.to(device)
            logits = model(batched_graph)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(lbls.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())
            
    val_auc = roc_auc_score(val_targets, val_probs, multi_class='ovr')
    val_macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
    
    print("\n--- Final Evaluation on Test Set ---")
    print(classification_report(val_targets, val_preds, target_names=['CN', 'AD', 'MCI'], zero_division=0))
    print(f"Final Multi-Class AUC-ROC (OVR): {val_auc:.4f}")
    print(f"Final Macro-F1: {val_macro_f1:.4f}")
    
    torch.save(model.state_dict(), 'gcn_finetuned_3class.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--unfreeze', action='store_true')
    args = parser.parse_args()
    
    train_finetune(epochs=args.epochs, frozen=not args.unfreeze)
