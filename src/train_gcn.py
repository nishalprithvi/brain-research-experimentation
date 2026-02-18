
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
from sklearn.model_selection import StratifiedKFold
from src.gcn_model import LatentDenseGCN

def train_gcn(latents_path, labels_path, epochs=100, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading latents from {latents_path}")
    X = torch.load(latents_path).float().to(device) # (N, 1, 16, 16)
    y = torch.load(labels_path).long().to(device)   # (N,)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Check class balance
    unique, counts = torch.unique(y, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Class Distribution: {dist}")
    
    # Simple Train/Test Split (First 80% train, rest test - stratified if possible)
    # Using sklearn for indices
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Just take first fold as validation
    train_idx, test_idx = next(kf.split(X.cpu(), y.cpu()))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 2. Model
    # input features=1 (latent value), hidden=32, classes=2
    # But wait, DenseGCN takes (B, 1, 16, 16).
    # Its node features are 'learned embeddings'. 
    # forward(z) uses z as Adjacency.
    # So input dim of z is implicitly 1. 
    # Wait, the GCN class __init__ has `in_features=1` for node embeddings? No.
    # `self.node_embedding = nn.Parameter(torch.randn(num_nodes, in_features))`
    # If in_features=1, embeddings are 1D. If 16, 16D.
    # The GCN *doesn't look at node features from data*. It only looks at *Adjacency* from data.
    # So valid.
    
    model = LatentDenseGCN(num_nodes=16, in_features=16, hidden_dim=32, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4) # Standard GCN settings
    criterion = nn.CrossEntropyLoss() # Weights for imbalance?
    
    # Weighted Loss? AD (1) is rare.
    n_cn = dist.get(0, 1)
    n_ad = dist.get(1, 1)
    weight = torch.tensor([1.0, n_cn / n_ad]).to(device)
    print(f"Using class weights: {weight}")
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # 3. Train Loop
    print("\n--- Starting Training ---")
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                outputs = model(xb)
                _, predicted = torch.max(outputs.data, 1)
                val_total += yb.size(0)
                val_correct += (predicted == yb).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'gcn_adni.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("Model saved to gcn_adni.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', type=str, default='./data/latents.pt')
    parser.add_argument('--labels', type=str, default='./data/labels.pt')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    if not os.path.exists(args.latents):
        print(f"Error: {args.latents} not found. Run extract_latents.py first.")
    else:
        train_gcn(args.latents, args.labels, epochs=args.epochs)
