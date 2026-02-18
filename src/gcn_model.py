
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentDenseGCN(nn.Module):
    def __init__(self, num_nodes=16, in_features=1, hidden_dim=32, n_classes=2):
        """
        A distinct GCN designed to operate on the 16x16 Latent Space of the VAE.
        
        The VAE outputs z of shape (Batch, 1, 16, 16).
        We treat this 16x16 matrix as the Weighted Adjacency Matrix (A) of a 16-node graph.
        
        Since 'z' represents the connection strength between 16 "super-regions" (latent spatial dims),
        this GCN learns to classify the disease state based on this compressed topology.
        
        We implement Dense GCN layers (A @ X @ W) to ensure full differentiability 
        for Classifier Guidance (we need gradient of Output w.r.t A).
        """
        super(LatentDenseGCN, self).__init__()
        
        self.num_nodes = num_nodes
        
        # Node features
        # If the latent matrix IS the structure, what are the features?
        # Option 1: Constant/Identity features.
        # Option 2: The diagonal of z?
        # Let's learn a feature embedding for each of the 16 nodes.
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, in_features))
        
        # GCN Layers
        # Layer 1
        self.w1 = nn.Linear(in_features, hidden_dim)
        
        # Layer 2
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer 3
        self.w3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Classification Head (Global Pooling -> Linear)
        self.fc = nn.Linear(hidden_dim * 1, n_classes) # *1 for pooling aggregation
        
    def forward(self, z):
        """
        Args:
            z: Latent batch (Batch, 1, 16, 16)
               We interpret this as the Adjacency Matrix A.
        """
        # 1. Prepare Adjacency A
        # z is (B, 1, 16, 16). Squeeze to (B, 16, 16).
        z_sq = z.squeeze(1)
        A = torch.sigmoid(z_sq) # Ensure non-negative and bounded [0, 1]
        
        # Enforce symmetry? Brain matrices are symmetric.
        # z might not be perfectly symmetric coming from VAE encoder, but should be close.
        # For graph theory, A should be symmetric.
        A = (A + A.transpose(1, 2)) / 2
        
        # Normalize A? (D^-0.5 A D^-0.5) or just raw weights?
        # For simple dense GCN, standard A is fine, but self-loops help.
        # Add self-loops (Identity)
        I = torch.eye(self.num_nodes, device=z.device).unsqueeze(0) # (1, 16, 16)
        A_hat = A + I
        
        # Degree Matrix D
        # D_ii = sum(A_ij)
        # Note: Since A is continuous (latent values), D is continuous.
        D_hat = torch.sum(A_hat, dim=2) + 1e-6 # (B, 16)
        D_inv_sqrt = torch.pow(D_hat, -0.5)
        
        # Turn D into diagonal matrices: (B, 16, 16)
        D_mat = torch.diag_embed(D_inv_sqrt)
        
        # Normalized A: D^-0.5 * A_hat * D^-0.5
        # (B, 16, 16) @ (B, 16, 16) @ (B, 16, 16)
        norm_A = torch.bmm(torch.bmm(D_mat, A_hat), D_mat)
        
        # 2. Graph Convolution
        # X is constant (Node Embeddings): (16, in_features) -> expand to (B, 16, in_features)
        batch_size = z.shape[0]
        X = self.node_embedding.expand(batch_size, -1, -1)
        
        # Layer 1: H1 = ReLU(norm_A @ X @ W1)
        XW1 = self.w1(X) # (B, 16, hidden)
        H1 = F.relu(torch.bmm(norm_A, XW1))
        
        # Layer 2: H2 = ReLU(norm_A @ H1 @ W2)
        H1W2 = self.w2(H1)
        H2 = F.relu(torch.bmm(norm_A, H1W2))
        
        # Layer 3
        H2W3 = self.w3(H2)
        H3 = F.relu(torch.bmm(norm_A, H2W3))
        
        # 3. Readout (Global Mean Pooling)
        # Average over nodes (dim 1)
        graph_emb = torch.mean(H3, dim=1) # (B, hidden)
        
        # 4. Classify
        logits = self.fc(graph_emb)
        
        return logits
