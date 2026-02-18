
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.nn import GlobalAttentionPooling, AvgPooling

class StandardGCN(nn.Module):
    def __init__(self, num_nodes=100, hidden_dim=64, n_classes=2):
        """
        Standard GCN for 100-node Brain Graphs.
        """
        super(StandardGCN, self).__init__()
        
        # Learnable node embeddings for the 100 ROIs
        # Shape: (100, hidden_dim)
        # This acts as the initial "node features" X.
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        
        # Layer 1
        self.conv1 = GraphConv(hidden_dim, hidden_dim * 2, allow_zero_in_degree=True)
        self.bn1 = nn.LayerNorm(hidden_dim * 2)
        
        # Layer 2
        self.conv2 = GraphConv(hidden_dim * 2, hidden_dim * 2, allow_zero_in_degree=True)
        self.bn2 = nn.LayerNorm(hidden_dim * 2)
        
        # Layer 3
        self.conv3 = GraphConv(hidden_dim * 2, hidden_dim, allow_zero_in_degree=True)
        self.bn3 = nn.LayerNorm(hidden_dim)
        
        # Pooling
        self.pooling = AvgPooling()
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, g, features=None, edge_weight=None):
        """
        g: DGLGraph batch
        features: Optional override for node features. 
                  If None, use self.node_embedding.
        edge_weight: Optional edge weights. If None, tries to retrieve from g.edata['feat']
        """
        # If g is a batch of N graphs, it has N*100 nodes.
        # We need to expand node_embedding to match.
        # But DGL handles batching by merging into one big graph.
        # We need to assign features to g.ndata['h']
        
        if features is None:
            # Replicate embedding for each graph in batch
            batch_size = g.batch_size
            h = self.node_embedding.repeat(batch_size, 1)
        else:
            h = features
            
        # Extract Edge Weights if not provided
        if edge_weight is None:
            # Check for common keys
            if 'feat' in g.edata:
                edge_weight = g.edata['feat']
            elif 'w' in g.edata:
                edge_weight = g.edata['w']
            elif 'weight' in g.edata:
                edge_weight = g.edata['weight']
            # If still None, GraphConv defaults to 1.0 (unweighted)
        
        # Layer 1
        h = self.conv1(g, h, edge_weight=edge_weight)
        h = self.bn1(h)
        h = F.relu(h)
        
        # Layer 2
        h = self.conv2(g, h, edge_weight=edge_weight)
        h = self.bn2(h)
        h = F.relu(h)
        
        # Layer 3
        h = self.conv3(g, h, edge_weight=edge_weight)
        h = self.bn3(h)
        h = F.relu(h)
        
        # Global Pooling
        hg = self.pooling(g, h)
        
        return self.fc(hg)
