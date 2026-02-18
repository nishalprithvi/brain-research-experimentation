
import dgl
from dgl.data.utils import load_graphs
import os

def inspect():
    bin_path = './data/adni_schaefer100.bin'
    print(f"Loading {bin_path}...")
    glist, _ = load_graphs(bin_path)
    print(f"Loaded {len(glist)} graphs.")
    
    for i in range(min(5, len(glist))):
        g = glist[i]
        print(f"\nGraph {i}:")
        print(f"Num Nodes: {g.num_nodes()}")
        print(f"Num Edges: {g.num_edges()}")
        print(f"Node Data Keys: {g.ndata.keys()}")
        if 'N_features' in g.ndata:
             print(f"N_features shape: {g.ndata['N_features'].shape}")
        if 'feat' in g.edata:
            print(f"E_feat shape: {g.edata['feat'].shape}")
        if 'E_features' in g.edata:
            print(f"E_features shape: {g.edata['E_features'].shape}")

if __name__ == "__main__":
    inspect()
