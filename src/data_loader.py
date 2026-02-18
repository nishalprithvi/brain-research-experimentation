
import dgl
import torch
import numpy as np
import os
import pandas as pd
from dgl.data.utils import load_graphs

def load_adni_data(data_dir='./data', bin_file='adni_schaefer100.bin', mapping_file='adni_schaefer100_mapping.txt', process_file='ADNI_process.csv'):
    """
    Step 1: Data Loader (DGL Version)
    
    1. Load DGL Graphs from .bin file.
    2. Load Mapping (Graph Index -> Subject ID).
    3. Load Metadata (Subject ID -> Diagnosis).
    4. Extract Adjacency Matrices and Normalize.
    """
    
    bin_path = os.path.join(data_dir, bin_file)
    mapping_path = os.path.join(data_dir, mapping_file)
    process_path = os.path.join(data_dir, process_file)
    
    print(f"Loading graphs from {bin_path}...")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary file not found: {bin_path}")
        
    # Load Graphs
    # load_graphs returns (graph_list, label_dict)
    glist, label_dict = load_graphs(bin_path)
    print(f"Loaded {len(glist)} graphs.")
    
    # Load Mapping
    # Assuming text file format: "sub-control002S0413,0,0" (SubjectID, GraphIdx, Label?)
    print(f"Loading mapping from {mapping_path}...")
    # Based on view_file earlier, format is: SubjectID, Index, Label?
    # e.g., "sub-control002S0413,0,0"
    mapping_df = pd.read_csv(mapping_path, header=None, names=['SubjectID', 'GraphIdx', 'Label'])
    
    # Load Metadata (Process CSV)
    labels = []
    
    if os.path.exists(process_path):
        print(f"Loading metadata from {process_path}...")
        meta_df = pd.read_csv(process_path)
        
        # Create Mapping: Normalized SubjectID -> Group
        # Normalize: Remove 'sub-control' from mapping, remove '_' from meta
        # Mapping: sub-control002S0413 -> 002S0413
        # Meta: 002_S_0413 -> 002S0413
        
        meta_df['NormalizedID'] = meta_df['Subject'].str.replace('_', '')
        subj_to_group = dict(zip(meta_df['NormalizedID'], meta_df['Group']))
        
        # Prepare Label List aligned with glist indices
        # We iterate 0..len(glist)-1. Find corresponding SubjectID from mapping_df.
        # mapping_df should be sorted by GraphIdx? It seems redundant if GraphIdx is just 0..N.
        # Let's create a dict: GraphIdx -> SubjectID
        
        idx_to_subj = dict(zip(mapping_df['GraphIdx'], mapping_df['SubjectID']))
        
        # Group Mapping
        group_map = {'CN': 0, 'AD': 1, 'MCI': 2, 'EMCI': 3, 'LMCI': 4, 'SMC': 5}
        
        for i in range(len(glist)):
            if i in idx_to_subj:
                raw_subj = idx_to_subj[i]
                norm_subj = raw_subj.replace('sub-control', '').replace('sub-patient', '') # Handle both?
                # Actually mapping file usually has sub-control... or sub-patient... ?
                # The sample showed sub-control.
                # Just remove prefixes safely?
                # Or simplistic: remove 'sub-control'.
                # Let's try removing 'sub-control' and 'sub-patient'.
                if norm_subj.startswith('sub-control'):
                    norm_subj = norm_subj.replace('sub-control', '')
                elif norm_subj.startswith('sub-patient'): 
                    norm_subj = norm_subj.replace('sub-patient', '')
                
                # Check dict
                if norm_subj in subj_to_group:
                    group_str = subj_to_group[norm_subj]
                    label = group_map.get(group_str, -1)
                else:
                    # Try fuzzy matching or just -1
                    # print(f"Warning: Subject {norm_subj} not found in metadata.")
                    label = -1
            else:
                label = -1
            labels.append(label)
    else:
        # If no metadata, return -1s
        labels = [-1] * len(glist)
    
    matrices = []
    
    # Process Graphs
    print("Extracting adjacency matrices...")
    for i, g in enumerate(glist):
        # 1. Get Adjacency Matrix
        num_nodes = g.num_nodes()
        src, dst = g.edges()
        
        weights = None
        if 'E_features' in g.edata:
            weights = g.edata['E_features']
        elif 'feat' in g.edata:
            weights = g.edata['feat']
            
        adj = torch.zeros((num_nodes, num_nodes))
        
        if weights is not None:
            if weights.dim() > 1:
                weights = weights.squeeze()
            adj[src, dst] = weights.float()
        else:
            adj[src, dst] = 1.0
            
        # Normalize to [0, 1]
        min_val = adj.min()
        max_val = adj.max()
        
        if max_val - min_val > 1e-6:
            adj = (adj - min_val) / (max_val - min_val)
        
        matrices.append(adj.numpy())
        
    return np.array(matrices), np.array(labels)

def load_adni_dgl_with_labels(data_dir='./data', bin_file='adni_schaefer100.bin', mapping_file='adni_schaefer100_mapping.txt', process_file='ADNI_process.csv'):
    """
    Returns (glist, labels)
    glist: list of DGLGraphs
    labels: np.array of labels
    """
    bin_path = os.path.join(data_dir, bin_file)
    mapping_path = os.path.join(data_dir, mapping_file)
    process_path = os.path.join(data_dir, process_file)
    
    # 1. Load Graphs
    glist, _ = load_graphs(bin_path)
    
    # Clean inconsistent features
    for g in glist:
        if 'N_features' in g.ndata:
            del g.ndata['N_features']
            
    # 2. Load Labels (Reuse logic)
    mapping_df = pd.read_csv(mapping_path, header=None, names=['SubjectID', 'GraphIdx', 'Label'])
    
    # SORT GUARANTEE: Ensure mapping_df is sorted by GraphIdx
    mapping_df = mapping_df.sort_values('GraphIdx').reset_index(drop=True)
    
    # We assume glist is already in order [0, 1, 2...] corresponding to GraphIdx
    # But just to be safe, we rely on the fact that dgl load_graphs preserves order.
    # The critical part is that mapping_df aligns with glist.
    
    labels = []
    
    if os.path.exists(process_path):
        meta_df = pd.read_csv(process_path)
        meta_df['NormalizedID'] = meta_df['Subject'].str.replace('_', '')
        subj_to_group = dict(zip(meta_df['NormalizedID'], meta_df['Group']))
        idx_to_subj = dict(zip(mapping_df['GraphIdx'], mapping_df['SubjectID']))
        group_map = {'CN': 0, 'AD': 1, 'MCI': 2, 'EMCI': 3, 'LMCI': 4, 'SMC': 5}
        
        for i in range(len(glist)):
            if i in idx_to_subj:
                raw_subj = idx_to_subj[i]
                norm_subj = raw_subj.replace('sub-control', '').replace('sub-patient', '')
                if norm_subj.startswith('sub-control'): norm_subj = norm_subj.replace('sub-control', '')
                elif norm_subj.startswith('sub-patient'): norm_subj = norm_subj.replace('sub-patient', '')
                
                if norm_subj in subj_to_group:
                    group_str = subj_to_group[norm_subj]
                    label = group_map.get(group_str, -1)
                else:
                    label = -1
            else:
                label = -1
            labels.append(label)
    else:
        labels = [-1] * len(glist)
        
    return glist, np.array(labels)

if __name__ == "__main__":
    # Test execution
    matrices, labels = load_adni_data()
    print(f"Extracted {len(matrices)} matrices.")
    print(f"Shape: {matrices[0].shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Check distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label Distribution: {dict(zip(unique, counts))}")
    # 0: CN, 1: AD
    
    # Save sample
    np.save('sample_dgl_matrix.npy', matrices[0])
    print("Saved sample_dgl_matrix.npy")
