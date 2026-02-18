import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    print(f"[Results Reproducibility] Setting global seed to {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations (may slow down training but ensures reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # OS environment (some libraries use this)
    os.environ['PYTHONHASHSEED'] = str(seed)
