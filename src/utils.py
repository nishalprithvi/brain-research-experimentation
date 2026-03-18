import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    """Sets random seeds and deterministic backend flags for reproducibility."""
    print(f"[Results Reproducibility] Setting global seed to {seed}...")

    # Must be set before CUDA kernels launch; safe to set repeatedly.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Enforce deterministic ops where supported; warn on unsupported kernels.
    torch.use_deterministic_algorithms(True, warn_only=True)
