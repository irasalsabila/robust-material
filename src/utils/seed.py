"""src/utils/seed.py — reproducible seed setting for CPU / CUDA / MPS."""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS does not expose a seed API but torch.manual_seed covers it
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
