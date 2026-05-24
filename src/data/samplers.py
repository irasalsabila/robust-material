"""
src/data/samplers.py
====================
Weighted random sampler for imbalanced classification.

Rules (from Phase 1 analysis)
------------------------------
crystal_system     : weighted sampler ON  (30x imbalance)
top10_space_group  : weighted sampler OFF (2x imbalance — standard CE sufficient)

Never apply weighted sampling to val or test splits.
"""

from __future__ import annotations

from typing import List

import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(
    labels: List[int],
    num_classes: int,
    num_samples: int | None = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler from a list of integer labels.

    Parameters
    ----------
    labels      : per-sample integer class labels
    num_classes : total number of classes
    num_samples : how many samples to draw per epoch (default: len(labels))
    replacement : sample with replacement (default True)

    Returns
    -------
    torch.utils.data.WeightedRandomSampler
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for lbl in labels:
        counts[lbl] += 1.0

    # weight per class = 1 / count  (unseen classes get weight 0)
    class_weights = torch.where(
        counts > 0,
        1.0 / counts,
        torch.zeros_like(counts),
    )

    # weight per sample
    sample_weights = torch.tensor(
        [class_weights[lbl].item() for lbl in labels],
        dtype=torch.float64,
    )

    n = num_samples if num_samples is not None else len(labels)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n,
        replacement=replacement,
    )


def should_use_weighted_sampler(task: str, split: str) -> bool:
    """Return True if weighted sampling should be applied.

    Only train split, only crystal_system task.
    """
    if split != "train":
        return False
    return task == "crystal_system"
