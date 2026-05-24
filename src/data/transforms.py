"""
src/data/transforms.py
======================
XRD pattern normalization and length standardization.

Supported normalizations
------------------------
none            — raw values as-is
minmax          — scale to [0, 1]
standard        — zero-mean, unit-variance (per pattern)
max_intensity   — divide by max absolute value (default)

All transforms operate on a 1-D numpy array and return a 1-D numpy array.
Padding uses zeros; truncation drops the tail.
"""

from __future__ import annotations

import numpy as np

VALID_NORMALIZATIONS = ("none", "minmax", "standard", "max_intensity")


def normalize(pattern: np.ndarray, method: str) -> np.ndarray:
    """Normalize a 1-D XRD intensity array.

    Parameters
    ----------
    pattern : np.ndarray, shape (L,)
    method  : one of VALID_NORMALIZATIONS

    Returns
    -------
    np.ndarray, shape (L,), dtype float32
    """
    if method not in VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization '{method}'. "
            f"Choose from {VALID_NORMALIZATIONS}."
        )

    x = pattern.astype(np.float32)

    if method == "none":
        return x

    if method == "minmax":
        lo, hi = x.min(), x.max()
        rng = hi - lo
        if rng < 1e-8:
            return np.zeros_like(x)
        return (x - lo) / rng

    if method == "standard":
        mu, sigma = x.mean(), x.std()
        if sigma < 1e-8:
            return np.zeros_like(x)
        return (x - mu) / sigma

    if method == "max_intensity":
        peak = np.abs(x).max()
        if peak < 1e-8:
            return np.zeros_like(x)
        return x / peak

    # unreachable
    return x


def pad_or_truncate(pattern: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or truncate to exactly target_length.

    Parameters
    ----------
    pattern       : np.ndarray, shape (L,)
    target_length : int

    Returns
    -------
    np.ndarray, shape (target_length,), dtype float32
    """
    x = pattern.astype(np.float32)
    current = len(x)

    if current == target_length:
        return x
    if current > target_length:
        return x[:target_length]
    # pad right with zeros
    pad = np.zeros(target_length - current, dtype=np.float32)
    return np.concatenate([x, pad])


def preprocess(
    pattern: np.ndarray,
    normalization: str = "max_intensity",
    target_length: int = 4500,
) -> np.ndarray:
    """Normalize then pad/truncate.

    Returns shape (target_length,), dtype float32.
    """
    x = normalize(pattern, normalization)
    x = pad_or_truncate(x, target_length)
    return x
