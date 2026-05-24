"""
src/data — XRD DataLoader package for Phase 2.

Public API
----------
from src.data import XRDDataset, XRDDataModule, build_weighted_sampler
"""

from .xrd_dataset import XRDDataset, MaterializedXRDDataset, materialized_cache_exists
from .datamodule import XRDDataModule
from .samplers import build_weighted_sampler

__all__ = [
    "XRDDataset",
    "MaterializedXRDDataset",
    "materialized_cache_exists",
    "XRDDataModule",
    "build_weighted_sampler",
]