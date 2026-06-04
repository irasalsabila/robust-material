"""
src/data/datamodule.py
======================
XRDDataModule — wires together XRDDataset + samplers into
train / val / test DataLoaders.

Usage
-----
    dm = XRDDataModule(
        manifest_path  = "outputs/data_audit/manifest.json",
        split_json_path= "outputs/dataset_splits/splits/split_D1_10pct.json",
        task           = "crystal_system",
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    test_loader  = dm.test_dataloader()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .xrd_dataset import (
    XRDDataset,
    MaterializedXRDDataset,
    get_materialized_cache_dir,
    materialized_cache_exists,
)
from .samplers import build_weighted_sampler, should_use_weighted_sampler

logger = logging.getLogger(__name__)


def _collate_fn(batch):
    """Custom collate: stack tensors, keep string fields as lists."""
    xs   = torch.stack([item["x"] for item in batch])
    ys   = torch.stack([item["y"] for item in batch])
    return {
        "x":             xs,
        "y":             ys,
        "sample_id":     [item["sample_id"]     for item in batch],
        "structure_id":  [item["structure_id"]  for item in batch],
        "domain":        [item["domain"]        for item in batch],
        "task":          [item["task"]          for item in batch],
    }


class XRDDataModule:
    """
    Parameters
    ----------
    manifest_path   : path to outputs/data_audit/manifest.json
    split_json_path : path to outputs/dataset_splits/splits/split_{D}_{S}pct.json
    task            : "crystal_system" | "top10_space_group"
    data_dir        : directory containing D*.zip files  (default: "data")
    normalization   : "none" | "minmax" | "standard" | "max_intensity"
    target_length   : pad/truncate length  (default: 4500)
    batch_size      : samples per batch  (default: 64)
    num_workers     : DataLoader workers  (default: 0)
    pin_memory      : pin tensors to CUDA memory  (default: False)
    label_map_dir   : where to save label_map_*.json
    use_weighted_sampler : override auto-detection (None = auto)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split_json_path: str | Path,
        task: str,
        data_dir: str | Path = "data",
        normalization: str = "max_intensity",
        target_length: int = 4500,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        label_map_dir: str | Path = "outputs/label_mappings",
        use_weighted_sampler: Optional[bool] = None,
        source: Optional[str] = None,
        materialized_base_dir: str | Path = "outputs/materialized",
    ) -> None:
        self.manifest_path    = Path(manifest_path)
        self.split_json_path  = Path(split_json_path)
        self.task             = task
        self.data_dir         = Path(data_dir)
        self.normalization    = normalization
        self.target_length    = target_length
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.pin_memory       = pin_memory
        self.label_map_dir    = Path(label_map_dir)
        self._use_weighted_sampler_override = use_weighted_sampler
        self._materialized_base_dir = Path(materialized_base_dir)

        # source: "zip", "materialized", or None (auto-detect)
        self._source = source

        self._train_ds = None
        self._val_ds   = None
        self._test_ds  = None

    # ── Common dataset kwargs ─────────────────────────────────────────────────

    def _ds_kwargs(self) -> dict:
        return dict(
            manifest_path   = self.manifest_path,
            split_json_path = self.split_json_path,
            task            = self.task,
            data_dir        = self.data_dir,
            normalization   = self.normalization,
            target_length   = self.target_length,
            label_map_dir   = self.label_map_dir,
        )

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _resolve_source(self) -> str:
        """Determine whether to use materialized cache or zip."""
        if self._source in ("zip", "materialized"):
            return self._source

        # Auto-detect: read domain/scale from split JSON
        import json
        with open(self.split_json_path, "r") as f:
            split_data = json.load(f)
        domain = split_data["domain"]
        scale = int(split_data["scale_pct"])

        if materialized_cache_exists(
            domain, scale, self.task, self._materialized_base_dir
        ):
            logger.info("Auto-detected materialized cache for %s_%dpct/%s",
                        domain, scale, self.task)
            return "materialized"
        return "zip"

    def setup(self) -> None:
        """Instantiate all three dataset splits."""
        source = self._resolve_source()

        if source == "materialized":
            self._setup_materialized()
        else:
            self._setup_zip()

        logger.info(
            "DataModule ready | source=%s train=%d val=%d test=%d",
            source, len(self._train_ds), len(self._val_ds), len(self._test_ds),
        )

    def _setup_zip(self) -> None:
        """Load datasets from zip archives (slow, no cache needed)."""
        kwargs = self._ds_kwargs()
        self._train_ds = XRDDataset(split="train", **kwargs)
        self._val_ds   = XRDDataset(split="val",   **kwargs)
        self._test_ds  = XRDDataset(split="test",  **kwargs)

    def _setup_materialized(self) -> None:
        """Load datasets from pre-materialized numpy arrays (fast)."""
        import json
        with open(self.split_json_path, "r") as f:
            split_data = json.load(f)
        domain = split_data["domain"]
        scale = int(split_data["scale_pct"])

        cache_dir = get_materialized_cache_dir(
            domain, scale, self.task, self._materialized_base_dir
        )
        self._train_ds = MaterializedXRDDataset(cache_dir, split="train", task=self.task)
        self._val_ds   = MaterializedXRDDataset(cache_dir, split="val",   task=self.task)
        self._test_ds  = MaterializedXRDDataset(cache_dir, split="test",  task=self.task)

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Call setup() first."
        ds = self._train_ds

        # Decide whether to use weighted sampler
        if self._use_weighted_sampler_override is not None:
            use_ws = self._use_weighted_sampler_override
        else:
            use_ws = should_use_weighted_sampler(self.task, "train")

        if use_ws:
            labels  = ds.get_sample_labels()
            sampler = build_weighted_sampler(labels, ds.num_classes)
            logger.info("Train loader: WeightedRandomSampler enabled (task=%s)", self.task)
            return DataLoader(
                ds,
                batch_size  = self.batch_size,
                sampler     = sampler,
                num_workers = self.num_workers,
                pin_memory  = self.pin_memory,
                collate_fn  = _collate_fn,
                drop_last   = False,
            )
        else:
            logger.info("Train loader: shuffle=True (task=%s)", self.task)
            return DataLoader(
                ds,
                batch_size  = self.batch_size,
                shuffle     = True,
                num_workers = self.num_workers,
                pin_memory  = self.pin_memory,
                collate_fn  = _collate_fn,
                drop_last   = False,
            )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "Call setup() first."
        return DataLoader(
            self._val_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
            collate_fn  = _collate_fn,
            drop_last   = False,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None, "Call setup() first."
        return DataLoader(
            self._test_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
            collate_fn  = _collate_fn,
            drop_last   = False,
        )

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def train_dataset(self):
        assert self._train_ds is not None, "Call setup() first."
        return self._train_ds

    @property
    def val_dataset(self):
        assert self._val_ds is not None, "Call setup() first."
        return self._val_ds

    @property
    def test_dataset(self):
        assert self._test_ds is not None, "Call setup() first."
        return self._test_ds

    @property
    def num_classes(self) -> int:
        assert self._train_ds is not None, "Call setup() first."
        return self._train_ds.num_classes

    @property
    def class_names(self) -> dict:
        assert self._train_ds is not None, "Call setup() first."
        return self._train_ds.class_names

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights from the training split."""
        assert self._train_ds is not None, "Call setup() first."
        return self._train_ds.compute_class_weights()