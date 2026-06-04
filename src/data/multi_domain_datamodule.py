"""
src/data/multi_domain_datamodule.py
====================================
Multi-domain DataModule that concatenates XRD datasets from several
domains into unified train / val / test splits.

Usage
-----
    dm = MultiDomainDataModule(
        manifest_path   = "outputs/data_audit/manifest.json",
        splits_dir      = "outputs/dataset_splits/splits",
        train_domains   = ["D1", "D2"],
        test_domain     = "D1",
        scale           = 10,
        task            = "crystal_system",
    )
    dm.setup()
    train_loader = dm.train_dataloader()   # concatenated D1+D2 train
    val_loader   = dm.val_dataloader()     # concatenated D1+D2 val
    test_loader  = dm.test_dataloader()    # D1 test only

Design
------
* The train/val split is taken from the UNION of all train_domains.
* The test split is taken from test_domain ONLY — this is the
  cross-domain test set.  test_domain does not need to be in
  train_domains (leave-one-domain-out).
* Structure-level leakage safety is preserved because each domain's
  split JSON uses the same structure-ID partition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader

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
    xs = torch.stack([item["x"] for item in batch])
    ys = torch.stack([item["y"] for item in batch])
    return {
        "x":            xs,
        "y":            ys,
        "sample_id":    [item["sample_id"]    for item in batch],
        "structure_id": [item["structure_id"] for item in batch],
        "domain":       [item["domain"]       for item in batch],
        "task":         [item["task"]         for item in batch],
    }


class MultiDomainDataModule:
    """
    Parameters
    ----------
    manifest_path   : path to outputs/data_audit/manifest.json
    splits_dir      : directory containing split_*.json files
    train_domains   : list of domain strings to include in training
    test_domain     : domain to use as test (and val) set
    scale           : integer scale percentage, e.g. 10 for 10%
    task            : "crystal_system" | "top10_space_group"
    data_dir        : directory containing D*.zip files
    normalization   : "none"|"minmax"|"standard"|"max_intensity"
    target_length   : pad/truncate length
    batch_size      : samples per batch
    num_workers     : DataLoader workers
    pin_memory      : pin tensors to CUDA memory
    label_map_dir   : where to save label_map_*.json
    source          : "zip"|"materialized"|None (auto-detect)
    materialized_base_dir : base dir for materialized cache
    """

    def __init__(
        self,
        manifest_path: str | Path,
        splits_dir: str | Path,
        train_domains: List[str],
        test_domain: str,
        scale: int,
        task: str,
        data_dir: str | Path = "data",
        normalization: str = "max_intensity",
        target_length: int = 4500,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        label_map_dir: str | Path = "outputs/label_mappings",
        source: Optional[str] = None,
        materialized_base_dir: str | Path = "outputs/materialized",
        use_weighted_sampler: Optional[bool] = None,
    ) -> None:
        if not train_domains:
            raise ValueError("train_domains must be a non-empty list")

        self.manifest_path    = Path(manifest_path)
        self.splits_dir       = Path(splits_dir)
        self.train_domains    = list(train_domains)
        self.test_domain      = test_domain
        self.scale            = int(scale)
        self.task             = task
        self.data_dir         = Path(data_dir)
        self.normalization    = normalization
        self.target_length    = target_length
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.pin_memory       = pin_memory
        self.label_map_dir    = Path(label_map_dir)
        self._use_weighted_sampler_override = use_weighted_sampler
        self._source          = source
        self._materialized_base_dir = Path(materialized_base_dir)

        self._train_ds: Optional[ConcatDataset] = None
        self._val_ds:   Optional[ConcatDataset] = None
        self._test_ds:  Optional[object]        = None  # single domain
        self._num_classes: Optional[int]        = None
        self._class_names: Optional[dict]       = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _split_json(self, domain: str) -> Path:
        return self.splits_dir / f"split_{domain}_{self.scale}pct.json"

    def _resolve_source(self, domain: str) -> str:
        if self._source in ("zip", "materialized"):
            return self._source
        if materialized_cache_exists(
            domain, self.scale, self.task, self._materialized_base_dir
        ):
            logger.info(
                "Auto-detected materialized cache for %s_%dpct/%s",
                domain, self.scale, self.task,
            )
            return "materialized"
        return "zip"

    def _load_dataset(self, domain: str, split: str):
        source = self._resolve_source(domain)
        if source == "materialized":
            cache_dir = get_materialized_cache_dir(
                domain, self.scale, self.task, self._materialized_base_dir
            )
            return MaterializedXRDDataset(cache_dir, split=split, task=self.task)
        # zip-based
        return XRDDataset(
            manifest_path   = self.manifest_path,
            split_json_path = self._split_json(domain),
            split           = split,
            task            = self.task,
            data_dir        = self.data_dir,
            normalization   = self.normalization,
            target_length   = self.target_length,
            label_map_dir   = self.label_map_dir,
        )

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Build all three dataset splits."""
        # ── Train: concatenate all train_domains ────────────────────────────
        train_datasets = [
            self._load_dataset(d, "train") for d in self.train_domains
        ]
        val_datasets = [
            self._load_dataset(d, "val") for d in self.train_domains
        ]

        # Derive class metadata from the first dataset (all domains share it)
        first = train_datasets[0]
        self._num_classes = first.num_classes
        self._class_names = first.class_names

        self._train_ds = (
            ConcatDataset(train_datasets)
            if len(train_datasets) > 1
            else train_datasets[0]
        )
        self._val_ds = (
            ConcatDataset(val_datasets)
            if len(val_datasets) > 1
            else val_datasets[0]
        )

        # ── Test: only the designated test_domain ───────────────────────────
        self._test_ds = self._load_dataset(self.test_domain, "test")

        n_train = sum(len(d) for d in train_datasets)
        n_val   = sum(len(d) for d in val_datasets)
        n_test  = len(self._test_ds)

        logger.info(
            "MultiDomainDataModule | train_domains=%s test_domain=%s "
            "task=%s scale=%d%% "
            "train=%d val=%d test=%d classes=%d",
            "+".join(self.train_domains), self.test_domain,
            self.task, self.scale,
            n_train, n_val, n_test, self._num_classes,
        )

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Call setup() first."
        use_ws = should_use_weighted_sampler(self.task, "train")

        if use_ws:
            # Collect labels from all underlying datasets in the ConcatDataset
            labels = self._collect_train_labels()
            sampler = build_weighted_sampler(labels, self._num_classes)
            logger.info(
                "Train loader: WeightedRandomSampler (task=%s, "
                "domains=%s)",
                self.task, self.train_domains,
            )
            return DataLoader(
                self._train_ds,
                batch_size  = self.batch_size,
                sampler     = sampler,
                num_workers = self.num_workers,
                pin_memory  = self.pin_memory,
                collate_fn  = _collate_fn,
                drop_last   = False,
            )

        return DataLoader(
            self._train_ds,
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

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        assert self._num_classes is not None, "Call setup() first."
        return self._num_classes

    @property
    def class_names(self) -> dict:
        assert self._class_names is not None, "Call setup() first."
        return self._class_names

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights from the concatenated training split."""
        assert self._train_ds is not None, "Call setup() first."
        import numpy as np
        labels = self._collect_train_labels()
        counts = np.zeros(self._num_classes, dtype=np.float64)
        for lbl in labels:
            counts[lbl] += 1
        counts = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum() * self._num_classes
        return torch.tensor(weights, dtype=torch.float32)

    def _collect_train_labels(self) -> list:
        """Gather integer labels for every training sample."""
        if isinstance(self._train_ds, ConcatDataset):
            labels = []
            for ds in self._train_ds.datasets:
                labels.extend(ds.get_sample_labels())
            return labels
        return self._train_ds.get_sample_labels()

    # ── Info ──────────────────────────────────────────────────────────────────

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

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MultiDomainDataModule("
            f"train={'+'.join(self.train_domains)}, "
            f"test={self.test_domain}, "
            f"scale={self.scale}%, task={self.task})"
        )
