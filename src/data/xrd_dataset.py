"""
src/data/xrd_dataset.py
=======================
PyTorch Dataset for XRD diffraction patterns.

Each sample is one augmented pattern (e.g. 1000104_1) belonging to a
structure.  The split JSON controls which *structure* IDs belong to
train / val / test; every augmentation of those structures is included.

Zip layout (per domain)
-----------------------
D1  : data/D1.zip  →  train/{sample_id}.csv
D2  : data/D2.zip  →  train_with_back/{sample_id}.csv
D3  : data/D3.zip  →  train_with_noise1/{sample_id}.csv
D4  : data/D4.zip  →  train_with_noise/{sample_id}.csv

CSV format: single column with header "y", 4500 float rows.

Returned dict
-------------
{
    "x"           : FloatTensor [1, L]   — normalised, padded/truncated
    "y"           : LongTensor  []       — integer class label
    "sample_id"   : str                  — e.g. "1000104_1"
    "structure_id": str                  — e.g. "1000104"
    "domain"      : str                  — "D1" / "D2" / "D3" / "D4"
    "task"        : str                  — "crystal_system" | "top10_space_group"
}
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import preprocess, VALID_NORMALIZATIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain → folder inside zip
# ---------------------------------------------------------------------------
DOMAIN_FOLDER: Dict[str, str] = {
    "D1": "train",
    "D2": "train_with_back",
    "D3": "train_with_noise1",
    "D4": "train_with_noise",
}

# ---------------------------------------------------------------------------
# Crystal-system label map  (string key → int label 0-6)
# ---------------------------------------------------------------------------
CRYSTAL_SYSTEM_MAP: Dict[str, int] = {
    "0": 0,  # cubic
    "1": 1,  # monoclinic
    "2": 2,  # orthorhombic
    "3": 3,  # triclinic
    "4": 4,  # hexagonal
    "5": 5,  # tetragonal
    "6": 6,  # trigonal
}
CRYSTAL_SYSTEM_NAMES: Dict[int, str] = {
    0: "cubic",
    1: "monoclinic",
    2: "orthorhombic",
    3: "triclinic",
    4: "hexagonal",
    5: "tetragonal",
    6: "trigonal",
}

# Canonical top-10 SG classes (by frequency at 20% scale, D1 reference)
TOP10_SG_CLASSES: List[int] = [62, 14, 225, 194, 2, 12, 166, 221, 139, 227]
TOP10_SG_NAMES: Dict[int, str] = {
    62: "Pnma",
    14: "P2_1/c",
    225: "Fm-3m",
    194: "P6_3/mmc",
    2: "P-1",
    12: "C2/m",
    166: "R-3m",
    221: "Pm-3m",
    139: "I4/mmm",
    227: "Fd-3m",
}
TOP10_SG_MAP: Dict[int, int] = {sg: idx for idx, sg in enumerate(TOP10_SG_CLASSES)}


def _build_label_map(task: str) -> Tuple[Dict, Dict]:
    """Return (class_to_idx, idx_to_name) for the given task."""
    if task == "crystal_system":
        class_to_idx = CRYSTAL_SYSTEM_MAP.copy()
        idx_to_name = {v: CRYSTAL_SYSTEM_NAMES[v] for v in CRYSTAL_SYSTEM_NAMES}
        return class_to_idx, idx_to_name
    if task == "top10_space_group":
        class_to_idx = {str(sg): idx for sg, idx in TOP10_SG_MAP.items()}
        idx_to_name = {idx: TOP10_SG_NAMES[sg] for sg, idx in TOP10_SG_MAP.items()}
        return class_to_idx, idx_to_name
    raise ValueError(f"Unknown task '{task}'. Choose 'crystal_system' or 'top10_space_group'.")


def save_label_mappings(output_dir: Path, task: str) -> None:
    """Persist label maps to outputs/phase2/label_mappings/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_to_idx, idx_to_name = _build_label_map(task)
    payload = {
        "task": task,
        "num_classes": len(idx_to_name),
        "class_to_idx": class_to_idx,
        "idx_to_name": {str(k): v for k, v in idx_to_name.items()},
    }
    out_path = output_dir / f"label_map_{task}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved label map: %s", out_path)


class XRDDataset(Dataset):
    """
    Parameters
    ----------
    manifest_path  : path to outputs/phase1/manifest.json
    split_json_path: path to outputs/phase1/splits/split_{D}_{S}pct.json
    split          : "train" | "val" | "test"
    task           : "crystal_system" | "top10_space_group"
    data_dir       : directory containing D1.zip … D4.zip  (default: "data/")
    normalization  : "none" | "minmax" | "standard" | "max_intensity"
    target_length  : pad/truncate to this length  (default: 4500)
    label_map_dir  : where to save label_map_*.json  (default: outputs/phase2/label_mappings/)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split_json_path: str | Path,
        split: str,
        task: str,
        data_dir: str | Path = "data",
        normalization: str = "max_intensity",
        target_length: int = 4500,
        label_map_dir: str | Path = "outputs/phase2/label_mappings",
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")
        if task not in ("crystal_system", "top10_space_group"):
            raise ValueError(
                f"task must be 'crystal_system' or 'top10_space_group', got '{task}'"
            )
        if normalization not in VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {VALID_NORMALIZATIONS}, got '{normalization}'"
            )

        self.split = split
        self.task = task
        self.normalization = normalization
        self.target_length = target_length
        self.data_dir = Path(data_dir)

        # ── Load manifest ────────────────────────────────────────────────────
        with open(manifest_path, "r") as f:
            raw = json.load(f)
        self._manifest: Dict = raw["structures"]

        # ── Load split JSON ──────────────────────────────────────────────────
        with open(split_json_path, "r") as f:
            split_data = json.load(f)

        self.domain: str = split_data["domain"]
        self.scale_pct: int = int(split_data["scale_pct"])

        # ── Resolve sample IDs for this split ────────────────────────────────
        # split JSON stores pattern IDs as "D1:1000104_1"
        raw_patterns: List[str] = split_data[f"{split}_patterns"]
        # strip domain prefix
        self._sample_ids: List[str] = [p.split(":", 1)[1] for p in raw_patterns]

        # ── Build structure_id lookup ─────────────────────────────────────────
        # sample_id = "{structure_id}_{aug_index}"
        self._structure_ids: List[str] = [
            sid.rsplit("_", 1)[0] for sid in self._sample_ids
        ]

        # ── Label map ────────────────────────────────────────────────────────
        self._class_to_idx, self._idx_to_name = _build_label_map(task)
        save_label_mappings(Path(label_map_dir), task)

        # ── Filter samples that have a valid label ────────────────────────────
        # For top10_space_group, structures outside the top-10 are excluded.
        valid_indices = []
        for i, struct_id in enumerate(self._structure_ids):
            if struct_id not in self._manifest:
                logger.warning("Structure %s not in manifest — skipping", struct_id)
                continue
            try:
                self._get_label(struct_id)
                valid_indices.append(i)
            except KeyError:
                # SG not in top-10 — skip silently
                pass

        self._sample_ids    = [self._sample_ids[i]    for i in valid_indices]
        self._structure_ids = [self._structure_ids[i] for i in valid_indices]

        # ── Open zip (keep handle open for fast repeated access) ─────────────
        zip_path = self.data_dir / f"{self.domain}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip not found: {zip_path}")
        self._zip = zipfile.ZipFile(zip_path, "r")
        self._zip_folder = DOMAIN_FOLDER[self.domain]

        logger.info(
            "XRDDataset | domain=%s scale=%d%% split=%s task=%s "
            "samples=%d norm=%s L=%d",
            self.domain, self.scale_pct, split, task,
            len(self._sample_ids), normalization, target_length,
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        return len(self._idx_to_name)

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._idx_to_name)

    @property
    def sample_ids(self) -> List[str]:
        return list(self._sample_ids)

    @property
    def structure_ids(self) -> List[str]:
        return list(self._structure_ids)

    # ── Label helpers ─────────────────────────────────────────────────────────

    def _get_label(self, structure_id: str) -> int:
        info = self._manifest[structure_id]
        if self.task == "crystal_system":
            cs_key = str(info["crystal_sys"])
            return self._class_to_idx[cs_key]
        if self.task == "top10_space_group":
            sg_no = int(info["space_group_no"])
            sg_key = str(sg_no)
            if sg_key not in self._class_to_idx:
                raise KeyError(sg_no)
            return self._class_to_idx[sg_key]
        raise ValueError(self.task)

    # ── Pattern loading ───────────────────────────────────────────────────────

    def _load_pattern(self, sample_id: str) -> np.ndarray:
        member = f"{self._zip_folder}/{sample_id}.csv"
        with self._zip.open(member) as fh:
            raw = fh.read().decode("utf-8")
        # CSV has header "y" then one float per line
        lines = raw.strip().splitlines()
        # skip header
        values = [float(line.strip()) for line in lines[1:] if line.strip()]
        return np.array(values, dtype=np.float32)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        sample_id   = self._sample_ids[idx]
        structure_id = self._structure_ids[idx]

        pattern = self._load_pattern(sample_id)
        pattern = preprocess(pattern, self.normalization, self.target_length)

        label = self._get_label(structure_id)

        x = torch.from_numpy(pattern).unsqueeze(0)  # [1, L]
        y = torch.tensor(label, dtype=torch.long)

        return {
            "x":            x,
            "y":            y,
            "sample_id":    sample_id,
            "structure_id": structure_id,
            "domain":       self.domain,
            "task":         self.task,
        }

    # ── Class weight computation ──────────────────────────────────────────────

    def compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for the current split.

        Returns FloatTensor of shape (num_classes,).
        Suitable for nn.CrossEntropyLoss(weight=...).
        """
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for struct_id in self._structure_ids:
            label = self._get_label(struct_id)
            counts[label] += 1

        # avoid division by zero for unseen classes
        counts = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.num_classes  # normalise
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_labels(self) -> List[int]:
        """Return a label for every sample (used by WeightedRandomSampler)."""
        return [self._get_label(sid) for sid in self._structure_ids]

    def __del__(self) -> None:
        try:
            self._zip.close()
        except Exception:
            pass


# ===========================================================================
# Materialized (numpy) dataset — fast loading from pre-processed .npy files
# ===========================================================================

class MaterializedXRDDataset(Dataset):
    """XRD Dataset backed by pre-materialized numpy arrays.

    Expects the directory structure created by materialize_dataset.py:
        {cache_dir}/x_{split}.npy
        {cache_dir}/y_{split}.npy
        {cache_dir}/sample_ids_{split}.json
        {cache_dir}/structure_ids_{split}.json
        {cache_dir}/metadata.json
        {cache_dir}/label_map.json

    Returns the same batch dict as XRDDataset for drop-in compatibility.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        task: str,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")

        self.cache_dir = Path(cache_dir)
        self.split = split
        self.task = task

        # Load metadata
        with open(self.cache_dir / "metadata.json", "r") as f:
            self._metadata = json.load(f)

        self.domain: str = self._metadata["domain"]
        self.scale_pct: int = self._metadata["scale_pct"]

        # Load label map
        with open(self.cache_dir / "label_map.json", "r") as f:
            lm = json.load(f)
        self._class_to_idx: Dict = lm["class_to_idx"]
        self._idx_to_name: Dict[int, str] = {int(k): v for k, v in lm["idx_to_name"].items()}

        # Load arrays (memory-mapped for large datasets)
        x_path = self.cache_dir / f"x_{split}.npy"
        y_path = self.cache_dir / f"y_{split}.npy"
        self._X = np.load(x_path, mmap_mode="r")  # [N, L]
        self._Y = np.load(y_path)                   # [N]

        # Load IDs
        with open(self.cache_dir / f"sample_ids_{split}.json", "r") as f:
            self._sample_ids: List[str] = json.load(f)
        with open(self.cache_dir / f"structure_ids_{split}.json", "r") as f:
            self._structure_ids: List[str] = json.load(f)

        assert len(self._X) == len(self._Y) == len(self._sample_ids)

        logger.info(
            "MaterializedXRDDataset | domain=%s scale=%d%% split=%s task=%s "
            "samples=%d (from %s)",
            self.domain, self.scale_pct, split, task,
            len(self._sample_ids), self.cache_dir.name,
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        return len(self._idx_to_name)

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._idx_to_name)

    @property
    def sample_ids(self) -> List[str]:
        return list(self._sample_ids)

    @property
    def structure_ids(self) -> List[str]:
        return list(self._structure_ids)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        x = torch.from_numpy(self._X[idx].copy()).unsqueeze(0)  # [1, L]
        y = torch.tensor(int(self._Y[idx]), dtype=torch.long)

        return {
            "x":            x,
            "y":            y,
            "sample_id":    self._sample_ids[idx],
            "structure_id": self._structure_ids[idx],
            "domain":       self.domain,
            "task":         self.task,
        }

    # ── Class weight computation ──────────────────────────────────────────────

    def compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for the current split."""
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for label in self._Y:
            counts[int(label)] += 1
        counts = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.num_classes
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_labels(self) -> List[int]:
        """Return a label for every sample (used by WeightedRandomSampler)."""
        return self._Y.tolist()


def get_materialized_cache_dir(
    domain: str,
    scale: int,
    task: str,
    base_dir: str | Path = "outputs/phase2/materialized",
) -> Path:
    """Return the expected cache directory for a given config."""
    return Path(base_dir) / f"{domain}_{scale}pct" / task


def materialized_cache_exists(
    domain: str,
    scale: int,
    task: str,
    base_dir: str | Path = "outputs/phase2/materialized",
) -> bool:
    """Check if a materialized cache exists and is complete."""
    cache_dir = get_materialized_cache_dir(domain, scale, task, base_dir)
    required = [
        "metadata.json", "label_map.json",
        "x_train.npy", "y_train.npy",
        "x_val.npy", "y_val.npy",
        "x_test.npy", "y_test.npy",
        "sample_ids_train.json", "sample_ids_val.json", "sample_ids_test.json",
        "structure_ids_train.json", "structure_ids_val.json", "structure_ids_test.json",
    ]
    return all((cache_dir / f).exists() for f in required)
