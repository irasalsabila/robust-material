"""
src/data/materialize_dataset.py
================================
Pre-process XRD patterns from zip archives into numpy arrays for fast training.

Reads patterns from zip, applies normalization + pad/truncate, and saves:
    outputs/materialized/{domain}_{scale}pct/{task}/
        x_train.npy          float32 [N_train, target_length]
        y_train.npy          int64   [N_train]
        x_val.npy            float32 [N_val, target_length]
        y_val.npy            int64   [N_val]
        x_test.npy           float32 [N_test, target_length]
        y_test.npy           int64   [N_test]
        sample_ids_train.json
        sample_ids_val.json
        sample_ids_test.json
        structure_ids_train.json
        structure_ids_val.json
        structure_ids_test.json
        metadata.json
        label_map.json

Usage
-----
    python src/data/materialize_dataset.py --domain D1 --scale 10 --task crystal_system
    python src/data/materialize_dataset.py --domain D1 --scale 10 --task top10_space_group
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.transforms import preprocess
from src.data.xrd_dataset import (
    DOMAIN_FOLDER,
    _build_label_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize XRD dataset from zip to numpy")
    p.add_argument("--domain", default="D1", choices=["D1", "D2", "D3", "D4"])
    p.add_argument("--scale", default=10, type=int, choices=[5, 10, 15, 20, 25, 30, 35])
    p.add_argument("--task", default="crystal_system",
                   choices=["crystal_system", "top10_space_group"])
    p.add_argument("--normalization", default="max_intensity",
                   choices=["none", "minmax", "standard", "max_intensity"])
    p.add_argument("--target-length", default=4500, type=int)
    p.add_argument("--output-dir", default=None,
                   help="Override output directory (default: outputs/phase2/materialized/)")
    return p.parse_args()


def load_manifest(manifest_path: Path) -> Dict:
    with open(manifest_path, "r") as f:
        raw = json.load(f)
    return raw["structures"]


def load_split(split_json_path: Path) -> Dict:
    with open(split_json_path, "r") as f:
        return json.load(f)


def get_label(manifest: Dict, structure_id: str, task: str, class_to_idx: Dict) -> int:
    """Get integer label for a structure. Raises KeyError if not in label map."""
    info = manifest[structure_id]
    if task == "crystal_system":
        cs_key = str(info["crystal_sys"])
        return class_to_idx[cs_key]
    if task == "top10_space_group":
        sg_key = str(int(info["space_group_no"]))
        if sg_key not in class_to_idx:
            raise KeyError(sg_key)
        return class_to_idx[sg_key]
    raise ValueError(task)


def materialize_split(
    split_name: str,
    split_data: Dict,
    manifest: Dict,
    task: str,
    class_to_idx: Dict,
    zip_file: zipfile.ZipFile,
    zip_folder: str,
    normalization: str,
    target_length: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Process one split (train/val/test) and return arrays + ID lists."""

    # Get pattern IDs from split JSON
    raw_patterns: List[str] = split_data[f"{split_name}_patterns"]
    sample_ids = [p.split(":", 1)[1] for p in raw_patterns]
    structure_ids = [sid.rsplit("_", 1)[0] for sid in sample_ids]

    # Filter to valid labels (for top10_space_group, exclude non-top-10)
    valid_sample_ids: List[str] = []
    valid_structure_ids: List[str] = []
    valid_labels: List[int] = []

    for sample_id, struct_id in zip(sample_ids, structure_ids):
        if struct_id not in manifest:
            continue
        try:
            label = get_label(manifest, struct_id, task, class_to_idx)
            valid_sample_ids.append(sample_id)
            valid_structure_ids.append(struct_id)
            valid_labels.append(label)
        except KeyError:
            pass  # SG not in top-10

    n = len(valid_sample_ids)
    logger.info("  %s: %d samples (filtered from %d patterns)",
                split_name, n, len(raw_patterns))

    # Allocate arrays
    X = np.zeros((n, target_length), dtype=np.float32)
    Y = np.array(valid_labels, dtype=np.int64)

    # Load and preprocess patterns
    t0 = time.time()
    errors = 0
    for i, sample_id in enumerate(valid_sample_ids):
        if (i + 1) % 5000 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info("    [%s] %d/%d (%.0f samples/s)", split_name, i + 1, n, rate)

        member = f"{zip_folder}/{sample_id}.csv"
        try:
            with zip_file.open(member) as fh:
                raw = fh.read().decode("utf-8")
            lines = raw.strip().splitlines()
            values = np.array(
                [float(line.strip()) for line in lines[1:] if line.strip()],
                dtype=np.float32,
            )
            X[i] = preprocess(values, normalization, target_length)
        except Exception as e:
            logger.warning("    Error loading %s: %s", sample_id, e)
            errors += 1

    elapsed = time.time() - t0
    logger.info("  %s done: %.1fs (%.0f samples/s), %d errors",
                split_name, elapsed, n / max(elapsed, 0.001), errors)

    # Verify finite
    if not np.isfinite(X).all():
        non_finite = (~np.isfinite(X)).sum()
        logger.warning("  WARNING: %d non-finite values in X_%s", non_finite, split_name)

    return X, Y, valid_sample_ids, valid_structure_ids


def verify_no_leakage(
    train_structs: List[str],
    val_structs: List[str],
    test_structs: List[str],
) -> List[str]:
    """Check no structure-level overlap between splits."""
    issues = []
    train_set = set(train_structs)
    val_set = set(val_structs)
    test_set = set(test_structs)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    if tv:
        issues.append(f"LEAKAGE: {len(tv)} structures in train ∩ val")
    if tt:
        issues.append(f"LEAKAGE: {len(tt)} structures in train ∩ test")
    if vt:
        issues.append(f"LEAKAGE: {len(vt)} structures in val ∩ test")
    return issues


def main() -> None:
    args = parse_args()

    manifest_path = ROOT / "outputs" / "data_audit" / "manifest.json"
    split_json_path = (
        ROOT / "outputs" / "dataset_splits" / "splits"
        / f"split_{args.domain}_{args.scale}pct.json"
    )
    data_dir = ROOT / "data"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            ROOT / "outputs" / "materialized"
            / f"{args.domain}_{args.scale}pct" / args.task
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("MATERIALIZE XRD DATASET")
    print("=" * 70)
    print(f"  domain        : {args.domain}")
    print(f"  scale         : {args.scale}%")
    print(f"  task          : {args.task}")
    print(f"  normalization : {args.normalization}")
    print(f"  target_length : {args.target_length}")
    print(f"  output_dir    : {out_dir}")
    print()

    # Load manifest and split
    logger.info("Loading manifest ...")
    manifest = load_manifest(manifest_path)
    logger.info("Manifest: %d structures", len(manifest))

    split_data = load_split(split_json_path)
    logger.info("Split: %s %d%%", split_data["domain"], int(split_data["scale_pct"]))

    # Build label map
    class_to_idx, idx_to_name = _build_label_map(args.task)
    logger.info("Task: %s (%d classes)", args.task, len(idx_to_name))

    # Open zip
    zip_path = data_dir / f"{args.domain}.zip"
    zip_folder = DOMAIN_FOLDER[args.domain]
    logger.info("Opening %s (folder: %s) ...", zip_path.name, zip_folder)
    zf = zipfile.ZipFile(zip_path, "r")

    # Process each split
    results = {}
    total_t0 = time.time()

    for split_name in ("train", "val", "test"):
        X, Y, sample_ids, structure_ids = materialize_split(
            split_name=split_name,
            split_data=split_data,
            manifest=manifest,
            task=args.task,
            class_to_idx=class_to_idx,
            zip_file=zf,
            zip_folder=zip_folder,
            normalization=args.normalization,
            target_length=args.target_length,
        )
        results[split_name] = {
            "X": X,
            "Y": Y,
            "sample_ids": sample_ids,
            "structure_ids": structure_ids,
        }

    zf.close()
    total_elapsed = time.time() - total_t0
    logger.info("All splits processed in %.1fs", total_elapsed)

    # Verify no leakage
    leakage_issues = verify_no_leakage(
        results["train"]["structure_ids"],
        results["val"]["structure_ids"],
        results["test"]["structure_ids"],
    )
    if leakage_issues:
        for issue in leakage_issues:
            logger.error(issue)
        print("\n!! LEAKAGE DETECTED — aborting !!")
        sys.exit(1)
    else:
        logger.info("Leakage check: CLEAN")

    # Save arrays and metadata
    logger.info("Saving to %s ...", out_dir)
    total_bytes = 0

    for split_name, data in results.items():
        x_path = out_dir / f"x_{split_name}.npy"
        y_path = out_dir / f"y_{split_name}.npy"
        np.save(x_path, data["X"])
        np.save(y_path, data["Y"])
        total_bytes += x_path.stat().st_size + y_path.stat().st_size

        with open(out_dir / f"sample_ids_{split_name}.json", "w") as f:
            json.dump(data["sample_ids"], f)
        with open(out_dir / f"structure_ids_{split_name}.json", "w") as f:
            json.dump(data["structure_ids"], f)

    # Save metadata
    metadata = {
        "domain": args.domain,
        "scale_pct": args.scale,
        "task": args.task,
        "normalization": args.normalization,
        "target_length": args.target_length,
        "num_classes": len(idx_to_name),
        "train_samples": len(results["train"]["sample_ids"]),
        "val_samples": len(results["val"]["sample_ids"]),
        "test_samples": len(results["test"]["sample_ids"]),
        "x_dtype": "float32",
        "y_dtype": "int64",
        "leakage_clean": True,
        "total_time_s": round(total_elapsed, 2),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save label map
    label_map = {
        "task": args.task,
        "num_classes": len(idx_to_name),
        "class_to_idx": class_to_idx,
        "idx_to_name": {str(k): v for k, v in idx_to_name.items()},
    }
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("MATERIALIZATION COMPLETE")
    print("=" * 70)
    print(f"  Output dir    : {out_dir}")
    print(f"  Total time    : {total_elapsed:.1f}s")
    print(f"  Disk size     : {total_bytes / 1024 / 1024:.1f} MB (arrays only)")
    print(f"  Leakage       : CLEAN")
    print()
    print("  Split shapes:")
    for split_name, data in results.items():
        print(f"    {split_name:5s}: X={data['X'].shape}  Y={data['Y'].shape}")
    print()
    print("  Files created:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:40s} {size_kb:>8.1f} KB")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()