"""
src/smoke_test_dataloader.py
============================
Smoke test for the Phase 2 XRD DataLoader.

Usage
-----
    python src/smoke_test_dataloader.py --domain D1 --scale 10 --task crystal_system
    python src/smoke_test_dataloader.py --domain D1 --scale 20 --task top10_space_group

Checks
------
* Train / val / test loaders build without error
* Batch shapes are correct: x=[B,1,L], y=[B]
* All tensors are finite (no NaN / Inf)
* No structure-level leakage between splits
* Label distribution printed per split
* Class weights printed
* Report saved to outputs/phase2/dataloader_smoke_test_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# Allow running from project root: python src/smoke_test_dataloader.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xrd_dataset import XRDDataset, CRYSTAL_SYSTEM_NAMES, TOP10_SG_NAMES, TOP10_SG_CLASSES
from src.data.datamodule import XRDDataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 DataLoader smoke test")
    p.add_argument("--domain", default="D1", choices=["D1", "D2", "D3", "D4"])
    p.add_argument("--scale",  default=10,   type=int, choices=[5, 10, 15, 20, 25, 30, 35])
    p.add_argument("--task",   default="crystal_system",
                   choices=["crystal_system", "top10_space_group"])
    p.add_argument("--batch-size",     default=64,   type=int)
    p.add_argument("--num-workers",    default=0,    type=int)
    p.add_argument("--normalization",  default="max_intensity",
                   choices=["none", "minmax", "standard", "max_intensity"])
    p.add_argument("--target-length",  default=4500, type=int)
    p.add_argument("--source",         default=None, choices=["zip", "materialized"],
                   help="Data source: 'zip' (raw), 'materialized' (numpy cache), or auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_name(label: int, task: str) -> str:
    if task == "crystal_system":
        return CRYSTAL_SYSTEM_NAMES.get(label, str(label))
    sg = TOP10_SG_CLASSES[label] if label < len(TOP10_SG_CLASSES) else label
    return TOP10_SG_NAMES.get(sg, str(sg))


def check_batch(batch: dict, task: str, target_length: int, split: str) -> list[str]:
    """Run assertions on one batch. Return list of warning strings."""
    warnings = []

    x = batch["x"]
    y = batch["y"]

    # Shape checks
    assert x.ndim == 3, f"Expected x.ndim=3, got {x.ndim}"
    assert x.shape[1] == 1, f"Expected x.shape[1]=1 (channel), got {x.shape[1]}"
    assert x.shape[2] == target_length, (
        f"Expected x.shape[2]={target_length}, got {x.shape[2]}"
    )
    assert y.ndim == 1, f"Expected y.ndim=1, got {y.ndim}"
    assert x.shape[0] == y.shape[0], "Batch size mismatch between x and y"

    # Finite check
    if not torch.isfinite(x).all():
        warnings.append(f"[{split}] Non-finite values in x (NaN or Inf detected)")
    if not torch.isfinite(y.float()).all():
        warnings.append(f"[{split}] Non-finite values in y")

    # dtype checks
    assert x.dtype == torch.float32, f"Expected x.dtype=float32, got {x.dtype}"
    assert y.dtype == torch.long,    f"Expected y.dtype=long, got {y.dtype}"

    return warnings


def check_leakage(
    train_ds: XRDDataset,
    val_ds:   XRDDataset,
    test_ds:  XRDDataset,
) -> list[str]:
    """Verify no structure-level overlap between splits."""
    train_structs = set(train_ds.structure_ids)
    val_structs   = set(val_ds.structure_ids)
    test_structs  = set(test_ds.structure_ids)

    issues = []
    tv = train_structs & val_structs
    tt = train_structs & test_structs
    vt = val_structs   & test_structs

    if tv:
        issues.append(f"LEAKAGE: {len(tv)} structures in both train and val: {sorted(tv)[:5]}")
    if tt:
        issues.append(f"LEAKAGE: {len(tt)} structures in both train and test: {sorted(tt)[:5]}")
    if vt:
        issues.append(f"LEAKAGE: {len(vt)} structures in both val and test: {sorted(vt)[:5]}")

    return issues


def label_distribution(dataset: XRDDataset, task: str) -> dict[str, int]:
    labels = dataset.get_sample_labels()
    counts = Counter(labels)
    return {
        f"{label_name(lbl, task)} ({lbl})": cnt
        for lbl, cnt in sorted(counts.items())
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    manifest_path   = PROJECT_ROOT / "outputs" / "data_audit" / "manifest.json"
    split_json_path = (
        PROJECT_ROOT / "outputs" / "dataset_splits" / "splits"
        / f"split_{args.domain}_{args.scale}pct.json"
    )
    output_dir = PROJECT_ROOT / "outputs" / "dataloader_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 2 DATALOADER SMOKE TEST")
    print("=" * 70)
    print(f"  domain      : {args.domain}")
    print(f"  scale       : {args.scale}%")
    print(f"  task        : {args.task}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  norm        : {args.normalization}")
    print(f"  target_len  : {args.target_length}")
    print()

    # ── Build DataModule ──────────────────────────────────────────────────────
    t0 = time.time()
    dm = XRDDataModule(
        manifest_path   = manifest_path,
        split_json_path = split_json_path,
        task            = args.task,
        data_dir        = PROJECT_ROOT / "data",
        normalization   = args.normalization,
        target_length   = args.target_length,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        label_map_dir   = PROJECT_ROOT / "outputs" / "label_mappings",
        source          = args.source,
        materialized_base_dir = PROJECT_ROOT / "outputs" / "materialized",
    )
    dm.setup()
    setup_time = time.time() - t0
    print(f"  setup time  : {setup_time:.2f}s")
    print()

    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    test_loader  = dm.test_dataloader()

    all_warnings: list[str] = []
    report_lines: list[str] = []

    # ── Leakage check ─────────────────────────────────────────────────────────
    leakage_issues = check_leakage(dm.train_dataset, dm.val_dataset, dm.test_dataset)
    leakage_status = "CLEAN — no structure-level leakage detected" if not leakage_issues else "FAIL"
    print(f"Leakage check : {leakage_status}")
    if leakage_issues:
        for issue in leakage_issues:
            print(f"  !! {issue}")
            all_warnings.append(issue)
    print()

    # ── Per-split checks ──────────────────────────────────────────────────────
    split_info: dict = {}

    for split_name, loader, dataset in [
        ("train", train_loader, dm.train_dataset),
        ("val",   val_loader,   dm.val_dataset),
        ("test",  test_loader,  dm.test_dataset),
    ]:
        print(f"── {split_name.upper()} ──────────────────────────────────────")
        print(f"  samples     : {len(dataset):,}")
        print(f"  batches     : {len(loader)}")

        # Fetch first batch
        t1 = time.time()
        batch = next(iter(loader))
        fetch_time = time.time() - t1

        x, y = batch["x"], batch["y"]
        print(f"  x shape     : {list(x.shape)}  dtype={x.dtype}")
        print(f"  y shape     : {list(y.shape)}  dtype={y.dtype}")
        print(f"  x range     : [{x.min():.4f}, {x.max():.4f}]")
        print(f"  first fetch : {fetch_time:.3f}s")

        # Assertions
        warns = check_batch(batch, args.task, args.target_length, split_name)
        all_warnings.extend(warns)
        for w in warns:
            print(f"  WARNING: {w}")

        # Label distribution
        dist = label_distribution(dataset, args.task)
        print(f"  label dist  :")
        for cls_name, cnt in dist.items():
            pct = 100.0 * cnt / len(dataset)
            print(f"    {cls_name:<35s} {cnt:>6,}  ({pct:5.1f}%)")

        split_info[split_name] = {
            "samples":  len(dataset),
            "batches":  len(loader),
            "x_shape":  list(x.shape),
            "y_shape":  list(y.shape),
            "x_min":    float(x.min()),
            "x_max":    float(x.max()),
            "label_distribution": dist,
        }
        print()

    # ── Class weights ─────────────────────────────────────────────────────────
    print("── CLASS WEIGHTS (from train split) ─────────────────────────────")
    cw = dm.class_weights()
    print(f"  shape       : {list(cw.shape)}")
    for idx, w in enumerate(cw.tolist()):
        name = label_name(idx, args.task)
        print(f"    [{idx}] {name:<30s}  weight={w:.4f}")
    print()

    # ── Sampler info ──────────────────────────────────────────────────────────
    from src.data.samplers import should_use_weighted_sampler
    ws_active = should_use_weighted_sampler(args.task, "train")
    print(f"WeightedRandomSampler : {'ACTIVE' if ws_active else 'INACTIVE'} for train")
    print(f"  (val/test always use sequential sampler)")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("── SUMMARY ──────────────────────────────────────────────────────")
    status = "PASS" if not all_warnings else f"PASS WITH {len(all_warnings)} WARNING(S)"
    print(f"  Status      : {status}")
    if all_warnings:
        for w in all_warnings:
            print(f"  WARNING: {w}")
    print()

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = output_dir / "dataloader_smoke_test_report.md"
    _write_report(
        report_path  = report_path,
        args         = args,
        split_info   = split_info,
        class_weights= cw,
        leakage_status = leakage_status,
        leakage_issues = leakage_issues,
        ws_active    = ws_active,
        all_warnings = all_warnings,
        setup_time   = setup_time,
        task         = args.task,
    )
    print(f"Report saved  : {report_path}")
    print("=" * 70 + "\n")

    if leakage_issues:
        sys.exit(1)


def _write_report(
    report_path: Path,
    args,
    split_info: dict,
    class_weights: torch.Tensor,
    leakage_status: str,
    leakage_issues: list,
    ws_active: bool,
    all_warnings: list,
    setup_time: float,
    task: str,
) -> None:
    lines = [
        "# Phase 2 DataLoader Smoke Test Report",
        "",
        f"**Domain:** {args.domain}  |  **Scale:** {args.scale}%  |  **Task:** {args.task}",
        f"**Normalization:** {args.normalization}  |  **Target length:** {args.target_length}",
        f"**Batch size:** {args.batch_size}  |  **Setup time:** {setup_time:.2f}s",
        "",
        "## Leakage Check",
        "",
        f"Status: **{leakage_status}**",
    ]
    if leakage_issues:
        for issue in leakage_issues:
            lines.append(f"- {issue}")
    lines += [""]

    lines += [
        "## Split Overview",
        "",
        "| Split | Samples | Batches | x shape | y shape | x min | x max |",
        "|-------|---------|---------|---------|---------|-------|-------|",
    ]
    for split_name, info in split_info.items():
        lines.append(
            f"| {split_name} | {info['samples']:,} | {info['batches']} "
            f"| {info['x_shape']} | {info['y_shape']} "
            f"| {info['x_min']:.4f} | {info['x_max']:.4f} |"
        )

    lines += ["", "## Label Distribution", ""]
    for split_name, info in split_info.items():
        lines.append(f"### {split_name.capitalize()}")
        lines.append("")
        lines.append("| Class | Count | % |")
        lines.append("|-------|-------|---|")
        total = info["samples"]
        for cls_name, cnt in info["label_distribution"].items():
            pct = 100.0 * cnt / total
            lines.append(f"| {cls_name} | {cnt:,} | {pct:.1f}% |")
        lines.append("")

    lines += [
        "## Class Weights (train split)",
        "",
        "| Index | Class | Weight |",
        "|-------|-------|--------|",
    ]
    for idx, w in enumerate(class_weights.tolist()):
        name = label_name(idx, task)
        lines.append(f"| {idx} | {name} | {w:.4f} |")

    lines += [
        "",
        "## Sampler",
        "",
        f"WeightedRandomSampler: **{'ACTIVE' if ws_active else 'INACTIVE'}** for train split.",
        "Val and test always use sequential (unweighted) sampling.",
        "",
        "## Warnings",
        "",
    ]
    if all_warnings:
        for w in all_warnings:
            lines.append(f"- {w}")
    else:
        lines.append("None.")

    lines += [
        "",
        "## Verdict",
        "",
        "**PASS**" if not leakage_issues else "**FAIL — leakage detected**",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
