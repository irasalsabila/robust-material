"""
src/experiments/cross_domain_eval.py
=====================================
Cross-domain generalization experiment runner.

Runs all five cross-domain experiment types described in the spec:

  1. Single-domain training  (train Di -> test D1..D4)
  2. Progressive domain mixing  (train D1, D1+D2, D1+D2+D3, D1+D2+D3+D4)
  3. Leave-one-domain-out  (train all-but-one -> test held-out)
  4. Clean-to-noisy robustness  (train D1 -> test D2, D3, D4)
  5. Mixed-domain robustness  (train all -> test each domain)

For every train/test combination the script:
  * Trains a model using train_baseline.py logic (or loads an existing checkpoint)
  * Evaluates on the designated test domain
  * Records all required metrics to CSV files

Outputs
-------
  outputs/cross_domain/
    cross_domain_matrix.csv
    progressive_domain_mixing.csv
    leave_one_domain_out.csv
    robustness_drop_summary.csv
    calibration_summary.csv

Usage
-----
    # Run everything
    python src/experiments/cross_domain_eval.py \\
        --model cnn1d --scale 10 --task crystal_system --epochs 30

    # Run only specific experiment types
    python src/experiments/cross_domain_eval.py \\
        --model cnn1d --scale 10 --task crystal_system --epochs 30 \\
        --experiments single progressive

    # Skip training, only aggregate already-trained runs
    python src/experiments/cross_domain_eval.py \\
        --model cnn1d --scale 10 --task crystal_system \\
        --aggregate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.multi_domain_datamodule import MultiDomainDataModule
from src.data.datamodule import XRDDataModule
from src.models import build_model
from src.utils.metrics import (
    compute_metrics,
    save_metrics,
    save_classification_report_csv,
    plot_confusion_matrix,
    compute_robustness_drop,
)
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DOMAINS = ["D1", "D2", "D3", "D4"]

# Noise severity label for reference
DOMAIN_NOISE: Dict[str, str] = {
    "D1": "low_noise",
    "D2": "background+low_noise",
    "D3": "medium_noise",
    "D4": "high_noise",
}

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Focal Loss (duplicated from train_baseline for self-containment)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        ce = torch.nn.functional.nll_loss(
            log_probs, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-domain generalization experiment runner"
    )
    p.add_argument("--model",         default="cnn1d", choices=["mlp", "cnn1d"])
    p.add_argument("--scale",         default=10, type=int,
                   choices=[5, 10, 15, 20, 25, 30, 35])
    p.add_argument("--task",          default="crystal_system",
                   choices=["crystal_system", "top10_space_group"])
    p.add_argument("--epochs",        default=30, type=int)
    p.add_argument("--batch-size",    default=64, type=int)
    p.add_argument("--lr",            default=1e-3, type=float)
    p.add_argument("--weight-decay",  default=1e-4, type=float)
    p.add_argument("--normalization", default="max_intensity",
                   choices=["none", "minmax", "standard", "max_intensity"])
    p.add_argument("--target-length", default=4500, type=int)
    p.add_argument("--num-workers",   default=0, type=int)
    p.add_argument("--seed",          default=42, type=int)
    p.add_argument("--patience",      default=5, type=int)
    p.add_argument("--loss",          default="ce", choices=["ce", "focal"])
    p.add_argument("--focal-gamma",   default=2.0, type=float)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--source",        default=None, choices=["zip", "materialized"])
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["single", "progressive", "leave_one_out",
                 "clean_to_noisy", "mixed"],
        choices=["single", "progressive", "leave_one_out",
                 "clean_to_noisy", "mixed"],
        help="Which experiment types to run",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training; only aggregate existing outputs into CSVs",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (default: outputs/cross_domain)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[Dict[int, str]] = None,
    prefix: str = "test_",
    task: str = "crystal_system",
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        total += x.size(0)

    topk = [3, 5] if task == "top10_space_group" else None
    metrics = compute_metrics(
        all_labels, all_preds, class_names,
        prefix=prefix, y_probs=all_probs, topk=topk,
    )
    return total_loss / total, metrics


# ---------------------------------------------------------------------------
# Full train + eval for one train/test domain combination
# ---------------------------------------------------------------------------

def run_experiment(
    train_domains: List[str],
    test_domain: str,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> Dict:
    """
    Train a model on train_domains and evaluate on test_domain.
    Returns a flat dict of result metrics suitable for CSV rows.
    """
    set_seed(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "outputs" / "data_audit" / "manifest.json"
    splits_dir    = ROOT / "outputs" / "dataset_splits" / "splits"

    is_multi = (
        len(train_domains) > 1
        or (len(train_domains) == 1 and train_domains[0] != test_domain)
    )

    if is_multi:
        dm = MultiDomainDataModule(
            manifest_path        = manifest_path,
            splits_dir           = splits_dir,
            train_domains        = train_domains,
            test_domain          = test_domain,
            scale                = args.scale,
            task                 = args.task,
            data_dir             = ROOT / "data",
            normalization        = args.normalization,
            target_length        = args.target_length,
            batch_size           = args.batch_size,
            num_workers          = args.num_workers,
            source               = args.source,
            materialized_base_dir= ROOT / "outputs" / "materialized",
        )
    else:
        split_json = splits_dir / f"split_{train_domains[0]}_{args.scale}pct.json"
        dm = XRDDataModule(
            manifest_path        = manifest_path,
            split_json_path      = split_json,
            task                 = args.task,
            data_dir             = ROOT / "data",
            normalization        = args.normalization,
            target_length        = args.target_length,
            batch_size           = args.batch_size,
            num_workers          = args.num_workers,
            source               = args.source,
            materialized_base_dir= ROOT / "outputs" / "materialized",
        )

    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    test_loader  = dm.test_dataloader()

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(
        args.model,
        input_length = args.target_length,
        num_classes  = dm.num_classes,
    ).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    cw = None if args.no_class_weights else dm.class_weights().to(device)
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, weight=cw)
    elif cw is not None:
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_macro_f1 = -1.0
    best_epoch        = 0
    patience_counter  = 0
    train_log: List[Dict] = []

    logger.info(
        "Training %s train=%s -> test=%s scale=%d%% task=%s",
        args.model, "+".join(train_domains), test_domain,
        args.scale, args.task,
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = run_eval(
            model, val_loader, criterion, device,
            class_names=dm.class_names, prefix="val_", task=args.task,
        )
        scheduler.step()
        elapsed = time.time() - t0

        val_f1  = val_metrics["val_macro_f1"]
        val_acc = val_metrics["val_accuracy"]
        train_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "val_f1":     round(val_f1,     6),
            "elapsed_s":  round(elapsed,    2),
        })

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch        = epoch
            patience_counter  = 0
            torch.save(
                {
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "val_macro_f1":       val_f1,
                    "val_accuracy":       val_acc,
                    "train_domains":      train_domains,
                    "test_domain":        test_domain,
                },
                run_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch, args.patience,
            )
            break

    # Save train log
    if train_log:
        with open(run_dir / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
            w.writeheader()
            w.writerows(train_log)

    # ── Reload best and evaluate on test domain ───────────────────────────
    ckpt = torch.load(
        run_dir / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Reloaded best model from epoch %d", best_epoch)

    _, test_metrics = run_eval(
        model, test_loader, criterion, device,
        class_names=dm.class_names, prefix="test_", task=args.task,
    )
    save_metrics(test_metrics, run_dir / "test_metrics.json")

    # ── Build result row ───────────────────────────────────────────────────
    row = _build_result_row(
        train_domains = train_domains,
        test_domain   = test_domain,
        task          = args.task,
        scale         = args.scale,
        architecture  = args.model,
        loss_type     = args.loss,
        metrics       = test_metrics,
        prefix        = "test_",
    )
    return row


def _build_result_row(
    train_domains: List[str],
    test_domain: str,
    task: str,
    scale: int,
    architecture: str,
    loss_type: str,
    metrics: Dict,
    prefix: str = "test_",
    robustness_drop: Optional[float] = None,
    macro_f1_drop: Optional[float] = None,
) -> Dict:
    """Flatten a metrics dict into a single CSV row."""
    p = prefix
    row: Dict = {
        "train_domains":    "+".join(train_domains),
        "test_domain":      test_domain,
        "task":             task,
        "scale_pct":        scale,
        "architecture":     architecture,
        "loss_type":        loss_type,
        "accuracy":         metrics.get(f"{p}accuracy",          ""),
        "macro_f1":         metrics.get(f"{p}macro_f1",          ""),
        "weighted_f1":      metrics.get(f"{p}weighted_f1",       ""),
        "balanced_accuracy":metrics.get(f"{p}balanced_accuracy", ""),
    }
    # Top-k (space group only)
    row["top_3_accuracy"] = metrics.get(f"{p}top_3_accuracy", "")
    row["top_5_accuracy"] = metrics.get(f"{p}top_5_accuracy", "")
    # Calibration
    row["ece"]               = metrics.get(f"{p}ece",               "")
    row["mean_confidence"]   = metrics.get(f"{p}mean_confidence",   "")
    row["overconfidence_gap"]= metrics.get(f"{p}overconfidence_gap","")
    # Robustness drop (filled in post-hoc)
    row["robustness_drop"]   = robustness_drop if robustness_drop is not None else ""
    row["macro_f1_drop"]     = macro_f1_drop   if macro_f1_drop   is not None else ""
    return row


def load_test_metrics(run_dir: Path, prefix: str = "test_") -> Optional[Dict]:
    """Load test_metrics.json from a run directory if it exists."""
    p = run_dir / "test_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def run_or_load(
    train_domains: List[str],
    test_domain: str,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    force_retrain: bool = False,
) -> Dict:
    """
    If a completed run exists (test_metrics.json present) and
    force_retrain is False, load and return the saved metrics.
    Otherwise train from scratch.
    """
    existing = load_test_metrics(run_dir)
    if existing is not None and not force_retrain:
        logger.info("Reusing existing run: %s", run_dir.name)
        return _build_result_row(
            train_domains = train_domains,
            test_domain   = test_domain,
            task          = args.task,
            scale         = args.scale,
            architecture  = args.model,
            loss_type     = args.loss,
            metrics       = existing,
            prefix        = "test_",
        )
    return run_experiment(
        train_domains, test_domain, args, run_dir, device
    )


# ---------------------------------------------------------------------------
# Run-name helpers
# ---------------------------------------------------------------------------

def run_name_for(
    train_domains: List[str],
    test_domain: str,
    args: argparse.Namespace,
) -> str:
    train_tag = "+".join(train_domains)
    return (
        f"{args.model}_train{train_tag}_test{test_domain}"
        f"_{args.scale}pct_{args.task}"
    )


# ---------------------------------------------------------------------------
# Robustness drop computation
# ---------------------------------------------------------------------------

def add_robustness_drops(
    rows: List[Dict],
    clean_domain: str = "D1",
) -> List[Dict]:
    """
    For each row, compute robustness_drop and macro_f1_drop relative to
    the clean D1 test accuracy of the same train configuration.
    """
    # Index clean-domain rows by train_domains key
    clean_acc: Dict[str, float] = {}
    clean_f1:  Dict[str, float] = {}
    for row in rows:
        if row["test_domain"] == clean_domain and row["accuracy"] != "":
            clean_acc[row["train_domains"]] = float(row["accuracy"])
            clean_f1[row["train_domains"]]  = float(row["macro_f1"])

    for row in rows:
        td = row["train_domains"]
        if row["test_domain"] != clean_domain and td in clean_acc and row["accuracy"] != "":
            row["robustness_drop"] = round(
                clean_acc[td] - float(row["accuracy"]), 4
            )
            row["macro_f1_drop"] = round(
                clean_f1[td] - float(row["macro_f1"]), 4
            )
    return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

ROW_FIELDS = [
    "train_domains", "test_domain", "task", "scale_pct",
    "architecture", "loss_type",
    "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
    "top_3_accuracy", "top_5_accuracy",
    "ece", "mean_confidence", "overconfidence_gap",
    "robustness_drop", "macro_f1_drop",
]


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved: %s (%d rows)", path, len(rows))


def write_calibration_csv(rows: List[Dict], path: Path) -> None:
    """Write calibration-focused CSV (per-head if multitask, else per-task)."""
    cal_fields = [
        "train_domains", "test_domain", "task", "scale_pct",
        "architecture", "loss_type",
        "accuracy", "ece", "mean_confidence", "overconfidence_gap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cal_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved calibration summary: %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Experiment builders
# ---------------------------------------------------------------------------

def exp_single_domain(
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
    aggregate_only: bool = False,
) -> List[Dict]:
    """
    Experiment 1: Single-domain training.
    For each training domain Di, evaluate on every test domain Dj.
    """
    rows: List[Dict] = []
    for train_d in DOMAINS:
        for test_d in DOMAINS:
            run_dir = out_root / "runs" / run_name_for([train_d], test_d, args)
            if aggregate_only:
                m = load_test_metrics(run_dir)
                if m is None:
                    logger.warning("Missing run: %s", run_dir.name)
                    continue
                row = _build_result_row(
                    [train_d], test_d, args.task, args.scale,
                    args.model, args.loss, m,
                )
            else:
                row = run_or_load([train_d], test_d, args, run_dir, device)
            rows.append(row)
    rows = add_robustness_drops(rows)
    return rows


def exp_progressive_mixing(
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
    aggregate_only: bool = False,
) -> List[Dict]:
    """
    Experiment 2: Progressive domain mixing.
    Train on D1, D1+D2, D1+D2+D3, D1+D2+D3+D4.
    For each setting, test on each domain separately.
    """
    rows: List[Dict] = []
    progressive_configs = [
        ["D1"],
        ["D1", "D2"],
        ["D1", "D2", "D3"],
        ["D1", "D2", "D3", "D4"],
    ]
    for train_domains in progressive_configs:
        for test_d in DOMAINS:
            run_dir = out_root / "runs" / run_name_for(train_domains, test_d, args)
            if aggregate_only:
                m = load_test_metrics(run_dir)
                if m is None:
                    logger.warning("Missing run: %s", run_dir.name)
                    continue
                row = _build_result_row(
                    train_domains, test_d, args.task, args.scale,
                    args.model, args.loss, m,
                )
            else:
                row = run_or_load(train_domains, test_d, args, run_dir, device)
            rows.append(row)
    rows = add_robustness_drops(rows)
    return rows


def exp_leave_one_out(
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
    aggregate_only: bool = False,
) -> List[Dict]:
    """
    Experiment 3: Leave-one-domain-out.
    Train on all domains except the test domain.
    """
    rows: List[Dict] = []
    for held_out in DOMAINS:
        train_domains = [d for d in DOMAINS if d != held_out]
        run_dir = out_root / "runs" / run_name_for(train_domains, held_out, args)
        if aggregate_only:
            m = load_test_metrics(run_dir)
            if m is None:
                logger.warning("Missing run: %s", run_dir.name)
                continue
            row = _build_result_row(
                train_domains, held_out, args.task, args.scale,
                args.model, args.loss, m,
            )
        else:
            row = run_or_load(train_domains, held_out, args, run_dir, device)
        rows.append(row)
    return rows


def exp_clean_to_noisy(
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
    aggregate_only: bool = False,
) -> List[Dict]:
    """
    Experiment 4: Clean-to-noisy robustness.
    Train D1 only, test on D1, D2, D3, D4.
    Reports robustness drop relative to D1 test accuracy.
    """
    train_domains = ["D1"]
    rows: List[Dict] = []
    for test_d in DOMAINS:
        run_dir = out_root / "runs" / run_name_for(train_domains, test_d, args)
        if aggregate_only:
            m = load_test_metrics(run_dir)
            if m is None:
                logger.warning("Missing run: %s", run_dir.name)
                continue
            row = _build_result_row(
                train_domains, test_d, args.task, args.scale,
                args.model, args.loss, m,
            )
        else:
            row = run_or_load(train_domains, test_d, args, run_dir, device)
        rows.append(row)
    rows = add_robustness_drops(rows, clean_domain="D1")
    return rows


def exp_mixed_robustness(
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
    aggregate_only: bool = False,
) -> List[Dict]:
    """
    Experiment 5: Mixed-domain robustness.
    Train on all four domains, test on each domain separately.
    Reports whether mixed-domain training reduces robustness drop.
    """
    train_domains = ["D1", "D2", "D3", "D4"]
    rows: List[Dict] = []
    for test_d in DOMAINS:
        run_dir = out_root / "runs" / run_name_for(train_domains, test_d, args)
        if aggregate_only:
            m = load_test_metrics(run_dir)
            if m is None:
                logger.warning("Missing run: %s", run_dir.name)
                continue
            row = _build_result_row(
                train_domains, test_d, args.task, args.scale,
                args.model, args.loss, m,
            )
        else:
            row = run_or_load(train_domains, test_d, args, run_dir, device)
        rows.append(row)
    rows = add_robustness_drops(rows, clean_domain="D1")
    return rows


# ---------------------------------------------------------------------------
# Robustness drop summary
# ---------------------------------------------------------------------------

def build_robustness_drop_summary(
    single_rows:    List[Dict],
    progressive_rows: List[Dict],
    clean_to_noisy: List[Dict],
    mixed_rows:     List[Dict],
) -> List[Dict]:
    """
    Aggregate robustness drop across all settings that have a D1
    reference point.  Returns rows ready for robustness_drop_summary.csv.
    """
    summary: List[Dict] = []

    def _add(rows: List[Dict], experiment_tag: str) -> None:
        for row in rows:
            if row.get("robustness_drop", "") != "":
                summary.append({
                    "experiment":       experiment_tag,
                    "train_domains":    row["train_domains"],
                    "test_domain":      row["test_domain"],
                    "task":             row["task"],
                    "scale_pct":        row["scale_pct"],
                    "architecture":     row["architecture"],
                    "loss_type":        row["loss_type"],
                    "clean_domain":     "D1",
                    "accuracy":         row["accuracy"],
                    "macro_f1":         row["macro_f1"],
                    "robustness_drop":  row["robustness_drop"],
                    "macro_f1_drop":    row["macro_f1_drop"],
                    "ece":              row["ece"],
                    "mean_confidence":  row["mean_confidence"],
                    "overconfidence_gap": row["overconfidence_gap"],
                })

    _add(single_rows,     "single_domain")
    _add(progressive_rows,"progressive_mixing")
    _add(clean_to_noisy,  "clean_to_noisy")
    _add(mixed_rows,      "mixed_domain")
    return summary


ROBUSTNESS_FIELDS = [
    "experiment", "train_domains", "test_domain", "task", "scale_pct",
    "architecture", "loss_type", "clean_domain",
    "accuracy", "macro_f1", "robustness_drop", "macro_f1_drop",
    "ece", "mean_confidence", "overconfidence_gap",
]


def write_robustness_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROBUSTNESS_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved robustness summary: %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT / "outputs" / "cross_domain"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output root: %s", out_root)
    logger.info("Experiments: %s", args.experiments)
    logger.info("Aggregate only: %s", args.aggregate_only)

    # Run experiment types
    single_rows:      List[Dict] = []
    progressive_rows: List[Dict] = []
    leave_out_rows:   List[Dict] = []
    clean_noisy_rows: List[Dict] = []
    mixed_rows:       List[Dict] = []

    if "single" in args.experiments:
        logger.info("=== Experiment 1: Single-domain training ===")
        single_rows = exp_single_domain(args, out_root, device, args.aggregate_only)
        write_csv(single_rows, out_root / "cross_domain_matrix.csv")

    if "progressive" in args.experiments:
        logger.info("=== Experiment 2: Progressive domain mixing ===")
        progressive_rows = exp_progressive_mixing(args, out_root, device, args.aggregate_only)
        write_csv(progressive_rows, out_root / "progressive_domain_mixing.csv")

    if "leave_one_out" in args.experiments:
        logger.info("=== Experiment 3: Leave-one-domain-out ===")
        leave_out_rows = exp_leave_one_out(args, out_root, device, args.aggregate_only)
        write_csv(leave_out_rows, out_root / "leave_one_domain_out.csv")

    if "clean_to_noisy" in args.experiments:
        logger.info("=== Experiment 4: Clean-to-noisy robustness ===")
        clean_noisy_rows = exp_clean_to_noisy(args, out_root, device, args.aggregate_only)
        # These rows are a subset of single_domain (D1 training)
        # but we write them separately for clarity
        write_csv(clean_noisy_rows, out_root / "clean_to_noisy.csv")

    if "mixed" in args.experiments:
        logger.info("=== Experiment 5: Mixed-domain robustness ===")
        mixed_rows = exp_mixed_robustness(args, out_root, device, args.aggregate_only)

    # Robustness drop summary (across all experiments)
    robustness_rows = build_robustness_drop_summary(
        single_rows, progressive_rows, clean_noisy_rows, mixed_rows
    )
    write_robustness_csv(robustness_rows, out_root / "robustness_drop_summary.csv")

    # Calibration summary (collect all ECE rows from every experiment)
    all_rows = single_rows + progressive_rows + leave_out_rows + clean_noisy_rows + mixed_rows
    write_calibration_csv(all_rows, out_root / "calibration_summary.csv")

    # Print final summary table
    print("\n" + "=" * 80)
    print("CROSS-DOMAIN EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'train_domains':<24} {'test':>5} {'acc':>7} {'f1':>7} "
          f"{'ece':>7} {'drop':>7}")
    print("-" * 60)
    for row in (single_rows + progressive_rows + leave_out_rows
                + clean_noisy_rows + mixed_rows):
        acc  = f"{float(row['accuracy']):.4f}"  if row['accuracy']  != "" else "  N/A "
        f1   = f"{float(row['macro_f1']):.4f}"  if row['macro_f1']  != "" else "  N/A "
        ece  = f"{float(row['ece']):.4f}"        if row['ece']       != "" else "  N/A "
        drop = (f"{float(row['robustness_drop']):.4f}"
                if row['robustness_drop'] != "" else "  N/A ")
        print(
            f"{row['train_domains']:<24} {row['test_domain']:>5} "
            f"{acc:>7} {f1:>7} {ece:>7} {drop:>7}"
        )
    print("=" * 80)
    print(f"\nAll CSVs saved to: {out_root}")


if __name__ == "__main__":
    main()
