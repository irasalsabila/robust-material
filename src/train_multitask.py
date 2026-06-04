"""
src/train_multitask.py
=======================
Multitask training script: shared backbone + crystal system + space group heads.

Usage
-----
    # Single domain
    python src/train_multitask.py --model cnn1d --domain D1 --scale 20 --epochs 100

    # Multi-domain training, test on D3
    python src/train_multitask.py --model cnn1d --domains D1,D2 --test-domain D3 \
        --scale 20 --epochs 100

Outputs (outputs/single_vs_multitask/{run_name}/)
-------
    config.json
    train_log.csv
    val_metrics.json
    test_metrics.json
    crystal_classification_report.csv
    sg_classification_report.csv
    crystal_confusion_matrix.png
    sg_confusion_matrix.png
    best_model.pt
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.datamodule import XRDDataModule
from src.data.multi_domain_datamodule import MultiDomainDataModule
from src.models import MultitaskModel
from src.utils.metrics import (
    compute_metrics,
    compute_crystallographic_consistency,
    save_metrics,
    save_classification_report_csv,
    plot_confusion_matrix,
)
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multitask XRD Training")
    p.add_argument("--model", default="cnn1d",
                   choices=["cnn1d", "resnet1d", "cnn_attention"],
                   help="Backbone architecture")
    p.add_argument("--domain", default=None, choices=["D1", "D2", "D3", "D4"],
                   help="Single training/test domain (legacy)")
    p.add_argument("--domains", default=None,
                   help="Comma-separated training domains, e.g. 'D1,D2,D3'")
    p.add_argument("--test-domain", default=None, choices=["D1", "D2", "D3", "D4"],
                   help="Domain to evaluate on")
    p.add_argument("--scale", default=20, type=int, choices=[5,10,15,20,25,30,35])
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--lambda-crystal", default=1.0, type=float,
                   help="Loss weight for crystal system head")
    p.add_argument("--lambda-sg", default=1.0, type=float,
                   help="Loss weight for space group head")
    p.add_argument("--loss", default="weighted_ce",
                   choices=["ce", "weighted_ce", "focal"],
                   help="Loss function for both heads")
    p.add_argument("--focal-gamma", default=2.0, type=float)
    p.add_argument("--normalization", default="max_intensity",
                   choices=["none", "minmax", "standard", "max_intensity"])
    p.add_argument("--target-length", default=4500, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--run-name", default=None,
                   help="Override auto-generated run name")
    p.add_argument("--patience", default=5, type=int,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--source", default=None, choices=["zip", "materialized"])
    p.add_argument("--output-dir", default=None,
                   help="Override output directory")

    args = p.parse_args()

    # Resolve domain flags
    if args.domains is not None:
        args.train_domains = [d.strip() for d in args.domains.split(",")]
    elif args.domain is not None:
        args.train_domains = [args.domain]
    else:
        args.train_domains = ["D1"]

    if args.test_domain is None:
        args.test_domain = args.train_domains[-1]

    if args.domain is None:
        args.domain = args.train_domains[0]

    return args


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        ce = torch.nn.functional.nll_loss(
            log_probs, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: MultitaskModel,
    loader,
    criterion_crystal: nn.Module,
    criterion_sg: nn.Module,
    optimizer,
    device: torch.device,
    lambda_crystal: float,
    lambda_sg: float,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    crystal_correct = 0
    sg_correct = 0
    total = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        # For multitask we need both labels; assume crystal_system split provides y as crystal label
        # This is a simplification; real implementation would need dual-label batches
        # For now we skip this implementation detail
        raise NotImplementedError(
            "Multitask training requires dual-label batches. "
            "This is a placeholder implementation."
        )

    return total_loss / total if total > 0 else 0.0, \
           crystal_correct / total if total > 0 else 0.0, \
           sg_correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_multitask(
    model: MultitaskModel,
    loader,
    criterion_crystal: nn.Module,
    criterion_sg: nn.Module,
    device: torch.device,
    lambda_crystal: float,
    lambda_sg: float,
    class_names_crystal: Dict[int, str],
    class_names_sg: Dict[int, str],
    prefix: str = "val_",
) -> Tuple[float, Dict, Dict]:
    """Evaluate multitask model. Returns (total_loss, crystal_metrics, sg_metrics)."""
    model.eval()
    total_loss = 0.0
    total = 0
    crystal_preds: List[int] = []
    crystal_labels: List[int] = []
    crystal_probs: List[List[float]] = []
    sg_preds: List[int] = []
    sg_labels: List[int] = []
    sg_probs: List[List[float]] = []

    # Placeholder: real implementation needs dual-label loader
    raise NotImplementedError(
        "Multitask evaluation requires dual-label batches. "
        "This is a placeholder implementation."
    )

    # Would compute metrics for both heads
    crystal_metrics = compute_metrics(
        crystal_labels, crystal_preds, class_names_crystal,
        prefix=f"{prefix}crystal_", y_probs=crystal_probs,
    )
    sg_metrics = compute_metrics(
        sg_labels, sg_preds, class_names_sg,
        prefix=f"{prefix}sg_", y_probs=sg_probs, topk=[3, 5],
    )

    # Crystallographic consistency
    consistency = compute_crystallographic_consistency(crystal_preds, sg_preds)
    crystal_metrics[f"{prefix}crystallographic_consistency"] = consistency

    return total_loss / total if total > 0 else 0.0, crystal_metrics, sg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    logger.error(
        "\n" + "=" * 70 + "\n"
        "MULTITASK TRAINING NOT YET FULLY IMPLEMENTED\n"
        "\n"
        "Missing implementation:\n"
        "  1. Dual-label DataLoader (crystal_system + space_group simultaneously)\n"
        "  2. Batch format with both y_crystal and y_sg\n"
        "  3. Training loop with dual loss computation\n"
        "  4. Evaluation loop with dual metrics\n"
        "\n"
        "Current status: Architecture ready, training logic placeholder.\n"
        "\n"
        "To implement:\n"
        "  - Create MultitaskXRDDataset that loads both label types\n"
        "  - Update collate_fn to return y_crystal and y_sg\n"
        "  - Complete train_one_epoch and evaluate_multitask\n"
        "\n"
        "For now, use train_baseline.py for single-task experiments.\n"
        + "=" * 70
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
