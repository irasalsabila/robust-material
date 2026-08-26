"""
src/train_multitask.py
======================

Multitask training for XRD classification.

Shared backbone:
    resnet1d or cnn_attention

Heads:
    crystal system head
    space group head

Usage
-----
python src/train_multitask.py --model resnet1d --domain D1 --scale 20 --epochs 100

python src/train_multitask.py --model cnn_attention --domain D1 --scale 25 --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.datamodule import XRDDataModule
from src.data.multi_domain_datamodule import MultiDomainDataModule
from src.models import MultitaskModel
from src.utils.metrics import (
    compute_metrics,
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


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------

class MultitaskDataset(Dataset):
    """
    Combines two datasets with the same XRD samples but different labels.

    crystal_dataset returns:
        {"x": ..., "y": crystal_label}

    sg_dataset returns:
        {"x": ..., "y": space_group_label}

    Output:
        {
            "x": ...,
            "y_crystal": ...,
            "y_sg": ...
        }
    """

    def __init__(self, crystal_dataset: Dataset, sg_dataset: Dataset) -> None:
        if len(crystal_dataset) != len(sg_dataset):
            raise ValueError(
                f"Dataset length mismatch: "
                f"crystal={len(crystal_dataset)}, sg={len(sg_dataset)}"
            )

        self.crystal_dataset = crystal_dataset
        self.sg_dataset = sg_dataset

    def __len__(self) -> int:
        return len(self.crystal_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        crystal_item = self.crystal_dataset[idx]
        sg_item = self.sg_dataset[idx]

        return {
            "x": crystal_item["x"],
            "y_crystal": crystal_item["y"],
            "y_sg": sg_item["y"],
        }


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        ce = torch.nn.functional.nll_loss(
            log_probs,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multitask XRD Training")

    p.add_argument(
        "--model",
        default="resnet1d",
        choices=["resnet1d", "cnn_attention"],
        help="Shared backbone architecture",
    )

    p.add_argument("--domain", default=None, choices=["D1", "D2", "D3", "D4"])
    p.add_argument("--domains", default=None)
    p.add_argument("--test-domain", default=None, choices=["D1", "D2", "D3", "D4"])

    p.add_argument("--scale", default=20, type=int, choices=[5, 10, 15, 20, 25, 30, 35, 50])
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)

    p.add_argument("--lambda-crystal", default=1.0, type=float)
    p.add_argument("--lambda-sg", default=1.0, type=float)

    p.add_argument(
        "--loss",
        default="weighted_ce",
        choices=["ce", "weighted_ce", "focal"],
    )

    p.add_argument("--focal-gamma", default=2.0, type=float)

    p.add_argument(
        "--normalization",
        default="max_intensity",
        choices=["none", "minmax", "standard", "max_intensity"],
    )

    p.add_argument("--target-length", default=4500, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--patience", default=10, type=int)
    p.add_argument("--source", default=None, choices=["zip", "materialized"])
    p.add_argument("--run-name", default=None)

    p.add_argument("--imbalance-mode", default="auto",
        choices=["auto", "natural", "weighted_loss", "oversample", "oversample_weighted_loss", "focal",],)

    args = p.parse_args()

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

    if args.imbalance_mode == "auto":
        args.crystal_imbalance_mode = "oversample_weighted_loss"
        args.sg_imbalance_mode = "natural"
    else:
        args.crystal_imbalance_mode = args.imbalance_mode
        args.sg_imbalance_mode = args.imbalance_mode

    return args


# ---------------------------------------------------------------------
# Datamodule builders
# ---------------------------------------------------------------------

def build_datamodule(args: argparse.Namespace, task: str):
    manifest_path = ROOT / "outputs" / "data_audit" / "manifest.json"
    splits_dir = ROOT / "outputs" / "dataset_splits" / "splits"

    is_multi = len(args.train_domains) > 1 or (
        len(args.train_domains) == 1
        and args.train_domains[0] != args.test_domain
    )

    if is_multi:
        dm = MultiDomainDataModule(
            manifest_path=manifest_path,
            splits_dir=splits_dir,
            train_domains=args.train_domains,
            test_domain=args.test_domain,
            scale=args.scale,
            task=task,
            data_dir=ROOT / "data",
            normalization=args.normalization,
            target_length=args.target_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_map_dir=ROOT / "outputs" / "label_mappings",
            source=args.source,
            materialized_base_dir=ROOT / "outputs" / "materialized",
        )
    else:
        split_json = splits_dir / f"split_{args.domain}_{args.scale}pct.json"

        dm = XRDDataModule(
            manifest_path=manifest_path,
            split_json_path=split_json,
            task=task,
            data_dir=ROOT / "data",
            normalization=args.normalization,
            target_length=args.target_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_map_dir=ROOT / "outputs" / "label_mappings",
            source=args.source,
            materialized_base_dir=ROOT / "outputs" / "materialized",
            use_weighted_sampler=(
                task == "crystal_system"
                and args.crystal_imbalance_mode in ["oversample", "oversample_weighted_loss"]
            ) or (
                task == "top10_space_group"
                and args.sg_imbalance_mode in ["oversample", "oversample_weighted_loss"]
            ),
        )

    dm.setup()
    return dm


def build_multitask_loaders(args, crystal_dm, sg_dm):
    crystal_train_loader = DataLoader(
        crystal_dm.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sg_train_loader = DataLoader(
        sg_dm.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    crystal_val_loader = DataLoader(
        crystal_dm.val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sg_val_loader = DataLoader(
        sg_dm.val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    crystal_test_loader = DataLoader(
        crystal_dm.test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sg_test_loader = DataLoader(
        sg_dm.test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        crystal_train_loader,
        sg_train_loader,
        crystal_val_loader,
        sg_val_loader,
        crystal_test_loader,
        sg_test_loader,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_logits(outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    crystal_logits = outputs["crystal_logits"]

    if "space_group_logits" in outputs:
        sg_logits = outputs["space_group_logits"]
    elif "sg_logits" in outputs:
        sg_logits = outputs["sg_logits"]
    else:
        raise KeyError("Model output must contain 'space_group_logits' or 'sg_logits'.")

    return crystal_logits, sg_logits


def build_criterion(args, crystal_dm, sg_dm, device):
    use_crystal_weights = args.crystal_imbalance_mode in [
        "weighted_loss",
        "oversample_weighted_loss",
    ]
    use_sg_weights = args.sg_imbalance_mode in [
        "weighted_loss",
        "oversample_weighted_loss",
    ]

    crystal_weight = crystal_dm.class_weights().to(device) if use_crystal_weights else None
    sg_weight = sg_dm.class_weights().to(device) if use_sg_weights else None

    if args.crystal_imbalance_mode == "focal":
        criterion_crystal = FocalLoss(
            gamma=args.focal_gamma,
            weight=crystal_weight,
        )
    else:
        criterion_crystal = nn.CrossEntropyLoss(weight=crystal_weight)

    if args.sg_imbalance_mode == "focal":
        criterion_sg = FocalLoss(
            gamma=args.focal_gamma,
            weight=sg_weight,
        )
    else:
        criterion_sg = nn.CrossEntropyLoss(weight=sg_weight)

    logger.info("Crystal imbalance mode: %s", args.crystal_imbalance_mode)
    logger.info("SG imbalance mode: %s", args.sg_imbalance_mode)

    return criterion_crystal, criterion_sg




def save_predictions_and_confusion(
    labels,
    probs,
    class_names,
    predictions_path,
    confusion_path,
    topk=(1, 3, 5),
):
    """
    Save compact prediction and confusion analysis files.

    predictions_path:
        One row per test sample with true label, top-1/top-3/top-5 labels,
        probabilities, correctness flags, and confidence.

    confusion_path:
        Aggregated class-pair counts in one file. For each true class and
        candidate class, stores how often the candidate appears as top-1,
        inside top-3, and inside top-5.
    """
    prediction_rows = []
    confusion_counts = {}
    true_counts = {}

    # Make class_names robust whether keys are int or str.
    def cname(idx):
        return class_names[idx] if idx in class_names else class_names[str(idx)]

    for i, (y_true, prob) in enumerate(zip(labels, probs)):
        prob_tensor = torch.tensor(prob, dtype=torch.float32)
        max_k = min(max(topk), len(prob))

        values, indices = torch.topk(prob_tensor, k=max_k)
        top_ids = [int(x) for x in indices.tolist()]
        top_probs = [float(x) for x in values.tolist()]

        y_true = int(y_true)
        y_top1 = top_ids[0]

        true_name = cname(y_true)
        top1_name = cname(y_top1)

        true_counts[true_name] = true_counts.get(true_name, 0) + 1

        row = {
            "sample_index": i,
            "true_label_id": y_true,
            "true_label": true_name,
            "top1_label_id": y_top1,
            "top1_label": top1_name,
            "top1_prob": top_probs[0],
            "correct": y_true == y_top1,
            "confidence": top_probs[0],
        }

        for k_original in topk:
            k = min(k_original, len(prob))
            k_ids = top_ids[:k]
            k_probs = top_probs[:k]
            k_names = [cname(j) for j in k_ids]

            row[f"top{k_original}_label_ids"] = json.dumps(k_ids)
            row[f"top{k_original}_labels"] = json.dumps(k_names)
            row[f"top{k_original}_probs"] = json.dumps(k_probs)
            row[f"top{k_original}_correct"] = y_true in k_ids

            for pred_id in k_ids:
                pred_name = cname(pred_id)
                key = (true_name, pred_name)

                if key not in confusion_counts:
                    confusion_counts[key] = {
                        "top1_count": 0,
                        "top3_count": 0,
                        "top5_count": 0,
                    }

                count_key = f"top{k_original}_count"
                if count_key in confusion_counts[key]:
                    confusion_counts[key][count_key] += 1

        prediction_rows.append(row)

    if prediction_rows:
        with open(predictions_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prediction_rows)

    confusion_rows = []

    for (true_name, pred_name), counts in confusion_counts.items():
        total = true_counts[true_name]

        confusion_rows.append({
            "true_class": true_name,
            "candidate_class": pred_name,
            "true_count": total,
            "top1_count": counts["top1_count"],
            "top3_count": counts["top3_count"],
            "top5_count": counts["top5_count"],
            "top1_rate": counts["top1_count"] / total if total else 0.0,
            "top3_rate": counts["top3_count"] / total if total else 0.0,
            "top5_rate": counts["top5_count"] / total if total else 0.0,
            "is_correct_class": true_name == pred_name,
        })

    confusion_rows.sort(
        key=lambda r: (
            r["true_class"],
            not r["is_correct_class"],
            -r["top1_count"],
            -r["top3_count"],
            -r["top5_count"],
            r["candidate_class"],
        )
    )

    if confusion_rows:
        with open(confusion_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(confusion_rows[0].keys()))
            writer.writeheader()
            writer.writerows(confusion_rows)


# ---------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    crystal_loader,
    sg_loader,
    criterion_crystal,
    criterion_sg,
    optimizer,
    device,
    lambda_crystal: float,
    lambda_sg: float,
):
    model.train()

    total_loss = 0.0

    crystal_correct = 0
    sg_correct = 0

    crystal_total = 0
    sg_total = 0

    sg_iter = iter(sg_loader)
    num_batches = len(crystal_loader)

    for step, crystal_batch in enumerate(crystal_loader, start=1):
        try:
            sg_batch = next(sg_iter)
        except StopIteration:
            sg_iter = iter(sg_loader)
            sg_batch = next(sg_iter)

        # ---------------- Crystal ----------------
        x_crystal = crystal_batch["x"].to(device, non_blocking=True)
        y_crystal = crystal_batch["y"].to(device, non_blocking=True)

        # ---------------- Space Group ----------------
        x_sg = sg_batch["x"].to(device, non_blocking=True)
        y_sg = sg_batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Crystal forward
        outputs = model(x_crystal)
        crystal_logits, _ = get_logits(outputs)
        loss_crystal = criterion_crystal(crystal_logits, y_crystal)

        # Space-group forward
        outputs = model(x_sg)
        _, sg_logits = get_logits(outputs)
        loss_sg = criterion_sg(sg_logits, y_sg)

        # Combined loss
        loss = (
            lambda_crystal * loss_crystal
            + lambda_sg * loss_sg
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x_crystal.size(0)

        crystal_correct += (
            crystal_logits.argmax(1) == y_crystal
        ).sum().item()

        sg_correct += (
            sg_logits.argmax(1) == y_sg
        ).sum().item()

        crystal_total += y_crystal.size(0)
        sg_total += y_sg.size(0)

        # Progress every 100 batches
        if step % 100 == 0 or step == num_batches:
            print(
                f"  Batch {step:4d}/{num_batches} | "
                f"Loss={total_loss / crystal_total:.4f} | "
                f"CrAcc={crystal_correct / crystal_total:.4f} | "
                f"SGAcc={sg_correct / sg_total:.4f}",
                flush=True,
            )

    return {
        "loss": total_loss / crystal_total,
        "crystal_acc": crystal_correct / crystal_total,
        "sg_acc": sg_correct / sg_total,
    }
    
@torch.no_grad()
def evaluate(
    model,
    crystal_loader,
    sg_loader,
    criterion_crystal,
    criterion_sg,
    device,
    lambda_crystal: float,
    lambda_sg: float,
    crystal_class_names,
    sg_class_names,
    prefix: str,
):
    model.eval()

    total_loss = 0.0
    total = 0

    crystal_labels = []
    crystal_preds = []
    crystal_probs = []

    sg_labels = []
    sg_preds = []
    sg_probs = []

    for batch in crystal_loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        outputs = model(x)
        crystal_logits, _ = get_logits(outputs)

        loss = criterion_crystal(crystal_logits, y)
        total_loss += lambda_crystal * loss.item() * x.size(0)
        total += x.size(0)

        probs = torch.softmax(crystal_logits, dim=1)
        crystal_labels.extend(y.cpu().tolist())
        crystal_preds.extend(crystal_logits.argmax(1).cpu().tolist())
        crystal_probs.extend(probs.cpu().tolist())

    for batch in sg_loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        outputs = model(x)
        _, sg_logits = get_logits(outputs)

        loss = criterion_sg(sg_logits, y)
        total_loss += lambda_sg * loss.item() * x.size(0)
        total += x.size(0)

        probs = torch.softmax(sg_logits, dim=1)
        sg_labels.extend(y.cpu().tolist())
        sg_preds.extend(sg_logits.argmax(1).cpu().tolist())
        sg_probs.extend(probs.cpu().tolist())

    crystal_metrics = compute_metrics(
        crystal_labels,
        crystal_preds,
        crystal_class_names,
        prefix=f"{prefix}crystal_",
        y_probs=crystal_probs,
        topk=[3, 5],
    )

    sg_metrics = compute_metrics(
        sg_labels,
        sg_preds,
        sg_class_names,
        prefix=f"{prefix}sg_",
        y_probs=sg_probs,
        topk=[3, 5],
    )

    metrics = {
        f"{prefix}loss": total_loss / total,
        **crystal_metrics,
        **sg_metrics,
    }

    return metrics, {
        "crystal_labels": crystal_labels,
        "crystal_preds": crystal_preds,
        "crystal_probs": crystal_probs,
        "sg_labels": sg_labels,
        "sg_preds": sg_preds,
        "sg_probs": sg_probs,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_tag = "+".join(args.train_domains)
    test_tag = args.test_domain

    if args.run_name:
        run_name = args.run_name
    elif len(args.train_domains) == 1 and args.train_domains[0] == args.test_domain:
        run_name = f"multitask_{args.model}_{args.domain}_{args.scale}pct"
    else:
        run_name = f"multitask_{args.model}_train{train_tag}_test{test_tag}_{args.scale}pct"

    out_dir = ROOT / "outputs" / "single_vs_multitask_new" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run: %s", run_name)
    logger.info("Device: %s", device)

    crystal_dm = build_datamodule(args, task="crystal_system")
    sg_dm = build_datamodule(args, task="top10_space_group")

    (
        crystal_train_loader, sg_train_loader,
        crystal_val_loader, sg_val_loader,
        crystal_test_loader, sg_test_loader,
    ) = build_multitask_loaders(args, crystal_dm, sg_dm)

    model = MultitaskModel(
        backbone=args.model,
        input_length=args.target_length,
        num_crystal_classes=crystal_dm.num_classes,
        num_sg_classes=sg_dm.num_classes,
    ).to(device)

    logger.info("Model: %s", args.model)
    logger.info("Params: %s", f"{model.count_parameters():,}")
    logger.info("Crystal classes: %d", crystal_dm.num_classes)
    logger.info("Space group classes: %d", sg_dm.num_classes)

    criterion_crystal, criterion_sg = build_criterion(
        args,
        crystal_dm,
        sg_dm,
        device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    config = {
        "run_name": run_name,
        "model": args.model,
        "train_domains": args.train_domains,
        "test_domain": args.test_domain,
        "scale": args.scale,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_crystal": args.lambda_crystal,
        "lambda_sg": args.lambda_sg,
        "loss": args.loss,
        "normalization": args.normalization,
        "target_length": args.target_length,
        "seed": args.seed,
        "num_params": model.count_parameters(),
        "num_crystal_classes": crystal_dm.num_classes,
        "num_sg_classes": sg_dm.num_classes,
        "crystal_class_names": {str(k): v for k, v in crystal_dm.class_names.items()},
        "sg_class_names": {str(k): v for k, v in sg_dm.class_names.items()},
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_val_score = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None
    log_rows: List[Dict] = []

    print("\n" + "=" * 72)
    print(
        f"Training {args.model.upper()} MULTITASK | "
        f"train={train_tag} -> test={test_tag} | "
        f"{args.scale}%"
    )
    print(
        f"Crystal={args.crystal_imbalance_mode} | "
        f"SG={args.sg_imbalance_mode}"
    )
    print(
        f"Device={device} | "
        f"Params={model.count_parameters():,}"
    )
    print("=" * 72)

    header = (
        f"{'Ep':>3}  "
        f"{'Loss':>8}  "
        f"{'CrAcc':>7}  "
        f"{'SGAcc':>7}  "
        f"{'CrF1':>7}  "
        f"{'SGF1':>7}  "
        f"{'Score':>7}  "
        f"{'LR':>8}  "
        f"{'Time':>6}"
    )

    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            model,
            crystal_train_loader,
            sg_train_loader,
            criterion_crystal,
            criterion_sg,
            optimizer,
            device,
            args.lambda_crystal,
            args.lambda_sg,
        )

        val_metrics, _ = evaluate(
            model,
            crystal_val_loader,
            sg_val_loader,
            criterion_crystal,
            criterion_sg,
            device,
            args.lambda_crystal,
            args.lambda_sg,
            crystal_dm.class_names,
            sg_dm.class_names,
            prefix="val_",
        )

        scheduler.step()
        elapsed = time.time() - t0

        val_crystal_f1 = val_metrics["val_crystal_macro_f1"]
        val_sg_f1 = val_metrics["val_sg_macro_f1"]

        val_score = 0.5 * val_crystal_f1 + 0.5 * val_sg_f1
        lr = scheduler.get_last_lr()[0]

        print(
                f"{epoch:>3}  "
                f"{train_stats['loss']:>8.4f}  "
                f"{train_stats['crystal_acc']:>7.4f}  "
                f"{train_stats['sg_acc']:>7.4f}  "
                f"{val_crystal_f1:>7.4f}  "
                f"{val_sg_f1:>7.4f}  "
                f"{val_score:>7.4f}  "
                f"{lr:>8.6f}  "
                f"{elapsed:>5.1f}s",
                flush=True,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_crystal_acc": train_stats["crystal_acc"],
            "train_sg_acc": train_stats["sg_acc"],
            "val_loss": val_metrics["val_loss"],
            "val_crystal_macro_f1": val_crystal_f1,
            "val_sg_macro_f1": val_sg_f1,
            "val_score": val_score,
            "lr": lr,
            "epoch_time_s": elapsed,
        }

        log_rows.append(row)

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0

            # Keep best weights in memory instead of saving to disk
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    with open(out_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    val_metrics, _ = evaluate(
        model,
        crystal_val_loader,
        sg_val_loader,
        criterion_crystal,
        criterion_sg,
        device,
        args.lambda_crystal,
        args.lambda_sg,
        crystal_dm.class_names,
        sg_dm.class_names,
        prefix="val_",
    )

    test_metrics, test_raw = evaluate(
        model,
        crystal_test_loader,
        sg_test_loader,
        criterion_crystal,
        criterion_sg,
        device,
        args.lambda_crystal,
        args.lambda_sg,
        crystal_dm.class_names,
        sg_dm.class_names,
        prefix="test_",
    )

    save_metrics(val_metrics, out_dir / "val_metrics.json")
    save_metrics(test_metrics, out_dir / "test_metrics.json")

    save_predictions_and_confusion(
        test_raw["crystal_labels"],
        test_raw["crystal_probs"],
        crystal_dm.class_names,
        out_dir / "crystal_predictions.csv",
        out_dir / "crystal_confusion.csv",
        topk=(1, 3, 5),
    )

    save_predictions_and_confusion(
        test_raw["sg_labels"],
        test_raw["sg_probs"],
        sg_dm.class_names,
        out_dir / "sg_predictions.csv",
        out_dir / "sg_confusion.csv",
        topk=(1, 3, 5),
    )

    save_classification_report_csv(
        test_raw["crystal_labels"],
        test_raw["crystal_preds"],
        crystal_dm.class_names,
        out_dir / "crystal_classification_report.csv",
    )

    save_classification_report_csv(
        test_raw["sg_labels"],
        test_raw["sg_preds"],
        sg_dm.class_names,
        out_dir / "sg_classification_report.csv",
    )

    plot_confusion_matrix(
        test_raw["crystal_labels"],
        test_raw["crystal_preds"],
        crystal_dm.class_names,
        out_dir / "crystal_confusion_matrix.png",
        title=f"{args.model.upper()} Multitask Crystal System",
    )

    plot_confusion_matrix(
        test_raw["sg_labels"],
        test_raw["sg_preds"],
        sg_dm.class_names,
        out_dir / "sg_confusion_matrix.png",
        title=f"{args.model.upper()} Multitask Space Group",
    )

    print("\n" + "=" * 80)
    print("MULTITASK TRAINING COMPLETE")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val score: {best_val_score:.4f}")
    print(f"Test crystal macro F1: {test_metrics['test_crystal_macro_f1']:.4f}")
    print(f"Test SG macro F1: {test_metrics['test_sg_macro_f1']:.4f}")
    print(f"Output dir: {out_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()