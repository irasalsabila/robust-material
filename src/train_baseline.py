"""
src/train_baseline.py

Phase 3 baseline training for XRD crystal classification.
Supports single-domain and multi-domain (cross-domain) training.

Usage
-----
    # Single domain
    python src/train_baseline.py --model cnn1d --domain D1 --scale 10 --task crystal_system --epochs 10

    # Multi-domain training, tested on D3
    python src/train_baseline.py --model cnn1d --domains D1,D2 --test-domain D3 \
        --scale 10 --task crystal_system --epochs 10

Outputs  (outputs/baselines/{model}/{space|crystal}/{scale}pct/{run_name}/)
-------
    config.json
    train_log.csv
    val_metrics.json
    test_metrics.json
    classification_report.csv
    confusion_matrix.png
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
from src.models import build_model
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
    p = argparse.ArgumentParser(description="Phase 3 XRD Baseline Training")
    p.add_argument("--model",    default="cnn1d", choices=["mlp", "cnn1d", "resnet1d", "cnn_attention"],
                   help="Model architecture")
    # Single-domain shorthand (kept for backward compat)
    p.add_argument("--domain",   default=None,    choices=["D1","D2","D3","D4"],
                   help="Single training/test domain (legacy; use --domains + --test-domain "
                        "for cross-domain experiments)")
    # Multi-domain flags
    p.add_argument("--domains",  default=None,
                   help="Comma-separated training domains, e.g. 'D1,D2,D3'. "
                        "Overrides --domain for training.")
    p.add_argument("--test-domain", default=None, choices=["D1","D2","D3","D4"],
                   help="Domain to evaluate on. Defaults to the single --domain "
                        "when --domains is not set.")
    p.add_argument("--scale",    default=10,       type=int, choices=[5,10,15,20,25,30,35, 50])
    p.add_argument("--task",     default="crystal_system",
                   choices=["crystal_system","top10_space_group"])
    p.add_argument("--epochs",   default=10,       type=int)
    p.add_argument("--batch-size",    default=64,  type=int)
    p.add_argument("--lr",            default=1e-3, type=float)
    p.add_argument("--weight-decay",  default=1e-4, type=float)
    p.add_argument("--normalization", default="max_intensity",
                   choices=["none","minmax","standard","max_intensity"])
    p.add_argument("--target-length", default=4500, type=int)
    p.add_argument("--num-workers",   default=4,    type=int,
                   help="DataLoader worker processes (0 = load in main process)")
    p.add_argument("--pin-memory", dest="pin_memory", action=argparse.BooleanOptionalAction,
                   default=None, help="Pin CUDA memory for faster H2D copy (default: auto=on CUDA)")
    p.add_argument("--amp", dest="amp", action=argparse.BooleanOptionalAction,
                   default=None, help="Enable mixed precision on CUDA (default: auto=on CUDA)")
    p.add_argument("--seed",          default=42,   type=int)
    p.add_argument("--run-name",      default=None,
                   help="Override auto-generated run name")
    p.add_argument("--patience",      default=5,    type=int,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--imbalance-mode", default="auto",
                   choices=["auto", "natural", "weighted_loss", "oversample",
                            "oversample_weighted_loss", "focal"],
                   help="Imbalance handling strategy (auto = task-dependent default)")
    p.add_argument("--focal-gamma", default=2.0, type=float,
                   help="Focal loss gamma parameter (default 2.0)")
    p.add_argument("--source", default=None, choices=["zip", "materialized"],
                   help="Data source: 'zip', 'materialized', or auto-detect (default)")
    # Legacy flags for backward compatibility
    p.add_argument("--no-class-weights", action="store_true",
                   help="(Deprecated) Use --imbalance-mode natural instead")
    p.add_argument("--loss", default=None, choices=["ce", "focal"],
                   help="(Deprecated) Use --imbalance-mode instead")
    args = p.parse_args()

    # Resolve domain flags
    if args.domains is not None:
        # --domains takes precedence
        args.train_domains = [d.strip() for d in args.domains.split(",")]
    elif args.domain is not None:
        args.train_domains = [args.domain]
    else:
        args.train_domains = ["D1"]

    if args.test_domain is None:
        # Default test domain = last training domain (or the only one)
        args.test_domain = args.train_domains[-1]

    # Backwards-compat: set args.domain to the first train domain for run naming
    if args.domain is None:
        args.domain = args.train_domains[0]

    # Handle legacy flags
    if args.loss is not None or args.no_class_weights:
        logger.warning("--loss and --no-class-weights are deprecated; use --imbalance-mode instead")
        if args.loss == "focal":
            args.imbalance_mode = "focal"
        elif args.no_class_weights:
            args.imbalance_mode = "natural"

    # Resolve auto mode based on task
    if args.imbalance_mode == "auto":
        if args.task == "crystal_system":
            # ~30x imbalance: use oversample + weighted loss by default
            args.imbalance_mode = "oversample_weighted_loss"
        else:
            # ~2x imbalance: natural training is sufficient
            args.imbalance_mode = "natural"
        logger.info("Auto-selected imbalance_mode=%s for task=%s",
                    args.imbalance_mode, args.task)

    return args


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler=None,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(x)
            loss   = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[Dict[int, str]] = None,
    prefix: str = "val_",
    task: str = "",
) -> Tuple[float, Dict]:
    """Run inference on a loader. Returns (avg_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    total      = 0
    all_preds:  List[int] = []
    all_labels: List[int] = []
    all_probs:  List[List[float]] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        logits = model(x)
        loss   = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds       = logits.argmax(dim=1)
        probs       = torch.softmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        total += x.size(0)

    avg_loss = total_loss / total
    topk = [3, 5] if task == "top10_space_group" else None
    metrics  = compute_metrics(
        all_labels,
        all_preds,
        class_names,
        prefix=prefix,
        y_probs=all_probs,
        topk=topk,
    )
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class focal loss  CE(p_t) * (1 - p_t)^gamma."""

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # class weights tensor or None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        ce = torch.nn.functional.nll_loss(
            log_probs, targets,
            weight=self.weight, reduction="none",
        )
        # pt = exp(-ce) which equals the probability of the true class
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = get_device()

    use_amp = args.amp if args.amp is not None else device.type == "cuda"
    pin_mem = args.pin_memory if args.pin_memory is not None else device.type == "cuda"
    if use_amp:
        logger.info("Mixed precision (AMP) enabled on %s", device)

    # ── Run name & output dir ─────────────────────────────────────────────
    train_tag = "+".join(args.train_domains)
    test_tag  = args.test_domain
    if args.run_name:
        run_name = args.run_name
    elif len(args.train_domains) == 1 and args.train_domains[0] == args.test_domain:
        # Original single-domain naming
        run_name = f"{args.model}_{args.domain}_{args.scale}pct_{args.task}"
    else:
        run_name = (
            f"{args.model}_train{train_tag}_test{test_tag}"
            f"_{args.scale}pct_{args.task}"
        )
    # Structured layout: outputs/baselines/{model}/{space|crystal}/{scale}pct/{run_name}
    kind = "space" if args.task == "top10_space_group" else "crystal"
    out_dir = ROOT / "outputs" / "baselines" / args.model / kind / f"{args.scale}pct" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run: %s", run_name)
    logger.info("Device: %s", device)
    logger.info("Output dir: %s", out_dir)
    logger.info("Train domains: %s", args.train_domains)
    logger.info("Test domain:   %s", args.test_domain)

    # ── DataModule ────────────────────────────────────────────────────────
    manifest_path = ROOT / "outputs" / "data_audit" / "manifest.json"
    splits_dir    = ROOT / "outputs" / "dataset_splits" / "splits"

    is_multi = len(args.train_domains) > 1 or (
        len(args.train_domains) == 1
        and args.train_domains[0] != args.test_domain
    )

    # Decide weighted sampler based on imbalance_mode
    use_weighted_sampler = args.imbalance_mode in ["oversample", "oversample_weighted_loss"]

    if is_multi:
        dm = MultiDomainDataModule(
            manifest_path        = manifest_path,
            splits_dir           = splits_dir,
            train_domains        = args.train_domains,
            test_domain          = args.test_domain,
            scale                = args.scale,
            task                 = args.task,
            data_dir             = ROOT / "data",
            normalization        = args.normalization,
            target_length        = args.target_length,
            batch_size           = args.batch_size,
            num_workers          = args.num_workers,
            pin_memory           = pin_mem,
            label_map_dir        = ROOT / "outputs" / "label_mappings",
            source               = args.source,
            materialized_base_dir= ROOT / "outputs" / "materialized",
        )
    else:
        split_json = splits_dir / f"split_{args.domain}_{args.scale}pct.json"
        dm = XRDDataModule(
            manifest_path        = manifest_path,
            split_json_path      = split_json,
            task                 = args.task,
            data_dir             = ROOT / "data",
            normalization        = args.normalization,
            target_length        = args.target_length,
            batch_size           = args.batch_size,
            num_workers          = args.num_workers,
            pin_memory           = pin_mem,
            label_map_dir        = ROOT / "outputs" / "label_mappings",
            source               = args.source,
            materialized_base_dir= ROOT / "outputs" / "materialized",
            use_weighted_sampler = use_weighted_sampler,
        )

    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    test_loader  = dm.test_dataloader()

    logger.info(
        "Dataset | train=%d val=%d test=%d classes=%d",
        len(dm.train_dataset), len(dm.val_dataset),
        len(dm.test_dataset), dm.num_classes,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        args.model,
        input_length = args.target_length,
        num_classes  = dm.num_classes,
    ).to(device)
    logger.info(
        "Model: %s | params=%s",
        args.model, f"{model.count_parameters():,}",
    )

    # ── Imbalance handling ───────────────────────────────────────────────
    # Determine loss and sampler based on imbalance_mode
    use_weighted_sampler = args.imbalance_mode in ["oversample", "oversample_weighted_loss"]
    use_class_weights = args.imbalance_mode in ["weighted_loss", "oversample_weighted_loss"]
    use_focal_loss = args.imbalance_mode == "focal"

    # Get class weights if needed
    cw = dm.class_weights().to(device) if use_class_weights else None
    if cw is not None:
        logger.info("Class weights: %s", [round(w, 4) for w in cw.cpu().tolist()])

    # Build loss function
    if use_focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=None)
        logger.info("Loss: FocalLoss(gamma=%.1f)", args.focal_gamma)
    elif use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=cw)
        logger.info("Loss: CrossEntropyLoss with class weights")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Loss: CrossEntropyLoss (no class weights)")

    # Log imbalance mode
    logger.info("Imbalance mode: %s", args.imbalance_mode)
    logger.info("  Sampler: %s", "WeightedRandomSampler" if use_weighted_sampler else "shuffle")
    logger.info("  Loss weights: %s", "enabled" if use_class_weights else "disabled")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Save config ───────────────────────────────────────────────────────
    config = {
        "run_name":        run_name,
        "model":           args.model,
        "domain":          args.domain,          # first / only train domain
        "train_domains":   args.train_domains,
        "test_domain":     args.test_domain,
        "scale_pct":       args.scale,
        "task":            args.task,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "normalization":   args.normalization,
        "target_length":   args.target_length,
        "seed":            args.seed,
        "device":          str(device),
        "num_params":      model.count_parameters(),
        "imbalance_mode":  args.imbalance_mode,
        "focal_gamma":     args.focal_gamma if use_focal_loss else None,
        "use_weighted_sampler": use_weighted_sampler,
        "use_class_weights": use_class_weights,
        "class_weights":   ([round(w, 6) for w in cw.cpu().tolist()] if cw is not None else []),
        "manifest_path":   str(manifest_path.relative_to(ROOT)),
        # For multi-domain: record per-domain split paths
        "splits_dir":      str(splits_dir.relative_to(ROOT)),
        "num_classes":     dm.num_classes,
        "class_names":     {str(k): v for k, v in dm.class_names.items()},
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_macro_f1 = -1.0
    best_epoch        = 0
    patience_counter  = 0
    log_rows: List[Dict] = []

    print("\n" + "=" * 72)
    print(f"  Training {args.model.upper()} | "
          f"train={'+'.join(args.train_domains)} -> test={args.test_domain} "
          f"{args.scale}% | {args.task}")
    print(f"  Device: {device} | Params: {model.count_parameters():,}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | Loss: {args.loss}")
    print("=" * 72)
    header = (f"{'Ep':>3}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
              f"{'ValLoss':>8}  {'ValAcc':>7}  {'MacroF1':>8}  "
              f"{'BalAcc':>7}  {'LR':>8}  {'Time':>6}")
    print(header)
    print("-" * len(header))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device,
            class_names=dm.class_names, prefix="val_",
            task=args.task,
        )
        scheduler.step()
        elapsed = time.time() - t0

        val_macro_f1 = val_metrics["val_macro_f1"]
        val_acc      = val_metrics["val_accuracy"]
        val_bacc     = val_metrics["val_balanced_accuracy"]
        current_lr   = scheduler.get_last_lr()[0]

        print(
            f"{epoch:>3}  {train_loss:>9.4f}  {train_acc:>8.4f}  "
            f"{val_loss:>8.4f}  {val_acc:>7.4f}  {val_macro_f1:>8.4f}  "
            f"{val_bacc:>7.4f}  {current_lr:>8.6f}  {elapsed:>5.1f}s"
        )

        log_rows.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "train_acc":    round(train_acc,  6),
            "val_loss":     round(val_loss,   6),
            "val_accuracy": round(val_acc,    6),
            "val_macro_f1": round(val_macro_f1, 6),
            "val_weighted_f1":       round(val_metrics["val_weighted_f1"], 6),
            "val_balanced_accuracy": round(val_bacc, 6),
            "lr":           round(current_lr, 8),
            "epoch_time_s": round(elapsed, 2),
            "imbalance_mode": args.imbalance_mode,
        })

        # ── Model selection ───────────────────────────────────────────────
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch        = epoch
            patience_counter  = 0
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1":    val_macro_f1,
                    "val_accuracy":    val_acc,
                    "config":          config,
                },
                out_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        # ── Early stopping ────────────────────────────────────────────────
        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch, args.patience,
            )
            break

    print("-" * len(header))
    print(f"  Best epoch: {best_epoch}  |  Best val macro F1: {best_val_macro_f1:.4f}")
    print()

    # ── Save train log ────────────────────────────────────────────────────
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    logger.info("Saved train log: %s", log_path)

    # ── Load best model for final evaluation ─────────────────────────────
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded best model from epoch %d", best_epoch)

    # ── Val metrics (best model) ──────────────────────────────────────────
    _, val_metrics_best = evaluate(
        model, val_loader, criterion, device,
        class_names=dm.class_names, prefix="val_",
        task=args.task,
    )
    save_metrics(val_metrics_best, out_dir / "val_metrics.json")

    print("VAL METRICS (best model):")
    for k, v in val_metrics_best.items():
        if k != "per_class":
            print(f"  {k}: {v}")

    # ── Test metrics ──────────────────────────────────────────────────────
    _, test_metrics = evaluate(
        model, test_loader, criterion, device,
        class_names=dm.class_names, prefix="test_",
        task=args.task,
    )
    save_metrics(test_metrics, out_dir / "test_metrics.json")

    print("\nTEST METRICS:")
    for k, v in test_metrics.items():
        if k != "per_class":
            print(f"  {k}: {v}")

    # ── Classification report CSV ─────────────────────────────────────────
    # Re-run inference to get raw predictions for CSV / confusion matrix
    @torch.no_grad()
    def get_preds(loader):
        model.eval()
        ys, ps = [], []
        for batch in loader:
            x = batch["x"].to(device)
            logits = model(x)
            ps.extend(logits.argmax(1).cpu().tolist())
            ys.extend(batch["y"].tolist())
        return ys, ps

    test_labels, test_preds = get_preds(test_loader)
    save_classification_report_csv(
        test_labels, test_preds, dm.class_names,
        out_dir / "classification_report.csv",
    )
    plot_confusion_matrix(
        test_labels, test_preds, dm.class_names,
        out_dir / "confusion_matrix.png",
        title=f"{args.model.upper()} Test Confusion Matrix — {args.task}",
    )

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print(f"  Run dir        : {out_dir}")
    print(f"  Train domains  : {'+'.join(args.train_domains)}")
    print(f"  Test domain    : {args.test_domain}")
    print(f"  Best epoch     : {best_epoch}")
    print(f"  Val macro F1   : {best_val_macro_f1:.4f}")
    print(f"  Test macro F1  : {test_metrics['test_macro_f1']:.4f}")
    print(f"  Test accuracy  : {test_metrics['test_accuracy']:.4f}")
    print(f"  Test ECE       : {test_metrics.get('test_ece', 'N/A')}")
    print(f"  Mean confidence: {test_metrics.get('test_mean_confidence', 'N/A')}")
    print(f"  Overconf gap   : {test_metrics.get('test_overconfidence_gap', 'N/A')}")
    print("  Files saved:")
    for fname in [
        "config.json", "train_log.csv", "val_metrics.json",
        "test_metrics.json", "classification_report.csv",
        "confusion_matrix.png", "best_model.pt",
    ]:
        exists = "OK" if (out_dir / fname).exists() else "MISSING"
        print(f"    [{exists}] {fname}")
    print("=" * 72 + "\n")

    # Delete checkpoint
    ckpt_path = out_dir / "best_model.pt"
    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    main()
