"""
src/evaluate.py

Standalone evaluation script. Loads a saved checkpoint and runs
inference on val or test split, printing and saving metrics.

Usage
-----
    python src/evaluate.py \
        --checkpoint outputs/baselines/cnn1d_D1_10pct_crystal_system/best_model.pt \
        --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.datamodule import XRDDataModule
from src.models import build_model
from src.utils.metrics import (
    compute_metrics,
    save_metrics,
    save_classification_report_csv,
    plot_confusion_matrix,
)
from src.utils.seed import set_seed


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved XRD baseline checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"]
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
        all_probs.extend(probs.cpu().tolist())
    return all_labels, all_preds, all_probs


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = get_device()

    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent

    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found in {run_dir}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)

    dm = XRDDataModule(
        manifest_path=ROOT / cfg["manifest_path"],
        split_json_path=ROOT / cfg["split_json_path"],
        task=cfg["task"],
        data_dir=ROOT / "data",
        normalization=cfg["normalization"],
        target_length=cfg["target_length"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    model = build_model(
        cfg["model"],
        input_length=cfg["target_length"],
        num_classes=dm.num_classes,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val macro_f1={ckpt.get('val_macro_f1', 0.0):.4f})")

    labels, preds, probs = run_inference(model, loader, device)
    topk = [3, 5] if cfg["task"] == "top10_space_group" else None
    metrics = compute_metrics(
        labels,
        preds,
        dm.class_names,
        prefix=f"{args.split}_",
        y_probs=probs,
        topk=topk,
    )

    print(f"{args.split.upper()} METRICS")
    for k, v in metrics.items():
        if k != "per_class":
            print(f"  {k}: {v}")
    print("  per_class:")
    for cls, vals in metrics["per_class"].items():
        print(f"    {cls}: {vals}")

    out_path = run_dir / f"{args.split}_metrics.json"
    save_metrics(metrics, out_path)
    print(f"Saved: {out_path}")

    save_classification_report_csv(
        labels, preds, dm.class_names,
        run_dir / f"{args.split}_classification_report.csv",
    )
    plot_confusion_matrix(
        labels, preds, dm.class_names,
        run_dir / f"{args.split}_confusion_matrix.png",
        title=f"{cfg['model'].upper()} {args.split} Confusion Matrix",
    )
    print("Saved confusion matrix and classification report.")


if __name__ == "__main__":
    main()
