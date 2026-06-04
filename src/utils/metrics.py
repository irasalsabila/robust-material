"""
src/utils/metrics.py

Compute, save, and plot classification metrics.

Metrics computed
----------------
- accuracy
- macro F1
- weighted F1
- balanced accuracy
- per-class precision / recall / F1 / support
- confusion matrix (saved as PNG)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def _compute_ece(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    if y_probs.ndim != 2:
        raise ValueError("y_probs must be 2D array of shape [n_samples, n_classes]")
    confidences = y_probs.max(axis=1)
    y_pred = y_probs.argmax(axis=1)
    accuracies = (y_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if not in_bin.any():
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / len(y_true)) * abs(bin_conf - bin_acc)
    return float(round(ece, 6))


def confidence_vs_accuracy_summary(
    y_true: List[int],
    y_probs: List[List[float]] | np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_probs_arr = np.asarray(y_probs, dtype=float)
    if y_probs_arr.ndim != 2:
        raise ValueError("y_probs must be a 2D array")
    confidences = y_probs_arr.max(axis=1)
    y_pred = y_probs_arr.argmax(axis=1)
    accuracies = (y_pred == y_true_arr).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    summary = []
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        count = int(in_bin.sum())
        if count == 0:
            summary.append({
                "bin_lower": float(bins[i]),
                "bin_upper": float(bins[i + 1]),
                "count": 0,
                "accuracy": float("nan"),
                "confidence": float("nan"),
            })
            continue
        summary.append({
            "bin_lower": float(bins[i]),
            "bin_upper": float(bins[i + 1]),
            "count": count,
            "accuracy": float(round(accuracies[in_bin].mean(), 4)),
            "confidence": float(round(confidences[in_bin].mean(), 4)),
        })
    return summary


def top_k_accuracy(
    y_true: List[int],
    y_probs: List[List[float]] | np.ndarray,
    k: int,
) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_probs_arr = np.asarray(y_probs, dtype=float)
    if y_probs_arr.ndim != 2:
        raise ValueError("y_probs must be a 2D array")
    return float(round(top_k_accuracy_score(y_true_arr, y_probs_arr, k=k, labels=np.arange(y_probs_arr.shape[1])), 4))


def compute_robustness_drop(
    clean_metrics: Dict,
    noisy_metrics: Dict,
    prefix: str = "",
) -> Dict[str, float]:
    p = prefix
    clean_acc = clean_metrics.get(f"{p}accuracy", 0.0)
    noisy_acc = noisy_metrics.get(f"{p}accuracy", 0.0)
    clean_f1  = clean_metrics.get(f"{p}macro_f1", 0.0)
    noisy_f1  = noisy_metrics.get(f"{p}macro_f1", 0.0)
    return {
        f"{p}robustness_drop": float(round(clean_acc - noisy_acc, 4)),
        f"{p}macro_f1_drop": float(round(clean_f1 - noisy_f1, 4)),
    }


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[Dict[int, str]] = None,
    prefix: str = "",
    y_probs: Optional[List[List[float]] | np.ndarray] = None,
    topk: Optional[List[int]] = None,
) -> Dict:
    """Return a dict of scalar metrics plus per-class breakdown.

    Parameters
    ----------
    y_true      : ground-truth integer labels
    y_pred      : predicted integer labels
    class_names : optional {label_int: name_str} mapping
    prefix      : optional string prefix for metric keys (e.g. "val_")
    y_probs     : optional predicted class probabilities
    topk        : optional list of top-k values to compute

    Returns
    -------
    dict with keys:
        {prefix}accuracy, {prefix}macro_f1, {prefix}weighted_f1,
        {prefix}balanced_accuracy, optional top-k accuracies, calibration metrics,
        per_class: {class_name: {precision, recall, f1, support}}
    """
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wf1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)

    # Per-class report
    labels = sorted(set(y_true) | set(y_pred))
    names  = (
        [class_names.get(l, str(l)) for l in labels]
        if class_names else [str(l) for l in labels]
    )
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": report[name]["precision"],
            "recall":    report[name]["recall"],
            "f1":        report[name]["f1-score"],
            "support":   int(report[name]["support"]),
        }
        for name in names
        if name in report
    }

    metrics = {
        f"{prefix}accuracy":          round(float(acc),  4),
        f"{prefix}macro_f1":          round(float(mf1),  4),
        f"{prefix}weighted_f1":       round(float(wf1),  4),
        f"{prefix}balanced_accuracy": round(float(bacc), 4),
        "per_class":                 per_class,
    }

    if topk is not None and y_probs is not None:
        for k in topk:
            if k <= 0:
                continue
            metrics[f"{prefix}top_{k}_accuracy"] = top_k_accuracy(y_true, y_probs, k)

    if y_probs is not None:
        y_probs_arr = np.asarray(y_probs, dtype=float)
        if y_probs_arr.ndim == 1:
            raise ValueError("y_probs must be a 2D array of class probabilities")
        if y_probs_arr.shape[0] != len(y_true):
            raise ValueError("y_probs length must match y_true length")
        metrics[f"{prefix}ece"] = _compute_ece(np.asarray(y_true, dtype=int), y_probs_arr)
        metrics[f"{prefix}mean_confidence"] = float(round(y_probs_arr.max(axis=1).mean(), 4))
        metrics[f"{prefix}overconfidence_gap"] = float(
            round(metrics[f"{prefix}mean_confidence"] - metrics[f"{prefix}accuracy"], 4)
        )
        metrics[f"{prefix}confidence_vs_accuracy_summary"] = (
            confidence_vs_accuracy_summary(y_true, y_probs_arr)
        )

    return metrics


def save_metrics(metrics: Dict, path: Path) -> None:
    """Write metrics dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_classification_report_csv(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[Dict[int, str]],
    path: Path,
) -> None:
    """Save per-class precision/recall/F1/support as CSV."""
    labels = sorted(set(y_true) | set(y_pred))
    names  = (
        [class_names.get(l, str(l)) for l in labels]
        if class_names else [str(l) for l in labels]
    )
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for name in names:
        if name in report:
            r = report[name]
            rows.append({
                "class":     name,
                "precision": round(r["precision"], 4),
                "recall":    round(r["recall"],    4),
                "f1":        round(r["f1-score"],  4),
                "support":   int(r["support"]),
            })
    # append macro / weighted averages
    for avg in ("macro avg", "weighted avg"):
        if avg in report:
            r = report[avg]
            rows.append({
                "class":     avg,
                "precision": round(r["precision"], 4),
                "recall":    round(r["recall"],    4),
                "f1":        round(r["f1-score"],  4),
                "support":   int(r["support"]),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[Dict[int, str]],
    path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Save a confusion matrix PNG.

    Parameters
    ----------
    normalize : if True, show row-normalised (recall) values
    """
    labels = sorted(set(y_true) | set(y_pred))
    names  = (
        [class_names.get(l, str(l)) for l in labels]
        if class_names else [str(l) for l in labels]
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_plot  = cm.astype(float) / row_sums
        fmt      = ".2f"
        cbar_label = "Recall (row-normalised)"
    else:
        cm_plot    = cm
        fmt        = "d"
        cbar_label = "Count"

    n = len(labels)
    fig_size = max(6, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, label=cbar_label)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=12, fontweight="bold")

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            txt = f"{val:{fmt}}" if fmt == ".2f" else str(val)
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
