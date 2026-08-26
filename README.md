# Robust Material — Hierarchical Crystallographic Classification from Powder XRD

Deep-learning pipeline for **hierarchical crystallographic classification from powder XRD
patterns**, studying whether **multitask learning** improves sample efficiency, robustness,
and calibration under limited-data and noisy conditions.

> **Research question:** *Can multitask learning improve sample efficiency and robustness for
> hierarchical crystallographic classification from powder XRD patterns under limited-data and
> noisy conditions?*

---

## Table of contents

- [Overview](#overview)
- [Research questions](#research-questions)
- [Data](#data)
- [Repository structure](#repository-structure)
- [Environment setup](#environment-setup)
- [Pipeline](#pipeline)
- [Models](#models)
- [Training](#training)
- [Experiments](#experiments)
- [Metrics](#metrics)
- [Outputs](#outputs)
- [Repository conventions](#repository-conventions)

---

## Overview

Each input is a 1D powder XRD intensity pattern of length **4500**. The pipeline predicts two
hierarchically related labels:

| Task | Target | Classes |
|---|---|---|
| `crystal_system` | Crystal system | 7 (≈30× class imbalance) |
| `top10_space_group` | Top-10 space groups | 10 (≈2× class imbalance) |

Data is organized into four **noise domains**:

| Domain | Noise | Severity |
|---|---|---|
| D1 | Low noise | Clean reference |
| D2 | Background + low noise | Mild |
| D3 | Medium noise | Moderate |
| D4 | High noise | Severe |

Experiments test sample efficiency (15–35% training scales), cross-domain generalization,
noise robustness, calibration, and class-imbalance handling.

---

## Research questions

1. **Sample efficiency** — Does a shared-backbone, dual-head multitask model improve macro-F1 on
   both tasks at limited training scales (15–25%)?
2. **Robustness** — Does multitask training reduce robustness drop when tested on noisier domains
   (D2–D4) vs single-task training?
3. **Cross-domain generalization** — Which training-domain composition best generalizes across
   noise conditions?
4. **Calibration** — Are multitask models better or worse calibrated (ECE, overconfidence gap)
   than single-task models?
5. **Imbalance** — Does class-weighted CE or focal loss further reduce robustness drop under
   noisy cross-domain testing?

---

## Data

Raw data are **not committed** (too large, ~2.5 GB per zip). Place the four zips in `data/`:

```
data/D1.zip   low noise
data/D2.zip   background + low noise
data/D3.zip   medium noise
data/D4.zip   high noise
```

Each zip contains ~692k CSV files (one per augmented XRD pattern, 4500 intensity values each).

### Split policy

All experiments use **stratified structure-level splits** (no pattern/augmentation leakage):

- Stratification: `crystal_system + filtered_space_group`
- Ratios: 70% train / 15% val / 15% test
- All augmentations of a structure stay in one split
- Val/test are never reweighted or resampled; class balancing is applied to train only
- Multi-domain training concatenates train sets; test is the designated test domain only

---

## Repository structure

```
robust-material/
├── configs/                 # Experiment + model YAML configs
│   ├── models/             # Per-architecture specs
├── src/
│   ├── data/               # Manifest, splits, materialization, dataloaders
│   ├── models/             # MLP, CNN1D, ResNet1D, CNNAttention, multitask
│   ├── experiments/        # Cross-domain evaluation runner
│   ├── utils/              # Metrics, seeding
│   ├── train_baseline.py   # Single-task / multi-domain training
│   ├── train_multitask.py  # Dual-head multitask training
│   └── evaluate.py         # Evaluation + metrics
├── data/                   # Raw zips (gitignored)
├── outputs/                # Generated artifacts (gitignored)
└── setup_venv.sh           # Recreate the PyTorch venv
```

`data/` and `outputs/` are git-ignored. Regenerate outputs from source (see below).

---

## Environment setup

```bash
bash setup_venv.sh          # creates ./venv with CUDA torch + deps
source venv/bin/activate
```

Or with your own environment:

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm einops
```

Tested with Python 3.12, PyTorch 2.5.1.

---

## Pipeline

### 1. Materialize a dataset (fast dataloading)

Reading straight from zips is slow; materialize to numpy caches first:

```bash
python src/data/materialize_dataset.py --domain D1 --scale 10 --task crystal_system
python src/data/materialize_dataset.py --domain D1 --scale 10 --task top10_space_group
```

Materialize all combinations for full experiments:

```bash
for domain in D1 D2 D3 D4; do
  for scale in 15 20 25 30 35; do
    python src/data/materialize_dataset.py --domain $domain --scale $scale --task crystal_system
    python src/data/materialize_dataset.py --domain $domain --scale $scale --task top10_space_group
  done
done
```

Output: `outputs/materialized/{domain}_{scale}pct/{task}/` (~40 GB for all 40 datasets).

### 2. Smoke test

```bash
python src/smoke_test_dataloader.py --domain D1 --scale 10 --task crystal_system --source materialized
```

Verifies batch shapes `[64, 1, 4500]`, no leakage, finite tensors, correct label distribution.

---

## Models

### Single-task backbones

| Model | Params (7 cls) | Notes |
|---|---|---|
| `mlp` | 2,471,431 | Flatten + 3 FC; lower-bound baseline |
| `cnn1d` | 684,359 | 3 conv blocks + global avg pool; primary baseline |
| `resnet1d` | 8,595,015 | Residual blocks; use lighter config at small scales |
| `cnn_attention` | 793,799 | CNN + multi-head self-attention |

### Multitask models (shared backbone + dual heads)

| Backbone | Total params | Crystal head | SG head |
|---|---|---|---|
| `cnn1d` | 817,618 | 7 | 10 |
| `resnet1d` | 8,661,970 | 7 | 10 |
| `cnn_attention` | 828,754 | 7 | 10 |

```python
out = model(x)            # x: [B, 1, 4500]
out["crystal_logits"]     # [B, 7]
out["space_group_logits"] # [B, 10]
```

### ResNet1D lighter config (recommended at 15–25% scale)

```python
ResNet1D(input_length=4500, num_classes=7,
         layers=[2,2,2], channels=[64,128,256])  # ~1.1M params
```

---

## Training

### Single-task (single domain)

```bash
python src/train_baseline.py \
  --model cnn1d --domain D1 --scale 20 \
  --task crystal_system --epochs 100 --source materialized
```

### Single-task with focal loss

```bash
python src/train_baseline.py \
  --model cnn1d --domain D1 --scale 20 \
  --task crystal_system --epochs 100 --loss focal --focal-gamma 2.0 \
  --source materialized
```

### Multi-domain training

```bash
python src/train_baseline.py \
  --model cnn1d --domains D1,D2 --test-domain D3 \
  --scale 20 --task crystal_system --epochs 100 --source materialized
```

### Multitask training

```bash
python src/train_multitask.py \
  --model cnn1d --domain D1 --scale 20 --epochs 100 --source materialized
```

### Imbalance handling

- **Crystal system (7 cls):** `WeightedRandomSampler` (auto) + class-weighted CE (default), or
  focal loss via `--loss focal`. Val/test never reweighted.
- **Top-10 space group (10 cls):** standard shuffle + standard CE. Val/test never reweighted.

---

## Experiments

Configs live in `configs/` (`data_scaling.yaml`, `single_vs_multitask.yaml`, `robustness.yaml`,
`cross_domain.yaml`, `imbalance.yaml`, `baselines.yaml`).

### Cross-domain runner (5 experiment types)

```bash
python src/experiments/cross_domain_eval.py \
  --model cnn1d --scale 20 --task crystal_system --epochs 100

# specific experiments only
python src/experiments/cross_domain_eval.py \
  --model cnn1d --scale 20 --task crystal_system --epochs 100 \
  --experiments single progressive leave_one_out

# aggregate from existing checkpoints (no retraining)
python src/experiments/cross_domain_eval.py \
  --model cnn1d --scale 20 --task crystal_system --aggregate-only
```

| # | Experiment | Design |
|---|---|---|
| 1 | Single-domain | train each domain → test all domains |
| 2 | Progressive mixing | D1 → D1+D2 → … → all; test each domain |
| 3 | Leave-one-out | train on 3 domains → test the held-out one |
| 4 | Clean→noisy | train D1 → test D2/D3/D4 (robustness drop) |
| 5 | Mixed robustness | train all → test each separately |

Recommended server order: debug (D1 10%), baseline (D1 20% all models), scaling
(D1 15–35%), cross-domain, robustness, imbalance ablation, full sweep.

---

## Metrics

- **Standard:** accuracy, macro/weighted F1, balanced accuracy, per-class P/R/F1, top-3/top-5 acc.
- **Robustness:** `clean_accuracy`, `noisy_accuracy`, `robustness_drop`, `macro_f1_drop`.
- **Calibration:** `ece` (10 bins), `mean_confidence`, `overconfidence_gap`, per-bin table.
- **Model selection:** validation macro-F1 (single-task) or mean of both heads' macro-F1 (multitask).

---

## Outputs

All under `outputs/` (git-ignored, regenerable):

```
outputs/
├── data_audit/            # manifest + class-balance audit
├── dataset_splits/        # split JSONs used by DataModule
├── label_mappings/        # label_map_{task}.json
├── materialized/          # numpy caches ({domain}_{scale}pct/{task}/)
├── baselines/             # single-task runs
├── cross_domain/          # cross-domain CSVs + runs
├── scaling/  single_vs_multitask/  robustness/  imbalance/
```

---

## Repository conventions

- `data/` and `outputs/` are excluded from git (large / regenerable).
- `venv/`, `__pycache__/`, `.ipynb_checkpoints/`, and editor/OS files are ignored.
- Source of truth is `src/` + `configs/`; all artifacts regenerate from those.
