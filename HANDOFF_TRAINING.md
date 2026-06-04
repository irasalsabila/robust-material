# Training Handoff Guide

> Private repository handoff for XRD crystal classification.
>
> **Paper title:** "Can multitask learning improve sample efficiency and robustness for hierarchical crystallographic classification from powder XRD patterns under limited-data and noisy conditions?"

---

## 1. Project Status

| Phase | Status | Description |
|---|---|---|
| Phase 1 | Complete | Manifest, stratified structure-level splits (28 configs D1-D4 × 5-35%), validation, class balance analysis |
| Phase 2 | Complete | DataLoader, materialization pipeline, smoke tests |
| Phase 3 | Complete | Baseline models (MLP, CNN1D) + training script with early stopping, metrics, calibration |
| Phase 4 | Ready | New architectures (ResNet1D, CNNAttention, MultitaskModel), cross-domain runner, all configs |
| Phase 5 | Pending GPU | Full experiments: scaling, single vs multitask, robustness, cross-domain |

**What works end-to-end (local):**
- Split JSONs for D1-D4 × 5/10/15/20/25/30/35% (stratified structure-level)
- No structure leakage, no augmentation leakage, validation confirmed
- DataLoader from zip (slow) or materialized numpy (fast)
- WeightedRandomSampler for crystal_system; standard shuffle for space group
- MLP, CNN1D: single-task training with class-weighted CE or focal loss
- ResNet1D, CNNAttention: new architectures, shape-validated
- MultitaskModel: shared backbone (CNN1D / ResNet1D / CNNAttention) + dual heads
- All robustness and calibration metrics: ECE, mean confidence, overconfidence gap, robustness drop, macro-F1 drop
- Multi-domain DataModule: concatenates train sets across domains
- Cross-domain experiment runner: all 5 experiment types
- Model selection: validation macro-F1

**What needs GPU time:**
- All full training runs (MPS is functional but slow; CUDA recommended)
- All scaling / multitask / cross-domain / robustness experiments

---

## 2. Research Questions

The paper addresses five core questions:

1. **Sample efficiency:** Does multitask learning (shared backbone, dual heads) improve macro-F1 on both crystal system and space group classification at limited training scales (15-25%)?
2. **Robustness:** Does multitask training reduce robustness drop when tested on noisier domains (D2, D3, D4) relative to single-task training?
3. **Cross-domain generalization:** Which training domain composition best generalizes across noise conditions?
4. **Calibration:** Are multitask models better or worse calibrated (ECE, overconfidence gap) than single-task models?
5. **Imbalance:** Does class-weighted CE or focal loss further reduce robustness drop under noisy cross-domain testing?

---

## 3. Data

Raw data zips are not committed (too large). Place in `data/`:

```
data/D1.zip   (~2.5 GB)  low noise
data/D2.zip   (~2.5 GB)  background + low noise
data/D3.zip   (~2.5 GB)  medium noise
data/D4.zip   (~2.5 GB)  high noise
```

Each zip: ~692k CSV files, one per augmented XRD pattern, 4500 intensity values each.

### Domain definitions

| Domain | Noise type | Severity |
|---|---|---|
| D1 | Low noise | Clean reference |
| D2 | Background + low noise | Mild |
| D3 | Medium noise | Moderate |
| D4 | High noise | Severe |

### Split policy

All experiments use **stratified structure-level splits**:
- Stratification: `crystal_system + filtered_space_group`
- Ratios: 70% train / 15% val / 15% test
- No structure leakage across splits
- No augmentation leakage (all augmentations of a structure stay in one split)
- Val and test sets are **never** reweighted or resampled
- Class balancing applied to train split only
- Multi-domain training concatenates train sets; val uses union of domain val sets; test is the designated test domain only

---

## 4. Materialization (Quick Start)

Materialize D1 10% for debugging:

```bash
python src/data/materialize_dataset.py --domain D1 --scale 10 --task crystal_system
python src/data/materialize_dataset.py --domain D1 --scale 10 --task top10_space_group
```

For full experiments, materialize all combinations:

```bash
for domain in D1 D2 D3 D4; do
  for scale in 15 20 25 30 35; do
    python src/data/materialize_dataset.py --domain $domain --scale $scale --task crystal_system
    python src/data/materialize_dataset.py --domain $domain --scale $scale --task top10_space_group
  done
done
```

Estimated total disk: ~40 GB for all 40 materialized datasets (4 domains × 5 scales × 2 tasks).

Output per dataset: `outputs/materialized/{domain}_{scale}pct/{task}/`

---

## 5. Smoke Test

```bash
python src/smoke_test_dataloader.py --domain D1 --scale 10 --task crystal_system --source materialized
```

Verifies: batch shapes `[64, 1, 4500]`, no leakage, finite tensors, correct label distribution.

Model shape smoke test (no data required):

```bash
python - <<'EOF'
import torch, sys; sys.path.insert(0,'.')
from src.models import build_model, MultitaskModel
x = torch.randn(4, 1, 4500)
for name in ['mlp','cnn1d','resnet1d','cnn_attention']:
    m = build_model(name, 4500, 7)
    print(name, m(x).shape, m.count_parameters(), 'params')
m = MultitaskModel('cnn1d',4500,7,10)
out = m(x)
print('multitask cs=',out['crystal_logits'].shape,'sg=',out['space_group_logits'].shape)
EOF
```

---

## 6. Models

### Single-task models

| Model | Params (7 cls) | Notes |
|---|---|---|
| `mlp` | 2,471,431 | Flatten + 3 FC layers; lower-bound baseline only |
| `cnn1d` | 684,359 | 3 conv blocks + global avg pool; primary baseline |
| `resnet1d` | 8,595,015 | Residual blocks; use lighter config at small scales |
| `cnn_attention` | 793,799 | CNN + multi-head self-attention |

### Multitask models (shared backbone + dual heads)

| Backbone | Total Params | Crystal head | SG head |
|---|---|---|---|
| `cnn1d` | 817,618 | 7 classes | 10 classes |
| `resnet1d` | 8,661,970 | 7 classes | 10 classes |
| `cnn_attention` | 828,754 | 7 classes | 10 classes |

Multitask model output:
```python
out = model(x)  # x: [B, 1, 4500]
out["crystal_logits"]      # [B, 7]
out["space_group_logits"]  # [B, 10]
```

### ResNet1D lighter config (recommended at 15-25% scale)

```python
ResNet1D(input_length=4500, num_classes=7,
         layers=[2,2,2], channels=[64,128,256])  # ~1.1M params
```

---

## 7. Training

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
# Train D1+D2, test on D3
python src/train_baseline.py \
  --model cnn1d --domains D1,D2 --test-domain D3 \
  --scale 20 --task crystal_system --epochs 100 --source materialized
```

### Cross-domain experiments (all 5 types)

```bash
# Run all cross-domain experiments
python src/experiments/cross_domain_eval.py \
  --model cnn1d --scale 20 --task crystal_system --epochs 100

# Run specific experiment types
python src/experiments/cross_domain_eval.py \
  --model cnn1d --scale 20 --task crystal_system --epochs 100 \
  --experiments single progressive leave_one_out

# Aggregate CSVs from already-trained checkpoints (no retraining)
python src/experiments/cross_domain_eval.py \
  --model cnn1d --scale 20 --task crystal_system \
  --aggregate-only
```

### Recommended experiment order on server

1. **Debug** D1 10% crystal_system CNN1D (fast iteration, ~2k train structures)
2. **Baseline** D1 20% all models, crystal_system + space group (4.6k structures)
3. **Scaling** D1 15-35% CNN1D single-task vs multitask
4. **Cross-domain** all 5 experiment types, CNN1D + ResNet1D
5. **Robustness** single-task vs multitask on D2/D3/D4
6. **Imbalance ablation** focal vs CE-weighted on D1
7. **Full sweep** ResNet1D + CNNAttention at best scale

---

## 8. Imbalance Handling

### Crystal System (7 classes, ~30x imbalance)

| Component | Setting |
|---|---|
| Train sampler | `WeightedRandomSampler` (auto-enabled) |
| Loss | `CrossEntropyLoss(weight=class_weights)` (default) |
| Alternative loss | `FocalLoss(gamma=2.0)` (via `--loss focal`) |
| Val/Test | Never reweighted |

### Top-10 Space Group (10 classes, ~2x imbalance)

| Component | Setting |
|---|---|
| Train sampler | Standard shuffle (no reweighting) |
| Loss | Standard `CrossEntropyLoss` (default) |
| Val/Test | Never reweighted |

---

## 9. Metrics

### Standard classification
- accuracy, macro F1, weighted F1, balanced accuracy
- per-class precision / recall / F1 / support
- top-3 and top-5 accuracy (space group task only)

### Robustness
- `clean_accuracy`: accuracy on D1 test
- `noisy_accuracy`: accuracy on D2/D3/D4 test
- `robustness_drop`: clean_accuracy - noisy_accuracy
- `macro_f1_drop`: clean_macro_F1 - noisy_macro_F1

### Calibration (per head)
- `ece`: Expected Calibration Error (10 bins)
- `mean_confidence`: mean of max softmax probability
- `overconfidence_gap`: mean_confidence - accuracy
- `confidence_vs_accuracy_summary`: per-bin calibration table

### Multitask only (pending implementation)
- `crystallographic_consistency`: fraction of predictions where predicted crystal system is consistent with the crystal system of the predicted space group

### Model selection criterion
- Single-task: validation macro F1
- Multitask: mean of crystal system and space group validation macro F1

---

## 10. Experiment Configurations

| Config | File | Purpose |
|---|---|---|
| Data scaling | `configs/data_scaling.yaml` | Sample efficiency sweep 15-35% |
| Single vs multitask | `configs/single_vs_multitask.yaml` | Core paper comparison |
| Robustness | `configs/robustness.yaml` | Noise robustness + calibration |
| Cross-domain | `configs/cross_domain.yaml` | All 5 cross-domain experiments |
| Imbalance | `configs/imbalance.yaml` | Loss function / sampler ablation |
| Baselines | `configs/baselines.yaml` | Training defaults + domain/scale lists |
| MLP | `configs/models/mlp.yaml` | MLP architecture spec |
| CNN1D | `configs/models/cnn1d.yaml` | CNN1D architecture spec |
| ResNet1D | `configs/models/resnet1d.yaml` | ResNet1D architecture spec |
| CNNAttention | `configs/models/cnn_attention.yaml` | CNNAttention architecture spec |

---

## 11. Cross-Domain Experiment Definitions

### Experiment 1: Single-domain training
Train on one domain; test on every domain.
```
train D1 -> test D1, D2, D3, D4
train D2 -> test D1, D2, D3, D4
train D3 -> test D1, D2, D3, D4
train D4 -> test D1, D2, D3, D4
```
Output: `outputs/cross_domain/cross_domain_matrix.csv`

### Experiment 2: Progressive domain mixing
Incrementally add noisier domains; test each domain separately.
```
train D1            -> test D1, D2, D3, D4
train D1+D2         -> test D1, D2, D3, D4
train D1+D2+D3      -> test D1, D2, D3, D4
train D1+D2+D3+D4   -> test D1, D2, D3, D4
```
Output: `outputs/cross_domain/progressive_domain_mixing.csv`

### Experiment 3: Leave-one-domain-out
```
train D2+D3+D4 -> test D1
train D1+D3+D4 -> test D2
train D1+D2+D4 -> test D3
train D1+D2+D3 -> test D4
```
Output: `outputs/cross_domain/leave_one_domain_out.csv`

### Experiment 4: Clean-to-noisy robustness
```
train D1 -> test D2, D3, D4
report robustness_drop relative to D1 test accuracy
```
Output: `outputs/cross_domain/clean_to_noisy.csv`

### Experiment 5: Mixed-domain robustness
```
train D1+D2+D3+D4 -> test D1, D2, D3, D4 separately
report whether mixed training reduces robustness_drop
```

Summary outputs:
- `outputs/cross_domain/robustness_drop_summary.csv`
- `outputs/cross_domain/calibration_summary.csv`

---

## 12. Output Structure

```
outputs/
├── data_audit/                      # Committed
│   ├── manifest.json
│   ├── splits/                      # split_{D}_{S}pct.json (28 files)
│   └── class_balance/
├── dataset_splits/splits/           # Split JSONs used by DataModule
├── label_mappings/                  # label_map_{task}.json
├── model_audit/
│   └── model_review.md              # Architecture audit report
├── materialized/                    # Regenerated numpy caches (not in git)
│   └── {domain}_{scale}pct/{task}/
│       ├── x_train.npy
│       ├── y_train.npy
│       └── ...
├── baselines/                       # Single-task training outputs
│   └── {run_name}/
│       ├── config.json
│       ├── train_log.csv
│       ├── best_model.pt
│       ├── val_metrics.json
│       ├── test_metrics.json
│       ├── classification_report.csv
│       └── confusion_matrix.png
├── cross_domain/                    # Cross-domain experiment outputs
│   ├── runs/{run_name}/
│   ├── cross_domain_matrix.csv
│   ├── progressive_domain_mixing.csv
│   ├── leave_one_domain_out.csv
│   ├── robustness_drop_summary.csv
│   └── calibration_summary.csv
├── scaling/                         # Data scaling experiment outputs
├── single_vs_multitask/             # Multitask vs single-task outputs
├── robustness/                      # Robustness experiment outputs
└── imbalance/                       # Imbalance ablation outputs
```

---

## 13. Batch Shape Reference

```python
batch = next(iter(train_loader))
batch["x"].shape      # torch.Size([64, 1, 4500])  float32
batch["y"].shape      # torch.Size([64])            int64
batch["sample_id"]    # list[str]  e.g. "1000104_1"
batch["structure_id"] # list[str]  e.g. "1000104"
batch["domain"]       # list[str]  e.g. "D1"
batch["task"]         # list[str]  e.g. "crystal_system"
```

---

## 14. Environment

Tested with:
- Python 3.12
- PyTorch 2.5.1
- numpy, pandas, scikit-learn, matplotlib

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

---

## 15. Remaining Work Before Server Training

| Priority | Task | Notes |
|---|---|---|
| High | Implement multitask training script | `train_baseline.py` handles single-task; needs dual-head loss for multitask |
| High | Add `crystallographic_consistency` metric | `src/utils/metrics.py` |
| High | Materialize all 40 datasets on server | 4 domains × 5 scales × 2 tasks |
| Medium | Test ResNet1D lighter config `[2,2,2]` | For 15-20% scale to reduce overfitting risk |
| Medium | Suppress CNNAttention `UserWarning` (cosmetic) | `norm_first=True` warning |
| Low | Verify split JSONs exist for 25/30/35% scales | Run `src/split_dataset.py` if missing |
