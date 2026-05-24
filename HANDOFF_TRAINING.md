"# Training Handoff Guide

> Private repository handoff for XRD crystal classification.

## 1. Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Manifest, splits (16 configs), validation, class balance analysis |
| Phase 2 | ✅ Complete | DataLoader, materialization pipeline, smoke tests |
| Phase 3 | 🔧 Ready | Baseline models (MLP + CNN1D) implemented, training script ready |

**What works end-to-end:**
- Split JSONs for D1–D4 × 5/10/15/20%
- Validation (no leakage, augmentation complete)
- Class balance analysis with paper report
- DataLoader from zip (slow) or materialized numpy (fast)
- WeightedRandomSampler for crystal_system
- MLP and CNN1D model architectures
- Training script with early stopping, model selection, metrics, confusion matrix

**What needs GPU time:**
- Actual training runs (MPS works but is slow; CUDA recommended)

---

## 2. Data Requirement

Raw data zips are **not committed** to the repository (too large).

Place these files in `data/`:
```
data/D1.zip   (~2.5 GB)
data/D2.zip   (~2.5 GB)
data/D3.zip   (~2.5 GB)
data/D4.zip   (~2.5 GB)
```

Each zip contains ~692k CSV files (one per augmented XRD pattern, 4500 intensity values each).

---

## 3. Materialize D1 10% (Quick Start)

Materialization reads patterns from zip once and saves as numpy arrays for fast training:

```bash
python src/data/materialize_dataset.py --domain D1 --scale 10 --task crystal_system
```

Output: `outputs/phase2/materialized/D1_10pct/crystal_system/`
- `x_train.npy` (681 MB), `x_val.npy` (146 MB), `x_test.npy` (146 MB)
- `y_*.npy`, `sample_ids_*.json`, `structure_ids_*.json`
- `metadata.json`, `label_map.json`

Takes ~2 minutes. After this, training loads data in <0.05s instead of minutes.

---

## 4. Smoke Test

```bash
python src/smoke_test_dataloader.py --domain D1 --scale 10 --task crystal_system --source materialized
```

Verifies:
- Batch shapes: `x=[64, 1, 4500]`, `y=[64]`
- No structure-level leakage
- All tensors finite
- Label distribution correct
- Class weights computed

---

## 5. Materialize All Recommended Datasets

For full experiments, materialize all domain × scale × task combinations:

```bash
# Debugging scale (10%)
for domain in D1 D2 D3 D4; do
  python src/data/materialize_dataset.py --domain $domain --scale 10 --task crystal_system
  python src/data/materialize_dataset.py --domain $domain --scale 10 --task top10_space_group
done

# Main experiment scale (20%)
for domain in D1 D2 D3 D4; do
  python src/data/materialize_dataset.py --domain $domain --scale 20 --task crystal_system
  python src/data/materialize_dataset.py --domain $domain --scale 20 --task top10_space_group
done
```

Estimated total disk: ~16 GB for all 16 materialized datasets.

---

## 6. Training

### Quick debug run (D1 10%)

```bash
python src/train_baseline.py \
  --model cnn1d \
  --domain D1 --scale 10 \
  --task crystal_system \
  --epochs 10 \
  --source materialized
```

### Full experiment plan

1. **Start:** D1 10% crystal_system (fast iteration, ~2k train structures)
2. **Scale up:** D1 20% crystal_system (4.6k structures, more stable SG distribution)
3. **Cross-domain:** D1–D4 20% crystal_system (same structures, different noise conditions)
4. **Space group:** D1 20% top10_space_group (10-class, well-balanced)

### Model options

| Model | Params | Notes |
|-------|--------|-------|
| `mlp` | ~2.5M | Simple baseline, flatten + 3 FC layers |
| `cnn1d` | ~684K | 3 conv blocks + global avg pool + FC head |

---

## 7. Imbalance Handling

### Crystal System (7 classes, 30x imbalance)

| Component | Setting |
|-----------|---------|
| Train sampler | `WeightedRandomSampler` (auto-enabled) |
| Loss | `CrossEntropyLoss(weight=class_weights)` |
| Val/Test | Never reweighted |

Triclinic has only ~12 train structures at 10% scale (288 patterns) vs ~360 for orthorhombic (8640 patterns).

### Top-10 Space Group (10 classes, ~2x imbalance)

| Component | Setting |
|-----------|---------|
| Train sampler | Standard shuffle (no reweighting needed) |
| Loss | Standard `CrossEntropyLoss` (class weights optional) |
| Val/Test | Never reweighted |

Well-balanced: train min=57, max=110 at 10% scale.

---

## 8. Expected Batch Shape

```python
batch = next(iter(train_loader))
batch[\"x\"].shape      # torch.Size([64, 1, 4500])  float32
batch[\"y\"].shape      # torch.Size([64])            int64
batch[\"sample_id\"]    # list of 64 strings, e.g. \"1000104_1\"
batch[\"structure_id\"] # list of 64 strings, e.g. \"1000104\"
batch[\"domain\"]       # list of 64 strings, e.g. \"D1\"
batch[\"task\"]         # list of 64 strings, e.g. \"crystal_system\"
```

---

## 9. Output Structure

```
outputs/
├── phase1/                          # Committed
│   ├── manifest.json
│   ├── splits/                      # 16 split JSONs
│   ├── validation/                  # validation_report.json
│   └── class_balance/               # CSVs, JSONs, MD, figures
├── phase2/                          # Partially committed
│   ├── label_mappings/              # Committed (small JSONs)
│   ├── dataloader_smoke_test_report.md  # Committed
│   └── materialized/               # .npy NOT committed (regenerate)
└── phase3/                          # NOT committed (training outputs)
    └── baseline/{run_name}/
        ├── config.json
        ├── train_log.csv
        ├── best_model.pt
        ├── val_metrics.json
        ├── test_metrics.json
        ├── classification_report.csv
        └── confusion_matrix.png
```

---

## 10. Key Metrics (Model Selection)

- **Primary:** Validation macro F1 (handles class imbalance)
- **Secondary:** Balanced accuracy, weighted F1
- **Reported:** Per-class precision/recall/F1, confusion matrix

---

## 11. Environment

Tested with:
- Python 3.12
- PyTorch 2.5.1
- numpy, pandas, scikit-learn, matplotlib

Install:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```
"