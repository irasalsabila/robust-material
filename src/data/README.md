"# src/data — XRD DataLoader Package

Phase 2 data loading for XRD diffraction pattern classification.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `xrd_dataset.py` | `XRDDataset` — PyTorch Dataset |
| `datamodule.py` | `XRDDataModule` — wires loaders together |
| `samplers.py` | `build_weighted_sampler` + policy helpers |
| `transforms.py` | Normalization and pad/truncate |

## Quick Start

```python
from src.data import XRDDataModule

dm = XRDDataModule(
  manifest_path   = \"outputs/data_audit/manifest.json\",
  split_json_path = \"outputs/dataset_splits/splits/split_D1_10pct.json\",
  task            = \"crystal_system\",   # or \"top10_space_group\"
)
dm.setup()

for batch in dm.train_dataloader():
    x = batch[\"x\"]            # FloatTensor [B, 1, 4500]
    y = batch[\"y\"]            # LongTensor  [B]
    sample_ids = batch[\"sample_id\"]   # list[str]
    break
```

## Returned Batch Keys

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `x` | `FloatTensor` | `[B, 1, L]` | Normalised XRD pattern |
| `y` | `LongTensor` | `[B]` | Integer class label |
| `sample_id` | `list[str]` | `B` | e.g. `\"1000104_1\"` |
| `structure_id` | `list[str]` | `B` | e.g. `\"1000104\"` |
| `domain` | `list[str]` | `B` | `\"D1\"` … `\"D4\"` |
| `task` | `list[str]` | `B` | task name |

## Tasks

### `crystal_system`
7 classes (cubic, monoclinic, orthorhombic, triclinic, hexagonal, tetragonal, trigonal).
Imbalance ratio ~30x — **WeightedRandomSampler ON** for train, class weights provided.

### `top10_space_group`
10 classes (Pnma, P2₁/c, Fm-3m, P6₃/mmc, P-1, C2/m, R-3m, Pm-3m, I4/mmm, Fd-3m).
Imbalance ratio ~2x — **WeightedRandomSampler OFF**, standard cross-entropy sufficient.
Structures whose SG is not in the top-10 are silently excluded from the dataset.

## Normalization Options

| Method | Description |
|--------|-------------|
| `max_intensity` | Divide by max absolute value **(default)** |
| `minmax` | Scale to [0, 1] |
| `standard` | Zero-mean, unit-variance per pattern |
| `none` | Raw values |

## Data Layout

```
data/
  D1.zip   →  train/{sample_id}.csv          (4500 float rows, header \"y\")
  D2.zip   →  train_with_back/{sample_id}.csv
  D3.zip   →  train_with_noise1/{sample_id}.csv
  D4.zip   →  train_with_noise/{sample_id}.csv
```

All patterns (train/val/test) are stored in the domain's single folder.
Split membership is controlled entirely by the split JSON.

## Sampler Policy

| Task | Split | Sampler |
|------|-------|---------|
| `crystal_system` | train | `WeightedRandomSampler` |
| `crystal_system` | val / test | Sequential (no reweighting) |
| `top10_space_group` | train | Shuffle |
| `top10_space_group` | val / test | Sequential |

Val and test distributions are **never** altered.

## Label Mappings

Saved automatically to `outputs/label_mappings/` on first dataset construction:
- `label_map_crystal_system.json`
- `label_map_top10_space_group.json`

## Smoke Test

```bash
python src/smoke_test_dataloader.py --domain D1 --scale 10 --task crystal_system
python src/smoke_test_dataloader.py --domain D1 --scale 20 --task top10_space_group
```

Saves report to `outputs/dataloader_validation/dataloader_smoke_test_report.md`.
"