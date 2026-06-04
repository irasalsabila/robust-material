# Model Architecture Review

**Project:** Multitask learning for hierarchical crystallographic classification from powder XRD patterns  
**Date:** 2026-06-04  
**Reviewer:** Automated audit via smoke test

---

## 1. Audit Summary

All five architectures were validated against input shape `[B, 1, 4500]` at `B=4`.
All produce correct output shapes and have correct parameter counts.
All are GPU/MPS/CPU compatible.

| Model | Task | Output Shape | Params | Multitask Backbone? |
|---|---|---|---|---|
| MLP | crystal_system | [B, 7] | 2,471,431 | No |
| MLP | top10_space_group | [B, 10] | 2,471,818 | No |
| CNN1D | crystal_system | [B, 7] | 684,359 | Yes |
| CNN1D | top10_space_group | [B, 10] | 685,130 | Yes |
| ResNet1D | crystal_system | [B, 7] | 8,595,015 | Yes |
| ResNet1D | top10_space_group | [B, 10] | 8,595,786 | Yes |
| CNNAttention | crystal_system | [B, 7] | 793,799 | Yes |
| CNNAttention | top10_space_group | [B, 10] | 794,186 | Yes |
| Multitask(cnn1d) | both heads | cs=[B,7] sg=[B,10] | 817,618 | — |
| Multitask(resnet1d) | both heads | cs=[B,7] sg=[B,10] | 8,661,970 | — |
| Multitask(cnn_attention) | both heads | cs=[B,7] sg=[B,10] | 828,754 | — |

---

## 2. MLP

**File:** `src/models/mlp.py`

### Architecture
- Input `[B, 1, L]` → Flatten → `[B, L]`
- Three hidden layers: Linear(L, 512) → BN → ReLU → Dropout × 3
- Output: Linear(128, num_classes)

### Shape Verification
- Input: `[B, 1, 4500]` ✓
- Output: `[B, num_classes]` ✓
- Flatten handles the channel-1 dimension correctly ✓

### Parameter Count
- crystal_system (7 classes): **2,471,431**
- top10_space_group (10 classes): **2,471,818**

### Strengths
- Simple, transparent baseline
- No spatial bias; treats XRD pattern as flat feature vector
- Fast single forward pass

### Limitations
- Cannot exploit local peak structure (no convolution)
- Highest parameter count among baselines; risk of overfitting at small scales (15-20%)
- No `forward_features` method — **not usable as MultitaskModel backbone**
- Sensitive to input normalization (max_intensity recommended)

### Required Fix
- None for single-task use. Not a multitask backbone.

---

## 3. CNN1D

**File:** `src/models/cnn1d.py`

### Architecture
- Input `[B, 1, L]`
- 3 × ConvBlock: Conv1d → BN → ReLU → Dropout → MaxPool(4)
- AdaptiveAvgPool1d(1) → Flatten → Linear(256, fc_dim=256) → BN → ReLU → Dropout → Linear(256, num_classes)

### Shape Trace (L=4500)
```
[B, 1, 4500] → [B, 64, 1125] → [B, 128, 281] → [B, 256, 70]
             → pool [B, 256, 1] → flatten [B, 256]
             → fc_dim [B, 256] → output [B, num_classes]
```

### Parameter Count
- crystal_system (7 classes): **684,359**
- top10_space_group (10 classes): **685,130**

### Adaptive Pooling
- `AdaptiveAvgPool1d(1)` is input-length agnostic ✓
- Works correctly regardless of exact sequence length after pooling ✓

### Multitask Compatibility
- `forward_features` accessible via backbone wrapper in `MultitaskModel._CNN1DBackbone` ✓
- Feature dim: `fc_dim = 256` ✓
- Multitask(cnn1d) total params: **817,618** (backbone + two heads) ✓

### Strengths
- Efficient: lowest params among CNN architectures
- Local peak feature extraction via 1-D convolutions
- AdaptiveAvgPool makes it robust to input-length variation
- Default backbone for multitask experiments

### Limitations
- Fixed receptive field per block; long-range XRD correlations not captured
- Three blocks may underfit complex noise patterns in D3/D4
- No residual connections; gradient flow degrades with depth

### Required Fix
- None. Architecture is correct and compatible.

---

## 4. ResNet1D

**File:** `src/models/resnet1d.py` (new)

### Architecture
- Input `[B, 1, L]`
- Stem: Conv1d(stride=2) → BN → ReLU → MaxPool(stride=2)  → `[B, 64, ~1125]`
- 4 Residual stages: [2,2,2,2] blocks, channels [64,128,256,256]
  - Stride-2 downsampling at first block of stages 1-3
  - Skip connection with 1×1 conv when dimensions change
- AdaptiveAvgPool1d(1) → Flatten → Dropout → Linear(256, num_classes)

### Shape Trace (L=4500)
```
[B, 1, 4500] → stem [B, 64, 1125] → stage0 [B, 64, 1125]
             → stage1 [B, 128, 563] → stage2 [B, 256, 282] → stage3 [B, 256, 282]
             → pool [B, 256, 1] → flatten [B, 256] → output [B, num_classes]
```

### Parameter Count
- Default [2,2,2,2] / [64,128,256,256]: **8,595,015 / 8,595,786**
- Note: this is large for small-scale experiments (15-20%)
- Recommended lighter config: `layers=[2,2,2]`, `channels=[64,128,256]` → ~1.1M

### Adaptive Pooling
- `AdaptiveAvgPool1d(1)` present ✓
- `forward_features` method exposed for MultitaskModel ✓

### Multitask Compatibility
- `MultitaskModel(backbone='resnet1d')` works ✓
- Total params: **8,661,970** (with default config)

### Strengths
- Residual connections prevent vanishing gradients in deeper networks
- Configurable depth for ablation (`layers=[2,2]` vs `[2,2,2,2]`)
- Better multi-scale feature capture than plain CNN1D

### Limitations
- Default config (8.6M params) is large relative to training set at 15-20% scale
- Slower per-epoch training than CNN1D
- Higher GPU memory requirement

### Recommended Action
- Use lighter config `layers=[2,2,2]`, `channels=[64,128,256]` for 15-20% scale experiments
- Default 4-stage config suitable for 30-35% scale experiments
- Add lighter preset to `configs/models/resnet1d.yaml` before server training

---

## 5. CNNAttention

**File:** `src/models/cnn_attention.py` (new)

### Architecture
- Input `[B, 1, L]`
- 3 × ConvBlock → `[B, 256, ~70]`
- 1×1 Conv projection → `[B, d_model=128, ~70]`
- Learnable positional encoding
- TransformerEncoderLayer(d_model=128, nhead=4, ff_dim=256, norm_first=True)
- AdaptiveAvgPool1d(1) → Flatten → Dropout → Linear(128, num_classes)

### Shape Trace (L=4500)
```
[B, 1, 4500] → cnn [B, 256, 70] → proj [B, 128, 70]
             → permute [B, 70, 128] → pos_enc [B, 70, 128]
             → transformer [B, 70, 128] → permute [B, 128, 70]
             → pool [B, 128, 1] → flatten [B, 128] → output [B, num_classes]
```

### Parameter Count
- crystal_system (7 classes): **793,799**
- top10_space_group (10 classes): **794,186**
- Multitask(cnn_attention): **828,754**

### Adaptive Pooling
- `AdaptiveAvgPool1d(1)` present ✓
- `forward_features` exposed ✓

### Multitask Compatibility
- `MultitaskModel(backbone='cnn_attention')` works ✓

### Note on Transformer Warning
- PyTorch emits a `UserWarning` about `enable_nested_tensor` when `norm_first=True`
- This is a cosmetic warning; behaviour is correct
- Can suppress with `torch.backends.cuda.enable_flash_sdp(False)` if needed

### Strengths
- Global attention over ~70-token sequence enables long-range peak correlation
- Pre-LN (norm_first=True) improves training stability
- Competitive params (~794K) relative to CNN1D (~685K)

### Limitations
- Attention over short sequences (~70 tokens post-pooling) may provide marginal benefit
- Learnable positional encoding requires sufficient training data
- TransformerEncoderLayer adds hyperparameters (d_model, num_heads, ff_dim)

### Required Fix
- None. Architecture is correct. Suppress warning optionally before server training.

---

## 6. MultitaskModel

**File:** `src/models/multitask.py` (new)

### Architecture
```
Input [B,1,L]
    └─> Backbone.forward_features(x) -> [B, feat_dim]
            ├─> CrystalSystemHead  -> crystal_logits  [B, 7]
            └─> SpaceGroupHead    -> space_group_logits [B, 10]
```

### Output Dict
```python
{
    "crystal_logits":     Tensor [B, 7],
    "space_group_logits": Tensor [B, 10],
}
```

### Verified Backbone Configurations
| Backbone | feat_dim | Total Params |
|---|---|---|
| cnn1d | 256 | 817,618 |
| resnet1d | 256 | 8,661,970 |
| cnn_attention | 128 | 828,754 |

### Head Architecture (CrystalSystemHead / SpaceGroupHead)
- Linear(feat_dim, hidden_dim=128) → BN → ReLU → Dropout(0.3) → Linear(128, num_classes)
- Kaiming initialization on all Linear layers

### Crystallographic Consistency Metric
- Not yet implemented in metrics.py
- Definition: fraction of predictions where predicted crystal system is consistent
  with the crystal system that contains the predicted space group
- Required for paper; should be added to `src/utils/metrics.py` before full experiments

---

## 7. Heads

**File:** `src/models/heads.py` (new)

- `CrystalSystemHead`: 7-class, optional hidden projection (128 by default)
- `SpaceGroupHead`: N-class (default 10), optional hidden projection
- Both follow identical interface: `forward(x: [B, feat_dim]) -> [B, num_classes]`
- Kaiming weight initialization ✓

---

## 8. Summary of Required Actions Before Server Training

| Priority | Action | File |
|---|---|---|
| High | Add `crystallographic_consistency` metric | `src/utils/metrics.py` |
| High | Add multitask training script / update `train_baseline.py` for dual-head loss | `src/train_baseline.py` or new `src/train_multitask.py` |
| Medium | Test ResNet1D lighter config `[2,2,2]` / `[64,128,256]` for small-scale experiments | `src/models/resnet1d.py` |
| Medium | Suppress TransformerEncoderLayer `UserWarning` | `src/models/cnn_attention.py` |
| Low | Add `forward_features` directly on MLP if needed as a degenerate backbone | `src/models/mlp.py` |

---

## 9. Split Policy (All Models)

All experiments use **stratified structure-level splits** with the following invariants:

- Stratification: `crystal_system + filtered_space_group` (fallback: crystal_system only)
- Ratios: 70% train / 15% val / 15% test
- No structure leakage across train/val/test
- No augmentation leakage (all augmentations of a structure stay in one split)
- Val and test sets are never reweighted or resampled
- Class balancing (WeightedRandomSampler, class-weighted loss) applied to train only
