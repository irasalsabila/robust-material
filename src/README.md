"""
XRD Crystallographic Classification - Phase 1 Scripts
======================================================

This directory contains production-ready scripts for dataset inspection,
manifest building, splitting, and validation.

All scripts are:
- Fully type-hinted
- Well-documented
- Production-style
- Robust against data corruption
- Leakage-aware

## Scripts

### 1. inspect_zips.py
Dataset inspection and discovery.

**Purpose**: 
  - Examine raw zip archive structure
  - Count and categorize file types
  - Detect metadata files
  - Extract structure ID patterns

**Usage**:
  python src/inspect_zips.py

**Outputs**:
  outputs/phase1/zip_inspection.json

**Key Features**:
  - Handles corrupted zip members gracefully
  - Reports sample filenames for verification
  - Detects metadata files (.csv, .json, .yaml)
  - Identifies structure identifiers

---

### 2. build_manifest.py
Structure-level manifest generation.

**Purpose**:
  - Extract structure-level metadata from annotations
  - Build structure registry with augmentation tracking
  - Compute dataset statistics
  - Group structures by space groups

**Usage**:
  python src/build_manifest.py

**Outputs**:
  outputs/phase1/manifest.json
  outputs/phase1/manifest_analysis.json

**Key Features**:
  - Reads all anno_train.csv and anno_val.csv files
  - Consolidates duplicate structure records
  - Tracks all augmentations per structure
  - Computes space group and label distributions
  - Identifies structure hierarchies

**Important**:
  - Output JSON is large (~100MB) due to manifest size
  - Contains 23,073 unique structures
  - Each structure has augmentation lists

---

### 3. split_dataset.py
Structure-level stratified splitting.

**Purpose**:
  - Create train/val/test splits at STRUCTURE level
  - Support stratification by space group
  - Generate scalable subsets (5%, 10%)
  - Ensure no leakage

**Usage**:
  python src/split_dataset.py

**Outputs**:
  outputs/phase1/split_full_dataset.json
  outputs/phase1/split_subset_5pct.json
  outputs/phase1/split_subset_10pct.json

**Key Features**:
  - Structure-level splitting (not pattern-level)
  - Stratified by top-10 space groups
  - Maintains 70/15/15 train/val/test ratio
  - Scalable subset generation
  - Reproducible with random seed

**Configuration**:
  - SplitConfig class for customization
  - Default: 70/15/15 with top-10 SG stratification
  - Random seed: 42

**Critical Design**:
  - All augmentations of structure go together
  - No structure appears in multiple splits
  - Patterns listed explicitly for traceability

---

### 4. validate_split.py
Split validation and leakage detection.

**Purpose**:
  - Verify no leakage between splits
  - Ensure augmentation consistency
  - Check split ratio compliance
  - Validate manifest references

**Usage**:
  python src/validate_split.py

**Outputs**:
  - Console report only
  - Exit code 0 if all valid, 1 if invalid

**Validation Checks**:
  1. No structure in multiple splits
  2. No pattern duplication
  3. All augmentations with structure
  4. All structures in manifest
  5. Split ratios within tolerance
  6. Data completeness

**Result**:
  ✓ ALL SPLITS ARE VALID - NO LEAKAGE DETECTED

---

## Pipeline Usage

### Full Pipeline (Sequential)
```bash
# 1. Inspect raw data
python src/inspect_zips.py

# 2. Build manifest
python src/build_manifest.py

# 3. Create splits
python src/split_dataset.py

# 4. Validate splits
python src/validate_split.py
```

### Individual Step
Each script is independent after manifest generation:
```bash
# Only need to run once:
python src/build_manifest.py

# Can re-run splitting with different configs:
# (Modify SplitConfig in split_dataset.py)
python src/split_dataset.py

# Validate independently:
python src/validate_split.py
```

---

## Configuration

### Modify Split Behavior
Edit `split_dataset.py`, class `SplitConfig`:

```python
config = SplitConfig(
    train_ratio=0.70,           # Train fraction
    val_ratio=0.15,             # Val fraction
    test_ratio=0.15,            # Test fraction
    stratify_by_space_group=True,  # Enable stratification
    top_k_space_groups=10,      # Top-k space groups to use
    random_seed=42              # For reproducibility
)
```

---

## Output Format

### manifest.json Structure
```json
{
  "total_structures": 23073,
  "structures": {
    "1522982": {
      "structure_id": "1522982",
      "domain": "D1",
      "label": 0,
      "formula": "Mn4 Ni8 Sn4",
      "space_group": "F-43m",
      "space_group_no": 216,
      "augmentations": ["1522982_1", "1522982_2", ..., "1522982_24"]
    },
    ...
  }
}
```

### split_*.json Structure
```json
{
  "split_name": "full_dataset",
  "num_structures": {"train": 16151, "val": 3461, "test": 3461, "total": 23073},
  "num_patterns": {"train": 387624, "val": 83064, "test": 83064, "total": 553752},
  "train_structures": ["1522982", "1525299", ...],
  "val_structures": [...],
  "test_structures": [...],
  "train_patterns": ["1522982_1", "1522982_2", ...],
  "val_patterns": [...],
  "test_patterns": [...],
  "stratification_info": {
    "train": {"14": {"count": 1190, "name": "P2_1/c", "pct": 7.4}, ...},
    "val": {...},
    "test": {...}
  }
}
```

---

## Performance Notes

### Execution Times (on M1 MacBook Pro)
- `inspect_zips.py`: ~30 seconds
- `build_manifest.py`: ~30 seconds
- `split_dataset.py`: <1 second
- `validate_split.py`: <1 second

### Memory Usage
- Manifest loading: ~500MB
- Split creation: <100MB
- Validation: <100MB

### Data Size
- Input zips: ~50MB each
- manifest.json: ~100MB
- split_*.json: 5-50MB each

---

## Error Handling

All scripts include:
- Type validation
- Missing file detection
- Corrupted data handling
- Graceful error reporting
- Comprehensive logging

Example error messages:
```
ERROR: Data directory not found: /path/to/data
ERROR: Manifest file not found: /path/to/manifest.json
ERROR: BadZipFile - corrupted archive detected
WARNING: Structure 12345 in split but not in manifest
```

---

## Dependencies

### Python Packages
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Stratified splitting
- zipfile: Archive handling (stdlib)
- json: Data serialization (stdlib)
- logging: Logging (stdlib)

### No External Dependencies Required For:
- Pure Python operations
- Zip file inspection
- JSON serialization
- Dataset validation

---

## Best Practices

1. **Always validate after splitting**
   ```bash
   python src/validate_split.py
   ```

2. **Use reproducible seeds**
   - Set `random_seed` in SplitConfig
   - Default: 42

3. **Check manifest before splitting**
   - Verify structure counts
   - Check space group distribution
   - Ensure data completeness

4. **Save split configs**
   - Document SplitConfig used
   - Record random seed
   - Include in experiment notes

---

## Troubleshooting

### Issue: "No annotation files found"
**Cause**: Zips don't contain anno_train.csv or anno_val.csv
**Solution**: Verify zip contents, check file paths

### Issue: "Manifest is very large"
**Cause**: 23,073 structures × augmentation data
**Solution**: Normal - split JSON into chunks if needed

### Issue: Split validation shows leakage
**Cause**: Likely a code bug (shouldn't happen with this implementation)
**Solution**: Check manifest integrity first, then split creation

---

## References

- Dataset Format: See `outputs/phase1/PHASE1_SUMMARY.md`
- Manifest Schema: Built from anno_train.csv/anno_val.csv
- Split Ratios: 70/15/15 (configurable)
- Stratification: Space group number (top-10)

---

## Author Notes

This implementation prioritizes:
- **Leakage Prevention**: Structure-level splitting enforced
- **Transparency**: All operations logged and traceable
- **Scalability**: Supports subset generation for experimentation
- **Robustness**: Handles corrupted data and edge cases
- **Production Quality**: Type hints, comprehensive validation, clear errors
"""

