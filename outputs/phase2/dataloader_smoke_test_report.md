# Phase 2 DataLoader Smoke Test Report

**Domain:** D1  |  **Scale:** 10%  |  **Task:** crystal_system
**Normalization:** max_intensity  |  **Target length:** 4500
**Batch size:** 64  |  **Setup time:** 0.02s

## Leakage Check

Status: **CLEAN — no structure-level leakage detected**

## Split Overview

| Split | Samples | Batches | x shape | y shape | x min | x max |
|-------|---------|---------|---------|---------|-------|-------|
| train | 38,736 | 606 | [64, 1, 4500] | [64] | -0.0108 | 1.0000 |
| val | 8,304 | 130 | [64, 1, 4500] | [64] | -0.0097 | 1.0000 |
| test | 8,328 | 131 | [64, 1, 4500] | [64] | -0.0116 | 1.0000 |

## Label Distribution

### Train

| Class | Count | % |
|-------|-------|---|
| cubic (0) | 7,968 | 20.6% |
| monoclinic (1) | 5,400 | 13.9% |
| orthorhombic (2) | 8,640 | 22.3% |
| triclinic (3) | 288 | 0.7% |
| hexagonal (4) | 7,968 | 20.6% |
| tetragonal (5) | 6,576 | 17.0% |
| trigonal (6) | 1,896 | 4.9% |

### Val

| Class | Count | % |
|-------|-------|---|
| cubic (0) | 1,704 | 20.5% |
| monoclinic (1) | 1,152 | 13.9% |
| orthorhombic (2) | 1,848 | 22.3% |
| triclinic (3) | 72 | 0.9% |
| hexagonal (4) | 1,704 | 20.5% |
| tetragonal (5) | 1,416 | 17.1% |
| trigonal (6) | 408 | 4.9% |

### Test

| Class | Count | % |
|-------|-------|---|
| cubic (0) | 1,728 | 20.7% |
| monoclinic (1) | 1,152 | 13.8% |
| orthorhombic (2) | 1,848 | 22.2% |
| triclinic (3) | 48 | 0.6% |
| hexagonal (4) | 1,728 | 20.7% |
| tetragonal (5) | 1,416 | 17.0% |
| trigonal (6) | 408 | 4.9% |

## Class Weights (train split)

| Index | Class | Weight |
|-------|-------|--------|
| 0 | cubic | 0.1868 |
| 1 | monoclinic | 0.2756 |
| 2 | orthorhombic | 0.1722 |
| 3 | triclinic | 5.1674 |
| 4 | hexagonal | 0.1868 |
| 5 | tetragonal | 0.2263 |
| 6 | trigonal | 0.7849 |

## Sampler

WeightedRandomSampler: **ACTIVE** for train split.
Val and test always use sequential (unweighted) sampling.

## Warnings

None.

## Verdict

**PASS**
