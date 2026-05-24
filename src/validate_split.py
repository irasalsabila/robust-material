"""
Phase 1.5: Split Validation (Multi-Domain)
==========================================
Validates all split_*.json files in outputs/phase1/splits/ for:
  1. No structure leakage across train/val/test
  2. Every structure belongs to exactly one split
  3. All augmentations for selected structures are included
  4. Expected augmentation count (24 per structure)
  5. Missing/extra augmentation reporting
  6. Train/val/test ratio verification
  7. Crystal-system distribution preservation
  8. Filtered SG distribution preservation
  9. Chi-square p-value for crystal system
 10. Chi-square p-value for filtered SG
 11. Jensen-Shannon divergence for train vs val and train vs test

Outputs:
  outputs/phase1/validation/validation_report.json

Exit code: 0 if all valid, 1 if any invalid.
"""

import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EXPECTED_AUG_PER_STRUCTURE = 24
RATIO_TOLERANCE = 0.05


@dataclass
class ValidationResult:
    split_name: str
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.is_valid = False
        self.errors.append(msg)
        logger.error("  [FAIL] %s: %s", self.split_name, msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("  [WARN] %s: %s", self.split_name, msg)


class SplitValidator:
    def __init__(self, manifest_path: Path):
        self.manifest = self._load_manifest(manifest_path)

    def _load_manifest(self, path: Path) -> Dict:
        logger.info("Loading manifest from %s", path)
        with open(path, "r") as f:
            data = json.load(f)
        return data["structures"]

    # --- Statistical helpers ---

    def _chi2(self, observed: List[int], expected: List[int]) -> Dict:
        obs = [float(x) for x in observed]
        exp = [max(float(x), 1e-8) for x in expected]
        chi2 = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
        df = max(len(obs) - 1, 1)
        p_value = None
        try:
            from scipy.stats import chi2 as scipy_chi2
            p_value = float(scipy_chi2.sf(chi2, df))
        except Exception:
            pass
        return {"chi2": round(chi2, 4), "df": df, "p_value": p_value}

    def _jsd(self, counts_a: List[int], counts_b: List[int]) -> float:
        sa, sb = sum(counts_a), sum(counts_b)
        if sa == 0 or sb == 0:
            return float("nan")
        pa = [x / sa for x in counts_a]
        pb = [x / sb for x in counts_b]
        m = [(a + b) / 2.0 for a, b in zip(pa, pb)]

        def kl(p, q):
            return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)

        return round(0.5 * kl(pa, m) + 0.5 * kl(pb, m), 6)

    def _category_counts(self, sids: List[str], key: str) -> Dict[str, int]:
        if key == "crystal_system":
            counts = Counter(
                self.manifest[sid]["crystal_sys"] for sid in sids if sid in self.manifest
            )
            return {str(k): int(v) for k, v in sorted(counts.items())}
        elif key == "space_group":
            counts = Counter(
                self.manifest[sid]["space_group_no"] for sid in sids if sid in self.manifest
            )
            return {str(k): int(v) for k, v in counts.most_common()}
        raise ValueError(f"Unknown key: {key}")

    def _compare_distributions(self, base: Dict[str, int], comp: Dict[str, int]) -> Dict:
        cats = sorted(set(base) | set(comp))
        obs = [base.get(c, 0) for c in cats]
        exp = [comp.get(c, 0) for c in cats]
        return {
            "chi2": self._chi2(obs, exp),
            "js_divergence": self._jsd(obs, exp),
        }

    # --- Core validation ---

    def validate_split(self, split_path: Path) -> ValidationResult:
        with open(split_path, "r") as f:
            sd = json.load(f)

        split_name = sd.get("split_name", split_path.stem)
        result = ValidationResult(split_name=split_name)

        train_s = sd.get("train_structures", [])
        val_s = sd.get("val_structures", [])
        test_s = sd.get("test_structures", [])
        train_p = sd.get("train_patterns", [])
        val_p = sd.get("val_patterns", [])
        test_p = sd.get("test_patterns", [])

        ts, vs, es = set(train_s), set(val_s), set(test_s)
        all_s = ts | vs | es
        total_s = len(all_s)
        total_p = len(train_p) + len(val_p) + len(test_p)

        domain = sd.get("domain", "")

        # 1. Basic checks
        if total_s == 0:
            result.fail("Zero structures.")
            return result

        # 2. Structure leakage
        for label, overlap in [("train/val", ts & vs), ("train/test", ts & es), ("val/test", vs & es)]:
            if overlap:
                result.fail(f"Structure leakage {label}: {len(overlap)} overlapping.")

        # 3. Pattern leakage
        tp, vp, ep = set(train_p), set(val_p), set(test_p)
        for label, overlap in [("train/val", tp & vp), ("train/test", tp & ep), ("val/test", vp & ep)]:
            if overlap:
                result.fail(f"Pattern leakage {label}: {len(overlap)} overlapping.")

        # 4. Manifest membership
        missing = [sid for sid in all_s if sid not in self.manifest]
        if missing:
            result.fail(f"{len(missing)} structures not in manifest.")

        # 5. Augmentation completeness
        # Pattern IDs are domain-qualified: "D1:1522982_1"
        # Strip domain prefix to check against manifest
        aug_errors = 0
        split_parts = {"train": tp, "val": vp, "test": ep}
        struct_to_split = {}
        for sname, sset in [("train", ts), ("val", vs), ("test", es)]:
            for sid in sset:
                struct_to_split[sid] = sname

        for sid in sorted(all_s):
            if sid not in self.manifest:
                continue
            declared_raw = set(self.manifest[sid]["augmentations"])
            # Expected domain-qualified patterns
            expected = {f"{domain}:{aug}" for aug in declared_raw}
            split_label = struct_to_split.get(sid, "train")
            found = expected & split_parts.get(split_label, set())
            missing_augs = expected - found

            if len(declared_raw) != EXPECTED_AUG_PER_STRUCTURE:
                result.warn(f"Structure {sid} has {len(declared_raw)} augs (expected {EXPECTED_AUG_PER_STRUCTURE}).")

            if missing_augs:
                aug_errors += 1

        if aug_errors > 0:
            result.fail(f"Augmentation incomplete: {aug_errors} structures have missing patterns.")

        # 6. Ratio checks
        train_ratio = len(train_s) / total_s
        val_ratio = len(val_s) / total_s
        test_ratio = len(test_s) / total_s

        if abs(train_ratio - 0.70) > RATIO_TOLERANCE:
            result.warn(f"Train ratio {train_ratio:.1%} deviates from 70%.")
        if abs(val_ratio - 0.15) > RATIO_TOLERANCE:
            result.warn(f"Val ratio {val_ratio:.1%} deviates from 15%.")
        if abs(test_ratio - 0.15) > RATIO_TOLERANCE:
            result.warn(f"Test ratio {test_ratio:.1%} deviates from 15%.")

        # 7-11. Distribution checks
        dist_stats: Dict = {}
        for cat_key in ["crystal_system", "space_group"]:
            train_counts = self._category_counts(train_s, cat_key)
            val_counts = self._category_counts(val_s, cat_key)
            test_counts = self._category_counts(test_s, cat_key)

            all_cats = sorted(set(train_counts) | set(val_counts) | set(test_counts))
            train_vec = [train_counts.get(c, 0) for c in all_cats]
            val_vec = [val_counts.get(c, 0) for c in all_cats]
            test_vec = [test_counts.get(c, 0) for c in all_cats]

            tv = self._compare_distributions(
                dict(zip(all_cats, train_vec)), dict(zip(all_cats, val_vec))
            )
            tt = self._compare_distributions(
                dict(zip(all_cats, train_vec)), dict(zip(all_cats, test_vec))
            )

            dist_stats[cat_key] = {
                "train": train_counts,
                "val": val_counts,
                "test": test_counts,
                "train_vs_val": tv,
                "train_vs_test": tt,
            }

        # Assemble stats
        result.stats = {
            "split_name": split_name,
            "domain": domain,
            "scale_pct": sd.get("scale_pct"),
            "total_structures": total_s,
            "total_patterns": total_p,
            "train_structures": len(train_s),
            "val_structures": len(val_s),
            "test_structures": len(test_s),
            "train_ratio": round(train_ratio, 4),
            "val_ratio": round(val_ratio, 4),
            "test_ratio": round(test_ratio, 4),
            "augmentation_errors": aug_errors,
            "distribution_stats": dist_stats,
        }

        status = "VALID" if result.is_valid else "INVALID"
        logger.info("  %s: %s (errors=%d warnings=%d)", split_name, status, len(result.errors), len(result.warnings))
        return result

    def validate_all(self, split_dir: Path) -> List[ValidationResult]:
        files = sorted(split_dir.glob("split_*.json"))
        if not files:
            logger.error("No split files in %s", split_dir)
            return []
        logger.info("Found %d split files.", len(files))
        return [self.validate_split(f) for f in files]


def print_report(results: List[ValidationResult]) -> None:
    print(f"\n{'='*72}")
    print("PHASE 1.5: SPLIT VALIDATION REPORT")
    print(f"{'='*72}")
    all_valid = all(r.is_valid for r in results)
    print(f"  Splits checked: {len(results)}")
    print(f"  Valid: {sum(1 for r in results if r.is_valid)}")
    print(f"  Invalid: {sum(1 for r in results if not r.is_valid)}")
    print(f"  Overall: {'PASS' if all_valid else 'FAIL'}")

    for r in results:
        s = r.stats
        ds = s.get("distribution_stats", {})
        cs_tv = ds.get("crystal_system", {}).get("train_vs_val", {})
        cs_tt = ds.get("crystal_system", {}).get("train_vs_test", {})
        sg_tv = ds.get("space_group", {}).get("train_vs_val", {})
        sg_tt = ds.get("space_group", {}).get("train_vs_test", {})

        status = "VALID" if r.is_valid else "INVALID"
        print(f"\n  {r.split_name}: {status}")
        print(f"    domain={s.get('domain')} scale={s.get('scale_pct')}%")
        print(f"    structs={s['total_structures']} patterns={s['total_patterns']}")
        print(f"    ratios: train={s['train_ratio']:.1%} val={s['val_ratio']:.1%} test={s['test_ratio']:.1%}")
        print(f"    aug_errors={s['augmentation_errors']}")
        print(f"    CS JSD: tv={cs_tv.get('js_divergence','?'):.4f} tt={cs_tt.get('js_divergence','?'):.4f}")
        print(f"    SG JSD: tv={sg_tv.get('js_divergence','?'):.4f} tt={sg_tt.get('js_divergence','?'):.4f}")

        if r.errors:
            for e in r.errors[:3]:
                print(f"    [ERROR] {e}")
        if r.warnings:
            for w in r.warnings[:3]:
                print(f"    [WARN] {w}")
            if len(r.warnings) > 3:
                print(f"    ... and {len(r.warnings)-3} more warnings")

    print(f"\n{'='*72}")
    print(f"{'ALL VALID' if all_valid else 'VALIDATION FAILED'}")
    print(f"{'='*72}\n")


def save_report(results: List[ValidationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "split_name": r.split_name,
            "is_valid": r.is_valid,
            "errors": r.errors,
            "warnings": r.warnings,
            "stats": r.stats,
        }
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved validation report to %s", output_path)


def main() -> int:
    project_root = Path(__file__).parent.parent
    manifest_path = project_root / "outputs" / "phase1" / "manifest.json"
    split_dir = project_root / "outputs" / "phase1" / "splits"
    output_path = project_root / "outputs" / "phase1" / "validation" / "validation_report.json"

    validator = SplitValidator(manifest_path)
    results = validator.validate_all(split_dir)

    if not results:
        logger.error("No splits found.")
        return 1

    print_report(results)
    save_report(results, output_path)

    all_valid = all(r.is_valid for r in results)
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())