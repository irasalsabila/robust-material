"""
Phase 1.5: Multi-Domain Structure-Level Dataset Splitting
=========================================================
Produces leakage-safe mini-datasets for every domain x scale combination.

Domains : D1 (clean), D2 (background), D3 (noise1), D4 (noise2)
Scales  : 5%, 10%, 15%, 20%
Splits  : 70% train / 15% val / 15% test  (structure-level)

Key design decisions
--------------------
* All four domains share the same 23,073 structure IDs and metadata.
  Domain = measurement condition, not a different crystal set.
* Splitting is done on the shared structure pool; the resulting structure-ID
  partition is then applied independently to each domain's pattern IDs.
* Pattern IDs are domain-qualified: "D1:1522982_1", "D2:1522982_1", etc.
* Stratification: crystal_sys + filtered SG (min_sg_freq threshold).
  Falls back to crystal_sys-only if any stratum < 2 members.
* SG frequency threshold is chosen per scale by testing [50,100,150,200]
  and picking the largest stable threshold.

Outputs
-------
    outputs/dataset_splits/splits/split_{DOMAIN}_{SCALE}pct.json   (28 files)
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DOMAINS = ["D1", "D2", "D3", "D4"]
SCALES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
SG_FREQ_THRESHOLDS = [50, 100, 150, 200]

CRYSTAL_SYSTEM_NAMES: Dict[str, str] = {
    "0": "cubic",
    "1": "monoclinic",
    "2": "orthorhombic",
    "3": "triclinic",
    "4": "hexagonal",
    "5": "tetragonal",
    "6": "trigonal",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


@dataclass
class SplitResult:
    split_name: str
    domain: str
    scale_pct: float
    selected_structure_count: int
    selected_pattern_count: int
    train_structures: List[str] = field(default_factory=list)
    val_structures: List[str] = field(default_factory=list)
    test_structures: List[str] = field(default_factory=list)
    train_patterns: List[str] = field(default_factory=list)
    val_patterns: List[str] = field(default_factory=list)
    test_patterns: List[str] = field(default_factory=list)
    crystal_system_distribution: Dict = field(default_factory=dict)
    top_space_group_distribution: Dict = field(default_factory=dict)
    sg_threshold_analysis: Dict = field(default_factory=dict)
    augmentation_summary: Dict = field(default_factory=dict)
    leakage_summary: Dict = field(default_factory=dict)
    stratification_method: str = ""
    chosen_sg_threshold: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        total_s = (len(self.train_structures) + len(self.val_structures)
                   + len(self.test_structures))
        total_p = (len(self.train_patterns) + len(self.val_patterns)
                   + len(self.test_patterns))
        return {
            "split_name": self.split_name,
            "domain": self.domain,
            "scale_pct": float(self.scale_pct),
            "selected_structure_count": int(self.selected_structure_count),
            "selected_pattern_count": int(self.selected_pattern_count),
            "split_ratios": {
                "train": round(len(self.train_structures) / total_s, 4) if total_s else 0.0,
                "val": round(len(self.val_structures) / total_s, 4) if total_s else 0.0,
                "test": round(len(self.test_structures) / total_s, 4) if total_s else 0.0,
            },
            "num_structures": {
                "train": len(self.train_structures),
                "val": len(self.val_structures),
                "test": len(self.test_structures),
                "total": total_s,
            },
            "num_patterns": {
                "train": len(self.train_patterns),
                "val": len(self.val_patterns),
                "test": len(self.test_patterns),
                "total": total_p,
            },
            "train_structures": self.train_structures,
            "val_structures": self.val_structures,
            "test_structures": self.test_structures,
            "train_patterns": self.train_patterns,
            "val_patterns": self.val_patterns,
            "test_patterns": self.test_patterns,
            "crystal_system_distribution": self.crystal_system_distribution,
            "top_space_group_distribution": self.top_space_group_distribution,
            "sg_threshold_analysis": self.sg_threshold_analysis,
            "augmentation_summary": self.augmentation_summary,
            "leakage_summary": self.leakage_summary,
            "stratification_method": self.stratification_method,
            "chosen_sg_threshold": int(self.chosen_sg_threshold),
            "warnings": self.warnings,
        }


class MultiDomainSplitter:
    """
    Produces leakage-safe splits for every domain x scale combination.

    All domains share the same 23,073 structure IDs and metadata.
    We compute one structure-level partition per scale and apply it to
    all domains. Pattern IDs are domain-qualified to keep them distinct.
    """

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.structures: Dict = self._load_manifest()
        self._domain_augs: Dict[str, Dict[str, List[str]]] = (
            self._build_domain_augmentations()
        )

    def _load_manifest(self) -> Dict:
        logger.info("Loading manifest from %s", self.manifest_path)
        with open(self.manifest_path, "r") as f:
            data = json.load(f)
        structs = data["structures"]
        logger.info("Manifest loaded: %d structures", len(structs))
        return structs

    def _build_domain_augmentations(self) -> Dict[str, Dict[str, List[str]]]:
        """Build domain-qualified augmentation lists. Format: '{DOMAIN}:{aug_id}'"""
        result: Dict[str, Dict[str, List[str]]] = {}
        for domain in DOMAINS:
            domain_augs: Dict[str, List[str]] = {}
            for sid, info in self.structures.items():
                orig_augs = info["augmentations"]
                domain_augs[sid] = [f"{domain}:{aug}" for aug in orig_augs]
            result[domain] = domain_augs
        return result

    # --- SG threshold helpers ---

    def _sg_threshold_analysis(self, structure_ids: List[str]) -> Dict:
        """Report SG class survival at each threshold for the sampled pool."""
        sg_counts = Counter(
            self.structures[sid]["space_group_no"] for sid in structure_ids
        )
        analysis: Dict = {}
        for thresh in SG_FREQ_THRESHOLDS:
            surviving = {sg: c for sg, c in sg_counts.items() if c >= thresh}
            min_val = int(min(surviving.values()) * VAL_RATIO) if surviving else 0
            analysis[str(thresh)] = {
                "surviving_sg_count": len(surviving),
                "min_sg_freq_in_sample": int(min(surviving.values())) if surviving else 0,
                "max_sg_freq_in_sample": int(max(surviving.values())) if surviving else 0,
                "min_expected_val_count": min_val,
                "stable": min_val >= 1 and len(surviving) >= 2,
            }
        return analysis

    def _choose_sg_threshold(
        self, structure_ids: List[str]
    ) -> Tuple[int, List[int]]:
        """Choose the largest stable SG threshold (val gets >=1 per SG)."""
        sg_counts = Counter(
            self.structures[sid]["space_group_no"] for sid in structure_ids
        )
        chosen = SG_FREQ_THRESHOLDS[0]
        for thresh in reversed(SG_FREQ_THRESHOLDS):
            surviving = {sg: c for sg, c in sg_counts.items() if c >= thresh}
            if not surviving:
                continue
            min_val = int(min(surviving.values()) * VAL_RATIO)
            if min_val >= 1 and len(surviving) >= 2:
                chosen = thresh
                break
        surviving_sgs = [sg for sg, c in sg_counts.items() if c >= chosen]
        return chosen, surviving_sgs

    # --- Stratification ---

    def _build_strata(
        self,
        structure_ids: List[str],
        surviving_sgs: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, str]:
        """Build stratification labels. Falls back to crystal_sys only if needed."""
        if surviving_sgs is None:
            _, surviving_sgs = self._choose_sg_threshold(structure_ids)

        sg_set = set(surviving_sgs)
        combined = []
        for sid in structure_ids:
            cs = self.structures[sid]["crystal_sys"]
            sg = self.structures[sid]["space_group_no"]
            sg_bin = sg if sg in sg_set else -1
            combined.append(f"{cs}_{sg_bin}")

        counts = Counter(combined)
        if any(c < 2 for c in counts.values()):
            small = {lbl: c for lbl, c in counts.items() if c < 2}
            logger.warning(
                "Combined strata has %d bins with <2 members: %s "
                "-- falling back to crystal_sys only.",
                len(small), small,
            )
            cs_labels = [
                str(self.structures[sid]["crystal_sys"])
                for sid in structure_ids
            ]
            return np.array(cs_labels), "crystal_system_only (fallback)"

        return (
            np.array(combined),
            f"crystal_system_plus_filtered_sg",
        )

    # --- Splitting helpers ---

    def _safe_split(
        self, ids: List[str], strata: np.ndarray, test_size: float, seed: int,
    ) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        if len(ids) < 2:
            return ids, [], strata, np.array([], dtype=object)
        counts = Counter(strata)
        if len(counts) > 1 and all(c >= 2 for c in counts.values()):
            try:
                a, b, sa, sb = train_test_split(
                    ids, strata, test_size=test_size,
                    stratify=strata, random_state=seed,
                )
                return list(a), list(b), np.array(sa), np.array(sb)
            except ValueError as exc:
                logger.warning("Stratified split failed (%s); using random.", exc)
        a, b = train_test_split(ids, test_size=test_size, random_state=seed)
        id2s = dict(zip(ids, strata))
        return (
            list(a), list(b),
            np.array([id2s[x] for x in a], dtype=object),
            np.array([id2s[x] for x in b], dtype=object),
        )

    def _three_way_split(
        self, ids: List[str], strata: np.ndarray
    ) -> Tuple[List[str], List[str], List[str]]:
        non_train = VAL_RATIO + TEST_RATIO
        train, temp, _, s_temp = self._safe_split(
            ids, strata, test_size=non_train, seed=RANDOM_SEED,
        )
        if len(temp) < 2:
            return sorted(train), sorted(temp), []
        val_frac = VAL_RATIO / non_train
        val, test, _, _ = self._safe_split(
            temp, s_temp, test_size=1.0 - val_frac, seed=RANDOM_SEED,
        )
        return sorted(train), sorted(val), sorted(test)

    # --- Subset sampling ---

    def _sample_subset(
        self, all_ids: List[str], fraction: float
    ) -> Tuple[List[str], List[str]]:
        warnings_list: List[str] = []
        n = max(1, int(round(len(all_ids) * fraction)))
        if n >= len(all_ids):
            return sorted(all_ids), warnings_list

        _, surviving_sgs = self._choose_sg_threshold(all_ids)
        strata, _ = self._build_strata(all_ids, surviving_sgs)
        counts = Counter(strata)

        if any(c < 2 for c in counts.values()):
            msg = (
                f"Subset sampling at {fraction*100:.0f}%: some strata <2 "
                "-- falling back to random sampling."
            )
            logger.warning(msg)
            warnings_list.append(msg)
            sampled, _ = train_test_split(
                all_ids, train_size=n, random_state=RANDOM_SEED,
            )
            return sorted(sampled), warnings_list

        try:
            sampled, _, _, _ = train_test_split(
                all_ids, strata,
                train_size=n, stratify=strata, random_state=RANDOM_SEED,
            )
            return sorted(sampled), warnings_list
        except ValueError as exc:
            msg = f"Stratified sampling failed ({exc}); using random."
            logger.warning(msg)
            warnings_list.append(msg)
            sampled, _ = train_test_split(
                all_ids, train_size=n, random_state=RANDOM_SEED,
            )
            return sorted(sampled), warnings_list

    # --- Distribution summaries ---

    def _cs_distribution(
        self, structure_ids: List[str], split_map: Dict[str, List[str]]
    ) -> Dict:
        all_cs = sorted(set(
            self.structures[sid]["crystal_sys"] for sid in structure_ids
        ))
        result: Dict = {}
        for cs in all_cs:
            entry: Dict = {
                "name": CRYSTAL_SYSTEM_NAMES.get(str(cs), str(cs)),
                "total": 0,
            }
            for sname, ids in split_map.items():
                cnt = sum(1 for sid in ids if self.structures[sid]["crystal_sys"] == cs)
                entry[sname] = int(cnt)
                entry["total"] += cnt
            entry["total"] = int(entry["total"])
            result[str(cs)] = entry
        return result

    def _sg_distribution(
        self, structure_ids: List[str], split_map: Dict[str, List[str]],
        surviving_sgs: List[int],
    ) -> Dict:
        sg_counts = Counter(
            self.structures[sid]["space_group_no"] for sid in structure_ids
        )
        result: Dict = {}
        for sg_no in sorted(surviving_sgs):
            sg_name = next(
                (self.structures[sid]["space_group"]
                 for sid in structure_ids
                 if self.structures[sid]["space_group_no"] == sg_no),
                "",
            )
            entry: Dict = {"name": sg_name, "total": int(sg_counts.get(sg_no, 0))}
            for sname, ids in split_map.items():
                cnt = sum(1 for sid in ids if self.structures[sid]["space_group_no"] == sg_no)
                entry[sname] = int(cnt)
            result[str(sg_no)] = entry
        return result

    # --- Augmentation and leakage ---

    def _gather_patterns(self, structure_ids: List[str], domain: str) -> List[str]:
        domain_augs = self._domain_augs[domain]
        patterns: List[str] = []
        for sid in structure_ids:
            patterns.extend(domain_augs[sid])
        return sorted(patterns)

    def _aug_summary(
        self, structure_ids: List[str], patterns: List[str], domain: str, label: str,
    ) -> Dict:
        domain_augs = self._domain_augs[domain]
        pat_set = set(patterns)
        missing_count = 0
        structs_missing = 0
        for sid in structure_ids:
            expected = set(domain_augs[sid])
            missing = expected - pat_set
            if missing:
                structs_missing += 1
                missing_count += len(missing)
        expected_total = sum(len(domain_augs[sid]) for sid in structure_ids)
        return {
            "split": label,
            "structure_count": int(len(structure_ids)),
            "expected_pattern_count": int(expected_total),
            "found_pattern_count": int(len(patterns)),
            "missing_pattern_count": int(missing_count),
            "structures_with_missing": int(structs_missing),
            "is_complete": missing_count == 0,
        }

    def _leakage_summary(
        self,
        train_s: List[str], val_s: List[str], test_s: List[str],
        train_p: List[str], val_p: List[str], test_p: List[str],
    ) -> Dict:
        ts, vs, es = set(train_s), set(val_s), set(test_s)
        tp, vp, ep = set(train_p), set(val_p), set(test_p)
        overlaps = {
            "train_val_structures": int(len(ts & vs)),
            "train_test_structures": int(len(ts & es)),
            "val_test_structures": int(len(vs & es)),
            "train_val_patterns": int(len(tp & vp)),
            "train_test_patterns": int(len(tp & ep)),
            "val_test_patterns": int(len(vp & ep)),
        }
        return {
            "is_leakage_free": all(v == 0 for v in overlaps.values()),
            "overlaps": overlaps,
        }

    # --- Public API ---

    def create_split(self, domain: str, fraction: float) -> SplitResult:
        """Create one domain x scale split."""
        pct = int(round(fraction * 100))
        split_name = f"split_{domain}_{pct}pct"
        logger.info("Creating %s ...", split_name)

        all_ids = sorted(self.structures.keys())
        sampled_ids, sample_warnings = self._sample_subset(all_ids, fraction)

        chosen_thresh, surviving_sgs = self._choose_sg_threshold(sampled_ids)
        strata, strat_method = self._build_strata(sampled_ids, surviving_sgs)

        if "fallback" in strat_method:
            sample_warnings.append(
                f"Stratification fell back to crystal_sys only for {split_name}."
            )

        sg_thresh_analysis = self._sg_threshold_analysis(sampled_ids)
        train_ids, val_ids, test_ids = self._three_way_split(sampled_ids, strata)

        train_patterns = self._gather_patterns(train_ids, domain)
        val_patterns = self._gather_patterns(val_ids, domain)
        test_patterns = self._gather_patterns(test_ids, domain)
        all_patterns = self._gather_patterns(sampled_ids, domain)

        split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        cs_dist = self._cs_distribution(sampled_ids, split_map)
        sg_dist = self._sg_distribution(sampled_ids, split_map, surviving_sgs)

        aug_summary = {
            "overall": self._aug_summary(sampled_ids, all_patterns, domain, "overall"),
            "train": self._aug_summary(train_ids, train_patterns, domain, "train"),
            "val": self._aug_summary(val_ids, val_patterns, domain, "val"),
            "test": self._aug_summary(test_ids, test_patterns, domain, "test"),
        }

        leakage = self._leakage_summary(
            train_ids, val_ids, test_ids,
            train_patterns, val_patterns, test_patterns,
        )

        result = SplitResult(
            split_name=split_name,
            domain=domain,
            scale_pct=float(pct),
            selected_structure_count=len(sampled_ids),
            selected_pattern_count=len(all_patterns),
            train_structures=train_ids,
            val_structures=val_ids,
            test_structures=test_ids,
            train_patterns=train_patterns,
            val_patterns=val_patterns,
            test_patterns=test_patterns,
            crystal_system_distribution=cs_dist,
            top_space_group_distribution=sg_dist,
            sg_threshold_analysis=sg_thresh_analysis,
            augmentation_summary=aug_summary,
            leakage_summary=leakage,
            stratification_method=strat_method,
            chosen_sg_threshold=chosen_thresh,
            warnings=sample_warnings,
        )

        logger.info(
            "  %s: %d structs (train=%d val=%d test=%d) patterns=%d "
            "leakage_free=%s strat=%s sg_thresh=%d",
            split_name, len(sampled_ids), len(train_ids), len(val_ids),
            len(test_ids), len(all_patterns), leakage["is_leakage_free"],
            strat_method, chosen_thresh,
        )
        for w in sample_warnings:
            logger.warning("  [WARN] %s", w)

        return result


def save_split(result: SplitResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{result.split_name}.json"
    with open(out, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Saved %s", out)
    return out


def print_split_report(result: SplitResult) -> None:
    d = result.to_dict()
    ns = d["num_structures"]
    np_ = d["num_patterns"]
    lk = d["leakage_summary"]
    aug = d["augmentation_summary"]["overall"]
    print(f"\n{'='*68}")
    print(f"  {result.split_name}  |  domain={result.domain}  scale={result.scale_pct:.0f}%")
    print(f"{'='*68}")
    print(f"  Structures : total={ns['total']}  train={ns['train']}  val={ns['val']}  test={ns['test']}")
    print(f"  Patterns   : total={np_['total']}  train={np_['train']}  val={np_['val']}  test={np_['test']}")
    print(f"  Ratios     : train={d['split_ratios']['train']:.1%}  val={d['split_ratios']['val']:.1%}  test={d['split_ratios']['test']:.1%}")
    print(f"  Strat      : {result.stratification_method}")
    print(f"  SG thresh  : {result.chosen_sg_threshold}  ({len(d['top_space_group_distribution'])} SGs)")
    print(f"  Leakage    : {'CLEAN' if lk['is_leakage_free'] else 'DETECTED'}")
    print(f"  Aug check  : {'OK' if aug['is_complete'] else 'INCOMPLETE'} (missing={aug['missing_pattern_count']})")
    for w in result.warnings:
        print(f"  WARN: {w}")


def main() -> None:
    project_root = Path(__file__).parent.parent
    manifest_path = project_root / "outputs" / "phase1" / "manifest.json"
    output_dir = project_root / "outputs" / "phase1" / "splits"

    splitter = MultiDomainSplitter(manifest_path)

    total = len(DOMAINS) * len(SCALES)
    done = 0
    for domain in DOMAINS:
        for fraction in SCALES:
            result = splitter.create_split(domain, fraction)
            save_split(result, output_dir)
            print_split_report(result)
            done += 1
            logger.info("Progress: %d / %d", done, total)

    logger.info("All %d splits complete.", total)


if __name__ == "__main__":
    main()