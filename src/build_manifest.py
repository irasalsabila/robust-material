"""
Phase 1: Build Dataset Manifest
Objective: Create structure-level manifest for leakage-safe splitting.
"""

import json
import logging
from collections import defaultdict, Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple
from zipfile import ZipFile

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class StructureInfo:
    """Information about a unique crystal structure."""
    structure_id: str
    domain: str
    label: int
    formula: str
    symbol_set: str
    space_group: str
    space_group_no: int
    crystal_sys: str
    lattice_a: float
    lattice_b: float
    lattice_c: float
    lattice_alpha: float
    lattice_beta: float
    lattice_gamma: float
    augmentations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


class ManifestBuilder:
    """Builds dataset manifest from annotation files."""
    
    def __init__(self, data_dir: Path):
        """Initialize with data directory."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def extract_structure_id(self, data_id: str) -> str:
        """
        Extract structure ID from dataId.
        
        Format: STRUCTURE_ID_AUGMENTATION
        Example: "1522982_1" -> "1522982"
        """
        parts = data_id.rsplit('_', 1)
        return parts[0] if len(parts) == 2 else data_id
    
    def read_annotations_from_zip(
        self,
        zip_path: Path
    ) -> pd.DataFrame:
        """Read all annotation files from a domain zip."""
        records = []
        domain = zip_path.stem  # e.g., "D1"
        
        try:
            with ZipFile(zip_path, 'r') as z:
                for anno_file in ['anno_train.csv', 'anno_val.csv']:
                    try:
                        df = pd.read_csv(z.open(anno_file))
                        df['domain'] = domain
                        df['annotation_file'] = anno_file
                        records.append(df)
                        logger.info(
                            f"✓ {domain}.zip/{anno_file}: "
                            f"{len(df)} patterns, "
                            f"{df['dataId'].nunique()} unique IDs"
                        )
                    except KeyError:
                        logger.warning(f"  {anno_file} not found in {domain}.zip")
        
        except Exception as e:
            logger.error(f"Error reading {zip_path}: {e}")
            return pd.DataFrame()
        
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    
    def build_manifest(self) -> Dict[str, StructureInfo]:
        """Build structure-level manifest from all domains."""
        manifest = {}
        
        # Read all annotations
        all_annotations = []
        zip_files = sorted(self.data_dir.glob("*.zip"))
        
        for zip_path in zip_files:
            df = self.read_annotations_from_zip(zip_path)
            if not df.empty:
                all_annotations.append(df)
        
        if not all_annotations:
            logger.warning("No annotation files found!")
            return manifest
        
        df_all = pd.concat(all_annotations, ignore_index=True)
        logger.info(f"\nTotal records loaded: {len(df_all)}")
        logger.info(f"Unique dataIds: {df_all['dataId'].nunique()}")
        
        # Build structure-level manifest
        for data_id, group in df_all.groupby('dataId'):
            # All rows for a dataId should be identical (same metadata)
            row = group.iloc[0]
            structure_id = self.extract_structure_id(data_id)
            
            if structure_id not in manifest:
                try:
                    manifest[structure_id] = StructureInfo(
                        structure_id=structure_id,
                        domain=row['domain'],
                        label=int(row['No']),
                        formula=str(row['formula']),
                        symbol_set=str(row['symbolSet']),
                        space_group=str(row['spaceGroup']),
                        space_group_no=int(row['spaceGroupNo']),
                        crystal_sys=str(row['crystalSys']),
                        lattice_a=float(row['a']),
                        lattice_b=float(row['b']),
                        lattice_c=float(row['c']),
                        lattice_alpha=float(row['alpha']),
                        lattice_beta=float(row['beta']),
                        lattice_gamma=float(row['gamma']),
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping {structure_id}: {e}")
                    continue
            
            # Track augmentation
            manifest[structure_id].augmentations.append(data_id)
        
        logger.info(f"\n✓ Manifest built: {len(manifest)} unique structures")
        return manifest
    
    def analyze_manifest(self, manifest: Dict[str, StructureInfo]) -> Dict:
        """Analyze manifest for dataset statistics."""
        
        structures = list(manifest.values())
        
        # Label distribution
        label_counts = Counter(s.label for s in structures)
        
        # Space group distribution
        sg_counts = Counter(s.space_group_no for s in structures)
        sg_names = Counter(s.space_group for s in structures)
        
        # Augmentation stats
        aug_counts = [len(s.augmentations) for s in structures]
        domain_counts = Counter(s.domain for s in structures)
        
        # Total patterns
        total_patterns = sum(len(s.augmentations) for s in structures)
        
        analysis = {
            "manifest_size": int(len(manifest)),
            "total_patterns": int(total_patterns),
            "avg_augmentations_per_structure": float(sum(aug_counts) / len(aug_counts)) if aug_counts else 0.0,
            "min_augmentations": int(min(aug_counts)) if aug_counts else 0,
            "max_augmentations": int(max(aug_counts)) if aug_counts else 0,
            "label_distribution": {int(k): int(v) for k, v in label_counts.most_common()},
            "space_group_distribution": {int(k): int(v) for k, v in sg_counts.most_common(10)},
            "space_group_names": {str(k): int(v) for k, v in sg_names.most_common(10)},
            "domain_distribution": {str(k): int(v) for k, v in domain_counts.most_common()},
        }
        
        return analysis


class ManifestReporter:
    """Generate reports from manifest."""
    
    @staticmethod
    def save_manifest_json(
        manifest: Dict[str, StructureInfo],
        output_path: Path
    ) -> None:
        """Save manifest to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "total_structures": len(manifest),
            "structures": {
                sid: struct.to_dict()
                for sid, struct in manifest.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✓ Manifest saved to {output_path}")
    
    @staticmethod
    def save_analysis_json(
        analysis: Dict,
        output_path: Path
    ) -> None:
        """Save analysis to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"✓ Analysis saved to {output_path}")
    
    @staticmethod
    def print_report(
        manifest: Dict[str, StructureInfo],
        analysis: Dict
    ) -> None:
        """Print human-readable report."""
        print("\n" + "="*80)
        print("PHASE 1: DATASET MANIFEST REPORT")
        print("="*80)
        
        print(f"\n📊 MANIFEST STATISTICS:")
        print(f"   Total unique structures: {analysis['manifest_size']}")
        print(f"   Total patterns (with augmentations): {analysis['total_patterns']}")
        print(f"   Avg augmentations per structure: {analysis['avg_augmentations_per_structure']:.2f}")
        print(f"   Augmentation range: {analysis['min_augmentations']}-{analysis['max_augmentations']}")
        
        print(f"\n📁 DOMAIN DISTRIBUTION:")
        for domain, count in analysis['domain_distribution'].items():
            print(f"   {domain}: {count} structures")
        
        print(f"\n🏷️  CLASS LABEL DISTRIBUTION:")
        # Show first 20 labels
        label_dist = analysis['label_distribution']
        num_labels = len(label_dist)
        for i, (label, count) in enumerate(sorted(label_dist.items())[:20]):
            pct = 100.0 * count / analysis['manifest_size']
            print(f"   Label {label:5d}: {count:5d} structures ({pct:5.1f}%)")
        
        if num_labels > 20:
            print(f"   ... and {num_labels - 20} more unique labels")
        
        print(f"\n🔬 TOP SPACE GROUPS (by structure count):")
        for sg_no, count in sorted(
            analysis['space_group_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]:
            # Find the space group name
            sg_name = "N/A"
            for name_str, count_val in analysis['space_group_names'].items():
                if count_val == count:  # Best effort match
                    sg_name = name_str
                    break
            print(f"   SG {sg_no:3d} ({sg_name:10s}): {count:5d} structures")
        
        print("\n" + "="*80)
        print("✓ Manifest ready for structure-level splitting!")
        print("="*80 + "\n")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / "phase1"
    
    logger.info("Building dataset manifest...")
    logger.info(f"Data directory: {data_dir}")
    
    # Build manifest
    builder = ManifestBuilder(data_dir)
    manifest = builder.build_manifest()
    analysis = builder.analyze_manifest(manifest)
    
    # Save and report
    reporter = ManifestReporter()
    reporter.save_manifest_json(manifest, output_dir / "manifest.json")
    reporter.save_analysis_json(analysis, output_dir / "manifest_analysis.json")
    reporter.print_report(manifest, analysis)
    
    logger.info("Manifest building complete!")


if __name__ == "__main__":
    main()
