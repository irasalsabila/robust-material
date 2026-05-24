"""
Phase 1: Dataset Inspection - XRD Crystallographic Classification
Objective: Inspect zip archives and build dataset understanding without leakage risk.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set
from zipfile import BadZipFile, ZipFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExtensionStats:
    """Statistics for file extensions."""
    count: int = 0
    sample_files: List[str] = field(default_factory=list)
    
    def add_sample(self, filename: str, max_samples: int = 5) -> None:
        """Add a sample filename if under the limit."""
        if len(self.sample_files) < max_samples:
            self.sample_files.append(filename)


@dataclass
class ZipInspectionResult:
    """Result of inspecting a single zip file."""
    zip_name: str
    total_files: int = 0
    extension_stats: Dict[str, ExtensionStats] = field(default_factory=dict)
    metadata_files: List[str] = field(default_factory=list)
    potential_structure_ids: Set[str] = field(default_factory=set)
    error: str | None = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "zip_name": self.zip_name,
            "total_files": self.total_files,
            "extension_stats": {
                ext: {
                    "count": stats.count,
                    "sample_files": stats.sample_files
                }
                for ext, stats in self.extension_stats.items()
            },
            "metadata_files": self.metadata_files,
            "potential_structure_ids": sorted(list(self.potential_structure_ids)),
            "error": self.error
        }


class ZipInspector:
    """Inspects zip files for dataset understanding."""
    
    METADATA_EXTENSIONS = {".csv", ".json", ".txt", ".yaml", ".yml", ".xml"}
    STRUCTURE_EXTENSIONS = {".cif", ".vasp", ".poscar", ".contcar"}
    PATTERN_EXTENSIONS = {".csv", ".dat", ".txt", ".npy", ".h5", ".hdf5"}
    
    def __init__(self, data_dir: Path):
        """Initialize inspector with data directory."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def extract_structure_id(self, filename: str) -> str | None:
        """
        Extract potential structure ID from filename.
        
        Looks for patterns like:
        - ICSD-12345, ICSD12345
        - MP-12345, MP12345
        - COD-12345, COD12345
        - UUID-like patterns
        - Digit-heavy patterns (likely structure IDs)
        """
        filename_clean = Path(filename).stem.upper()
        
        # Specific database patterns
        patterns = [
            r"(ICSD[-_]?\d+)",
            r"(MP[-_]?\d+)",
            r"(COD[-_]?\d+)",
            r"(PNMA[-_]?\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_clean)
            if match:
                return match.group(1)
        
        # UUID-like pattern
        uuid_pattern = r"[0-9a-f]{8}[-_]?[0-9a-f]{4}[-_]?[0-9a-f]{4}[-_]?[0-9a-f]{4}[-_]?[0-9a-f]{12}"
        if re.search(uuid_pattern, filename_clean, re.IGNORECASE):
            return "UUID_PATTERN"
        
        return None
    
    def inspect_zip(self, zip_path: Path) -> ZipInspectionResult:
        """Inspect a single zip file."""
        result = ZipInspectionResult(zip_name=zip_path.name)
        extension_stats: Dict[str, ExtensionStats] = defaultdict(ExtensionStats)
        metadata_files: List[str] = []
        structure_ids: Set[str] = set()
        
        try:
            with ZipFile(zip_path, 'r') as z:
                file_list = z.namelist()
                result.total_files = len(file_list)
                
                for filename in file_list:
                    # Skip directories
                    if filename.endswith('/'):
                        continue
                    
                    # Get extension
                    ext = Path(filename).suffix.lower()
                    if not ext:
                        ext = "(no_extension)"
                    
                    extension_stats[ext].count += 1
                    extension_stats[ext].add_sample(filename)
                    
                    # Detect metadata files
                    if ext in self.METADATA_EXTENSIONS:
                        metadata_files.append(filename)
                    
                    # Extract potential structure IDs
                    struct_id = self.extract_structure_id(filename)
                    if struct_id:
                        structure_ids.add(struct_id)
                
                result.extension_stats = extension_stats
                result.metadata_files = metadata_files
                result.potential_structure_ids = structure_ids
                
                logger.info(
                    f"✓ {zip_path.name}: {result.total_files} files, "
                    f"{len(extension_stats)} extensions, "
                    f"{len(metadata_files)} metadata files, "
                    f"{len(structure_ids)} unique structure IDs"
                )
        
        except BadZipFile as e:
            result.error = f"BadZipFile: {str(e)}"
            logger.error(f"✗ {zip_path.name}: Corrupted zip file - {e}")
        except Exception as e:
            result.error = f"Unexpected error: {str(e)}"
            logger.error(f"✗ {zip_path.name}: {type(e).__name__} - {e}")
        
        return result
    
    def inspect_all(self) -> Dict[str, ZipInspectionResult]:
        """Inspect all zip files in data directory."""
        results = {}
        zip_files = sorted(self.data_dir.glob("*.zip"))
        
        if not zip_files:
            logger.warning(f"No .zip files found in {self.data_dir}")
            return results
        
        logger.info(f"Found {len(zip_files)} zip files to inspect")
        
        for zip_path in zip_files:
            result = self.inspect_zip(zip_path)
            results[zip_path.name] = result
        
        return results


class InspectionReporter:
    """Generates reports from inspection results."""
    
    @staticmethod
    def save_json(
        results: Dict[str, ZipInspectionResult],
        output_path: Path
    ) -> None:
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "inspection_summary": {
                "total_zips": len(results),
                "successful": sum(1 for r in results.values() if r.error is None),
                "failed": sum(1 for r in results.values() if r.error is not None),
            },
            "zip_details": {
                name: result.to_dict()
                for name, result in results.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✓ Results saved to {output_path}")
    
    @staticmethod
    def print_summary(results: Dict[str, ZipInspectionResult]) -> None:
        """Print human-readable summary."""
        print("\n" + "="*80)
        print("PHASE 1: ZIP INSPECTION SUMMARY")
        print("="*80)
        
        total_zips = len(results)
        successful = sum(1 for r in results.values() if r.error is None)
        failed = total_zips - successful
        
        print(f"\n📦 ZIPS PROCESSED: {successful}/{total_zips}")
        if failed > 0:
            print(f"   ⚠️  FAILED: {failed}")
        
        # Aggregate stats
        all_extensions: Dict[str, int] = defaultdict(int)
        all_metadata_files: Set[str] = set()
        all_structure_ids: Set[str] = set()
        total_files = 0
        
        for name, result in results.items():
            if result.error:
                print(f"   ✗ {name}: {result.error}")
                continue
            
            total_files += result.total_files
            
            for ext, stats in result.extension_stats.items():
                all_extensions[ext] += stats.count
            
            all_metadata_files.update(result.metadata_files)
            all_structure_ids.update(result.potential_structure_ids)
        
        print(f"\n📊 FILE STATISTICS:")
        print(f"   Total files across all zips: {total_files}")
        print(f"   Unique extensions: {len(all_extensions)}")
        
        print(f"\n📁 TOP EXTENSIONS:")
        for ext, count in sorted(
            all_extensions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:
            print(f"   {ext:15s}: {count:6d} files")
        
        if all_metadata_files:
            print(f"\n📄 METADATA FILES DETECTED: {len(all_metadata_files)}")
            for fname in sorted(all_metadata_files)[:10]:
                print(f"   - {fname}")
            if len(all_metadata_files) > 10:
                print(f"   ... and {len(all_metadata_files) - 10} more")
        
        if all_structure_ids:
            print(f"\n🔍 STRUCTURE ID PATTERNS: {len(all_structure_ids)}")
            for sid in sorted(all_structure_ids):
                print(f"   - {sid}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / "phase1"
    output_file = output_dir / "zip_inspection.json"
    
    # Run inspection
    logger.info(f"Starting Phase 1 zip inspection...")
    logger.info(f"Data directory: {data_dir}")
    
    inspector = ZipInspector(data_dir)
    results = inspector.inspect_all()
    
    # Generate reports
    reporter = InspectionReporter()
    reporter.save_json(results, output_file)
    reporter.print_summary(results)
    
    logger.info("Phase 1 inspection complete!")


if __name__ == "__main__":
    main()
