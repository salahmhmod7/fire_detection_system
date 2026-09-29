"""
Master Pipeline Orchestrator (Phase 1)
Executes Validation -> Leakage Check -> Export -> Distribution Analysis.
"""
import argparse
import logging
import json
from pathlib import Path

from validate_data import DataValidator
from check_leakage import LeakageDetector
from export_curated import DataExporter
from analyze_distribution import DistributionAnalyzer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Fire/Smoke Data Pipeline")
    parser.add_argument("--root_dir", type=str, default="D:/fire_detection_system", help="Project root directory")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    root_path = Path(args.root_dir)
    raw_dir = root_path / "data" / "raw"
    processed_dir = root_path / "data" / "processed"

    if not raw_dir.exists():
        logger.error(f"Raw data directory not found at {raw_dir}")
        return

    logger.info("=== STARTING PHASE 1 DATA PIPELINE ===")

    # Step 1: Validate Data
    validator = DataValidator(raw_dir)
    valid_images = validator.run()

    # Step 2: Check Leakage
    detector = LeakageDetector(valid_images)
    safe_images = detector.detect_and_remediate()

    # Step 3: Export Curated
    exporter = DataExporter(safe_images, raw_dir, processed_dir)
    exporter.export()

    # Step 4: Analyze Distribution
    analyzer = DistributionAnalyzer(processed_dir)
    report = analyzer.analyze()

    # Save Executive Summary
    summary = {
        "pipeline_metrics": {
            "corrupt_files_dropped": len(validator.corrupt_images),
            "orphan_files_dropped": len(validator.orphan_files),
            "invalid_labels_dropped": len(validator.invalid_labels),
            "leaked_frames_removed": len(detector.leaked_files)
        },
        "dataset_analysis": report
    }

    summary_file = processed_dir / "phase1_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    logger.info(f"=== PHASE 1 COMPLETE ===")
    logger.info(f"Executive Summary saved to {summary_file}")

if __name__ == "__main__":
    main()