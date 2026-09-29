"""
Dataset Curation & Standardized Export Module
Migrates safe data and generates the final configuration file.
"""
import shutil
import logging
import yaml
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataExporter:
    def __init__(self, safe_images: Dict[str, List[Path]], raw_dir: Path, processed_dir: Path):
        self.safe_images = safe_images
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def export(self) -> None:
        logger.info(f"Exporting curated dataset to {self.processed_dir}...")
        
        # Clean output directory if it exists
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        total_copied = 0
        for split, paths in self.safe_images.items():
            if not paths:
                continue
                
            img_out_dir = self.processed_dir / split / "images"
            lbl_out_dir = self.processed_dir / split / "labels"
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(paths, desc=f"Exporting {split}"):
                lbl_path = self.raw_dir / split / "labels" / f"{img_path.stem}.txt"
                
                shutil.copy2(img_path, img_out_dir / img_path.name)
                if lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_out_dir / lbl_path.name)
                total_copied += 1

        self.generate_yaml()
        logger.info(f"Export Complete. Migrated {total_copied} validated files.")

    def generate_yaml(self) -> None:
        raw_yaml_path = self.raw_dir / "data.yaml"
        class_names = []
        
        if raw_yaml_path.exists():
            with open(raw_yaml_path, 'r') as f:
                raw_data = yaml.safe_load(f)
                class_names = raw_data.get('names', [])
        else:
            logger.warning("No raw data.yaml found. Generating generic class names.")
            class_names = ["class_0", "class_1"] # Fallback

        processed_yaml = {
            "path": str(self.processed_dir.absolute()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names
        }

        with open(self.processed_dir / "data.yaml", 'w') as f:
            yaml.dump(processed_yaml, f, sort_keys=False)