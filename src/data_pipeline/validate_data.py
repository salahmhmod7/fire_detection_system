"""
Data Integrity & Cleaning Module
Validates images and YOLO annotation formats.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.splits = ["train", "valid", "test"]
        self.valid_images: Dict[str, List[Path]] = {split: [] for split in self.splits}
        self.corrupt_images: List[Path] = []
        self.orphan_files: List[Path] = []
        self.invalid_labels: List[Path] = []
        self.empty_labels: List[Path] = [] # Background images

    def check_image_integrity(self, img_path: Path) -> bool:
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verifies file integrity without decoding the whole image
            return True
        except (UnidentifiedImageError, IOError, SyntaxError):
            return False

    def validate_yolo_label(self, label_path: Path) -> bool:
        if not label_path.exists():
            return False
        
        if label_path.stat().st_size == 0:
            self.empty_labels.append(label_path)
            return True # Empty files are valid (background images)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            
            try:
                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
                # Validate normalized boundaries and positive dimensions
                if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0):
                    return False
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    return False
            except ValueError:
                return False
        return True

    def run(self) -> Dict[str, List[Path]]:
        logger.info("Starting Data Validation...")
        
        for split in self.splits:
            img_dir = self.raw_dir / split / "images"
            lbl_dir = self.raw_dir / split / "labels"
            
            if not img_dir.exists() or not lbl_dir.exists():
                logger.warning(f"Split '{split}' missing images or labels directory. Skipping.")
                continue

            images = list(img_dir.glob("*.*"))
            labels = list(lbl_dir.glob("*.txt"))
            
            img_stems = {img.stem for img in images}
            lbl_stems = {lbl.stem for lbl in labels}

            # 1. Orphan Check
            orphan_imgs = [img for img in images if img.stem not in lbl_stems]
            orphan_lbls = [lbl for lbl in labels if lbl.stem not in img_stems]
            self.orphan_files.extend(orphan_imgs + orphan_lbls)

            # 2. Integrity and Label Format Check
            paired_images = [img for img in images if img.stem in lbl_stems]
            
            for img_path in tqdm(paired_images, desc=f"Validating {split}"):
                lbl_path = lbl_dir / f"{img_path.stem}.txt"

                if not self.check_image_integrity(img_path):
                    self.corrupt_images.append(img_path)
                    continue

                if not self.validate_yolo_label(lbl_path):
                    self.invalid_labels.append(lbl_path)
                    continue

                self.valid_images[split].append(img_path)

        logger.info(f"Validation Complete. Found {len(self.corrupt_images)} corrupt images, "
                    f"{len(self.orphan_files)} orphans, {len(self.invalid_labels)} invalid labels.")
        return self.valid_images