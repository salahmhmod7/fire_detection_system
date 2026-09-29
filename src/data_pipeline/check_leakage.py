"""
Data Leakage & Sequence Analysis Module
Prevents highly correlated frames from crossing split boundaries.
"""
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import imagehash
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LeakageDetector:
    def __init__(self, valid_images: Dict[str, List[Path]]):
        self.valid_images = valid_images
        self.safe_images: Dict[str, List[Path]] = {split: [] for split in valid_images.keys()}
        self.leaked_files: List[Path] = []
        self.hash_dict: Dict[str, Tuple[str, Path]] = {}  # hash -> (split, path)

    def extract_group_id(self, filename: str) -> str:
        """
        Extracts video source prefix. 
        Example: 'fire-53-preprocessed-4.jpg' -> 'fire-53'
        """
        match = re.match(r"^([a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)?)[_-]", filename)
        return match.group(1) if match else filename

    def detect_and_remediate(self) -> Dict[str, List[Path]]:
        logger.info("Starting Leakage Detection (Group & Perceptual Hash)...")
        
        train_groups: Set[str] = set()
        
        # 1. Map Train Groups
        for img_path in self.valid_images.get("train", []):
            group_id = self.extract_group_id(img_path.name)
            train_groups.add(group_id)
            
            # Compute hash for exact/near duplicate detection
            try:
                img_hash = str(imagehash.average_hash(Image.open(img_path)))
                self.hash_dict[img_hash] = ("train", img_path)
                self.safe_images["train"].append(img_path)
            except Exception as e:
                logger.error(f"Failed to hash {img_path.name}: {e}")

        # 2. Check Valid and Test Splits
        for split in ["valid", "test"]:
            for img_path in tqdm(self.valid_images.get(split, []), desc=f"Checking leakage in {split}"):
                group_id = self.extract_group_id(img_path.name)
                
                # Check Group Leakage
                if group_id in train_groups:
                    logger.debug(f"Group Leakage detected: {img_path.name} belongs to train group '{group_id}'.")
                    self.leaked_files.append(img_path)
                    continue
                
                # Check Hash Leakage (Near Duplicates)
                try:
                    img_hash = str(imagehash.average_hash(Image.open(img_path)))
                    if img_hash in self.hash_dict:
                        conflict_split, conflict_path = self.hash_dict[img_hash]
                        logger.debug(f"Hash Collision: {img_path.name} ({split}) conflicts with {conflict_path.name} ({conflict_split})")
                        self.leaked_files.append(img_path)
                        continue
                        
                    self.hash_dict[img_hash] = (split, img_path)
                    self.safe_images[split].append(img_path)
                except Exception as e:
                    logger.error(f"Failed to hash {img_path.name}: {e}")

        logger.info(f"Leakage Detection Complete. Removed {len(self.leaked_files)} leaked/duplicate frames.")
        return self.safe_images