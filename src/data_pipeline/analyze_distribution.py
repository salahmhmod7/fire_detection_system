"""
Class Imbalance & Bounding Box Metrics Module
Analyzes the processed dataset to inform model architecture and loss weights.
"""
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class DistributionAnalyzer:
    def __init__(self, processed_dir: Path, image_size: int = 640):
        self.processed_dir = processed_dir
        self.image_size = image_size
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.box_sizes: Dict[str, int] = {"small": 0, "medium": 0, "large": 0}
        self.aspect_ratios: List[float] = []
        self.class_names: List[str] = []

    def load_class_names(self):
        yaml_path = self.processed_dir / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                self.class_names = data.get('names', [])
        else:
            self.class_names = []

    def analyze(self) -> Dict[str, Any]:
        logger.info("Analyzing dataset distribution and bounding box profiles...")
        self.load_class_names()
        
        lbl_dirs = [self.processed_dir / split / "labels" for split in ["train", "valid", "test"]]
        
        for lbl_dir in lbl_dirs:
            if not lbl_dir.exists():
                continue
            for lbl_path in lbl_dir.glob("*.txt"):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        
                        cls_id = int(parts[0])
                        w, h = float(parts[3]), float(parts[4])
                        
                        self.class_counts[cls_id] += 1
                        
                        # COCO Size Profiling (Area in pixels)
                        area = (w * self.image_size) * (h * self.image_size)
                        if area < 1024:  # 32x32
                            self.box_sizes["small"] += 1
                        elif area <= 9216:  # 96x96
                            self.box_sizes["medium"] += 1
                        else:
                            self.box_sizes["large"] += 1
                            
                        # Aspect Ratio (Width / Height)
                        if h > 0:
                            self.aspect_ratios.append(w / h)

        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        total_boxes = sum(self.class_counts.values())
        nc = len(self.class_names) if self.class_names else len(self.class_counts)
        
        # Calculate Class Weights (Inverse frequency formulation)
        weights = {}
        if total_boxes > 0 and nc > 0:
            for cls_id, count in self.class_counts.items():
                name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                weight = total_boxes / (nc * count) if count > 0 else 0
                weights[name] = round(weight, 4)

        avg_ar = float(np.mean(self.aspect_ratios)) if self.aspect_ratios else 0.0

        report = {
            "total_annotations": total_boxes,
            "class_distribution": {self.class_names[k] if k < len(self.class_names) else k: v for k, v in self.class_counts.items()},
            "coco_box_profiles": self.box_sizes,
            "average_aspect_ratio": round(avg_ar, 2),
            "recommended_class_weights": weights
        }
        
        logger.info(f"Distribution Analysis Complete. Recommend weights: {weights}")
        return report