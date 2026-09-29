"""
Dynamic Training Configuration Module
Generates optimal YOLO hyperparameters based on Phase 1 data metrics.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TrainingConfigurator:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.summary_path = processed_dir / "phase1_summary.json"
        
    def load_phase1_metrics(self) -> Dict[str, Any]:
        if not self.summary_path.exists():
            logger.warning(f"Phase 1 summary not found at {self.summary_path}. Using default configurations.")
            return {}
        with open(self.summary_path, 'r') as f:
            return json.load(f)

    def generate_hyperparameters(self) -> Dict[str, Any]:
        """
        Calculates optimal hyperparameters for Fire/Smoke detection.
        """
        metrics = self.load_phase1_metrics()
        dist_analysis = metrics.get("dataset_analysis", {})
        
        # Base production hyperparameters for robust edge CV
        hyperparams = {
            "optimizer": "auto",       # AdamW is often selected automatically for fine-tuning
            "cos_lr": True,            # Cosine learning rate scheduling prevents plateauing
            "lr0": 0.01,               # Initial learning rate
            "lrf": 0.01,               # Final learning rate (lr0 * lrf)
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            
            # Augmentations tailored for Fire/Smoke
            "hsv_h": 0.015,            # Keep hue variations low (fire color is critical)
            "hsv_s": 0.7,              # High saturation variance (handles different lighting)
            "hsv_v": 0.4,              # High value variance (handles day/night changes)
            "degrees": 0.0,            # Avoid rotation (fire/smoke behaves differently under gravity)
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,             # Horizontal flips are physically valid
            "flipud": 0.0,             # Vertical flips are invalid (smoke goes UP)
            "mosaic": 1.0,             # Crucial for small object detection (distant smoke)
            "mixup": 0.1,              # Light mixup reduces overfitting on backgrounds
        }

        # Check for class imbalance to trigger Focal Loss
        class_distribution = dist_analysis.get("class_distribution", {})
        if class_distribution:
            counts = list(class_distribution.values())
            if max(counts) / (min(counts) + 1e-6) > 3.0:
                logger.info("Severe class imbalance detected. Enabling Focal Loss (fl_gamma).")
                hyperparams["fl_gamma"] = 1.5  # Focuses training on hard, minority examples

        return hyperparams