"""
Model Evaluation Module
Runs independent testing on the hold-out test set to verify true generalization.
"""
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, weights_path: Path, data_yaml: Path):
        self.weights_path = weights_path
        self.data_yaml = data_yaml

    def evaluate(self) -> dict:
        logger.info(f"Loading best weights from {self.weights_path} for evaluation...")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        model = YOLO(self.weights_path)

        logger.info("Running evaluation on hold-out TEST split...")
        # Override the split to explicitly evaluate on the 'test' partition
        metrics = model.val(
            data=str(self.data_yaml),
            split='test',
            save_json=True,
            plots=True,
            conf=0.25,      # Confidence threshold
            iou=0.45        # NMS IOU threshold
        )
        
        # Extract core metrics for logging
        mAP50 = round(metrics.box.map50, 4)
        mAP50_95 = round(metrics.box.map, 4)
        
        logger.info(f"Evaluation Complete. mAP@50: {mAP50} | mAP@50-95: {mAP50_95}")
        return {"mAP50": mAP50, "mAP50_95": mAP50_95}