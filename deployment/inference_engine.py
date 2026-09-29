"""
Production Edge Inference Engine
Provides frame-by-frame pipeline processing with adaptive thresholding.
"""
import time
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class EdgeInferenceEngine:
    def __init__(self, model_path: Path, conf_thresholds: Dict[str, float] = None):
        self.model_path = model_path
        # YOLO class transparently handles .pt, .onnx, and .engine formats
        self.model = YOLO(str(model_path), task="detect")
        
        # Adaptive Thresholding: Lower smoke threshold to solve the recall bottleneck
        self.conf_thresholds = conf_thresholds or {
            "fire": 0.40,
            "light": 0.50,
            "smoke": 0.25  # Lower threshold increases recall for faint smoke
        }
        
        # Safe class names mapping with fallback
        if hasattr(self.model, "names") and self.model.names:
            self.class_names = self.model.names
        else:
            self.class_names = {0: "fire", 1: "light", 2: "smoke"}

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Processes a single BGR frame, executes inference, and filters detections.
        """
        start_time = time.perf_counter()
        
        # Run inference on CPU explicitly
        results = self.model.predict(
            source=frame,
            verbose=False,
            imgsz=640,
            iou=0.45,
            device="cpu"  # Force execution on CPU to avoid ONNX-GPU runtime prompts
        )[0]
        
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = self.class_names.get(cls_id, "unknown")
                min_conf = self.conf_thresholds.get(class_name, 0.35)
                
                # Apply class-specific adaptive threshold
                if conf >= min_conf:
                    detections.append({
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": float(conf),
                        "bbox": [int(b) for b in box] # [x1, y1, x2, y2]
                    })

        inference_time_ms = (time.perf_counter() - start_time) * 1000
        return frame, detections, inference_time_ms