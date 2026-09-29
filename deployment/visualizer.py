"""
Industrial HUD Visualizer
Renders bounding boxes, alert status overlay, and performance telemetry.
"""
import cv2
import numpy as np
from typing import List, Dict, Any

class StreamVisualizer:
    def __init__(self):
        self.colors = {
            "fire": (0, 0, 255),     # Red
            "smoke": (128, 128, 128), # Gray
            "light": (0, 255, 255)   # Yellow
        }

    def draw(self, frame: np.ndarray, detections: List[Dict[str, Any]], fps: float) -> np.ndarray:
        output_frame = frame.copy()
        hazard_detected = False

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = self.colors.get(cls_name, (0, 255, 0))

            if cls_name in ["fire", "smoke"]:
                hazard_detected = True

            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            label = f"{cls_name.upper()} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(output_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw HUD Header
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(output_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Alarm Banner
        if hazard_detected:
            cv2.rectangle(output_frame, (0, output_frame.shape[0] - 40), (output_frame.shape[1], output_frame.shape[0]), (0, 0, 255), -1)
            cv2.putText(output_frame, "CRITICAL ALARM: HAZARD DETECTED", (10, output_frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return output_frame
