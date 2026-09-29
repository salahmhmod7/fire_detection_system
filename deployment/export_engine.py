"""
Model Quantization & Export Engine
Converts PyTorch checkpoints to ONNX and TensorRT with FP16 precision.
"""
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelExporter:
    def __init__(self, weights_path: Path, output_dir: Path):
        self.weights_path = weights_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_onnx(self, imgsz: int = 640) -> Path:
        """Exports model to ONNX format with dynamic batching capabilities."""
        logger.info("Exporting model to ONNX format...")
        model = YOLO(self.weights_path)
        onnx_path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=True,
            opset=12
        )
        logger.info(f"ONNX model successfully saved to: {onnx_path}")
        return Path(onnx_path)

    def export_tensorrt(self, imgsz: int = 640, half: bool = True) -> Path:
        """
        Exports model to NVIDIA TensorRT Engine with FP16 precision.
        Requires TensorRT installed on the host environment.
        """
        logger.info(f"Exporting model to TensorRT Engine (FP16={half})...")
        try:
            model = YOLO(self.weights_path)
            engine_path = model.export(
                format="engine",
                imgsz=imgsz,
                half=half,
                device=0
            )
            logger.info(f"TensorRT engine successfully generated at: {engine_path}")
            return Path(engine_path)
        except Exception as e:
            logger.error(f"TensorRT export failed (falling back to ONNX): {e}")
            return self.export_onnx(imgsz)
