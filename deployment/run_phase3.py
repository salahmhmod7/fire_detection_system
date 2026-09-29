"""
Phase 3 Master Deployment Runner
Executes model export and runs a real-time optimized inference stream.
"""
import winsound
import argparse
import logging
import cv2
import time
from pathlib import Path
from export_engine import ModelExporter
from inference_engine import EdgeInferenceEngine
from visualizer import StreamVisualizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Production Edge Deployment")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=str, default="0", help="RTSP stream URL, video file path, or 0 for webcam")
    parser.add_argument("--export_format", type=str, choices=["onnx", "engine", "none"], default="onnx", help="Target edge format")
    parser.add_argument("--export_only", action="store_true", help="Export model and exit without running inference")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights file not found at {weights_path}")
        return

    # تحديد مسار ملف الصوت
    sound_file = Path("alarm.wav") if Path("alarm.wav").exists() else Path("deployment/alarm.wav")
    if sound_file.exists():
        logger.info(f"Loaded alarm sound from: {sound_file}")
    else:
        logger.warning("alarm.wav file not found! Sound will be disabled.")

    engine_dir = weights_path.parent.parent / "deployment_engines"
    active_model_path = weights_path

    if args.export_format != "none":
        exporter = ModelExporter(weights_path, engine_dir)
        if args.export_format == "onnx":
            active_model_path = exporter.export_onnx(imgsz=640)
        elif args.export_format == "engine":
            active_model_path = exporter.export_tensorrt(imgsz=640, half=True)
            
    if args.export_only:
        logger.info("Export complete. Exiting (--export_only was set).")
        return

    logger.info(f"Initializing Edge Inference Engine with {active_model_path.name}...")
    engine = EdgeInferenceEngine(active_model_path)
    visualizer = StreamVisualizer()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {args.source}")
        return

    logger.info("Starting real-time inference... Press 'q' to quit.")
    
    prev_time = time.perf_counter()
    cv2.namedWindow("Industrial Fire & Smoke Detection", cv2.WINDOW_NORMAL)

    # متغيّر لتتبع حالة الصوت لتجنب إعادة تشغيله في كل فريم
    is_playing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("End of video stream or stream lost. Exiting.")
            break

        frame, detections, inf_time_ms = engine.process_frame(frame)

        current_time = time.perf_counter()
        fps = 1.0 / (max(current_time - prev_time, 1e-5))
        prev_time = current_time

        output_frame = visualizer.draw(frame, detections, fps)

        # -------------------------------------------------------------
        # 🔊 تشغيل إنذار الـ WAV بسلامة وبدون تقطيع
        has_hazard = any(det["class_name"] in ["fire", "smoke"] for det in detections)

        if sound_file.exists():
            if has_hazard:
                if not is_playing:
                    # SND_LOOP (للتكرار) | SND_ASYNC (بدون تجميد الكاميرا) | SND_FILENAME
                    winsound.PlaySound(str(sound_file), winsound.SND_ASYNC | winsound.SND_LOOP | winsound.SND_FILENAME)
                    is_playing = True
            else:
                if is_playing:
                    winsound.PlaySound(None, winsound.SND_PURGE) # إيقاف الصوت فوراً
                    is_playing = False
        # -------------------------------------------------------------

        cv2.imshow("Industrial Fire & Smoke Detection", output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    winsound.PlaySound(None, winsound.SND_PURGE)
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Pipeline terminated safely.")

if __name__ == "__main__":
    main()