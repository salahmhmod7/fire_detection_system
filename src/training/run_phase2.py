"""
Master Pipeline Orchestrator (Phase 2)
Executes Hyperparameter Config -> YOLO Training Loop -> Testing Evaluation.
"""
import argparse
import logging
from pathlib import Path

# Relative imports from src.training
from src.training.train_config import TrainingConfigurator
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fire/Smoke Model Training")
    parser.add_argument("--root_dir", type=str, default="D:/fire_detection_system", help="Project root directory")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Base model size (n, s, m, l, x)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if VRAM runs out)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    root_path = Path(args.root_dir)
    processed_dir = root_path / "data" / "processed"
    data_yaml = processed_dir / "data.yaml"

    logger.info("=== STARTING PHASE 2 MODEL TRAINING ===")

    # Step 1: Configure Hyperparameters dynamically based on Phase 1
    configurator = TrainingConfigurator(processed_dir)
    hyperparams = configurator.generate_hyperparameters()
    logger.info(f"Applied Hyperparameters: {hyperparams}")

    # Step 2: Train Model
    trainer = ModelTrainer(root_path, model_version=args.model)
    trainer.initialize_model()
    best_weights = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        hyperparams=hyperparams
    )

    # Step 3: Evaluate on Test Set
    evaluator = ModelEvaluator(best_weights, data_yaml)
    metrics = evaluator.evaluate()

    logger.info(f"=== PHASE 2 COMPLETE ===")
    logger.info(f"Final Test Metrics: {metrics}")
    logger.info("Check D:/fire_detection_system/runs/train/fire_smoke_v1/ for plots and confusion matrices.")

if __name__ == "__main__":
    main()