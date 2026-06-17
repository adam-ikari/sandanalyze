"""Train a custom YOLOv8-seg model for sand grain detection.

Uses auto-generated labels from traditional CV detection.
For best results, labels should be manually reviewed and corrected.

Usage:
    uv run python train_yolo.py --epochs 100 --imgsz 640
    uv run python train_yolo.py --data data/yolo_dataset/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def train_yolo(
    data_yaml: str,
    model: str = "yolov8n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    project: str = "runs/segment",
    name: str = "sand_grain",
) -> str:
    """Train a YOLOv8-seg model.

    Parameters
    ----------
    data_yaml : str
        Path to data.yaml file.
    model : str
        Base model to use.
    epochs : int
        Number of training epochs.
    imgsz : int
        Image size for training.
    batch : int
        Batch size.
    device : str
        Device to use (cpu, 0, 0,1, etc.).
    project : str
        Project directory for saving results.
    name : str
        Experiment name.

    Returns
    -------
    str
        Path to the best model weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required. Install with: uv add ultralytics")

    # Load base model
    yolo_model = YOLO(model)

    logger.info("Starting YOLO training...")
    logger.info("  Data: %s", data_yaml)
    logger.info("  Model: %s", model)
    logger.info("  Epochs: %d", epochs)
    logger.info("  Image size: %d", imgsz)
    logger.info("  Batch: %d", batch)
    logger.info("  Device: %s", device)

    # Train
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
    )

    # Find best weights
    best_weights = Path(project) / name / "weights" / "best.pt"
    if best_weights.exists():
        logger.info("Training complete. Best weights: %s", best_weights)
        return str(best_weights)
    else:
        logger.warning("Best weights not found at expected path: %s", best_weights)
        return ""


def validate_model(weights_path: str, data_yaml: str) -> dict:
    """Validate a trained model.

    Parameters
    ----------
    weights_path : str
        Path to model weights.
    data_yaml : str
        Path to data.yaml.

    Returns
    -------
    dict
        Validation metrics.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    results = model.val(data=data_yaml)

    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }

    return metrics


def export_model(weights_path: str, format: str = "onnx") -> str:
    """Export model to different formats.

    Parameters
    ----------
    weights_path : str
        Path to model weights.
    format : str
        Export format (onnx, torchscript, etc.).

    Returns
    -------
    str
        Path to exported model.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    exported = model.export(format=format)

    logger.info("Model exported to: %s", exported)
    return exported


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for sand grain detection")
    parser.add_argument("--data", default="data/yolo_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device (cpu, 0, etc.)")
    parser.add_argument("--project", default="runs/segment", help="Project directory")
    parser.add_argument("--name", default="sand_grain", help="Experiment name")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    parser.add_argument("--export-format", default="onnx", help="Export format")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Train
    best_weights = train_yolo(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    if best_weights and args.export:
        export_model(best_weights, format=args.export_format)

    print(f"\n{'='*50}")
    print(f"Training Complete")
    print(f"{'='*50}")
    print(f"Best weights: {best_weights}")
    if args.export:
        print(f"Exported to: {args.export_format}")
    print(f"\nTo use in app.py, update the model path in YOLODetector")


if __name__ == "__main__":
    main()
