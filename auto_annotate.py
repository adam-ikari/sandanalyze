"""Auto-annotate images using traditional detection results for YOLO training.

Generates YOLO-format label files from traditional CV detection.
These can be used as starting points for manual refinement or
for training a custom YOLO model.

YOLO format: <class_id> <x_center> <y_center> <width> <height>
All values are normalized to [0, 1].
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.preprocessor import PreprocessConfig, preprocess
from core.traditional import detect_grains

logger = logging.getLogger(__name__)

# Single class: sand grain
CLASS_NAME = "grain"


def generate_yolo_label(
    image_path: str,
    output_label_path: str,
    config: PreprocessConfig,
) -> dict:
    """Generate YOLO-format labels for an image using traditional detection.

    Parameters
    ----------
    image_path : str
        Path to input image.
    output_label_path : str
        Path to save YOLO label file.
    config : PreprocessConfig
        Preprocessing configuration.

    Returns
    -------
    dict
        Results including grain count and status.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "error": "Could not read image"}

    h, w = img.shape[:2]

    try:
        mask = preprocess(img, config)
        grains = detect_grains(mask, min_area=config.min_area)

        labels = []
        for grain in grains:
            x, y, bw, bh = cv2.boundingRect(grain.contour)

            # Normalize to [0, 1]
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        with open(output_label_path, "w") as f:
            f.write("\n".join(labels) + "\n")

        return {
            "status": "success",
            "grain_count": len(grains),
            "image_size": (w, h),
        }

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def create_yolo_dataset(
    train_dir: str,
    test_dir: str,
    output_dir: str,
    config: PreprocessConfig,
) -> dict:
    """Create a complete YOLO dataset from train and test images.

    Structure:
        output_dir/
        ├── images/
        │   ├── train/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   └── test/
        └── data.yaml

    Parameters
    ----------
    train_dir : str
        Directory with training images.
    test_dir : str
        Directory with test images.
    output_dir : str
        Output directory for YOLO dataset.
    config : PreprocessConfig
        Preprocessing configuration.

    Returns
    -------
    dict
        Summary of created dataset.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    results = {"train": 0, "test": 0, "errors": []}

    for split, input_dir in [("train", train_dir), ("test", test_dir)]:
        img_out = images_dir / split
        lbl_out = labels_dir / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for filename in sorted(os.listdir(input_dir)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(input_dir, filename)
            label_filename = Path(filename).stem + ".txt"
            label_path = lbl_out / label_filename

            result = generate_yolo_label(image_path, str(label_path), config)

            if result["status"] == "success":
                # Copy image to output
                img_dest = img_out / filename
                import shutil
                shutil.copy2(image_path, img_dest)
                results[split] += 1
                logger.info(
                    "%s: %d labels generated",
                    filename,
                    result["grain_count"],
                )
            else:
                results["errors"].append(f"{filename}: {result['error']}")
                logger.error("%s: %s", filename, result["error"])

    # Create data.yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/test",  # Use test as validation for now
        "test": "images/test",
        "names": {0: CLASS_NAME},
        "nc": 1,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("Dataset config saved to: %s", yaml_path)

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Auto-annotate images for YOLO training")
    parser.add_argument("train_dir", help="Directory with training images")
    parser.add_argument("test_dir", help="Directory with test images")
    parser.add_argument("output_dir", help="Output directory for YOLO dataset")
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--adaptive-block", type=int, default=11)
    parser.add_argument("--adaptive-c", type=int, default=2)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--watershed", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = PreprocessConfig(
        blur_kernel=args.blur_kernel,
        adaptive_block_size=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        morph_kernel_size=args.morph_kernel,
        min_area=args.min_area,
        use_clahe=args.clahe,
        use_watershed=args.watershed,
    )

    results = create_yolo_dataset(
        args.train_dir,
        args.test_dir,
        args.output_dir,
        config,
    )

    print(f"\n{'='*50}")
    print(f"YOLO Dataset Created")
    print(f"{'='*50}")
    print(f"Train images: {results['train']}")
    print(f"Test images: {results['test']}")
    if results["errors"]:
        print(f"Errors: {len(results['errors'])}")
        for e in results["errors"]:
            print(f"  - {e}")
    print(f"\nDataset location: {args.output_dir}")
    print(f"Config file: {args.output_dir}/data.yaml")
    print(f"\nTo train: yolo segment train data={args.output_dir}/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
