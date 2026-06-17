"""Evaluate traditional CV detection on cropped images.

Runs the traditional detection pipeline on all test images and reports metrics.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from core.preprocessor import PreprocessConfig, preprocess
from core.traditional import detect_grains

logger = logging.getLogger(__name__)


def evaluate_image(image_path: str, config: PreprocessConfig) -> dict:
    """Evaluate traditional detection on a single image.

    Returns
    -------
    dict
        Results including grain count, processing time, etc.
    """
    import time

    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "error": "Could not read image"}

    h, w = img.shape[:2]

    try:
        start = time.time()
        mask = preprocess(img, config)
        grains = detect_grains(mask, min_area=config.min_area)
        elapsed = time.time() - start

        return {
            "status": "success",
            "filename": os.path.basename(image_path),
            "width": w,
            "height": h,
            "grain_count": len(grains),
            "processing_time": elapsed,
        }
    except Exception as exc:
        return {
            "status": "error",
            "filename": os.path.basename(image_path),
            "error": str(exc),
        }


def evaluate_directory(input_dir: str, config: PreprocessConfig) -> list[dict]:
    """Evaluate all images in a directory."""
    results = []
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(input_dir, filename)
        result = evaluate_image(image_path, config)
        results.append(result)

        if result["status"] == "success":
            logger.info(
                "%s: %d grains in %.3fs",
                filename,
                result["grain_count"],
                result["processing_time"],
            )
        else:
            logger.error("%s: %s", filename, result.get("error", "Unknown error"))

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate traditional CV detection")
    parser.add_argument("input_dir", help="Directory containing images to evaluate")
    parser.add_argument("--output", "-o", help="Output CSV file for results")
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--adaptive-block", type=int, default=11)
    parser.add_argument("--adaptive-c", type=int, default=2)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE")
    parser.add_argument("--watershed", action="store_true", help="Enable watershed")
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

    results = evaluate_directory(args.input_dir, config)

    # Summary
    success_results = [r for r in results if r["status"] == "success"]
    if success_results:
        counts = [r["grain_count"] for r in success_results]
        times = [r["processing_time"] for r in success_results]

        print(f"\n{'='*50}")
        print(f"Evaluation Summary")
        print(f"{'='*50}")
        print(f"Images evaluated: {len(success_results)}/{len(results)}")
        print(f"Grain count: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")
        print(f"Processing time: min={min(times):.3f}s, max={max(times):.3f}s, avg={sum(times)/len(times):.3f}s")
        print(f"Total grains detected: {sum(counts)}")

    # Save results to CSV
    if args.output and success_results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "width", "height", "grain_count", "processing_time"])
            writer.writeheader()
            for r in success_results:
                writer.writerow({
                    "filename": r["filename"],
                    "width": r["width"],
                    "height": r["height"],
                    "grain_count": r["grain_count"],
                    "processing_time": f"{r['processing_time']:.4f}",
                })
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
