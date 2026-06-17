"""Crop black bars from sand grain images.

The images have black bars at top and bottom from the phone camera app.
This script crops them to keep only the actual sand grain content.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default crop boundaries (found by analyzing all images)
DEFAULT_CROP = {
    "y_start": 730,
    "y_end": 1820,
    "x_start": 0,
    "x_end": 1080,
}


def find_content_boundaries(image_path: str, threshold: int = 5) -> tuple[int, int, int, int]:
    """Find the content region by detecting non-black areas.

    Returns
    -------
    tuple
        (y_start, y_end, x_start, x_end)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find vertical boundaries (rows)
    row_means = gray.mean(axis=1)
    y_start = 0
    for i in range(h):
        if row_means[i] > threshold:
            y_start = i
            break

    y_end = h
    for i in range(h - 1, -1, -1):
        if row_means[i] > threshold:
            y_end = i + 1
            break

    # Find horizontal boundaries (columns)
    col_means = gray.mean(axis=0)
    x_start = 0
    for i in range(w):
        if col_means[i] > threshold:
            x_start = i
            break

    x_end = w
    for i in range(w - 1, -1, -1):
        if col_means[i] > threshold:
            x_end = i + 1
            break

    return y_start, y_end, x_start, x_end


def crop_image(
    image_path: str,
    output_path: str,
    y_start: int | None = None,
    y_end: int | None = None,
    x_start: int | None = None,
    x_end: int | None = None,
) -> tuple[int, int]:
    """Crop an image and save the result.

    Parameters
    ----------
    image_path : str
        Path to input image.
    output_path : str
        Path to save cropped image.
    y_start, y_end, x_start, x_end : int | None
        Crop boundaries. If None, auto-detects from image.

    Returns
    -------
    tuple
        (new_width, new_height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]

    # Auto-detect if not provided
    if y_start is None or y_end is None or x_start is None or x_end is None:
        y_start, y_end, x_start, x_end = find_content_boundaries(image_path)

    # Ensure boundaries are valid
    y_start = max(0, y_start)
    x_start = max(0, x_start)
    y_end = min(h, y_end)
    x_end = min(w, x_end)

    cropped = img[y_start:y_end, x_start:x_end]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cropped)

    return cropped.shape[1], cropped.shape[0]


def process_directory(
    input_dir: str,
    output_dir: str,
    y_start: int | None = None,
    y_end: int | None = None,
    x_start: int | None = None,
    x_end: int | None = None,
    auto_detect: bool = False,
) -> list[dict]:
    """Process all images in a directory.

    Parameters
    ----------
    input_dir : str
        Directory containing images.
    output_dir : str
        Directory to save cropped images.
    y_start, y_end, x_start, x_end : int | None
        Crop boundaries. If auto_detect is True, these are ignored.
    auto_detect : bool
        If True, auto-detect boundaries for each image.

    Returns
    -------
    list[dict]
        List of processing results.
    """
    results = []
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(image_extensions):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            if auto_detect:
                y_s, y_e, x_s, x_e = find_content_boundaries(input_path)
            else:
                y_s, y_e, x_s, x_e = y_start, y_end, x_start, x_end

            new_w, new_h = crop_image(input_path, output_path, y_s, y_e, x_s, x_e)

            results.append({
                "filename": filename,
                "status": "success",
                "size": (new_w, new_h),
            })
            logger.info("Cropped %s -> %dx%d", filename, new_w, new_h)

        except Exception as exc:
            results.append({
                "filename": filename,
                "status": "error",
                "error": str(exc),
            })
            logger.error("Failed to crop %s: %s", filename, exc)

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Crop black bars from sand grain images")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output directory for cropped images")
    parser.add_argument("--y-start", type=int, help="Top crop boundary")
    parser.add_argument("--y-end", type=int, help="Bottom crop boundary")
    parser.add_argument("--x-start", type=int, help="Left crop boundary")
    parser.add_argument("--x-end", type=int, help="Right crop boundary")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect boundaries for each image")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without cropping")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Use defaults if not specified
    y_start = args.y_start or DEFAULT_CROP["y_start"]
    y_end = args.y_end or DEFAULT_CROP["y_end"]
    x_start = args.x_start if args.x_start is not None else DEFAULT_CROP["x_start"]
    x_end = args.x_end if args.x_end is not None else DEFAULT_CROP["x_end"]

    if args.dry_run:
        print(f"Would crop images from {args.input_dir} to {args.output_dir}")
        print(f"Crop boundaries: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")
        return

    results = process_directory(
        args.input_dir,
        args.output_dir,
        y_start=y_start,
        y_end=y_end,
        x_start=x_start,
        x_end=x_end,
        auto_detect=args.auto_detect,
    )

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    print(f"\nProcessed {len(results)} images: {success_count} success, {error_count} errors")

    if success_count > 0:
        sizes = [r["size"] for r in results if r["status"] == "success"]
        print(f"Output size: {sizes[0][0]}x{sizes[0][1]}")


if __name__ == "__main__":
    main()
