"""Compare detection results with and without the post-processing filter on 25 images.

The post-processing filter in core/detector.py is:
    if area < 1000 and circularity < 0.3: continue

This script runs detection twice per image (with and without the filter),
records statistics, and produces a comparison chart + report.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure the project root is on the path so we can import core modules
PROJECT_ROOT = Path("/home/gem/project/sandanalyze/.worktrees/tech-debt-cleanup")
sys.path.insert(0, str(PROJECT_ROOT))

from core.detector import detect_grains, DetectionResult  # noqa: E402
from core.preprocessor import auto_tune_for_microscope  # noqa: E402


def detect_grains_without_filter(
    image: np.ndarray,
    config,
    min_area: int = 1000,
    max_area: int = 15000,
    border_margin: int = 5,
    hull_expansion_ratio: float = 1.5,
    floc_config=None,
    crop_black_background: bool = True,
) -> list[DetectionResult]:
    """Run the v6 detection pipeline *without* the post-processing filter.

    This is a copy of detect_grains with the filter block commented out so
    we can compare counts directly.
    """
    from core.detector import FlocculationConfig
    import numpy as np
    import cv2

    if floc_config is None:
        floc_config = FlocculationConfig()

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h_img, w_img = gray.shape[:2]

    # Step 1: Optional black background crop
    x, y, w, h = 0, 0, w_img, h_img
    if crop_black_background:
        _, bright_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
        largest_bright_area = 0
        max_label = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > largest_bright_area:
                largest_bright_area = stats[i, cv2.CC_STAT_AREA]
                max_label = i
        x = stats[max_label, cv2.CC_STAT_LEFT]
        y = stats[max_label, cv2.CC_STAT_TOP]
        w = stats[max_label, cv2.CC_STAT_WIDTH]
        h = stats[max_label, cv2.CC_STAT_HEIGHT]

    roi_gray = gray[y : y + h, x : x + w]

    # Step 2: Adaptive threshold in ROI
    roi_thresh = cv2.adaptiveThreshold(
        roi_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.adaptive_block_size,
        config.adaptive_c,
    )

    # Step 3: Morphological open
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
    )
    opened = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iter or 1)

    # Step 4: Edge filtering based on ROI boundary using connected components
    num_labels, labels, stats_arr, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered = np.zeros_like(opened)
    for i in range(1, num_labels):
        area = stats_arr[i, cv2.CC_STAT_AREA]
        lx = stats_arr[i, cv2.CC_STAT_LEFT]
        ly = stats_arr[i, cv2.CC_STAT_TOP]
        lw = stats_arr[i, cv2.CC_STAT_WIDTH]
        lh = stats_arr[i, cv2.CC_STAT_HEIGHT]

        touches_border = (
            lx <= 0 or ly <= 0 or lx + lw >= w or ly + lh >= h
        )

        if touches_border and area >= floc_config.min_area:
            allow_border = True
        else:
            allow_border = not touches_border

        if min_area <= area <= max_area and allow_border:
            filtered[labels == i] = 255

    # Step 5: Contour detection
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Per-contour feature computation + hull smoothing / mask filling
    results = []
    for cnt in contours:
        cnt_global = cnt + np.array([x, y])

        temp_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt_global], -1, 255, thickness=cv2.FILLED)
        area = float(cv2.countNonZero(temp_mask))

        perimeter = cv2.arcLength(cnt_global, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        hull = cv2.convexHull(cnt_global)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        is_potential_floc = (
            area >= floc_config.min_area and circularity < 0.1 and convexity < 0.5
        )

        if circularity < 0.02 and area > 5000 and not is_potential_floc:
            continue

        x2, y2, bw, bh = cv2.boundingRect(cnt_global)
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 0

        if len(cnt_global) >= 5:
            ellipse = cv2.fitEllipse(cnt_global)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
        else:
            major_axis = np.sqrt(area * 4 / np.pi)
            minor_axis = major_axis

        is_floc = False
        if area >= floc_config.min_area and area <= floc_config.max_area:
            conditions_met = sum([
                circularity <= floc_config.max_circularity,
                convexity <= floc_config.max_convexity,
                aspect_ratio >= floc_config.max_aspect_ratio,
            ])
            if conditions_met >= 2:
                is_floc = True
            elif conditions_met == 1 and circularity < 0.1 and convexity < 0.5:
                is_floc = True

        # ------------------------------------------------------------------
        # POST-PROCESSING FILTER REMOVED FOR THIS RUN
        # if area < 1000 and circularity < 0.3:
        #     continue
        # ------------------------------------------------------------------

        if hull_area / area < hull_expansion_ratio:
            final_contour = hull
        else:
            final_contour = cnt_global

        grain_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(grain_mask, [final_contour], -1, 255, thickness=cv2.FILLED)

        results.append(
            DetectionResult(
                contour=final_contour,
                mask=grain_mask,
                area=area,
                perimeter=perimeter,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                major_axis=major_axis,
                minor_axis=minor_axis,
                convexity=convexity,
                is_flocculation=is_floc,
            )
        )

    results.sort(key=lambda r: r.area, reverse=True)
    return results


def main():
    test_dir = Path("/home/gem/project/sandanalyze/data/test")
    image_paths = sorted(test_dir.glob("*.png"))
    if len(image_paths) != 25:
        print(f"WARNING: Expected 25 images, found {len(image_paths)}")

    records = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not load {img_path}")
            continue

        config, det_params = auto_tune_for_microscope(image)
        min_area = det_params.get("min_area", 1000)
        max_area = det_params.get("max_area", 15000)

        # Run WITH filter (current code)
        results_with = detect_grains(
            image,
            config,
            min_area=min_area,
            max_area=max_area,
        )
        # Run WITHOUT filter
        results_without = detect_grains_without_filter(
            image,
            config,
            min_area=min_area,
            max_area=max_area,
        )

        # Identify filtered-out detections
        filtered_out = []
        for r in results_without:
            if r.area < 1000 and r.circularity < 0.3:
                filtered_out.append(r)

        records.append(
            {
                "image": img_path.name,
                "without_filter": len(results_without),
                "with_filter": len(results_with),
                "filtered_count": len(filtered_out),
                "filtered_areas": [r.area for r in filtered_out],
                "filtered_circularities": [r.circularity for r in filtered_out],
            }
        )

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    total_without = sum(r["without_filter"] for r in records)
    total_with = sum(r["with_filter"] for r in records)
    total_filtered = sum(r["filtered_count"] for r in records)

    avg_without = total_without / len(records) if records else 0
    avg_with = total_with / len(records) if records else 0
    avg_filtered = total_filtered / len(records) if records else 0

    all_filtered_areas = [
        area for r in records for area in r["filtered_areas"]
    ]
    all_filtered_circularities = [
        c for r in records for c in r["filtered_circularities"]
    ]

    avg_filtered_area = np.mean(all_filtered_areas) if all_filtered_areas else 0.0
    avg_filtered_circularity = (
        np.mean(all_filtered_circularities) if all_filtered_circularities else 0.0
    )

    print(f"Images processed: {len(records)}")
    print(f"Average grains WITHOUT filter: {avg_without:.2f}")
    print(f"Average grains WITH filter:    {avg_with:.2f}")
    print(f"Average filtered out:          {avg_filtered:.2f}")
    print(f"Avg area of filtered:          {avg_filtered_area:.2f}")
    print(f"Avg circularity of filtered:   {avg_filtered_circularity:.4f}")

    # ------------------------------------------------------------------
    # Comparison chart
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Post-Processing Filter Comparison (25 Images)", fontsize=14, fontweight="bold")

    # 1. Bar chart: per-image counts
    ax = axes[0, 0]
    x = np.arange(len(records))
    width = 0.35
    ax.bar(x - width / 2, [r["without_filter"] for r in records], width, label="Without filter")
    ax.bar(x + width / 2, [r["with_filter"] for r in records], width, label="With filter")
    ax.set_xlabel("Image index")
    ax.set_ylabel("Grain count")
    ax.set_title("Grain Counts per Image")
    ax.legend()

    # 2. Histogram: filtered-out counts
    ax = axes[0, 1]
    ax.hist([r["filtered_count"] for r in records], bins=range(0, max(r["filtered_count"] for r in records) + 2), edgecolor="black")
    ax.set_xlabel("Filtered-out detections per image")
    ax.set_ylabel("Number of images")
    ax.set_title("Distribution of Filtered Detections")

    # 3. Histogram: areas of filtered detections
    ax = axes[1, 0]
    if all_filtered_areas:
        ax.hist(all_filtered_areas, bins=30, edgecolor="black")
    ax.set_xlabel("Area (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Area Distribution of Filtered Detections")
    ax.axvline(x=1000, color="r", linestyle="--", label="Filter threshold (1000)")
    ax.legend()

    # 4. Histogram: circularity of filtered detections
    ax = axes[1, 1]
    if all_filtered_circularities:
        ax.hist(all_filtered_circularities, bins=30, edgecolor="black")
    ax.set_xlabel("Circularity")
    ax.set_ylabel("Count")
    ax.set_title("Circularity Distribution of Filtered Detections")
    ax.axvline(x=0.3, color="r", linestyle="--", label="Filter threshold (0.3)")
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    chart_path = Path("/home/gem/project/sandanalyze/cv-parameter-tuning-workspace/iteration-1/post_filter_test.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to: {chart_path}")

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    report_path = Path("/home/gem/project/sandanalyze/cv-parameter-tuning-workspace/iteration-1/post_filter_test_report.md")
    with open(report_path, "w") as f:
        f.write("# Post-Processing Filter Comparison Report\n\n")
        f.write(f"**Date:** 2026-06-19\n\n")
        f.write(f"**Images tested:** {len(records)}\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Average grains WITHOUT filter:** {avg_without:.2f}\n")
        f.write(f"- **Average grains WITH filter:**    {avg_with:.2f}\n")
        f.write(f"- **Average filtered out:**           {avg_filtered:.2f}\n")
        f.write(f"- **Avg area of filtered:**            {avg_filtered_area:.2f} px\n")
        f.write(f"- **Avg circularity of filtered:**    {avg_filtered_circularity:.4f}\n\n")
        f.write("## Per-Image Results\n\n")
        f.write("| Image | Without | With | Filtered |\n")
        f.write("|-------|---------|------|----------|\n")
        for r in records:
            f.write(f"| {r['image']} | {r['without_filter']} | {r['with_filter']} | {r['filtered_count']} |\n")
        f.write("\n")
        f.write("## Analysis\n\n")
        f.write("The post-processing filter `if area < 1000 and circularity < 0.3: continue` ")
        f.write("removes small, non-circular detections that are likely noise or fragments.\n\n")
        if avg_filtered > 0:
            f.write("### Are filtered detections actually noise?\n\n")
            f.write(f"- Average area of filtered detections: **{avg_filtered_area:.2f} px** (threshold: 1000)\n")
            f.write(f"- Average circularity: **{avg_filtered_circularity:.4f}** (threshold: 0.3)\n")
            f.write("- These values indicate the filtered objects are small and irregular, ")
            f.write("consistent with noise or broken grain fragments.\n")
        else:
            f.write("No detections were filtered out by this rule across the test set.\n")

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
