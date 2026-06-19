#!/usr/bin/env python3
"""Multi-feature adaptive parameter tuning comparison test.

Compares fixed default parameters vs adaptive multi-feature tuning
on 25 microscope images.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Ensure project root is on path
sys.path.insert(0, "/home/gem/project/sandanalyze/.worktrees/tech-debt-cleanup")

from core.detector import detect_grains
from core.preprocessor import PreprocessConfig, auto_tune_for_microscope

TEST_DIR = "/home/gem/project/sandanalyze/data/test"
OUTPUT_DIR = "/home/gem/project/sandanalyze/cv-parameter-tuning-workspace/iteration-1"
CHART_PATH = os.path.join(OUTPUT_DIR, "multi_feature_test.png")
REPORT_PATH = os.path.join(OUTPUT_DIR, "multi_feature_test_report.md")

# Fixed defaults
FIXED_CONFIG = PreprocessConfig(
    blur_kernel=5,
    adaptive_block_size=51,
    adaptive_c=5,
    min_area=800,
)
FIXED_MIN_AREA = 800
FIXED_MAX_AREA = 15000


def run_fixed(image):
    """Run detection with fixed default parameters."""
    results = detect_grains(
        image,
        config=FIXED_CONFIG,
        min_area=FIXED_MIN_AREA,
        max_area=FIXED_MAX_AREA,
    )
    return len(results)


def run_adaptive(image):
    """Run detection with adaptive multi-feature tuning."""
    config, det_params = auto_tune_for_microscope(image)
    results = detect_grains(
        image,
        config=config,
        min_area=det_params["min_area"],
        max_area=det_params["max_area"],
    )
    return len(results), config, det_params


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all 25 images
    image_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])
    print(f"Found {len(image_files)} images in {TEST_DIR}")

    results = []
    all_features = []

    for fname in image_files:
        path = os.path.join(TEST_DIR, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"  WARNING: Could not load {fname}")
            continue

        # Fixed
        fixed_count = run_fixed(image)

        # Adaptive
        adaptive_count, adaptive_config, det_params = run_adaptive(image)

        # Also get features for correlation analysis
        from core.preprocessor import analyze_image_characteristics
        features = analyze_image_characteristics(image)

        results.append({
            "filename": fname,
            "fixed_count": fixed_count,
            "adaptive_count": adaptive_count,
            "diff": adaptive_count - fixed_count,
            "config": adaptive_config,
            "det_params": det_params,
            "features": features,
        })
        all_features.append(features)

        print(f"  {fname}: fixed={fixed_count}, adaptive={adaptive_count}, "
              f"blur={adaptive_config.blur_kernel}, min_area={det_params['min_area']}, "
              f"block={adaptive_config.adaptive_block_size}, C={adaptive_config.adaptive_c}")

    # Statistics
    fixed_counts = [r["fixed_count"] for r in results]
    adaptive_counts = [r["adaptive_count"] for r in results]

    fixed_mean = np.mean(fixed_counts)
    adaptive_mean = np.mean(adaptive_counts)
    fixed_std = np.std(fixed_counts)
    adaptive_std = np.std(adaptive_counts)

    better_adaptive = sum(1 for r in results if r["adaptive_count"] > r["fixed_count"])
    better_fixed = sum(1 for r in results if r["fixed_count"] > r["adaptive_count"])
    equal = sum(1 for r in results if r["fixed_count"] == r["adaptive_count"])

    # Parameter selection frequency
    blur_selections = Counter(r["config"].blur_kernel for r in results)
    min_area_selections = Counter(r["det_params"]["min_area"] for r in results)
    block_selections = Counter(r["config"].adaptive_block_size for r in results)
    c_selections = Counter(r["config"].adaptive_c for r in results)

    # Correlation between features and selected parameters
    feature_names = ["noise", "brightness", "contrast", "clarity", "texture_complexity"]
    blur_values = [r["config"].blur_kernel for r in results]
    min_area_values = [r["det_params"]["min_area"] for r in results]

    correlations = {}
    for feat in feature_names:
        feat_values = [r["features"][feat] for r in results]
        correlations[f"{feat}_vs_blur"] = np.corrcoef(feat_values, blur_values)[0, 1]
        correlations[f"{feat}_vs_min_area"] = np.corrcoef(feat_values, min_area_values)[0, 1]

    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multi-Feature Adaptive vs Fixed Default Parameter Tuning\n(25 Microscope Images)", fontsize=14)

    # 1. Bar chart: grain counts per image
    ax = axes[0, 0]
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width/2, fixed_counts, width, label="Fixed Defaults", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, adaptive_counts, width, label="Adaptive Multi-Feature", color="darkorange", alpha=0.8)
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Grain Count")
    ax.set_title("Grain Count per Image")
    ax.legend()
    ax.set_xticks(x[::5])
    ax.set_xticklabels([f"{i}" for i in x[::5]])

    # 2. Scatter: adaptive vs fixed
    ax = axes[0, 1]
    ax.scatter(fixed_counts, adaptive_counts, alpha=0.7, edgecolors="black")
    max_val = max(max(fixed_counts), max(adaptive_counts)) * 1.1
    ax.plot([0, max_val], [0, max_val], "r--", label="y=x (equal)")
    ax.set_xlabel("Fixed Default Count")
    ax.set_ylabel("Adaptive Multi-Feature Count")
    ax.set_title("Adaptive vs Fixed (Points above line = adaptive higher)")
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # 3. Difference histogram
    ax = axes[1, 0]
    diffs = [r["diff"] for r in results]
    ax.hist(diffs, bins=15, color="seagreen", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No difference")
    ax.set_xlabel("Adaptive - Fixed (grain count)")
    ax.set_ylabel("Number of Images")
    ax.set_title(f"Difference Distribution\nMean diff = {np.mean(diffs):+.1f}")
    ax.legend()

    # 4. Parameter selection frequency
    ax = axes[1, 1]
    param_labels = []
    param_counts = []
    colors = []
    for blur in sorted(blur_selections.keys()):
        for ma in sorted(min_area_selections.keys()):
            count = sum(1 for r in results if r["config"].blur_kernel == blur and r["det_params"]["min_area"] == ma)
            if count > 0:
                param_labels.append(f"blur={blur}\nmin_area={ma}")
                param_counts.append(count)
                colors.append("steelblue" if blur == 5 else "darkorange" if blur == 3 else "crimson")

    ax.barh(param_labels, param_counts, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Number of Images")
    ax.set_title("Adaptive Parameter Selection Frequency")
    for i, v in enumerate(param_counts):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {CHART_PATH}")

    # Write report
    report_lines = [
        "# Multi-Feature Adaptive Parameter Tuning Comparison Report",
        "",
        f"**Date:** 2026-06-19",
        f"**Images tested:** {len(results)}",
        f"**Fixed defaults:** blur={FIXED_CONFIG.blur_kernel}, block={FIXED_CONFIG.adaptive_block_size}, C={FIXED_CONFIG.adaptive_c}, min_area={FIXED_MIN_AREA}",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Fixed Defaults | Adaptive Multi-Feature |",
        "|--------|---------------|------------------------|",
        f"| Average grain count | {fixed_mean:.1f} | {adaptive_mean:.1f} |",
        f"| Standard deviation | {fixed_std:.1f} | {adaptive_std:.1f} |",
        f"| Total grains detected | {sum(fixed_counts)} | {sum(adaptive_counts)} |",
        "",
        "## Comparison",
        "",
        f"- **Adaptive higher:** {better_adaptive} images",
        f"- **Fixed higher:** {better_fixed} images",
        f"- **Equal:** {equal} images",
        "",
        "## Per-Image Results",
        "",
        "| Image | Fixed | Adaptive | Diff | blur | block | C | min_area |",
        "|-------|-------|----------|------|------|-------|---|----------|",
    ]

    for r in results:
        report_lines.append(
            f"| {r['filename']} | {r['fixed_count']} | {r['adaptive_count']} | "
            f"{r['diff']:+.0f} | {r['config'].blur_kernel} | {r['config'].adaptive_block_size} | "
            f"{r['config'].adaptive_c} | {r['det_params']['min_area']} |"
        )

    report_lines.extend([
        "",
        "## Parameter Selection Analysis",
        "",
        "### blur_kernel selection",
        "",
    ])
    for blur, count in sorted(blur_selections.items()):
        pct = count / len(results) * 100
        report_lines.append(f"- blur={blur}: {count} images ({pct:.0f}%)")

    report_lines.extend([
        "",
        "### min_area selection",
        "",
    ])
    for ma, count in sorted(min_area_selections.items()):
        pct = count / len(results) * 100
        report_lines.append(f"- min_area={ma}: {count} images ({pct:.0f}%)")

    report_lines.extend([
        "",
        "### adaptive_block_size selection",
        "",
    ])
    for block, count in sorted(block_selections.items()):
        pct = count / len(results) * 100
        report_lines.append(f"- block_size={block}: {count} images ({pct:.0f}%)")

    report_lines.extend([
        "",
        "### adaptive_c selection",
        "",
    ])
    for c, count in sorted(c_selections.items()):
        pct = count / len(results) * 100
        report_lines.append(f"- C={c}: {count} images ({pct:.0f}%)")

    report_lines.extend([
        "",
        "## Feature-Parameter Correlations",
        "",
        "| Feature | vs blur_kernel | vs min_area |",
        "|---------|---------------|-------------|",
    ])
    for feat in feature_names:
        report_lines.append(
            f"| {feat} | {correlations[f'{feat}_vs_blur']:.3f} | {correlations[f'{feat}_vs_min_area']:.3f} |"
        )

    report_lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    if adaptive_mean > fixed_mean:
        report_lines.append(
            f"Adaptive tuning detects on average **{adaptive_mean - fixed_mean:.1f} more grains per image** "
            f"({(adaptive_mean/fixed_mean - 1)*100:+.1f}%)."
        )
    else:
        report_lines.append(
            f"Adaptive tuning detects on average **{fixed_mean - adaptive_mean:.1f} fewer grains per image** "
            f"({(adaptive_mean/fixed_mean - 1)*100:+.1f}%)."
        )

    report_lines.extend([
        "",
        "The multi-feature approach selects parameters based on:",
        "- **Noise level**: drives blur_kernel and min_area",
        "- **Brightness**: drives adaptive_c",
        "- **Contrast**: drives blur_kernel and adaptive_block_size",
        "",
        "Strong positive correlations indicate features successfully guide parameter selection.",
        "",
    ])

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to: {REPORT_PATH}")
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Images tested: {len(results)}")
    print(f"Fixed avg: {fixed_mean:.1f} +/- {fixed_std:.1f}")
    print(f"Adaptive avg: {adaptive_mean:.1f} +/- {adaptive_std:.1f}")
    print(f"Adaptive higher on {better_adaptive}/{len(results)} images")
    print(f"Fixed higher on {better_fixed}/{len(results)} images")
    print(f"Equal on {equal}/{len(results)} images")


if __name__ == "__main__":
    main()
