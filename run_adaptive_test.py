#!/usr/bin/env python3
"""Compare adaptive tuning vs fixed defaults on 25 microscope images."""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, '/home/gem/project/sandanalyze/.worktrees/tech-debt-cleanup')

from core.detector import detect_grains
from core.preprocessor import PreprocessConfig, auto_tune_for_microscope

# Paths
IMAGE_DIR = '/home/gem/project/sandanalyze/data/test'
OUTPUT_DIR = '/home/gem/project/sandanalyze/cv-parameter-tuning-workspace/iteration-1'
OUTPUT_CHART = os.path.join(OUTPUT_DIR, 'adaptive_test.png')
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, 'adaptive_test_report.md')

# Fixed defaults config
fixed_config = PreprocessConfig(
    blur_kernel=5,
    adaptive_block_size=51,
    adaptive_c=5,
    morph_kernel_size=3,
    morph_open_iter=1,
    morph_close_iter=1,
    min_area=800,
    use_clahe=True,
)


def load_images(directory):
    """Load all images from directory, sorted."""
    image_files = sorted(
        [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    )
    images = {}
    for fname in image_files:
        path = os.path.join(directory, fname)
        img = cv2.imread(path)
        if img is not None:
            images[fname] = img
    return images


def count_grains(image, config):
    """Run detection and return grain count."""
    results = detect_grains(image, config)
    return len(results)


def main():
    print("Loading images...")
    images = load_images(IMAGE_DIR)
    print(f"Loaded {len(images)} images.")

    fixed_counts = []
    adaptive_counts = []
    image_names = []
    adaptive_configs = []

    for fname, img in images.items():
        print(f"Processing {fname}...")
        fixed_count = count_grains(img, fixed_config)
        adaptive_cfg = auto_tune_for_microscope(img)
        adaptive_count = count_grains(img, adaptive_cfg)

        fixed_counts.append(fixed_count)
        adaptive_counts.append(adaptive_count)
        image_names.append(fname)
        adaptive_configs.append(adaptive_cfg)
        print(f"  Fixed: {fixed_count}, Adaptive: {adaptive_count}")

    fixed_counts = np.array(fixed_counts)
    adaptive_counts = np.array(adaptive_counts)

    # Statistics
    avg_fixed = np.mean(fixed_counts)
    avg_adaptive = np.mean(adaptive_counts)
    std_fixed = np.std(fixed_counts)
    std_adaptive = np.std(adaptive_counts)
    better_count = np.sum(adaptive_counts > fixed_counts)
    worse_count = np.sum(adaptive_counts < fixed_counts)
    same_count = np.sum(adaptive_counts == fixed_counts)

    print(f"\n=== Statistics ===")
    print(f"Average grain count (fixed):     {avg_fixed:.2f} (std={std_fixed:.2f})")
    print(f"Average grain count (adaptive):  {avg_adaptive:.2f} (std={std_adaptive:.2f})")
    print(f"Images where adaptive > fixed: {better_count}")
    print(f"Images where adaptive < fixed: {worse_count}")
    print(f"Images where adaptive == fixed: {same_count}")

    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bar chart: fixed vs adaptive per image
    ax = axes[0, 0]
    x = np.arange(len(image_names))
    width = 0.35
    ax.bar(x - width/2, fixed_counts, width, label='Fixed', alpha=0.8)
    ax.bar(x + width/2, adaptive_counts, width, label='Adaptive', alpha=0.8)
    ax.set_xlabel('Image')
    ax.set_ylabel('Grain Count')
    ax.set_title('Grain Count: Fixed vs Adaptive per Image')
    ax.set_xticks(x)
    ax.set_xticklabels([n[:6] for n in image_names], rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Histogram of counts
    ax = axes[0, 1]
    ax.hist(fixed_counts, bins=15, alpha=0.5, label='Fixed')
    ax.hist(adaptive_counts, bins=15, alpha=0.5, label='Adaptive')
    ax.axvline(avg_fixed, color='blue', linestyle='--', label=f'Fixed avg={avg_fixed:.1f}')
    ax.axvline(avg_adaptive, color='orange', linestyle='--', label=f'Adaptive avg={avg_adaptive:.1f}')
    ax.set_xlabel('Grain Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Grain Counts')
    ax.legend()
    ax.grid(alpha=0.3)

    # Scatter: fixed vs adaptive
    ax = axes[1, 0]
    ax.scatter(fixed_counts, adaptive_counts, alpha=0.7)
    max_val = max(fixed_counts.max(), adaptive_counts.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax.set_xlabel('Fixed Grain Count')
    ax.set_ylabel('Adaptive Grain Count')
    ax.set_title('Adaptive vs Fixed (closer to y=x = more similar)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Difference bar chart
    ax = axes[1, 1]
    diffs = adaptive_counts - fixed_counts
    colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in diffs]
    ax.bar(x, diffs, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Image')
    ax.set_ylabel('Adaptive - Fixed')
    ax.set_title('Difference in Grain Count (Adaptive - Fixed)')
    ax.set_xticks(x)
    ax.set_xticklabels([n[:6] for n in image_names], rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {OUTPUT_CHART}")

    # Write report
    report = f"""# Adaptive vs Fixed Parameter Tuning Report

## Test Setup
- **Images tested:** {len(image_names)}
- **Image directory:** `{IMAGE_DIR}`
- **Fixed config:** blur=5, block=51, min_area=800
- **Adaptive function:** `auto_tune_for_microscope(image)`

## Summary Statistics

| Metric | Fixed Defaults | Adaptive Tuning |
|--------|---------------|-----------------|
| Average grain count | {avg_fixed:.2f} | {avg_adaptive:.2f} |
| Standard deviation | {std_fixed:.2f} | {std_adaptive:.2f} |
| Min count | {int(fixed_counts.min())} | {int(adaptive_counts.min())} |
| Max count | {int(fixed_counts.max())} | {int(adaptive_counts.max())} |

## Comparison

- **Images where adaptive > fixed:** {better_count} ({better_count/len(image_names)*100:.1f}%)
- **Images where adaptive < fixed:** {worse_count} ({worse_count/len(image_names)*100:.1f}%)
- **Images where adaptive == fixed:** {same_count} ({same_count/len(image_names)*100:.1f}%)

## Per-Image Results

| Image | Fixed | Adaptive | Diff | Adaptive Config (blur, block, min_area) |
|-------|-------|----------|------|------------------------------------------|
"""
    for i, fname in enumerate(image_names):
        cfg = adaptive_configs[i]
        report += f"| {fname} | {fixed_counts[i]} | {adaptive_counts[i]} | {adaptive_counts[i]-fixed_counts[i]:+d} | blur={cfg.blur_kernel}, block={cfg.adaptive_block_size}, min_area={cfg.min_area} |\n"

    report += f"""
## Interpretation

- Adaptive tuning adjusts `blur_kernel` and `min_area` based on image noise level.
- On average, adaptive tuning {'increases' if avg_adaptive > avg_fixed else 'decreases'} grain count by {abs(avg_adaptive - avg_fixed):.2f} grains per image.
- The adaptive approach is {'better' if better_count > worse_count else 'worse'} on {max(better_count, worse_count)} out of {len(image_names)} images.

## Chart

![Comparison Chart](adaptive_test.png)
"""

    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)
    print(f"Report saved to: {OUTPUT_REPORT}")


if __name__ == '__main__':
    main()
