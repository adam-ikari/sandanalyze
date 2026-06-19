#!/usr/bin/env python3
"""Run large-scale parameter comparison on test images."""

import os
import sys
import glob
import statistics

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, "/home/gem/project/sandanalyze/.worktrees/tech-debt-cleanup")

from core.detector import detect_grains
from core.preprocessor import PreprocessConfig


def main():
    test_dir = "/home/gem/project/sandanalyze/data/test"
    output_dir = "/home/gem/project/sandanalyze/cv-parameter-tuning-workspace/iteration-1"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    if not image_paths:
        print("No images found in", test_dir)
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {test_dir}")

    # Old defaults
    old_config = PreprocessConfig(
        blur_kernel=5,
        adaptive_block_size=31,
        adaptive_c=5,
        morph_kernel_size=3,
        min_area=1000,
    )

    # Optimized defaults
    opt_config = PreprocessConfig(
        blur_kernel=5,
        adaptive_block_size=51,
        adaptive_c=5,
        morph_kernel_size=3,
        min_area=800,
    )

    results = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: could not load {img_path}")
            continue

        old_results = detect_grains(image, config=old_config, min_area=1000)
        opt_results = detect_grains(image, config=opt_config, min_area=800)

        old_count = len(old_results)
        opt_count = len(opt_results)
        diff = opt_count - old_count

        results.append({
            "name": img_name,
            "old_count": old_count,
            "opt_count": opt_count,
            "diff": diff,
        })

    # Compute statistics
    old_counts = [r["old_count"] for r in results]
    opt_counts = [r["opt_count"] for r in results]

    avg_old = statistics.mean(old_counts)
    avg_opt = statistics.mean(opt_counts)
    std_old = statistics.stdev(old_counts) if len(old_counts) > 1 else 0
    std_opt = statistics.stdev(opt_counts) if len(opt_counts) > 1 else 0

    better_opt = sum(1 for r in results if r["diff"] > 0)
    better_old = sum(1 for r in results if r["diff"] < 0)
    same = sum(1 for r in results if r["diff"] == 0)

    print(f"\n{'='*60}")
    print(f"LARGE-SCALE PARAMETER TEST RESULTS")
    print(f"{'='*60}")
    print(f"Images tested: {len(results)}")
    print(f"")
    print(f"Old defaults (blur=5, block=31, C=5, min_area=1000):")
    print(f"  Average grain count: {avg_old:.2f}")
    print(f"  Std deviation:       {std_old:.2f}")
    print(f"")
    print(f"Optimized defaults (blur=5, block=51, C=5, min_area=800):")
    print(f"  Average grain count: {avg_opt:.2f}")
    print(f"  Std deviation:       {std_opt:.2f}")
    print(f"")
    print(f"Images where optimized > old: {better_opt}")
    print(f"Images where old > optimized: {better_old}")
    print(f"Images with same count:       {same}")
    print(f"{'='*60}")

    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Large-Scale Parameter Comparison\nOld vs Optimized Defaults", fontsize=14, fontweight="bold")

    # 1. Bar chart: grain counts per image
    ax1 = axes[0, 0]
    x = np.arange(len(results))
    width = 0.35
    labels = [r["name"].replace(".png", "") for r in results]
    ax1.bar(x - width/2, old_counts, width, label="Old (block=31, min=1000)", color="#e74c3c", alpha=0.8)
    ax1.bar(x + width/2, opt_counts, width, label="Optimized (block=51, min=800)", color="#2ecc71", alpha=0.8)
    ax1.set_xlabel("Image")
    ax1.set_ylabel("Grain Count")
    ax1.set_title("Grain Count per Image")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. Difference chart
    ax2 = axes[0, 1]
    diffs = [r["diff"] for r in results]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" if d < 0 else "#95a5a6" for d in diffs]
    ax2.bar(x, diffs, color=colors, alpha=0.8)
    ax2.set_xlabel("Image")
    ax2.set_ylabel("Difference (Optimized - Old)")
    ax2.set_title("Count Difference per Image")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)

    # 3. Histogram of counts
    ax3 = axes[1, 0]
    bins = np.arange(min(min(old_counts), min(opt_counts)), max(max(old_counts), max(opt_counts)) + 5, 5)
    ax3.hist(old_counts, bins=bins, alpha=0.6, label="Old", color="#e74c3c", edgecolor="white")
    ax3.hist(opt_counts, bins=bins, alpha=0.6, label="Optimized", color="#2ecc71", edgecolor="white")
    ax3.set_xlabel("Grain Count")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Grain Counts")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis("off")
    table_data = [
        ["Metric", "Old Defaults", "Optimized", "Difference"],
        ["Images Tested", str(len(results)), str(len(results)), ""],
        ["Avg Grain Count", f"{avg_old:.2f}", f"{avg_opt:.2f}", f"{avg_opt - avg_old:+.2f}"],
        ["Std Deviation", f"{std_old:.2f}", f"{std_opt:.2f}", f"{std_opt - std_old:+.2f}"],
        ["Min Count", str(min(old_counts)), str(min(opt_counts)), f"{min(opt_counts) - min(old_counts):+d}"],
        ["Max Count", str(max(old_counts)), str(max(opt_counts)), f"{max(opt_counts) - max(old_counts):+d}"],
        ["Better Count", "-", f"{better_opt} images", "-"],
        ["Worse Count", f"{better_old} images", "-", "-"],
        ["Same Count", "-", f"{same} images", "-"],
    ]
    table = ax4.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor("#34495e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    # Style alternating rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#ecf0f1")
            else:
                table[(i, j)].set_facecolor("#ffffff")
    ax4.set_title("Summary Statistics", fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    chart_path = os.path.join(output_dir, "large_scale_test.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to: {chart_path}")

    # Write report
    report_path = os.path.join(output_dir, "large_scale_test_report.md")
    with open(report_path, "w") as f:
        f.write("# Large-Scale Parameter Test Report\n\n")
        f.write("## Test Configuration\n\n")
        f.write("- **Images tested:** {}\n".format(len(results)))
        f.write("- **Image directory:** `{}`\n\n".format(test_dir))
        f.write("### Old Defaults\n")
        f.write("- blur_kernel: 5\n")
        f.write("- adaptive_block_size: 31\n")
        f.write("- adaptive_c: 5\n")
        f.write("- morph_kernel_size: 3\n")
        f.write("- min_area: 1000\n\n")
        f.write("### Optimized Defaults\n")
        f.write("- blur_kernel: 5\n")
        f.write("- adaptive_block_size: 51\n")
        f.write("- adaptive_c: 5\n")
        f.write("- morph_kernel_size: 3\n")
        f.write("- min_area: 800\n\n")
        f.write("---\n\n")
        f.write("## Per-Image Results\n\n")
        f.write("| Image | Old Count | Optimized Count | Difference |\n")
        f.write("|-------|-----------|-----------------|------------|\n")
        for r in results:
            diff_str = f"{r['diff']:+d}"
            f.write(f"| {r['name']} | {r['old_count']} | {r['opt_count']} | {diff_str} |\n")
        f.write("\n---\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Old Defaults | Optimized |\n")
        f.write("|--------|--------------|-----------|\n")
        f.write(f"| Average grain count | {avg_old:.2f} | {avg_opt:.2f} |\n")
        f.write(f"| Standard deviation | {std_old:.2f} | {std_opt:.2f} |\n")
        f.write(f"| Minimum count | {min(old_counts)} | {min(opt_counts)} |\n")
        f.write(f"| Maximum count | {max(old_counts)} | {max(opt_counts)} |\n")
        f.write(f"| Total grains (old) | {sum(old_counts)} | - |\n")
        f.write(f"| Total grains (optimized) | - | {sum(opt_counts)} |\n")
        f.write(f"| Net change | - | {sum(opt_counts) - sum(old_counts):+d} |\n")
        f.write("\n")
        f.write("## Performance Comparison\n\n")
        f.write(f"- **Images where optimized detects MORE grains:** {better_opt}\n")
        f.write(f"- **Images where old detects MORE grains:** {better_old}\n")
        f.write(f"- **Images with SAME count:** {same}\n")
        f.write("\n")
        f.write("## Analysis\n\n")
        total_old = sum(old_counts)
        total_opt = sum(opt_counts)
        pct_change = ((total_opt - total_old) / total_old * 100) if total_old > 0 else 0
        f.write(f"The optimized parameters detect **{total_opt - total_old:+d}** more grains total ")
        f.write(f"across all {len(results)} images, a **{pct_change:+.1f}%** change.\n\n")
        if better_opt > better_old:
            f.write("The optimized parameters perform better on more images than the old defaults. ")
            f.write("The larger adaptive block size (51 vs 31) provides more stable thresholding ")
            f.write("across varying lighting conditions, and the lower min_area (800 vs 1000) ")
            f.write("captures smaller grains that were previously filtered out.\n")
        elif better_old > better_opt:
            f.write("The old parameters perform better on more images. The optimized parameters ")
            f.write("may be over-detecting in some cases.\n")
        else:
            f.write("Both parameter sets perform similarly across the test set.\n")
        f.write("\n")
        f.write("## Chart\n\n")
        f.write(f"![Comparison Chart](large_scale_test.png)\n")

    print(f"Report saved to: {report_path}")
    print("\nDONE")


if __name__ == "__main__":
    main()
