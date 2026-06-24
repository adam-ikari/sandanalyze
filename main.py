"""SandAnalyze - 沙粒形态分析系统入口。

Usage:
    sandanalyze                           # Launch Streamlit web app
    sandanalyze --port 8080               # Launch with custom port
    sandanalyze batch input/ output/      # Batch process images
    sandanalyze batch input/ output/ \
        --adaptive-block-size 51 \
        --adaptive-c 2 \
        --max-area 15000 \
        --min-area 1000

CLI commands:
    web          Launch Streamlit web app (default)
    batch        Batch process images in a directory
    detect       Detect grains in a single image
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from core.batch import process_batch, process_single_image
from core.detector import FlocculationConfig
from core.pipeline import run_detection_pipeline
from core.preprocessor import PreprocessConfig, auto_detect_preset, auto_tune_for_microscope


def _build_preprocess_config(args: argparse.Namespace) -> PreprocessConfig:
    """Build PreprocessConfig from CLI arguments."""
    config = PreprocessConfig()

    if args.preset:
        config = PreprocessConfig.from_preset(args.preset)
    elif args.auto_tune:
        # Auto-tune requires an image, handled by the caller
        pass
    else:
        config = PreprocessConfig(
            blur_kernel=args.blur_kernel,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
            morph_kernel_size=args.morph_kernel_size,
            min_area=args.min_area,
            use_clahe=args.use_clahe,
            edge_clip_limit=args.edge_clip_limit,
            edge_blur_kernel=args.edge_blur_kernel,
            edge_adaptive_block_size=args.edge_adaptive_block_size,
            edge_adaptive_c=args.edge_adaptive_c,
            texture_window=args.texture_window,
            texture_std_threshold=args.texture_std_threshold,
            texture_diff_threshold=args.texture_diff_threshold,
        )

    return config


def _build_flocculation_config(args: argparse.Namespace) -> FlocculationConfig:
    """Build FlocculationConfig from CLI arguments."""
    return FlocculationConfig(
        min_area=args.floc_min_area,
        max_area=args.floc_max_area,
        min_circularity=args.floc_min_circularity,
        max_circularity=args.floc_max_circularity,
        min_convexity=args.floc_min_convexity,
        max_convexity=args.floc_max_convexity,
        max_aspect_ratio=args.floc_max_aspect_ratio,
    )


def cmd_web(args: argparse.Namespace) -> int:
    """Launch Streamlit web app."""
    from streamlit.web import cli as stcli

    port_arg = []
    if args.port:
        port_arg = ["--server.port", str(args.port)]

    sys.argv = ["streamlit", "run", "app.py", *port_arg]
    return stcli.main()


def cmd_batch(args: argparse.Namespace) -> int:
    """Batch process images in a directory."""
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    # Build config
    config = _build_preprocess_config(args)
    floc_config = _build_flocculation_config(args)

    # Auto-tune if requested (use first image for tuning)
    if args.auto_tune:
        image_files = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        if image_files:
            sample_image = cv2.imread(str(image_files[0]), cv2.IMREAD_COLOR)
            if sample_image is not None:
                detected_preset = auto_detect_preset(sample_image)
                config = PreprocessConfig.from_preset(detected_preset)
                config, detection_params = auto_tune_for_microscope(sample_image)
                print(f"Auto-detected preset: {detected_preset}")
                print(f"  blur={config.blur_kernel}, block={config.adaptive_block_size}, "
                      f"adaptive_c={config.adaptive_c}, min_area={detection_params.get('min_area', config.min_area)}")
                # Override min_area from auto-tune
                if "min_area" in detection_params:
                    config.min_area = detection_params["min_area"]

    # Run batch processing
    print(f"Processing images from {input_dir} -> {output_dir}")
    print(f"Config: blur={config.blur_kernel}, block={config.adaptive_block_size}, "
          f"adaptive_c={config.adaptive_c}, morph_kernel={config.morph_kernel_size}, "
          f"min_area={config.min_area}, max_area={args.max_area}")
    print(f"Flocculation: min_area={floc_config.min_area}, max_area={floc_config.max_area}")
    print("-" * 60)

    summary = process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config,
        floc_config=floc_config,
        border_margin=args.border_margin,
        generate_pdf=not args.no_pdf,
    )

    print("-" * 60)
    print(f"Batch complete: {summary.successful}/{summary.total_images} succeeded")
    print(f"Total grains detected: {summary.total_grains}")
    print(f"Success rate: {summary.success_rate:.1f}%")

    if summary.failed > 0:
        print("\nFailed images:")
        for result in summary.results:
            if not result.success:
                print(f"  - {result.image_path}: {result.error_message}")
        return 1

    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    """Detect grains in a single image and print results."""
    image_path = args.image_path
    output_dir = args.output_dir

    if not os.path.isfile(image_path):
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        return 1

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Build config
    config = _build_preprocess_config(args)
    floc_config = _build_flocculation_config(args)

    # Auto-tune if requested
    if args.auto_tune:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            detected_preset = auto_detect_preset(image)
            config = PreprocessConfig.from_preset(detected_preset)
            config, detection_params = auto_tune_for_microscope(image)
            print(f"Auto-detected preset: {detected_preset}")
            if "min_area" in detection_params:
                config.min_area = detection_params["min_area"]

    print(f"Processing: {image_path}")
    print(f"Config: blur={config.blur_kernel}, block={config.adaptive_block_size}, "
          f"adaptive_c={config.adaptive_c}, min_area={config.min_area}, max_area={args.max_area}")
    print("-" * 60)

    result = process_single_image(
        image_path=image_path,
        output_dir=output_dir or ".",
        config=config,
        floc_config=floc_config,
        border_margin=args.border_margin,
        generate_pdf=not args.no_pdf,
    )

    if not result.success:
        print(f"Error: {result.error_message}", file=sys.stderr)
        return 1

    print(f"Grains detected: {result.grain_count}")
    if result.statistics:
        stats = result.statistics
        print(f"Mean circularity: {stats.circularity_mean:.4f}")
        print(f"Mean sphericity: {stats.sphericity_mean:.4f}")
        print(f"Mean d_eq: {stats.d_eq_mean:.4f}")
        if stats.zingg_counts:
            print("Classification:")
            for key, cnt in stats.zingg_counts.items():
                pct = cnt / stats.count * 100 if stats.count > 0 else 0
                print(f"  {key}: {cnt} ({pct:.1f}%)")
        if stats.flocculation_count > 0:
            print(f"Flocculation: {stats.flocculation_count} ({stats.flocculation_ratio:.1%})")

    if output_dir:
        print(f"\nOutputs saved to {output_dir}:")
        print(f"  CSV: {result.output_csv_path}")
        print(f"  Annotated image: {result.output_image_path}")
        if result.output_pdf_path:
            print(f"  PDF: {result.output_pdf_path}")

    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common detection arguments to a subparser."""
    # Preprocessing parameters
    prep = parser.add_argument_group("Preprocessing Parameters")
    prep.add_argument(
        "--blur-kernel", type=int, default=5,
        help="Gaussian blur kernel size (odd, default: 5)",
    )
    prep.add_argument(
        "--adaptive-block-size", type=int, default=51,
        help="Adaptive threshold block size (odd, default: 51)",
    )
    prep.add_argument(
        "--adaptive-c", type=int, default=2,
        help="Adaptive threshold constant C (default: 2, lower = more sensitive)",
    )
    prep.add_argument(
        "--morph-kernel-size", type=int, default=3,
        help="Morphological opening kernel size (odd, default: 3)",
    )
    prep.add_argument(
        "--min-area", type=int, default=1000,
        help="Minimum grain area in pixels (default: 1000)",
    )
    prep.add_argument(
        "--max-area", type=int, default=15000,
        help="Maximum grain area in pixels, triggers splitting if exceeded (default: 15000)",
    )
    prep.add_argument(
        "--use-clahe", action="store_true", default=True,
        help="Enable CLAHE enhancement (default: True)",
    )
    prep.add_argument(
        "--no-clahe", action="store_false", dest="use_clahe",
        help="Disable CLAHE enhancement",
    )
    prep.add_argument(
        "--preset", choices=["default", "macro_sand", "microscope", "shadow"],
        help="Use a predefined parameter preset",
    )
    prep.add_argument(
        "--auto-tune", action="store_true",
        help="Automatically tune parameters based on image characteristics",
    )

    # Edge branch parameters
    edge = parser.add_argument_group("Edge Branch Parameters")
    edge.add_argument(
        "--edge-clip-limit", type=float, default=3.0,
        help="Edge branch CLAHE clip limit (default: 3.0)",
    )
    edge.add_argument(
        "--edge-blur-kernel", type=int, default=5,
        help="Edge branch blur kernel (odd, default: 5)",
    )
    edge.add_argument(
        "--edge-adaptive-block-size", type=int, default=31,
        help="Edge branch adaptive block size (odd, default: 31)",
    )
    edge.add_argument(
        "--edge-adaptive-c", type=int, default=2,
        help="Edge branch adaptive C (default: 2)",
    )

    # Texture branch parameters
    texture = parser.add_argument_group("Texture Branch Parameters")
    texture.add_argument(
        "--texture-window", type=int, default=15,
        help="Texture analysis window size (odd, default: 15)",
    )
    texture.add_argument(
        "--texture-std-threshold", type=float, default=10.0,
        help="Texture standard deviation threshold (default: 10.0)",
    )
    texture.add_argument(
        "--texture-diff-threshold", type=float, default=10.0,
        help="Texture difference threshold (default: 10.0)",
    )

    # Detection options
    det = parser.add_argument_group("Detection Options")
    det.add_argument(
        "--border-margin", type=int, default=5,
        help="Border margin for edge filtering in pixels (default: 5)",
    )
    det.add_argument(
        "--hull-expansion-ratio", type=float, default=1.5,
        help="Hull expansion ratio threshold (default: 1.5)",
    )
    det.add_argument(
        "--no-splitting", action="store_true",
        help="Disable oversized component splitting (enabled by default)",
    )

    # Flocculation parameters
    floc = parser.add_argument_group("Flocculation Detection")
    floc.add_argument(
        "--floc-min-area", type=int, default=3000,
        help="Minimum area for flocculation detection (default: 3000)",
    )
    floc.add_argument(
        "--floc-max-area", type=int, default=50000,
        help="Maximum area for flocculation detection (default: 50000)",
    )
    floc.add_argument(
        "--floc-min-circularity", type=float, default=0.01,
        help="Minimum circularity for flocculation (default: 0.01)",
    )
    floc.add_argument(
        "--floc-max-circularity", type=float, default=0.3,
        help="Maximum circularity for flocculation (default: 0.3)",
    )
    floc.add_argument(
        "--floc-min-convexity", type=float, default=0.2,
        help="Minimum convexity for flocculation (default: 0.2)",
    )
    floc.add_argument(
        "--floc-max-convexity", type=float, default=0.7,
        help="Maximum convexity for flocculation (default: 0.7)",
    )
    floc.add_argument(
        "--floc-max-aspect-ratio", type=float, default=5.0,
        help="Maximum aspect ratio for flocculation (default: 5.0)",
    )


def main() -> int:
    """Main entry point for SandAnalyze CLI."""
    parser = argparse.ArgumentParser(
        prog="sandanalyze",
        description="SandAnalyze - Sand grain morphology analysis system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sandanalyze                              # Launch web app
  sandanalyze --port 8080                  # Launch web app on port 8080
  sandanalyze batch input/ output/         # Batch process all images
  sandanalyze batch input/ output/ --auto-tune
  sandanalyze batch input/ output/ \
      --adaptive-block-size 51 --adaptive-c 2 --max-area 15000
  sandanalyze detect image.png --output-dir results/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Web command (default) ──────────────────────────────────────────────────
    web_parser = subparsers.add_parser(
        "web", help="Launch Streamlit web app (default if no command given)",
    )
    web_parser.add_argument(
        "--port", type=int,
        help="Port for Streamlit server",
    )

    # ── Batch command ──────────────────────────────────────────────────────────
    batch_parser = subparsers.add_parser(
        "batch", help="Batch process images in a directory",
    )
    batch_parser.add_argument(
        "input_dir", help="Directory containing input images",
    )
    batch_parser.add_argument(
        "output_dir", help="Directory for output files",
    )
    batch_parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip PDF report generation",
    )
    _add_common_args(batch_parser)

    # ── Detect command ───────────────────────────────────────────────────────
    detect_parser = subparsers.add_parser(
        "detect", help="Detect grains in a single image",
    )
    detect_parser.add_argument(
        "image_path", help="Path to input image",
    )
    detect_parser.add_argument(
        "--output-dir", "-o",
        help="Directory for output files (prints to stdout if omitted)",
    )
    detect_parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip PDF report generation",
    )
    _add_common_args(detect_parser)

    # Parse args - default to web if no command given
    args = parser.parse_args()

    # Ensure port attribute exists when defaulting to web
    if not hasattr(args, "port"):
        args.port = None

    if args.command == "web":
        return cmd_web(args)
    elif args.command == "batch":
        return cmd_batch(args)
    elif args.command == "detect":
        return cmd_detect(args)
    else:
        # Default: launch web app
        return cmd_web(args)


if __name__ == "__main__":
    sys.exit(main())
