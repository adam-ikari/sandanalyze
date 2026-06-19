"""Batch processing module for sand grain analysis.

Supports processing multiple images in a directory with progress tracking
and summary statistics.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.classifier import classify_grain
from core.detector import detect_grains, FlocculationConfig
from core.exporter import export_csv, export_annotated_image
from core.morphology import compute_morphology, compute_statistics, GrainMorphology, GrainStatistics
from core.preprocessor import PreprocessConfig
from core.report import generate_pdf_report
from core.traditional import GrainContour


@dataclass
class BatchResult:
    """Result of processing a single image."""

    image_path: str
    success: bool
    grain_count: int = 0
    morphologies: list[GrainMorphology] = field(default_factory=list)
    statistics: GrainStatistics | None = None
    error_message: str = ""
    output_csv_path: str = ""
    output_image_path: str = ""
    output_pdf_path: str = ""


@dataclass
class BatchSummary:
    """Summary across all processed images."""

    total_images: int = 0
    successful: int = 0
    failed: int = 0
    total_grains: int = 0
    results: list[BatchResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.successful / self.total_images * 100


def process_single_image(
    image_path: str,
    output_dir: str,
    config: PreprocessConfig | None = None,
    floc_config: FlocculationConfig | None = None,
    border_margin: int = 5,
    generate_pdf: bool = True,
) -> BatchResult:
    """Process a single image and export results.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_dir : str
        Directory for output files.
    config : PreprocessConfig | None, optional
        Preprocessing configuration. Uses defaults if None.
    floc_config : FlocculationConfig | None, optional
        Flocculation detection config. Uses defaults if None.
    border_margin : int, optional
        Distance from border for edge filtering.
    generate_pdf : bool, optional
        Whether to generate PDF report.

    Returns
    -------
    BatchResult
        Result of processing the image.
    """
    if config is None:
        config = PreprocessConfig()
    if floc_config is None:
        floc_config = FlocculationConfig()

    result = BatchResult(image_path=image_path, success=False)

    try:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            result.error_message = f"Failed to load image: {image_path}"
            return result

        # Detect grains using v6 single-step pipeline
        detections = detect_grains(
            image,
            config=config,
            min_area=config.min_area,
            max_area=15000,  # EXP003 default
            border_margin=border_margin,
            floc_config=floc_config,
        )

        # Compute morphology and classify
        morphologies = []
        grains = []
        for d in detections:
            gc = GrainContour(contour=d.contour, mask=d.mask)
            grains.append(gc)
            morph = compute_morphology(d.contour, d.mask)
            morph.shape_class = classify_grain(morph.aspect_ratio, d.is_flocculation)
            morph.is_flocculation = d.is_flocculation
            morph.confidence = 0.9 if d.is_flocculation else 0.95
            morphologies.append(morph)

        # Compute statistics
        stats = compute_statistics(morphologies)

        # Export results
        basename = Path(image_path).stem
        os.makedirs(output_dir, exist_ok=True)

        # CSV
        csv_path = os.path.join(output_dir, f"{basename}_results.csv")
        export_csv(morphologies, csv_path)
        result.output_csv_path = csv_path

        # Annotated image
        annotated_path = os.path.join(output_dir, f"{basename}_annotated.png")
        export_annotated_image(image, grains, annotated_path, morphologies=morphologies)
        result.output_image_path = annotated_path

        # PDF report
        if generate_pdf:
            pdf_path = os.path.join(output_dir, f"{basename}_report.pdf")
            generate_pdf_report(image_path, morphologies, stats, pdf_path,
                                  annotated_image_path=annotated_path)
            result.output_pdf_path = pdf_path

        result.success = True
        result.grain_count = len(morphologies)
        result.morphologies = morphologies
        result.statistics = stats

    except Exception as exc:
        result.error_message = str(exc)

    return result


def process_batch(
    input_dir: str,
    output_dir: str,
    config: PreprocessConfig | None = None,
    floc_config: FlocculationConfig | None = None,
    border_margin: int = 5,
    generate_pdf: bool = True,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> BatchSummary:
    """Process all images in a directory.

    Parameters
    ----------
    input_dir : str
        Directory containing input images.
    output_dir : str
        Directory for output files.
    config : PreprocessConfig | None, optional
        Preprocessing configuration.
    floc_config : FlocculationConfig | None, optional
        Flocculation detection config.
    border_margin : int, optional
        Distance from border for edge filtering.
    generate_pdf : bool, optional
        Whether to generate PDF reports.
    extensions : tuple[str, ...], optional
        Supported image file extensions.

    Returns
    -------
    BatchSummary
        Summary of batch processing results.
    """
    summary = BatchSummary()

    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

    summary.total_images = len(image_files)

    for image_path in sorted(image_files):
        result = process_single_image(
            str(image_path),
            output_dir,
            config=config,
            floc_config=floc_config,
            border_margin=border_margin,
            generate_pdf=generate_pdf,
        )

        if result.success:
            summary.successful += 1
            summary.total_grains += result.grain_count
        else:
            summary.failed += 1

        summary.results.append(result)

    return summary
