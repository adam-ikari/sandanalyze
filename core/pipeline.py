"""Shared detection pipeline for sand grain morphology analysis.

This module provides a single entry-point ``run_detection_pipeline`` that
orchestrates grain detection, morphology computation, classification, and
aggregate statistics — replacing the duplicated logic found in ``app.py``
and ``core/batch.py``.
"""

import numpy as np

from core.classifier import classify_grain
from core.detector import detect_grains, FlocculationConfig
from core.morphology import (
    compute_morphology,
    compute_statistics,
    GrainContour,
    GrainMorphology,
    GrainStatistics,
)
from core.preprocessor import PreprocessConfig


def run_detection_pipeline(
    image: np.ndarray,
    config: PreprocessConfig,
    min_area: int = 1000,
    max_area: int = 15000,
    border_margin: int = 5,
    hull_expansion_ratio: float = 1.5,
    floc_config: FlocculationConfig | None = None,
    crop_black_background: bool = True,
) -> tuple[list[GrainContour], list[GrainMorphology], GrainStatistics]:
    """Run the full detection pipeline on a raw image.

    Steps:
        1. Detect grains using :func:`core.detector.detect_grains`.
        2. Build :class:`core.morphology.GrainContour` objects.
        3. Compute morphology for each grain.
        4. Classify each grain (Zingg + flocculation).
        5. Compute aggregate statistics.

    Args:
        image: Raw input image (BGR or grayscale).
        config: Preprocessing configuration.
        min_area: Minimum grain area.
        max_area: Maximum grain area.
        border_margin: Distance from border to filter (kept for API compatibility).
        hull_expansion_ratio: Threshold for using convex hull vs mask filling.
        floc_config: Flocculation detection config. Uses defaults if None.
        crop_black_background: Whether to crop black background before processing.

    Returns:
        Tuple of (grains, morphologies, statistics).
    """
    results = detect_grains(
        image=image,
        config=config,
        min_area=min_area,
        max_area=max_area,
        border_margin=border_margin,
        hull_expansion_ratio=hull_expansion_ratio,
        floc_config=floc_config,
        crop_black_background=crop_black_background,
    )

    grains: list[GrainContour] = []
    morphologies: list[GrainMorphology] = []

    for result in results:
        grain = GrainContour(contour=result.contour, mask=result.mask)
        grains.append(grain)

        morph = compute_morphology(result.contour, result.mask)
        morph.is_flocculation = result.is_flocculation
        morph.shape_class = classify_grain(morph.aspect_ratio, result.is_flocculation)
        morph.confidence = 0.9 if result.is_flocculation else 0.95
        morphologies.append(morph)

    statistics = compute_statistics(morphologies)
    return grains, morphologies, statistics
