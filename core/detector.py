"""Grain detection module with flocculation and edge filtering support."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class FlocculationConfig:
    """Configuration for flocculation detection."""

    min_area: int = 5000
    max_area: int = 50000
    min_circularity: float = 0.01
    max_circularity: float = 0.3
    min_convexity: float = 0.2
    max_convexity: float = 0.7
    max_aspect_ratio: float = 5.0


@dataclass
class DetectionResult:
    """Result of grain detection."""

    contour: np.ndarray
    mask: np.ndarray
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    major_axis: float
    minor_axis: float
    convexity: float
    is_flocculation: bool
    is_edge: bool


def detect_flocculation(contour: np.ndarray, config: FlocculationConfig) -> bool:
    """Detect if a contour is a flocculation (cluster of grains).

    Uses combined criteria: large area + low circularity + low convexity + high aspect ratio.
    Must satisfy area condition + at least 2 other conditions.

    Args:
        contour: Contour to check.
        config: Flocculation detection configuration.

    Returns:
        True if the contour is detected as flocculation.
    """
    area = cv2.contourArea(contour)
    if area < config.min_area or area > config.max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    x, y, bw, bh = cv2.boundingRect(contour)
    aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0

    # Combined criteria: must satisfy area + at least 1 other condition
    conditions = [
        circularity <= config.max_circularity,  # Low circularity
        convexity <= config.max_convexity,       # Low convexity
        aspect_ratio >= config.max_aspect_ratio,  # High aspect ratio
    ]

    return sum(conditions) >= 1


def is_edge_grain(contour: np.ndarray, image_shape: Tuple[int, ...],
                   border_margin: int = 5) -> bool:
    """Check if a grain touches or is too close to the image border.

    Args:
        contour: Contour to check.
        image_shape: Shape of the original image (h, w) or (h, w, c).
        border_margin: Minimum distance from border.

    Returns:
        True if the grain is at the edge.
    """
    h, w = image_shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)

    return (
        x <= border_margin or
        y <= border_margin or
        x + bw >= w - border_margin or
        y + bh >= h - border_margin
    )


def detect_grains(mask: np.ndarray, image_shape: Tuple[int, ...],
                  min_area: int = 50, max_area: int = 50000,
                  border_margin: int = 5,
                  floc_config: FlocculationConfig = None) -> List[DetectionResult]:
    """Detect grains from preprocessed mask with flocculation and edge filtering.

    Args:
        mask: Preprocessed binary mask.
        image_shape: Original image shape for edge filtering.
        min_area: Minimum grain area.
        max_area: Maximum grain area.
        border_margin: Distance from border to filter.
        floc_config: Flocculation detection config. Uses defaults if None.

    Returns:
        List of DetectionResult objects.
    """
    if floc_config is None:
        floc_config = FlocculationConfig()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    h, w = image_shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)

        # Area filtering
        if area < min_area or area > max_area:
            continue

        # Edge filtering
        is_edge = is_edge_grain(contour, image_shape, border_margin)

        # Flocculation detection
        is_floc = detect_flocculation(contour, floc_config)

        # Compute morphology
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
        else:
            major_axis = np.sqrt(area * 4 / np.pi)
            minor_axis = major_axis

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        # Create mask for this grain
        grain_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(grain_mask, [contour], -1, 255, thickness=cv2.FILLED)

        results.append(DetectionResult(
            contour=contour,
            mask=grain_mask,
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            major_axis=major_axis,
            minor_axis=minor_axis,
            convexity=convexity,
            is_flocculation=is_floc,
            is_edge=is_edge,
        ))

    # Sort by area descending
    results.sort(key=lambda r: r.area, reverse=True)
    return results
