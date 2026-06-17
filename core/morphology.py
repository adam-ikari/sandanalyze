"""Morphological parameter computation for sand grain analysis."""

from dataclasses import dataclass, field
import math
from typing import List

import cv2
import numpy as np


@dataclass
class GrainMorphology:
    """Morphological parameters for a single sand grain."""

    area: float
    perimeter: float
    circularity: float
    d_eq: float
    major_axis: float
    minor_axis: float
    aspect_ratio: float
    sphericity: float
    convexity: float
    feret_max: float
    feret_min: float
    # New fields for v2.0
    is_flocculation: bool = False
    shape_class: str = ""
    confidence: float = 0.0


# Zingg classification colors (BGR format for OpenCV)
ZINGG_COLORS = {
    "spherical": (0, 255, 0),    # Green
    "rod-like": (0, 0, 255),     # Red
    "discoidal": (255, 0, 0),    # Blue
}

# Extended classification colors (BGR format for OpenCV)
CLASSIFICATION_COLORS = {
    "spherical": (0, 255, 0),      # Green
    "rod-like": (0, 0, 255),       # Red
    "discoidal": (255, 0, 0),      # Blue
    "flocculation": (0, 255, 255),  # Yellow (BGR)
}


def get_classification_color(shape_class: str) -> tuple[int, int, int]:
    """Get the color for a classification.

    Args:
        shape_class: One of "spherical", "rod-like", "discoidal", "flocculation".

    Returns:
        BGR color tuple.
    """
    return CLASSIFICATION_COLORS.get(shape_class, (128, 128, 128))


def zingg_classify(aspect_ratio: float) -> str:
    """Classify a grain using Zingg shape classification.

    Args:
        aspect_ratio: The aspect ratio of the grain (major_axis / minor_axis).

    Returns:
        One of: "spherical", "rod-like", "discoidal".
    """
    if aspect_ratio < 1.5:
        return "spherical"
    elif aspect_ratio < 2.5:
        return "rod-like"
    else:
        return "discoidal"


def get_zingg_color(aspect_ratio: float) -> tuple[int, int, int]:
    """Get the color for a Zingg classification.

    Args:
        aspect_ratio: The aspect ratio of the grain.

    Returns:
        BGR color tuple.
    """
    classification = zingg_classify(aspect_ratio)
    return ZINGG_COLORS[classification]


@dataclass
class GrainStatistics:
    """Aggregate statistics across multiple sand grains."""

    count: int
    area_mean: float = 0.0
    area_std: float = 0.0
    area_median: float = 0.0
    circularity_mean: float = 0.0
    circularity_std: float = 0.0
    circularity_median: float = 0.0
    d_eq_mean: float = 0.0
    d_eq_std: float = 0.0
    d_eq_median: float = 0.0
    aspect_ratio_mean: float = 0.0
    aspect_ratio_std: float = 0.0
    aspect_ratio_median: float = 0.0
    sphericity_mean: float = 0.0
    sphericity_std: float = 0.0
    sphericity_median: float = 0.0
    convexity_mean: float = 0.0
    convexity_std: float = 0.0
    convexity_median: float = 0.0
    # Updated for four-class system
    zingg_counts: dict = field(default_factory=lambda: {"spherical": 0, "rod-like": 0, "discoidal": 0, "flocculation": 0})
    zingg_colors: dict = field(default_factory=dict)
    d_eq_values: List[float] = field(default_factory=list)
    circularity_values: List[float] = field(default_factory=list)
    sphericity_values: List[float] = field(default_factory=list)
    # New: flocculation stats
    flocculation_count: int = 0
    flocculation_ratio: float = 0.0


def _feret_diameters(contour: np.ndarray) -> tuple[float, float]:
    """Compute Feret max and min diameters via rotating calipers (1-degree sampling)."""
    # Get the convex hull to reduce points and ensure convexity for calipers
    hull = cv2.convexHull(contour)
    pts = hull.reshape(-1, 2)
    if len(pts) < 2:
        return 0.0, 0.0

    feret_max = 0.0
    feret_min = float("inf")

    for angle_deg in range(0, 180, 1):
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Project points onto the direction perpendicular to the rotating edge
        # For a given angle, the width is the extent along the perpendicular direction
        projections = [pt[0] * cos_a + pt[1] * sin_a for pt in pts]
        width = max(projections) - min(projections)

        if width > feret_max:
            feret_max = width
        if width < feret_min:
            feret_min = width

    return feret_max, feret_min


def compute_morphology(contour: np.ndarray, mask: np.ndarray) -> GrainMorphology:
    """Compute morphological parameters for a single sand grain.

    Args:
        contour: The contour of the grain (from cv2.findContours).
        mask: Binary mask of the grain region.

    Returns:
        GrainMorphology object with computed parameters.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        circularity = 0.0
    else:
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        circularity = min(circularity, 1.0)

    d_eq = math.sqrt((4 * area) / math.pi)

    # Fit ellipse for major/minor axis
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
    else:
        # Fallback for very small contours
        major_axis = d_eq
        minor_axis = d_eq

    if minor_axis == 0:
        aspect_ratio = 1.0
        sphericity = 1.0
    else:
        aspect_ratio = major_axis / minor_axis
        sphericity = minor_axis / major_axis

    # Convexity = area / convex hull area
    hull = cv2.convexHull(contour)
    convex_hull_area = cv2.contourArea(hull)
    if convex_hull_area == 0:
        convexity = 0.0
    else:
        convexity = area / convex_hull_area

    # Feret diameters via rotating calipers
    feret_max, feret_min = _feret_diameters(contour)

    return GrainMorphology(
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        d_eq=d_eq,
        major_axis=major_axis,
        minor_axis=minor_axis,
        aspect_ratio=aspect_ratio,
        sphericity=sphericity,
        convexity=convexity,
        feret_max=feret_max,
        feret_min=feret_min,
    )


def compute_statistics(morphologies: List[GrainMorphology]) -> GrainStatistics:
    """Compute aggregate statistics across multiple grains.

    Args:
        morphologies: List of GrainMorphology objects.

    Returns:
        GrainStatistics with means, stds, medians, and Zingg classification counts.
    """
    if not morphologies:
        return GrainStatistics(count=0)

    def _stats(values: List[float]) -> tuple[float, float, float]:
        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        median = float(np.median(arr))
        return mean, std, median

    areas = [m.area for m in morphologies]
    circularities = [m.circularity for m in morphologies]
    d_eqs = [m.d_eq for m in morphologies]
    aspect_ratios = [m.aspect_ratio for m in morphologies]
    sphericities = [m.sphericity for m in morphologies]
    convexities = [m.convexity for m in morphologies]

    area_mean, area_std, area_median = _stats(areas)
    circ_mean, circ_std, circ_median = _stats(circularities)
    d_eq_mean, d_eq_std, d_eq_median = _stats(d_eqs)
    ar_mean, ar_std, ar_median = _stats(aspect_ratios)
    sph_mean, sph_std, sph_median = _stats(sphericities)
    conv_mean, conv_std, conv_median = _stats(convexities)

    # Four-class classification counts
    zingg_counts: dict[str, int] = {"spherical": 0, "rod-like": 0, "discoidal": 0, "flocculation": 0}
    for m in morphologies:
        if m.is_flocculation:
            zingg_counts["flocculation"] += 1
        elif m.aspect_ratio < 1.5:
            zingg_counts["spherical"] += 1
        elif m.aspect_ratio < 2.5:
            zingg_counts["rod-like"] += 1
        else:
            zingg_counts["discoidal"] += 1

    # Per-grain classification colors
    zingg_colors = {}
    for i, m in enumerate(morphologies):
        if m.is_flocculation:
            zingg_colors[i] = CLASSIFICATION_COLORS["flocculation"]
        else:
            zingg_colors[i] = get_zingg_color(m.aspect_ratio)

    flocculation_count = zingg_counts["flocculation"]
    flocculation_ratio = flocculation_count / len(morphologies) if morphologies else 0.0

    return GrainStatistics(
        count=len(morphologies),
        area_mean=area_mean,
        area_std=area_std,
        area_median=area_median,
        circularity_mean=circ_mean,
        circularity_std=circ_std,
        circularity_median=circ_median,
        d_eq_mean=d_eq_mean,
        d_eq_std=d_eq_std,
        d_eq_median=d_eq_median,
        aspect_ratio_mean=ar_mean,
        aspect_ratio_std=ar_std,
        aspect_ratio_median=ar_median,
        sphericity_mean=sph_mean,
        sphericity_std=sph_std,
        sphericity_median=sph_median,
        convexity_mean=conv_mean,
        convexity_std=conv_std,
        convexity_median=conv_median,
        zingg_counts=zingg_counts,
        zingg_colors=zingg_colors,
        d_eq_values=d_eqs,
        circularity_values=circularities,
        sphericity_values=sphericities,
    )
