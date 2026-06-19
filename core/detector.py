"""Grain detection module with v6 pipeline.

The v6 pipeline operates on the raw image in a single step:
  1. Crop black background (optional)
  2. Adaptive threshold in ROI
  3. Morphological open
  4. Edge filtering based on ROI boundary
  5. Contour detection
  6. Per-contour feature computation + hull smoothing / mask filling
"""

from dataclasses import dataclass

import cv2
import numpy as np

from core.preprocessor import PreprocessConfig


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
    """Result of grain detection.

    Contains preliminary morphology values computed during detection for
    filtering purposes (area filtering, flocculation detection, circularity
    filtering). For authoritative morphology measurements, use
    :func:`core.morphology.compute_morphology`.
    """

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


def detect_grains(
    image: np.ndarray,
    config: PreprocessConfig,
    min_area: int = 1000,
    max_area: int = 15000,
    border_margin: int = 5,
    hull_expansion_ratio: float = 1.5,
    floc_config: FlocculationConfig = None,
    crop_black_background: bool = True,
) -> list[DetectionResult]:
    """Detect grains from raw image using the v6 pipeline.

    Args:
        image: Raw input image (BGR or grayscale).
        config: Preprocessing configuration.
        min_area: Minimum grain area.
        max_area: Maximum grain area.
        border_margin: Distance from border to filter (unused in v6, kept for API compatibility).
        hull_expansion_ratio: Threshold for using convex hull vs mask filling.
        floc_config: Flocculation detection config. Uses defaults if None.
        crop_black_background: Whether to crop black background before processing.

    Returns:
        List of DetectionResult objects sorted by area descending.
    """
    if floc_config is None:
        floc_config = FlocculationConfig()

    # Convert to grayscale
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

    roi_gray = gray[y:y+h, x:x+w]

    # Step 2: Adaptive threshold in ROI
    roi_thresh = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, config.adaptive_block_size, config.adaptive_c
    )

    # Step 3: Morphological open
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
    )
    opened = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iter or 1)

    # Step 4: Edge filtering based on ROI boundary using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered = np.zeros_like(opened)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        lx = stats[i, cv2.CC_STAT_LEFT]
        ly = stats[i, cv2.CC_STAT_TOP]
        lw = stats[i, cv2.CC_STAT_WIDTH]
        lh = stats[i, cv2.CC_STAT_HEIGHT]

        # Check if component touches ROI boundary
        touches_border = (
            lx <= 0 or ly <= 0 or
            lx + lw >= w or ly + lh >= h
        )

        # Allow border-touching components if large enough to be flocculation
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
        # Convert to global coordinates
        cnt_global = cnt + np.array([x, y])

        # Compute area using filled mask (more accurate than contourArea for shapes with holes)
        temp_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt_global], -1, 255, thickness=cv2.FILLED)
        area = float(cv2.countNonZero(temp_mask))

        # Skip area re-checking — connectedComponents filtering in Step 4 is sufficient.
        # contourArea / filled mask overestimates for shapes with interior holes
        # (e.g. U-shape, C-shape), causing valid grains to be discarded.

        perimeter = cv2.arcLength(cnt_global, True)

        # Compute circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Filter out shapes with very low circularity (likely incomplete/edge grains)
        # Skip this filter for very small grains as they naturally have lower circularity
        # Also skip for potential flocculation clusters (low circularity + low convexity)
        # Compute convexity first to check if it's a flocculation
        hull = cv2.convexHull(cnt_global)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        is_potential_floc = (
            area >= floc_config.min_area and
            circularity < 0.1 and
            convexity < 0.5
        )

        if circularity < 0.02 and area > 5000 and not is_potential_floc:
            continue

        # Compute aspect ratio from bounding rect
        x2, y2, bw, bh = cv2.boundingRect(cnt_global)
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 0

        # Compute major/minor axis from ellipse fit
        if len(cnt_global) >= 5:
            ellipse = cv2.fitEllipse(cnt_global)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
        else:
            major_axis = np.sqrt(area * 4 / np.pi)
            minor_axis = major_axis

        # Flocculation detection (inline)
        # Require at least 2 out of 3 conditions to be met for more strict detection
        is_floc = False
        if area >= floc_config.min_area and area <= floc_config.max_area:
            conditions_met = sum([
                circularity <= floc_config.max_circularity,
                convexity <= floc_config.max_convexity,
                aspect_ratio >= floc_config.max_aspect_ratio,
            ])
            # Require at least 2 conditions OR very low circularity with low convexity
            if conditions_met >= 2:
                is_floc = True
            elif conditions_met == 1 and circularity < 0.1 and convexity < 0.5:
                is_floc = True

        # Hull smoothing / mask filling
        if hull_area / area < hull_expansion_ratio:
            final_contour = hull
        else:
            # Use original contour but create filled mask
            final_contour = cnt_global

        # Create grain mask (full image size)
        grain_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(grain_mask, [final_contour], -1, 255, thickness=cv2.FILLED)

        results.append(DetectionResult(
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
        ))

    # Sort by area descending
    results.sort(key=lambda r: r.area, reverse=True)
    return results
