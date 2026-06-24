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


def _split_oversized_component(comp_mask: np.ndarray, min_area: int) -> list[tuple[np.ndarray, tuple]]:
    """Split an oversized component into individual grains.

    Uses a multi-step approach:
    1. Morphological opening to separate touching grains (best for sand)
    2. Distance transform + watershed for well-separated grains
    3. Progressive morphological erosion for touching grains

    Args:
        comp_mask: Binary mask of the oversized component
        min_area: Minimum area for a valid grain

    Returns:
        List of (region_mask, bbox) tuples for each split region
    """
    original_area = cv2.countNonZero(comp_mask)

    # Method 1: Morphological opening (best for sand grains)
    regions = _try_morphological_open_split(comp_mask, min_area)
    if len(regions) >= 2:
        return regions

    # Method 2: Distance transform + watershed
    regions = _try_watershed_split(comp_mask, min_area)
    if len(regions) >= 2:
        return regions

    # Method 3: Progressive morphological erosion
    regions = _try_morphological_split(comp_mask, min_area)
    if len(regions) >= 2:
        return regions

    # No successful split - return empty list
    return []


def _try_morphological_open_split(comp_mask: np.ndarray, min_area: int) -> list[tuple[np.ndarray, tuple]]:
    """Try morphological opening to separate touching grains.

    Uses opening to find grain centers, then uses these as markers
    for watershed segmentation on the original component mask.
    Tries multiple opening sizes and returns the best split.

    Args:
        comp_mask: Binary mask of the oversized component
        min_area: Minimum area for a valid grain

    Returns:
        List of (region_mask, bbox) tuples
    """
    original_area = cv2.countNonZero(comp_mask)
    best_regions = []

    # Try different opening sizes - prefer more regions
    for open_size in [5, 7, 9, 11, 13, 15]:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        opened = cv2.morphologyEx(comp_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

        # Find connected components in opened image - these are grain centers
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

        # Need at least 2 valid markers
        valid_markers = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                valid_markers.append(i)

        if len(valid_markers) < 2:
            continue

        # Use opened components as markers for watershed on original
        markers = np.zeros(comp_mask.shape, dtype=np.int32)
        for i in valid_markers:
            markers[labels == i] = i

        # Background is 1, markers start from 2
        markers = markers + 1
        markers[comp_mask == 0] = 0

        # Apply watershed on original component
        comp_color = cv2.cvtColor(comp_mask, cv2.COLOR_GRAY2BGR)
        result = cv2.watershed(comp_color, markers)

        # Extract regions from watershed result
        regions = []
        for marker in np.unique(result):
            if marker <= 1:  # Skip background and border
                continue
            region_mask = ((result == marker) & (comp_mask > 0)).astype(np.uint8) * 255
            region_area = cv2.countNonZero(region_mask)
            if region_area >= min_area:
                ys, xs = np.where(region_mask > 0)
                if len(xs) > 0:
                    bbox = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
                    regions.append((region_mask, bbox))

        # Validate split
        if len(regions) >= 2:
            total_area = sum(cv2.countNonZero(rm) for rm, _ in regions)
            # Must preserve at least 50% of original area
            if total_area >= 0.5 * original_area:
                # Check no single region dominates
                valid = True
                for region_mask, _ in regions:
                    if cv2.countNonZero(region_mask) > 0.8 * total_area:
                        valid = False
                        break
                if valid:
                    # Keep the best split (most regions)
                    if len(regions) > len(best_regions):
                        best_regions = regions

    return best_regions


def _try_watershed_split(comp_mask: np.ndarray, min_area: int) -> list[tuple[np.ndarray, tuple]]:
    """Try watershed splitting on a component."""
    # Distance transform to find peaks
    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []

    # Normalize and threshold
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, dist_thresh = cv2.threshold(dist_norm, 0.15 * dist_norm.max(), 255, cv2.THRESH_BINARY)
    dist_thresh = dist_thresh.astype(np.uint8)

    # Find markers
    num_markers, markers = cv2.connectedComponents(dist_thresh, connectivity=8)
    if num_markers <= 2:
        return []

    # Apply watershed
    markers = markers + 1
    markers[comp_mask == 0] = 0
    comp_color = cv2.cvtColor(comp_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(comp_color, markers)

    # Extract regions
    regions = []
    for marker in np.unique(markers):
        if marker <= 1:
            continue
        region_mask = (markers == marker).astype(np.uint8) * 255
        region_area = cv2.countNonZero(region_mask)
        if region_area >= min_area:
            ys, xs = np.where(region_mask > 0)
            if len(xs) > 0:
                bbox = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
                regions.append((region_mask, bbox))

    # Validate split
    if len(regions) < 2:
        return []

    total_area = sum(cv2.countNonZero(rm) for rm, _ in regions)
    if total_area < 0.5 * cv2.countNonZero(comp_mask):
        return []

    for region_mask, _ in regions:
        if cv2.countNonZero(region_mask) > 0.8 * total_area:
            return []

    return regions


def _try_morphological_split(comp_mask: np.ndarray, min_area: int) -> list[tuple[np.ndarray, tuple]]:
    """Try morphological erosion to separate touching grains.

    Uses progressively larger erosion kernels to find the best split.
    Dilates back with a smaller kernel to avoid re-merging.

    Args:
        comp_mask: Binary mask of the oversized component
        min_area: Minimum area for a valid grain

    Returns:
        List of (region_mask, bbox) tuples
    """
    # Try different erosion sizes (progressively larger)
    for erosion_size in [3, 5, 7, 9, 11]:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        eroded = cv2.erode(comp_mask, erode_kernel, iterations=1)

        # Find connected components in eroded image
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)

        regions = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            # Extract this component
            lx = stats[i, cv2.CC_STAT_LEFT]
            ly = stats[i, cv2.CC_STAT_TOP]
            lw = stats[i, cv2.CC_STAT_WIDTH]
            lh = stats[i, cv2.CC_STAT_HEIGHT]

            component_mask = (labels[ly:ly+lh, lx:lx+lw] == i).astype(np.uint8) * 255

            # Dilate back with smaller kernel to approximate original size
            # but not so much that components re-merge
            dilate_size = max(3, erosion_size - 2)
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
            )
            dilated = cv2.dilate(component_mask, dilate_kernel, iterations=1)

            # Mask with original to keep within bounds
            dilated = cv2.bitwise_and(dilated, comp_mask[ly:ly+lh, lx:lx+lw])

            if cv2.countNonZero(dilated) >= min_area:
                regions.append((dilated, (lx, ly, lw, lh)))

        # Only return if we got multiple valid regions
        if len(regions) >= 2:
            # Validate: check that no single region dominates
            total_area = sum(cv2.countNonZero(rm) for rm, _ in regions)
            if total_area >= 0.5 * cv2.countNonZero(comp_mask):
                # Check if any single region is too large
                valid = True
                for region_mask, _ in regions:
                    if cv2.countNonZero(region_mask) > 0.8 * total_area:
                        valid = False
                        break
                if valid:
                    return regions

    return []


def _split_by_concave_points(comp_mask: np.ndarray, min_area: int) -> list[tuple[np.ndarray, tuple]]:
    """Split component using convexity defects (concave points).

    If splitting fails, returns empty list to avoid adding oversized
    components as single grains.

    Args:
        comp_mask: Binary mask of the component
        min_area: Minimum area for a valid grain

    Returns:
        List of (region_mask, bbox) tuples
    """
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    cnt = contours[0]
    if len(cnt) < 5:
        return []

    # Convex hull and defects
    try:
        hull = cv2.convexHull(cnt, returnPoints=False)
        if len(hull) < 3:
            return []
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:
        return []

    if defects is None or len(defects) == 0:
        return []

    # Find deep defects that could be split points
    split_points = []
    for i in range(len(defects)):
        s, e, f, d = defects[i, 0]
        # Deep defect indicates a potential split point
        if d > 10000:  # depth threshold
            far = tuple(cnt[f][0])
            split_points.append(far)

    if len(split_points) < 2:
        # Not enough split points, don't add as single region
        # (avoids adding oversized components)
        return []

    # For simplicity, if we have split points but can't easily split,
    # don't add the component (avoids false positives from oversized regions)
    return []


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

    # Analyze image for dark regions and auto-tune adaptive_c
    # Use image-wide brightness analysis for more accurate tuning
    h_img, w_img = gray.shape[:2]

    # Compute image brightness statistics
    mean_brightness = gray.mean()
    median_brightness = np.median(gray)
    p10_brightness = np.percentile(gray, 10)

    # Check bottom-left quadrant for dark regions (common location for large dark grains)
    bl_region = gray[h_img // 2 :, : w_img // 2]
    bl_dark_ratio = np.count_nonzero(bl_region < 80) / bl_region.size

    # Auto-tune adaptive_c based on image characteristics
    # Use 10th percentile for robustness (ignores bright grain outliers)
    effective_adaptive_c = config.adaptive_c
    if p10_brightness < 5 and median_brightness < 10:
        # Very dark image (like sample 25): use adaptive_c=0 for maximum sensitivity
        effective_adaptive_c = 0
    elif p10_brightness < 20 and median_brightness < 30:
        # Moderately dark image: use adaptive_c=1
        effective_adaptive_c = min(config.adaptive_c, 1)
    elif bl_dark_ratio > 0.3:
        # Bottom-left has dark regions
        effective_adaptive_c = min(config.adaptive_c, 1)

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

    # Step 2: Adaptive threshold in ROI with auto-tuned adaptive_c
    roi_thresh = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, config.adaptive_block_size, effective_adaptive_c
    )

    # Step 3: Morphological open
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
    )
    opened = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iter or 1)

    # Step 4: Connected components filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered = np.zeros_like(opened)
    oversized_components = []  # Track components that are too large but might be splittable

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        lx = stats[i, cv2.CC_STAT_LEFT]
        ly = stats[i, cv2.CC_STAT_TOP]
        lw = stats[i, cv2.CC_STAT_WIDTH]
        lh = stats[i, cv2.CC_STAT_HEIGHT]

        # Check if component touches ROI boundary
        touches_border = (
            lx <= border_margin or ly <= border_margin or
            lx + lw >= w - border_margin or ly + lh >= h - border_margin
        )

        # Reject border-touching components to avoid circular boundary artifacts
        # This prevents the microscope's circular field of view from being detected as grains
        if touches_border:
            allow_border = False
        else:
            allow_border = True

        if min_area <= area <= max_area and allow_border:
            filtered[labels == i] = 255
        elif area > max_area and allow_border:
            # Oversized component - might be multiple touching grains
            # Store for potential splitting
            oversized_components.append({
                "label": i,
                "area": area,
                "bbox": (lx, ly, lw, lh),
            })

    # Step 4b: Try to split oversized components
    split_count = 0
    for comp in oversized_components:
        lx, ly, lw, lh = comp["bbox"]
        comp_mask = (labels[ly:ly+lh, lx:lx+lw] == comp["label"]).astype(np.uint8) * 255

        # Try splitting
        split_regions = _split_oversized_component(comp_mask, min_area)

        if len(split_regions) >= 2:
            split_count += 1
            for region_mask, region_bbox in split_regions:
                rx, ry, rw, rh = region_bbox
                # Place region into filtered image
                global_x = lx + rx
                global_y = ly + ry
                # Ensure we don't go out of bounds
                end_x = min(global_x + rw, filtered.shape[1])
                end_y = min(global_y + rh, filtered.shape[0])
                actual_w = end_x - global_x
                actual_h = end_y - global_y
                if actual_w > 0 and actual_h > 0:
                    filtered[global_y:global_y+actual_h, global_x:global_x+actual_w] = cv2.bitwise_or(
                        filtered[global_y:global_y+actual_h, global_x:global_x+actual_w],
                        region_mask[:actual_h, :actual_w]
                    )

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

        # Filter out large circular contours that are likely the image border
        # These have high area and high circularity (close to 1.0)
        if area > 50000 and circularity > 0.5:
            # Likely the circular border of the image, skip
            continue

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
        # Relaxed for grains near max_area threshold as they may be split regions
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

            # Additional check: if area is close to max_area, it's likely a legitimate
            # large grain rather than flocculation (split regions can have irregular contours)
            if is_floc and area > max_area * 0.8:
                # Only keep as flocculation if very irregular
                if circularity > 0.05 and convexity > 0.5:
                    is_floc = False

        # Post-processing: filter out likely noise/fragments
        # Small, non-circular detections are likely noise
        # Relaxed threshold: allow small grains if they are somewhat circular
        if area < 1500 and circularity < 0.15:
            continue

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
