"""Multi-scale grain detection data structures."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.preprocessor import PreprocessConfig, preprocess


@dataclass
class GrainCandidate:
    """Represents a candidate grain with morphology and position features."""

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
    border_distance: float = 0.0
    solidity: float = 0.0


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale grain detection.

    Attributes:
        large_scale: Preprocessing config for large grains.
        medium_scale: Preprocessing config for medium grains.
        small_scale: Preprocessing config for small grains.
        shadow_enhance: Whether to apply shadow enhancement.
    """

    large_scale: PreprocessConfig
    medium_scale: PreprocessConfig
    small_scale: PreprocessConfig
    shadow_enhance: bool = True


def preprocess_all_scales(
    image: np.ndarray, config: MultiScaleConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run preprocessing at three scales simultaneously.

    Args:
        image: Input image (grayscale or color).
        config: MultiScaleConfig with large, medium, and small scale configs.

    Returns:
        Tuple of (large_mask, medium_mask, small_mask), each a binary uint8 ndarray.
    """
    large_mask = preprocess(image, config.large_scale)
    medium_mask = preprocess(image, config.medium_scale)
    small_mask = preprocess(image, config.small_scale)
    return large_mask, medium_mask, small_mask


def _count_components(mask: np.ndarray) -> int:
    """Count connected components in a binary mask (excluding background).

    Args:
        mask: Binary mask.

    Returns:
        Number of connected components (excluding background).
    """
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return num_labels - 1


def merge_multiscale_results(masks, iou_threshold=0.5):
    """Combine binary masks from multiple scales and remove duplicates.

    Args:
        masks: List of binary masks (uint8 ndarrays) from different scales.
        iou_threshold: IoU threshold above which overlapping components
            are considered duplicates. The component with better circularity
            is kept.

    Returns:
        Single merged binary mask (uint8).
    """
    if not masks:
        return np.array([], dtype=np.uint8)

    # Combine all masks using bitwise OR
    h, w = masks[0].shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        if mask is not None and mask.size > 0:
            combined = cv2.bitwise_or(combined, mask)

    # Find all connected components in the combined mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )

    if num_labels <= 1:
        return combined

    # Extract individual component masks and compute circularity for each
    components = []
    for label_id in range(1, num_labels):
        component_mask = np.uint8(labels == label_id) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = contours[0]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0.0
        components.append(
            {
                "mask": component_mask,
                "area": area,
                "circularity": circularity,
            }
        )

    if not components:
        return np.zeros((h, w), dtype=np.uint8)

    # Group overlapping components using IoU
    n = len(components)
    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        queue = [i]
        while queue:
            idx = queue.pop()
            for j in range(n):
                if visited[j]:
                    continue
                # Compute IoU between components idx and j
                intersection = cv2.countNonZero(
                    cv2.bitwise_and(components[idx]["mask"], components[j]["mask"])
                )
                if intersection == 0:
                    continue
                union = cv2.countNonZero(
                    cv2.bitwise_or(components[idx]["mask"], components[j]["mask"])
                )
                if union == 0:
                    continue
                iou = intersection / union
                if iou > iou_threshold:
                    visited[j] = True
                    group.append(j)
                    queue.append(j)
        groups.append(group)

    # For each group, keep the component with the best circularity
    result = np.zeros((h, w), dtype=np.uint8)
    for group in groups:
        best_idx = max(group, key=lambda i: components[i]["circularity"])
        result = cv2.bitwise_or(result, components[best_idx]["mask"])

    return result
