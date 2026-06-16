"""Traditional OpenCV-based grain detection.

This module provides contour-based grain detection from preprocessed masks.
It supports both binary masks and multi-label watershed masks.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GrainContour:
    """Represents a detected grain via its contour and binary mask."""

    contour: np.ndarray
    mask: np.ndarray


def detect_grains(mask: np.ndarray, min_area: int = 50) -> list[GrainContour]:
    """Detect grains from a preprocessed mask.

    Parameters
    ----------
    mask : np.ndarray
        Preprocessed mask. Can be either a binary mask (0/255 or 0/1)
        or a multi-label watershed mask where each label > 0 represents
        a distinct grain.
    min_area : int, optional
        Minimum contour area to keep a grain. Default is 50.

    Returns
    -------
    list[GrainContour]
        List of GrainContour objects sorted by area (descending).
    """
    grains: list[GrainContour] = []

    if mask is None or mask.size == 0:
        return grains

    # Determine if this is a multi-label watershed mask or binary mask
    unique_values = np.unique(mask)

    # Remove background label (0) for comparison
    non_bg = unique_values[unique_values != 0]
    is_watershed = len(non_bg) > 1 and not np.array_equal(non_bg, [255]) and not np.array_equal(non_bg, [1])

    if is_watershed:
        # Multi-label watershed mask: extract contours per label
        for label in non_bg:
            label_mask = (mask == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    grain_mask = np.zeros_like(mask, dtype=np.uint8)
                    cv2.drawContours(grain_mask, [contour], -1, 255, thickness=cv2.FILLED)
                    grains.append(GrainContour(contour=contour, mask=grain_mask))
    else:
        # Binary mask: extract all contours at once
        # Normalize binary mask to 0/255
        binary_mask = mask.copy()
        if binary_mask.max() <= 1:
            binary_mask = (binary_mask * 255).astype(np.uint8)
        else:
            binary_mask = binary_mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                grain_mask = np.zeros_like(mask, dtype=np.uint8)
                cv2.drawContours(grain_mask, [contour], -1, 255, thickness=cv2.FILLED)
                grains.append(GrainContour(contour=contour, mask=grain_mask))

    # Sort by area descending
    grains.sort(key=lambda g: cv2.contourArea(g.contour), reverse=True)
    return grains
