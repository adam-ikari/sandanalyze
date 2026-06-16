"""Image preprocessing module for sand grain segmentation.

Provides configurable preprocessing pipeline to convert raw sand images
into binary masks suitable for morphological analysis.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Configuration for the image preprocessing pipeline.

    Attributes:
        blur_kernel: Size of the Gaussian blur kernel (must be odd).
        adaptive_block_size: Block size for adaptive threshold (must be odd).
        adaptive_c: Constant subtracted from the mean in adaptive threshold.
        morph_kernel_size: Size of the morphological operation kernel.
        morph_open_iter: Number of morphological open iterations.
        morph_close_iter: Number of morphological close iterations.
        min_area: Minimum contour area to keep in the final mask.
        use_clahe: Whether to apply CLAHE contrast enhancement.
        use_watershed: Whether to apply watershed segmentation.
        watershed_thresh_ratio: Ratio for distance transform threshold (default 0.5).
    """

    blur_kernel: int = 5
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    morph_kernel_size: int = 3
    morph_open_iter: int = 1
    morph_close_iter: int = 1
    min_area: int = 50
    use_clahe: bool = True
    use_watershed: bool = True
    watershed_thresh_ratio: float = 0.5


def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Run the full preprocessing pipeline on a sand image.

    Pipeline:
        1. Convert to grayscale if needed.
        2. Optional CLAHE contrast enhancement.
        3. Gaussian blur.
        4. Adaptive thresholding.
        5. Morphological open/close.
        6. Optional watershed segmentation.
        7. Area filtering.

    Args:
        image: Input image (grayscale or color).
        config: Preprocessing configuration. Uses defaults if None.

    Returns:
        Binary mask (uint8) with foreground grains as 255 and background as 0.
    """
    if config is None:
        config = PreprocessConfig()

    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Optional CLAHE
    if config.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 3. Gaussian blur
    blurred = cv2.GaussianBlur(gray, (config.blur_kernel, config.blur_kernel), 0)

    # 4. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.adaptive_block_size,
        config.adaptive_c,
    )

    # 5. Morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
    )
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iter)

    # 6. Optional watershed
    if config.use_watershed:
        mask = _apply_watershed(closed, config.watershed_thresh_ratio)
    else:
        mask = closed

    # 7. Area filtering
    mask = _filter_by_area(mask, config.min_area)

    return mask


def _apply_watershed(binary_mask: np.ndarray, thresh_ratio: float = 0.5) -> np.ndarray:
    """Separate touching grains using the watershed algorithm.

    Uses distance transform to find sure foreground regions, then applies
    the watershed algorithm to separate touching grains.

    Args:
        binary_mask: Binary mask with foreground as 255.
        thresh_ratio: Ratio for distance transform threshold (default 0.5).

    Returns:
        Binary mask with separated grains.
    """
    # Ensure binary mask is clean
    binary = np.zeros_like(binary_mask)
    binary[binary_mask > 0] = 255

    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Threshold to get sure foreground
    _, sure_fg = cv2.threshold(dist_transform, thresh_ratio * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Sure background
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background is 1
    markers[unknown == 255] = 0  # Unknown is 0

    # Watershed
    # Create a color image for watershed
    color_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)

    # Create output mask: keep only regions that are not boundaries
    output = np.zeros_like(binary)
    output[markers > 1] = 255

    return output


def _filter_by_area(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove small connected components from a binary mask.

    Args:
        mask: Binary mask.
        min_area: Minimum area threshold.

    Returns:
        Filtered binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 255
    return output
