"""Shadow region enhancement and validation module.

Provides CLAHE-based shadow enhancement to improve grain detection
in shadowed regions, and local contrast validation to distinguish
real grains from uniform shadow areas.
"""

import cv2
import numpy as np


def enhance_shadow_regions(image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE to enhance shadow regions in an image.

    Args:
        image: Input image (grayscale or color/BGR).
        clip_limit: CLAHE clip limit. Higher values give more contrast.

    Returns:
        Enhanced image with same number of channels as input.
    """
    if image is None or image.size == 0:
        return image

    # Create CLAHE with the specified clip limit
    tile_grid_size = (8, 8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 3 and image.shape[2] == 3:
        # Color image: convert to LAB, apply CLAHE to L channel, convert back
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image: apply CLAHE directly
        enhanced = clahe.apply(image)

    return enhanced


def validate_local_contrast(region: np.ndarray, min_std: float = 15.0) -> bool:
    """Validate if a region has sufficient local contrast to be a real grain.

    Args:
        region: Image region (numpy array) to validate.
        min_std: Minimum standard deviation threshold.

    Returns:
        True if the region's standard deviation is >= min_std, False otherwise.
    """
    if region is None or region.size == 0:
        return False

    std = np.std(region.astype(np.float64))
    return bool(std >= min_std)
