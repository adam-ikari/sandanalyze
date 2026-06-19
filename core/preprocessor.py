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
    """

    blur_kernel: int = 5
    adaptive_block_size: int = 51
    adaptive_c: int = 5
    morph_kernel_size: int = 3
    morph_open_iter: int = 1
    morph_close_iter: int = 1
    min_area: int = 800
    use_clahe: bool = True


def estimate_image_noise(image: np.ndarray) -> float:
    """Estimate noise level in a microscope image.

    Uses a combination of Laplacian variance and local standard deviation
to estimate noise. Higher values indicate more noise/clutter.

    Args:
        image: Input image (grayscale or color).

    Returns:
        Noise level (0-255, higher = more noise).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Method 1: Laplacian variance (edge-based noise estimate)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_noise = np.std(laplacian)

    # Method 2: Local standard deviation (texture-based noise estimate)
    # Compute std in small windows to capture local variations
    kernel_size = 5
    local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
    local_std = np.sqrt(np.abs(local_sq_mean - local_mean ** 2))
    local_noise = np.mean(local_std)

    # Combine both methods (weighted average)
    # Laplacian is more sensitive to high-frequency noise
    # Local std is more sensitive to texture variations
    combined_noise = 0.6 * laplacian_noise + 0.4 * local_noise

    return float(combined_noise)


def auto_tune_for_microscope(
    image: np.ndarray,
) -> tuple[PreprocessConfig, dict[str, int | float]]:
    """Automatically tune parameters based on image noise level.

    Analyzes image noise and returns appropriate preprocessing parameters
    AND detection parameters. The detection parameters (min_area, max_area)
    are passed directly to detect_grains().

    Args:
        image: Input microscope image.

    Returns:
        Tuple of (PreprocessConfig, detection_params dict).
        detection_params contains:
            - min_area: Minimum grain area for detect_grains()
            - max_area: Maximum grain area for detect_grains()
    """
    noise = estimate_image_noise(image)

    if noise < 20:
        # Low noise - can afford lower min_area
        return (
            PreprocessConfig(
                blur_kernel=3,
                adaptive_block_size=51,
                adaptive_c=5,
                morph_kernel_size=3,
                min_area=600,
            ),
            {"min_area": 600, "max_area": 15000},
        )
    elif noise < 40:
        # Moderate noise - balanced parameters
        return (
            PreprocessConfig(
                blur_kernel=5,
                adaptive_block_size=51,
                adaptive_c=5,
                morph_kernel_size=3,
                min_area=800,
            ),
            {"min_area": 800, "max_area": 15000},
        )
    else:
        # High noise - conservative parameters
        return (
            PreprocessConfig(
                blur_kernel=7,
                adaptive_block_size=51,
                adaptive_c=5,
                morph_kernel_size=5,
                min_area=1200,
            ),
            {"min_area": 1200, "max_area": 15000},
        )


def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Run the full preprocessing pipeline on a sand image.

    Pipeline:
        1. Convert to grayscale if needed.
        2. Optional CLAHE contrast enhancement.
        3. Gaussian blur.
        4. Adaptive thresholding.
        5. Morphological open/close.
        6. Area filtering.

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

    # 6. Area filtering
    mask = _filter_by_area(closed, config.min_area)

    return mask


def crop_black_background(image: np.ndarray, threshold: int = 30) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop black background from image.

    Finds the largest bright connected component and crops to its bounding box.

    Args:
        image: Input image (grayscale or color).
        threshold: Brightness threshold for background separation.

    Returns:
        Tuple of (cropped_image, (x, y, w, h)) where (x, y, w, h) are the
        bounds of the cropped region in the original image coordinates.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold to separate bright foreground from dark background
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Find the largest component (excluding background which is label 0)
    largest_label = 0
    largest_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i

    if largest_label == 0:
        # No bright component found, return original image and full bounds
        h, w = image.shape[:2]
        return image.copy(), (0, 0, w, h)

    # Get bounding box of the largest component
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # Crop the original image to the bounding box
    cropped = image[y : y + h, x : x + w]

    return cropped, (x, y, w, h)


def filter_edge_grains(mask: np.ndarray, border_margin: int = 5) -> np.ndarray:
    """Remove grains that touch or are too close to the image border.

    Args:
        mask: Binary mask with grains as 255.
        border_margin: Minimum distance from border (pixels).

    Returns:
        Filtered binary mask with edge grains removed.
    """
    if mask is None or mask.size == 0:
        return mask

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_mask = np.zeros_like(mask)

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)

        # Check if contour touches or is too close to border
        touches_border = (
            x <= border_margin
            or y <= border_margin
            or x + bw >= w - border_margin
            or y + bh >= h - border_margin
        )

        if not touches_border:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_mask


def auto_tune_params(image: np.ndarray) -> PreprocessConfig:
    """Automatically tune preprocessing parameters based on image characteristics.

    Uses image brightness, contrast, and estimated grain size to determine
    optimal parameters.

    Args:
        image: Input grayscale or color image.

    Returns:
        Tuned PreprocessConfig.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Estimate image characteristics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    # Adjust blur kernel based on noise level
    if std_brightness > 60:
        blur_kernel = 7
    elif std_brightness > 40:
        blur_kernel = 5
    else:
        blur_kernel = 3

    # Ensure odd
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    # Adjust adaptive block based on image size
    h, w = gray.shape
    min_dim = min(h, w)
    if min_dim > 1000:
        adaptive_block = 21
    elif min_dim > 500:
        adaptive_block = 15
    else:
        adaptive_block = 11

    # Ensure odd and >= 3
    adaptive_block = max(3, adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1)

    # Adjust C based on brightness
    if mean_brightness < 100:
        adaptive_c = 5
    elif mean_brightness < 150:
        adaptive_c = 2
    else:
        adaptive_c = -2

    # Estimate min_area from image size
    estimated_grain_area = (min_dim / 50) ** 2
    min_area = max(50, int(estimated_grain_area * 0.5))

    return PreprocessConfig(
        blur_kernel=blur_kernel,
        adaptive_block_size=adaptive_block,
        adaptive_c=adaptive_c,
        min_area=min_area,
    )


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
