"""Texture and edge filtering for grain candidate validation.

This module provides the TextureEdgeValidator class which uses texture features
(LBP, GLCM) and edge analysis to distinguish real sand grains from noise,
lens edges, and other artifacts.

scikit-image is an optional dependency; OpenCV fallbacks are provided for all
texture features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ValidationConfig:
    """Configuration for texture/edge validation.

    Attributes:
        texture_score_threshold: Minimum composite texture score for a valid grain.
        edge_direction_threshold: Minimum edge direction consistency for a valid grain.
        edge_closure_threshold: Minimum ratio of contour pixels overlapping Canny edges.
        lens_edge_margin: Fraction of image border considered "near edge" for lens detection.
        lens_edge_circularity: Minimum circularity to be considered a lens edge.
        lens_edge_min_area: Minimum area (pixels) for a lens edge candidate.
        noise_max_texture_score: Maximum texture score for noise classification.
        noise_max_edge_strength: Maximum average Sobel magnitude for noise.
        noise_max_area: Maximum area (pixels) for noise classification.
    """

    texture_score_threshold: float = 0.3
    edge_direction_threshold: float = 0.6
    edge_closure_threshold: float = 0.1
    lens_edge_margin: float = 0.05
    lens_edge_circularity: float = 0.7
    lens_edge_min_area: float = 50000
    noise_max_texture_score: float = 0.3
    noise_max_edge_strength: float = 30.0
    noise_max_area: int = 500


class TextureEdgeValidator:
    """Validate grain candidates using texture and edge features.

    Uses LBP/GLCM texture analysis and Sobel/Canny edge features to filter out
    lens edges, noise, and other non-grain artifacts. Falls back to OpenCV-based
    approximations when scikit-image is not installed.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize validator with optional custom configuration.

        Args:
            config: ValidationConfig instance. Uses defaults if None.
        """
        self.config = config if config is not None else ValidationConfig()
        self._has_skimage = self._check_skimage()

    @staticmethod
    def _check_skimage() -> bool:
        """Check whether scikit-image is available."""
        try:
            import skimage  # noqa: F401

            return True
        except ImportError:
            return False

    def validate(self, candidate: object, full_image: np.ndarray) -> bool:
        """Validate a grain candidate.

        Steps:
        1. Reject lens-edge artifacts (large circular objects near border).
        2. Reject noise (small, low-texture, weak-edge regions).
        3. Compute composite texture+edge score and compare to threshold.
        4. Check edge closure against threshold.

        Args:
            candidate: GrainCandidate instance (must have area, circularity,
                contour, border_distance attributes).
            full_image: Original input image (grayscale or BGR).

        Returns:
            True if candidate passes all validation checks.
        """
        if self._is_lens_edge(candidate, full_image):
            return False

        if self._is_noise(candidate, full_image):
            return False

        composite = self._compute_composite_score(candidate, full_image)
        if composite < self.config.texture_score_threshold:
            return False

        roi_gray, roi_offset = self._extract_roi_with_offset(candidate, full_image)
        if roi_gray is None:
            return False

        # Shift contour to ROI-local coordinates for edge closure
        contour_local = candidate.contour.copy()
        if contour_local.ndim == 3:
            contour_local = contour_local.reshape(-1, 2)
        contour_local = contour_local - np.array(roi_offset)

        closure = compute_edge_closure(contour_local, roi_gray)
        if closure < self.config.edge_closure_threshold:
            return False

        return True

    def _is_lens_edge(self, candidate: object, full_image: np.ndarray) -> bool:
        """Detect lens-edge artifacts: large, circular objects near image border.

        Args:
            candidate: GrainCandidate instance.
            full_image: Original input image.

        Returns:
            True if candidate matches lens-edge profile.
        """
        if getattr(candidate, "area", 0) < self.config.lens_edge_min_area:
            return False
        if getattr(candidate, "circularity", 0) < self.config.lens_edge_circularity:
            return False
        h, w = full_image.shape[:2]
        margin = self.config.lens_edge_margin
        border_dist = getattr(candidate, "border_distance", float("inf"))
        # border_distance is pixels; treat as lens edge if within margin of border
        max_border_dist = max(h, w) * margin
        if border_dist <= max_border_dist:
            return True
        return False

    def _is_noise(self, candidate: object, full_image: np.ndarray) -> bool:
        """Detect noise: small regions with weak texture and weak edges.

        Args:
            candidate: GrainCandidate instance.
            full_image: Original input image.

        Returns:
            True if candidate matches noise profile.
        """
        area = getattr(candidate, "area", 0)
        if area > self.config.noise_max_area:
            return False

        roi_gray = self._extract_roi(candidate, full_image)
        if roi_gray is None:
            return True

        texture = self._compute_texture_score(roi_gray)
        edge_strength = compute_edge_strength(roi_gray)

        if texture <= self.config.noise_max_texture_score and edge_strength <= self.config.noise_max_edge_strength:
            return True
        return False

    def _compute_composite_score(self, candidate: object, full_image: np.ndarray) -> float:
        """Combine texture and edge features into a single [0, 1] score.

        Args:
            candidate: GrainCandidate instance.
            full_image: Original input image.

        Returns:
            Composite score in [0, 1].
        """
        roi_gray = self._extract_roi(candidate, full_image)
        if roi_gray is None:
            return 0.0

        texture = self._compute_texture_score(roi_gray)
        edge_strength = compute_edge_strength(roi_gray)
        # Normalize edge strength roughly to [0, 1] assuming max ~255
        edge_norm = min(edge_strength / 100.0, 1.0)
        edge_dir = compute_edge_direction_consistency(roi_gray)

        # Weighted combination
        score = 0.4 * texture + 0.3 * edge_norm + 0.3 * edge_dir
        return float(np.clip(score, 0.0, 1.0))

    def _compute_texture_score(self, roi_gray: np.ndarray) -> float:
        """Compute texture consistency score for a grayscale ROI.

        Uses skimage LBP/GLCM when available, otherwise OpenCV fallback.

        Args:
            roi_gray: Grayscale ROI image.

        Returns:
            Texture score in [0, 1].
        """
        if self._has_skimage:
            lbp = extract_lbp_features(roi_gray)
            glcm = extract_glcm_features(roi_gray)
            return compute_texture_consistency_score(lbp, glcm)
        else:
            lbp = extract_lbp_features_opencv(roi_gray)
            glcm = extract_glcm_features_opencv(roi_gray)
            return compute_texture_consistency_score(lbp, glcm)

    def _extract_roi(
        self, candidate: object, full_image: np.ndarray
    ) -> np.ndarray | None:
        """Extract a grayscale ROI around the candidate with 10 px padding.

        Args:
            candidate: GrainCandidate instance (must have contour attribute).
            full_image: Original input image.

        Returns:
            Grayscale ROI or None if extraction fails.
        """
        result = self._extract_roi_with_offset(candidate, full_image)
        return result[0] if result is not None else None

    def _extract_roi_with_offset(
        self, candidate: object, full_image: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int]] | None:
        """Extract a grayscale ROI and its top-left offset in the full image.

        Args:
            candidate: GrainCandidate instance (must have contour attribute).
            full_image: Original input image.

        Returns:
            Tuple of (grayscale ROI, (x1, y1) offset) or None if extraction fails.
        """
        contour = getattr(candidate, "contour", None)
        if contour is None or len(contour) == 0:
            return None

        # Convert to grayscale if needed
        if len(full_image.shape) == 3:
            gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = full_image

        x, y, w, h = cv2.boundingRect(contour)
        pad = 10
        H, W = gray.shape
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return roi, (x1, y1)


# ---------------------------------------------------------------------------
# Texture feature functions (scikit-image)
# ---------------------------------------------------------------------------


def extract_lbp_features(roi_gray: np.ndarray) -> np.ndarray:
    """Extract uniform LBP histogram from a grayscale ROI.

    Requires scikit-image.

    Args:
        roi_gray: Grayscale ROI image.

    Returns:
        Normalized histogram array with 10 bins.
    """
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        return extract_lbp_features_opencv(roi_gray)

    if roi_gray.size == 0:
        return np.zeros(10, dtype=np.float64)

    # Uniform LBP with P=8, R=1
    lbp = local_binary_pattern(roi_gray, P=8, R=1, method="uniform")
    # Uniform patterns have values 0..9 (10 bins)
    n_bins = int(lbp.max()) + 1 if lbp.max() >= 0 else 10
    n_bins = min(max(n_bins, 10), 10)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_glcm_features(roi_gray: np.ndarray) -> dict[str, float]:
    """Extract GLCM texture features from a grayscale ROI.

    Requires scikit-image. Computes contrast, dissimilarity, homogeneity,
    energy, and correlation at distance 1 over four angles.

    Args:
        roi_gray: Grayscale ROI image.

    Returns:
        Dictionary of feature names to averaged float values.
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        return extract_glcm_features_opencv(roi_gray)

    if roi_gray.size == 0:
        return {
            "contrast": 0.0,
            "dissimilarity": 0.0,
            "homogeneity": 0.0,
            "energy": 0.0,
            "correlation": 0.0,
        }

    # Quantize to 32 levels for efficiency
    roi_q = (roi_gray // 8).astype(np.uint8)
    levels = 32

    try:
        glcm = graycomatrix(
            roi_q,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels,
            symmetric=True,
            normed=True,
        )
    except Exception:
        return {
            "contrast": 0.0,
            "dissimilarity": 0.0,
            "homogeneity": 0.0,
            "energy": 0.0,
            "correlation": 0.0,
        }

    features = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        try:
            vals = graycoprops(glcm, prop)
            features[prop] = float(np.mean(vals))
        except Exception:
            features[prop] = 0.0
    return features


def compute_texture_consistency_score(
    lbp_features: np.ndarray, glcm_features: dict[str, float]
) -> float:
    """Compute a unified texture consistency score in [0, 1].

    Higher values indicate more structured, grain-like texture.

    Args:
        lbp_features: Normalized LBP histogram.
        glcm_features: Dictionary of GLCM features.

    Returns:
        Score in [0, 1].
    """
    if lbp_features is None or len(lbp_features) == 0:
        lbp_score = 0.0
    else:
        # Entropy-like measure: non-uniform histogram = structured texture
        # Avoid log(0)
        p = lbp_features[lbp_features > 0]
        if len(p) > 0:
            entropy = -np.sum(p * np.log(p))
            max_ent = np.log(len(lbp_features))
            lbp_score = entropy / max_ent if max_ent > 0 else 0.0
        else:
            lbp_score = 0.0

    contrast = glcm_features.get("contrast", 0.0)
    energy = glcm_features.get("energy", 0.0)
    homogeneity = glcm_features.get("homogeneity", 0.0)

    # Normalize contrast roughly to [0, 1] (typical max ~100 for 32-level GLCM)
    contrast_norm = min(contrast / 50.0, 1.0)
    # Energy is already in [0, 1]
    energy_norm = np.clip(energy, 0.0, 1.0)
    # Homogeneity is already in [0, 1]
    homo_norm = np.clip(homogeneity, 0.0, 1.0)

    score = 0.3 * lbp_score + 0.3 * contrast_norm + 0.2 * energy_norm + 0.2 * homo_norm
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Edge feature functions (OpenCV)
# ---------------------------------------------------------------------------


def compute_edge_strength(roi_gray: np.ndarray) -> float:
    """Compute average Sobel edge magnitude.

    Args:
        roi_gray: Grayscale ROI image.

    Returns:
        Average Sobel magnitude across the ROI.
    """
    if roi_gray.size == 0:
        return 0.0

    sobel_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(np.mean(magnitude))


def compute_edge_direction_consistency(roi_gray: np.ndarray) -> float:
    """Compute edge direction concentration in [0, 1].

    Uses Sobel gradients and measures how concentrated edge directions are
    via the resultant vector magnitude. Sand grains often have consistent
    edge directions around their perimeter.

    Args:
        roi_gray: Grayscale ROI image.

    Returns:
        Direction consistency in [0, 1]. Higher = more consistent directions.
    """
    if roi_gray.size == 0:
        return 0.0

    sobel_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Only consider pixels with significant gradient
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    mask = magnitude > np.mean(magnitude) + np.std(magnitude)
    if not np.any(mask):
        return 0.0

    angles = np.arctan2(sobel_y[mask], sobel_x[mask])
    # Resultant vector length / number of vectors = concentration
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
    return float(np.clip(r, 0.0, 1.0))


def compute_edge_closure(contour: np.ndarray, roi_gray: np.ndarray) -> float:
    """Compute ratio of contour pixels that overlap Canny edges.

    Measures how well the contour is supported by actual image edges.
    The contour is expected to be in ROI-local coordinates (i.e. already
    shifted so the top-left of the ROI is (0, 0)).

    Args:
        contour: Nx2 contour array (or Nx1x2 from OpenCV) in ROI-local coords.
        roi_gray: Grayscale ROI image.

    Returns:
        Closure ratio in [0, 1].
    """
    if roi_gray.size == 0 or contour is None or len(contour) == 0:
        return 0.0

    # Ensure contour is Nx2
    if contour.ndim == 3:
        pts = contour.reshape(-1, 2)
    else:
        pts = contour

    # Build edge map
    edges = cv2.Canny(roi_gray, 50, 150)
    if cv2.countNonZero(edges) == 0:
        return 0.0

    # Count contour pixels that land on edges
    h, w = edges.shape
    on_edge = 0
    total = 0
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            total += 1
            if edges[yi, xi] > 0:
                on_edge += 1

    return on_edge / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# OpenCV fallback functions
# ---------------------------------------------------------------------------


def extract_lbp_features_opencv(roi_gray: np.ndarray) -> np.ndarray:
    """OpenCV fallback for LBP: local standard-deviation histogram.

    Computes local std in 3x3 neighborhoods as a texture proxy.

    Args:
        roi_gray: Grayscale ROI image.

    Returns:
        Normalized histogram array with 10 bins.
    """
    if roi_gray.size == 0:
        return np.zeros(10, dtype=np.float64)

    # Local std using Gaussian blur difference as proxy
    blur = cv2.GaussianBlur(roi_gray.astype(np.float32), (3, 3), 0)
    diff = np.abs(roi_gray.astype(np.float32) - blur)
    hist, _ = np.histogram(diff.ravel(), bins=10, range=(0, 50))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_glcm_features_opencv(roi_gray: np.ndarray) -> dict[str, float]:
    """OpenCV fallback for GLCM: Laplacian/Sobel-based texture features.

    Args:
        roi_gray: Grayscale ROI image.

    Returns:
        Dictionary with keys contrast, dissimilarity, homogeneity, energy, correlation.
    """
    if roi_gray.size == 0:
        return {
            "contrast": 0.0,
            "dissimilarity": 0.0,
            "homogeneity": 0.0,
            "energy": 0.0,
            "correlation": 0.0,
        }

    # Laplacian variance as contrast proxy
    lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
    contrast = float(np.var(lap))

    # Sobel-based features
    sobel_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    energy = float(np.mean(magnitude**2))
    # Normalize to roughly comparable ranges
    energy = min(energy / 5000.0, 1.0)
    contrast = min(contrast / 500.0, 1.0)

    # Dissimilarity proxy: mean absolute Laplacian
    dissimilarity = float(np.mean(np.abs(lap)))
    dissimilarity = min(dissimilarity / 50.0, 1.0)

    # Homogeneity: inverse of contrast
    homogeneity = 1.0 / (1.0 + contrast)

    # Correlation: not well approximated; use energy as proxy
    correlation = energy

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
    }
