"""Lightweight false-positive filtering for grain candidates.

Provides SimpleValidator that filters obvious false positives:
- Lens-edge artifacts (large circular objects near border)
- Noise/fragments (very small, near-border, or extremely low contrast)
- Background blobs (low contrast, no distinct edges)

Designed to be fast, simple, and conservative — only reject obvious non-grains.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ValidationConfig:
    """Configuration for lightweight false-positive filtering.

    Attributes:
        lens_edge_margin: Fraction of image border considered "near edge".
        lens_edge_circularity: Minimum circularity to be considered a lens edge.
        lens_edge_min_area: Minimum area (pixels) for a lens edge candidate.
        noise_max_area: Maximum area (pixels) for noise classification.
        min_contrast: Minimum ROI std to be considered a real grain.
    """

    lens_edge_margin: float = 0.05
    lens_edge_circularity: float = 0.7
    lens_edge_min_area: float = 50000
    noise_max_area: int = 500
    min_contrast: float = 5.0


class SimpleValidator:
    """Lightweight validator that only rejects obvious false positives.

    Conservative approach: only filter what is clearly NOT a grain.
    Real sand grains (even low-contrast or uniform-textured) are preserved.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize validator with optional custom configuration.

        Args:
            config: ValidationConfig instance. Uses defaults if None.
        """
        self.config = config if config is not None else ValidationConfig()

    def validate(self, candidate: object, full_image: np.ndarray) -> bool:
        """Validate a grain candidate.

        Only rejects obvious false positives. Real grains are preserved.

        Args:
            candidate: GrainCandidate instance (must have area, circularity,
                contour, border_distance attributes).
            full_image: Original input image (grayscale or BGR).

        Returns:
            True if candidate is likely a real grain.
        """
        # 1. Reject lens-edge artifacts
        if self._is_lens_edge(candidate, full_image):
            return False

        # 2. Reject noise/fragments
        if self._is_noise(candidate, full_image):
            return False

        # 3. Reject extremely low contrast (background blobs)
        if self._is_low_contrast(candidate, full_image):
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
        max_border_dist = max(h, w) * margin

        return border_dist <= max_border_dist

    def _is_noise(self, candidate: object, full_image: np.ndarray) -> bool:
        """Detect noise: very small regions.

        Args:
            candidate: GrainCandidate instance.
            full_image: Original input image.

        Returns:
            True if candidate is likely noise.
        """
        area = getattr(candidate, "area", 0)
        return area < self.config.noise_max_area

    def _is_low_contrast(self, candidate: object, full_image: np.ndarray) -> bool:
        """Detect background blobs: extremely low contrast regions.

        Args:
            candidate: GrainCandidate instance.
            full_image: Original input image.

        Returns:
            True if candidate has extremely low contrast (likely background).
        """
        roi = self._extract_roi(candidate, full_image)
        if roi is None:
            return True

        # Very low std means uniform region (background)
        return float(np.std(roi)) < self.config.min_contrast

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
        contour = getattr(candidate, "contour", None)
        if contour is None or len(contour) == 0:
            return None

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
        return roi if roi.size > 0 else None
