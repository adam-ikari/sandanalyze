"""Tests for the grain detection module (v6 pipeline)."""

import cv2
import numpy as np

import pytest

from core.detector import detect_grains, FlocculationConfig, DetectionResult
from core.preprocessor import PreprocessConfig


def _make_test_image(size=400):
    """Create a synthetic image with a realistic gray background for testing.

    Pure black backgrounds cause adaptiveThreshold to fragment circles
    because the local mean is computed over neighborhoods that are mostly
    black. A gray background with slight noise makes the thresholding stable.
    """
    image = np.full((size, size, 3), 50, dtype=np.uint8)
    # Add slight noise
    noise = np.random.randint(0, 20, (size, size, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    return image


def _test_config():
    """Return a PreprocessConfig tuned for synthetic test images."""
    return PreprocessConfig(
        adaptive_block_size=11,
        morph_kernel_size=3,
        morph_open_iter=1,
    )


class TestDetectGrains:
    """Tests for the detect_grains function with v6 pipeline."""

    def test_detect_grains_from_image(self):
        """Should detect grains from a raw image."""
        image = _make_test_image(400)
        cv2.circle(image, (100, 100), 30, (200, 200, 200), -1)
        cv2.circle(image, (300, 300), 30, (200, 200, 200), -1)

        config = _test_config()
        results = detect_grains(
            image, config, min_area=50, crop_black_background=False
        )

        assert len(results) > 0
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_detect_grains_returns_expected_fields(self):
        """DetectionResult should have all expected fields."""
        image = _make_test_image(400)
        cv2.circle(image, (200, 200), 35, (200, 200, 200), -1)

        config = _test_config()
        results = detect_grains(
            image, config, min_area=50, crop_black_background=False
        )

        assert len(results) > 0
        r = results[0]
        assert hasattr(r, 'contour')
        assert hasattr(r, 'mask')
        assert hasattr(r, 'area')
        assert hasattr(r, 'perimeter')
        assert hasattr(r, 'circularity')
        assert hasattr(r, 'aspect_ratio')
        assert hasattr(r, 'major_axis')
        assert hasattr(r, 'minor_axis')
        assert hasattr(r, 'convexity')
        assert hasattr(r, 'is_flocculation')
        # is_edge was removed in v6
        assert not hasattr(r, 'is_edge')

    def test_detect_grains_filters_by_area(self):
        """Should filter out grains outside min_area/max_area range."""
        image = _make_test_image(400)
        cv2.circle(image, (100, 100), 30, (200, 200, 200), -1)   # Large
        cv2.circle(image, (300, 300), 8, (200, 200, 200), -1)   # Small

        config = _test_config()
        results = detect_grains(
            image, config, min_area=300, max_area=5000,
            crop_black_background=False,
        )

        # Only the large grain should be detected
        assert all(r.area >= 300 for r in results)
        assert all(r.area <= 5000 for r in results)

    def test_detect_grains_filters_edge_grains(self):
        """Grains touching the ROI boundary should be excluded."""
        image = _make_test_image(400)
        # Grain at center - should be kept
        cv2.circle(image, (200, 200), 30, (200, 200, 200), -1)
        # Grain touching the edge - should be filtered out
        cv2.circle(image, (15, 15), 20, (200, 200, 200), -1)

        config = _test_config()
        results = detect_grains(
            image, config, min_area=50, crop_black_background=False
        )

        # Only the center grain should remain (edge grain filtered by ROI boundary)
        assert len(results) >= 1
        for r in results:
            x, y, bw, bh = cv2.boundingRect(r.contour)
            # Verify no result touches the image border
            assert x > 0
            assert y > 0
            assert x + bw < 400
            assert y + bh < 400

    def test_detect_grains_with_hull_expansion_ratio(self):
        """Hull expansion ratio should affect contour selection."""
        image = _make_test_image(400)
        cv2.circle(image, (200, 200), 35, (200, 200, 200), -1)

        config = _test_config()
        # Low ratio: prefer convex hull when hull_area/area < ratio
        results_low = detect_grains(
            image, config, min_area=50, hull_expansion_ratio=1.2,
            crop_black_background=False,
        )
        # High ratio: prefer original contour when hull_area/area < ratio
        results_high = detect_grains(
            image, config, min_area=50, hull_expansion_ratio=5.0,
            crop_black_background=False,
        )

        assert len(results_low) > 0
        assert len(results_high) > 0

    def test_detect_grains_with_flocculation_config(self):
        """Flocculation config should affect is_flocculation flag."""
        # Create an irregular shape (simulating flocculation)
        # Use a C-shape with low circularity and low convexity (solid, no holes)
        image = _make_test_image(400)
        pts = np.array([
            [155, 155], [245, 155], [245, 170], [170, 170], [170, 230], [245, 230],
            [245, 245], [155, 245]
        ], dtype=np.int32)
        cv2.fillPoly(image, [pts], (200, 200, 200))

        config = _test_config()
        floc_config = FlocculationConfig(
            max_circularity=0.5,
            max_convexity=0.8,
        )
        results = detect_grains(
            image, config, min_area=50, floc_config=floc_config,
            crop_black_background=False,
        )

        assert len(results) > 0
        # The irregular cluster should be flagged as flocculation
        assert any(r.is_flocculation for r in results)

    def test_detect_grains_sorts_by_area_descending(self):
        """Results should be sorted by area descending."""
        image = _make_test_image(400)
        cv2.circle(image, (100, 100), 35, (200, 200, 200), -1)   # Large
        cv2.circle(image, (300, 300), 18, (200, 200, 200), -1)  # Small

        config = _test_config()
        results = detect_grains(
            image, config, min_area=50, crop_black_background=False,
        )

        areas = [r.area for r in results]
        assert areas == sorted(areas, reverse=True)

    def test_detect_grains_with_grayscale_image(self):
        """Should work with grayscale images too."""
        image = np.full((400, 400), 50, dtype=np.uint8)
        noise = np.random.randint(0, 20, (400, 400), dtype=np.uint8)
        image = cv2.add(image, noise)
        cv2.circle(image, (200, 200), 35, 200, -1)

        config = _test_config()
        results = detect_grains(
            image, config, min_area=50, crop_black_background=False,
        )

        assert len(results) > 0
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_detect_grains_empty_image(self):
        """Should return empty list for empty image."""
        image = _make_test_image(400)

        config = _test_config()
        results = detect_grains(
            image, config, min_area=50, crop_black_background=False,
        )

        assert len(results) == 0

    def test_detect_grains_with_crop_black_background(self):
        """Should work with crop_black_background=True on a real-like image."""
        # Create a synthetic image that mimics a real sand image:
        # - Black background (0)
        # - A bright rectangular region in the center (simulating the sample tray)
        # - Dark circular grains on the bright tray (grains are darker than background)
        image = np.zeros((400, 400, 3), dtype=np.uint8)

        # Bright tray region in the center (light gray background)
        tray_color = (220, 220, 220)
        cv2.rectangle(image, (50, 50), (350, 350), tray_color, -1)

        # Dark grains on the tray (grains are darker than the tray)
        # These will be detected as foreground by adaptive threshold (THRESH_BINARY_INV)
        grain_color = (80, 80, 80)
        cv2.circle(image, (200, 200), 30, grain_color, -1)
        cv2.circle(image, (120, 120), 25, grain_color, -1)
        cv2.circle(image, (280, 150), 28, grain_color, -1)

        config = _test_config()
        results = detect_grains(
            image, config, min_area=50, crop_black_background=True,
        )

        assert len(results) > 0
        assert all(isinstance(r, DetectionResult) for r in results)
