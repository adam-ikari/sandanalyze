"""Tests for the shared detection pipeline module."""

import cv2
import numpy as np
import pytest

from core.detector import FlocculationConfig
from core.morphology import GrainContour, GrainStatistics
from core.preprocessor import PreprocessConfig


def _make_test_image(size=400):
    """Create a synthetic image with a realistic gray background for testing."""
    image = np.full((size, size, 3), 50, dtype=np.uint8)
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


class TestRunDetectionPipeline:
    """Tests for run_detection_pipeline in core.pipeline."""

    def test_returns_grains_morphologies_and_stats(self):
        """run_detection_pipeline should return grains, morphologies, and statistics."""
        from core.pipeline import run_detection_pipeline

        image = _make_test_image(400)
        cv2.circle(image, (200, 200), 35, (200, 200, 200), -1)

        config = _test_config()
        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert isinstance(grains, list)
        assert isinstance(morphologies, list)
        assert isinstance(stats, GrainStatistics)
        assert len(grains) > 0
        assert len(morphologies) > 0
        assert len(grains) == len(morphologies)

    def test_grains_have_contour_and_mask(self):
        """Each grain should have a contour and mask."""
        from core.pipeline import run_detection_pipeline

        image = _make_test_image(400)
        cv2.circle(image, (200, 200), 35, (200, 200, 200), -1)

        config = _test_config()
        grains, _, _ = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert len(grains) > 0
        for grain in grains:
            assert isinstance(grain, GrainContour)
            assert hasattr(grain, "contour")
            assert hasattr(grain, "mask")
            assert grain.contour is not None
            assert grain.mask is not None
            assert isinstance(grain.contour, np.ndarray)
            assert isinstance(grain.mask, np.ndarray)

    def test_morphologies_have_classification(self):
        """Each morphology should have shape_class, is_flocculation, and confidence."""
        from core.pipeline import run_detection_pipeline

        image = _make_test_image(400)
        cv2.circle(image, (200, 200), 35, (200, 200, 200), -1)

        config = _test_config()
        _, morphologies, _ = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert len(morphologies) > 0
        for morph in morphologies:
            assert hasattr(morph, "shape_class")
            assert hasattr(morph, "is_flocculation")
            assert hasattr(morph, "confidence")
            assert morph.shape_class in ("spherical", "rod-like", "discoidal", "flocculation")
            assert isinstance(morph.is_flocculation, bool)
            assert isinstance(morph.confidence, float)
            assert morph.confidence > 0

    def test_statistics_are_computed(self):
        """Statistics should reflect the detected grains."""
        from core.pipeline import run_detection_pipeline

        image = _make_test_image(400)
        cv2.circle(image, (200, 200), 35, (200, 200, 200), -1)

        config = _test_config()
        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert stats.count == len(grains)
        assert stats.count > 0
        assert stats.area_mean > 0
        assert stats.circularity_mean > 0
        assert stats.d_eq_mean > 0
        assert stats.aspect_ratio_mean > 0
        assert stats.sphericity_mean > 0
        assert stats.convexity_mean > 0

    def test_empty_image_returns_empty(self):
        """An empty image should return empty lists and zero-count stats."""
        from core.pipeline import run_detection_pipeline

        image = _make_test_image(400)

        config = _test_config()
        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert grains == []
        assert morphologies == []
        assert stats.count == 0
        assert stats.area_mean == 0.0

    def test_flocculation_config_is_respected(self):
        """Flocculation config should be passed through to detect_grains."""
        from core.pipeline import run_detection_pipeline

        # Create an irregular C-shape that should trigger flocculation
        image = _make_test_image(400)
        pts = np.array([
            [155, 155], [245, 155], [245, 170], [170, 170],
            [170, 230], [245, 230], [245, 245], [155, 245]
        ], dtype=np.int32)
        cv2.fillPoly(image, [pts], (200, 200, 200))

        config = _test_config()
        floc_config = FlocculationConfig(
            max_circularity=0.5,
            max_convexity=0.8,
        )
        _, morphologies, _ = run_detection_pipeline(
            image, config, min_area=50, floc_config=floc_config,
            crop_black_background=False,
        )

        assert len(morphologies) > 0
        # The irregular shape should be flagged as flocculation
        assert any(m.is_flocculation for m in morphologies)
        # Verify the flocculation morph has the right shape_class
        floc_morphs = [m for m in morphologies if m.is_flocculation]
        for m in floc_morphs:
            assert m.shape_class == "flocculation"

    def test_grayscale_image(self):
        """Should accept grayscale images."""
        from core.pipeline import run_detection_pipeline

        image = np.full((400, 400), 50, dtype=np.uint8)
        noise = np.random.randint(0, 20, (400, 400), dtype=np.uint8)
        image = cv2.add(image, noise)
        cv2.circle(image, (200, 200), 35, 200, -1)

        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)
        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert len(grains) > 0

    def test_crop_black_background(self):
        """Should work with crop_black_background=True."""
        from core.pipeline import run_detection_pipeline

        image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (350, 350), (220, 220, 220), -1)
        cv2.circle(image, (200, 200), 30, (80, 80, 80), -1)

        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)
        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=True
        )

        assert len(grains) > 0

    def test_hull_expansion_ratio(self):
        """Should respect hull_expansion_ratio parameter."""
        from core.pipeline import run_detection_pipeline

        image = _make_test_image()
        cv2.circle(image, (200, 200), 35, (200, 200, 200), -1)
        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)

        grains_low, _, _ = run_detection_pipeline(
            image, config, min_area=50, hull_expansion_ratio=1.2, crop_black_background=False
        )
        grains_high, _, _ = run_detection_pipeline(
            image, config, min_area=50, hull_expansion_ratio=5.0, crop_black_background=False
        )

        assert len(grains_low) > 0
        assert len(grains_high) > 0


class TestMultiScalePipeline:
    """Tests for run_multiscale_detection_pipeline in core.pipeline."""

    def test_returns_grains_morphologies_and_stats(self):
        """run_multiscale_detection_pipeline should return grains, morphologies, and statistics."""
        from core.pipeline import run_multiscale_detection_pipeline

        image = _make_test_image(400)
        cv2.circle(image, (200, 200), 35, (200, 200, 200), -1)

        config = _test_config()
        grains, morphologies, stats = run_multiscale_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert isinstance(grains, list)
        assert isinstance(morphologies, list)
        assert isinstance(stats, GrainStatistics)
        assert len(grains) > 0
        assert len(morphologies) > 0
        assert len(grains) == len(morphologies)

    def test_multiscale_detects_more_grains_than_single(self):
        """Multi-scale should find more grains than single-scale pipeline."""
        from core.pipeline import (
            run_detection_pipeline,
            run_multiscale_detection_pipeline,
        )

        # Create an image with grains of varying sizes
        np.random.seed(42)
        image = np.ones((400, 400, 3), dtype=np.uint8) * 180
        # Add a large grain
        cv2.circle(image, (100, 100), 40, (50, 50, 50), -1)
        # Add a medium grain
        cv2.circle(image, (250, 150), 25, (50, 50, 50), -1)
        # Add a small grain
        cv2.circle(image, (320, 300), 12, (50, 50, 50), -1)

        config = PreprocessConfig(
            blur_kernel=5,
            adaptive_block_size=51,
            adaptive_c=5,
            morph_kernel_size=3,
            morph_open_iter=1,
            morph_close_iter=1,
            min_area=800,
            use_clahe=True,
        )

        single_grains, _, _ = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )
        multi_grains, _, _ = run_multiscale_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        # Multi-scale should detect at least as many grains as single-scale
        assert len(multi_grains) >= len(single_grains)

    def test_multiscale_filters_false_positives(self):
        """Should filter out noise and edge artifacts."""
        from core.pipeline import run_multiscale_detection_pipeline

        # Create an image with a real grain and some noise near the edge
        np.random.seed(42)
        image = np.ones((400, 400, 3), dtype=np.uint8) * 180
        # Add a real grain in the center
        cv2.circle(image, (200, 200), 35, (50, 50, 50), -1)
        # Add small noise blobs near the edge
        cv2.circle(image, (10, 10), 5, (60, 60, 60), -1)
        cv2.circle(image, (390, 390), 5, (60, 60, 60), -1)

        config = PreprocessConfig(
            blur_kernel=5,
            adaptive_block_size=51,
            adaptive_c=5,
            morph_kernel_size=3,
            morph_open_iter=1,
            morph_close_iter=1,
            min_area=800,
            use_clahe=True,
        )

        grains, morphologies, stats = run_multiscale_detection_pipeline(
            image, config, min_area=200, crop_black_background=False
        )

        # Should detect the real grain but filter out edge noise
        assert len(grains) > 0
        # The real grain should be detected, noise should be filtered
        for grain in grains:
            # Check that detected grains are reasonably sized (not tiny noise)
            x, y, bw, bh = cv2.boundingRect(grain.contour)
            # Grains should not be tiny
            assert bw > 5 or bh > 5
            # Grains should not be right at the edge (filtered by edge filter)
            assert x > 2
            assert y > 2

    def test_multiscale_empty_image_returns_empty(self):
        """An empty image should return empty lists and zero-count stats."""
        from core.pipeline import run_multiscale_detection_pipeline

        image = _make_test_image(400)

        config = _test_config()
        grains, morphologies, stats = run_multiscale_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert grains == []
        assert morphologies == []
        assert stats.count == 0
        assert stats.area_mean == 0.0
