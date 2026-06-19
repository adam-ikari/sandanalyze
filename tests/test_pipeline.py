"""Tests for the shared detection pipeline module."""

import cv2
import numpy as np
import pytest

from core.detector import FlocculationConfig
from core.morphology import GrainStatistics
from core.preprocessor import PreprocessConfig
from core.traditional import GrainContour


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
