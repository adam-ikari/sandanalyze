"""Integration tests for the full sand analysis pipeline."""

import os

import cv2
import numpy as np
import pytest

from core.detector import detect_grains as detect_grains_v6
from core.preprocessor import PreprocessConfig, preprocess
from core.morphology import compute_morphology, compute_statistics


class TestFullPipelineSynthetic:
    """Integration tests using synthetic grain images."""

    def test_pipeline_without_watershed(self, sample_grain_image):
        """Test full pipeline with synthetic image."""
        # Step 1: Preprocess
        config = PreprocessConfig(use_clahe=False, min_area=50)
        mask = preprocess(sample_grain_image, config)

        # Verify mask is binary and has content
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()).issubset({0, 255})
        assert np.sum(mask) > 0, "Preprocessed mask should contain foreground pixels"

        # Step 2: Detect grains using v6 pipeline
        image_color = cv2.cvtColor(sample_grain_image, cv2.COLOR_GRAY2BGR)
        results = detect_grains_v6(image_color, config, min_area=50, crop_black_background=False)
        assert len(results) > 0, "Should detect at least one grain"

        # Step 3: Compute morphology for each grain
        morphologies = []
        for result in results:
            morph = compute_morphology(result.contour, result.mask)
            morphologies.append(morph)

        # Verify morphologies are computed
        assert len(morphologies) == len(results)
        for morph in morphologies:
            assert morph.area > 0
            assert morph.perimeter > 0
            assert 0 < morph.circularity <= 1.0
            assert morph.d_eq > 0
            assert morph.major_axis > 0
            assert morph.minor_axis > 0
            assert morph.aspect_ratio >= 1.0
            assert 0 < morph.sphericity <= 1.0
            assert 0 < morph.convexity <= 1.0
            assert morph.feret_max > 0
            assert morph.feret_min > 0

        # Step 4: Compute statistics
        stats = compute_statistics(morphologies)

        # Verify statistics
        assert stats.count == len(results)
        assert stats.area_mean > 0
        assert stats.area_std >= 0
        assert stats.area_median > 0
        assert 0 < stats.circularity_mean <= 1.0
        assert stats.circularity_std >= 0
        assert 0 < stats.circularity_median <= 1.0
        assert stats.d_eq_mean > 0
        assert stats.d_eq_std >= 0
        assert stats.d_eq_median > 0
        assert stats.aspect_ratio_mean >= 1.0
        assert stats.aspect_ratio_std >= 0
        assert stats.aspect_ratio_median >= 1.0
        assert 0 < stats.sphericity_mean <= 1.0
        assert stats.sphericity_std >= 0
        assert 0 < stats.sphericity_median <= 1.0
        assert 0 < stats.convexity_mean <= 1.0
        assert stats.convexity_std >= 0
        assert 0 < stats.convexity_median <= 1.0
        assert len(stats.d_eq_values) == stats.count
        assert len(stats.circularity_values) == stats.count
        assert len(stats.sphericity_values) == stats.count
        assert isinstance(stats.zingg_counts, dict)


class TestFullPipelineRealImage:
    """Integration tests using a real sand image."""

    def test_pipeline_with_real_image(self, real_sand_image_path):
        """Test full pipeline with the real sand image."""
        # Skip if real image not available
        if not os.path.exists(real_sand_image_path):
            pytest.skip(f"Real image not found at {real_sand_image_path}")

        # Load real image
        image = cv2.imread(real_sand_image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            pytest.skip(f"Could not load real image from {real_sand_image_path}")

        # Step 1: Preprocess
        config = PreprocessConfig(use_clahe=True, min_area=50)
        mask = preprocess(image, config)

        # Verify mask
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()).issubset({0, 255})
        assert np.sum(mask) > 0, "Preprocessed mask should contain foreground pixels"

        # Step 2: Detect grains using v6 pipeline
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        results = detect_grains_v6(image_color, config, min_area=50, crop_black_background=False)
        assert len(results) > 0, "Should detect at least one grain in real image"

        # Step 3: Compute morphology for each grain
        morphologies = []
        for result in results:
            morph = compute_morphology(result.contour, result.mask)
            morphologies.append(morph)

        assert len(morphologies) == len(results)
        for morph in morphologies:
            assert morph.area > 0
            assert morph.perimeter > 0
            assert 0 < morph.circularity <= 1.0
            assert morph.d_eq > 0
            assert morph.major_axis > 0
            assert morph.minor_axis > 0
            assert morph.aspect_ratio >= 1.0
            assert 0 < morph.sphericity <= 1.0
            assert 0 < morph.convexity <= 1.0
            assert morph.feret_max > 0
            assert morph.feret_min > 0

        # Step 4: Compute statistics
        stats = compute_statistics(morphologies)

        # Verify statistics are reasonable
        assert stats.count == len(results)
        assert stats.count > 0
        assert stats.area_mean > 0
        assert stats.area_std >= 0
        assert stats.area_median > 0
        assert 0 < stats.circularity_mean <= 1.0
        assert stats.circularity_std >= 0
        assert 0 < stats.circularity_median <= 1.0
        assert stats.d_eq_mean > 0
        assert stats.d_eq_std >= 0
        assert stats.d_eq_median > 0
        assert stats.aspect_ratio_mean >= 1.0
        assert stats.aspect_ratio_std >= 0
        assert stats.aspect_ratio_median >= 1.0
        assert 0 < stats.sphericity_mean <= 1.0
        assert stats.sphericity_std >= 0
        assert 0 < stats.sphericity_median <= 1.0
        assert 0 < stats.convexity_mean <= 1.0
        assert stats.convexity_std >= 0
        assert 0 < stats.convexity_median <= 1.0
        assert len(stats.d_eq_values) == stats.count
        assert len(stats.circularity_values) == stats.count
        assert len(stats.sphericity_values) == stats.count
        assert isinstance(stats.zingg_counts, dict)

        # Zingg counts should sum to total count
        total_zingg = sum(stats.zingg_counts.values())
        assert total_zingg == stats.count
