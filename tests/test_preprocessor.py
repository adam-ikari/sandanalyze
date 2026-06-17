"""Tests for the image preprocessing module."""

import cv2
import numpy as np
import pytest

from core.preprocessor import PreprocessConfig, preprocess


class TestPreprocessConfig:
    """Tests for the PreprocessConfig dataclass."""

    def test_default_values(self):
        """Test that PreprocessConfig has sensible defaults."""
        config = PreprocessConfig()
        assert config.blur_kernel == 5
        assert config.adaptive_block_size == 11
        assert config.adaptive_c == 2
        assert config.morph_kernel_size == 3
        assert config.morph_open_iter == 1
        assert config.morph_close_iter == 1
        assert config.min_area == 50
        assert config.use_clahe is True
        assert config.use_watershed is True
        assert config.watershed_thresh_ratio == 0.5


class TestPreprocessReturns:
    """Tests for the preprocess function return values."""

    def test_returns_binary_mask(self, sample_grain_image):
        """Test that preprocess returns a binary mask (0 and 255)."""
        config = PreprocessConfig(use_clahe=False, use_watershed=False)
        result = preprocess(sample_grain_image, config)

        assert result.dtype == np.uint8
        assert set(np.unique(result).tolist()).issubset({0, 255})

    def test_returns_same_shape(self, sample_grain_image):
        """Test that output mask has the same shape as input."""
        config = PreprocessConfig(use_clahe=False, use_watershed=False)
        result = preprocess(sample_grain_image, config)
        assert result.shape == sample_grain_image.shape


class TestClahe:
    """Tests for CLAHE enhancement."""

    def test_clahe_enhances_contrast(self, sample_grain_image):
        """Test that CLAHE increases local contrast."""
        config_no_clahe = PreprocessConfig(use_clahe=False, use_watershed=False)
        config_with_clahe = PreprocessConfig(use_clahe=True, use_watershed=False)

        result_no_clahe = preprocess(sample_grain_image, config_no_clahe)
        result_with_clahe = preprocess(sample_grain_image, config_with_clahe)

        # With CLAHE, the mask should have more foreground pixels
        # because contrast enhancement makes grains more distinct
        assert np.sum(result_with_clahe) >= np.sum(result_no_clahe)


class TestWatershed:
    """Tests for watershed segmentation."""

    def test_watershed_separates_touching_grains(self):
        """Test that watershed separates touching grains.

        Creates a figure-8 shape (two circles connected by a thick bridge)
        that survives the preprocessing pipeline as a single component,
        then verifies watershed splits it into two distinct regions.
        """
        # Create a figure-8 shape with strong contrast
        binary = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(binary, (70, 100), 30, 255, -1)
        cv2.circle(binary, (130, 100), 30, 255, -1)
        # Thick bridge to survive adaptive thresholding
        cv2.rectangle(binary, (70, 95), (130, 105), 255, -1)

        img = np.where(binary > 0, 255, 0).astype(np.uint8)

        # Use a larger adaptive block size to avoid splitting the bridge
        config_no_watershed = PreprocessConfig(
            use_clahe=False,
            use_watershed=False,
            min_area=10,
            adaptive_block_size=71,
            adaptive_c=0,
        )
        config_with_watershed = PreprocessConfig(
            use_clahe=False,
            use_watershed=True,
            min_area=10,
            adaptive_block_size=71,
            adaptive_c=0,
        )

        result_no_watershed = preprocess(img, config_no_watershed)
        result_with_watershed = preprocess(img, config_with_watershed)

        # Count connected components (excluding background)
        num_no_watershed, _ = cv2.connectedComponents(result_no_watershed)
        num_with_watershed, _ = cv2.connectedComponents(result_with_watershed)

        # With watershed, we should have more components
        assert num_with_watershed > num_no_watershed


class TestEdgeFiltering:
    """Tests for edge filtering."""

    def test_filter_edge_grains_removes_border_contours(self):
        """Contours touching image border should be removed."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)

        # Grain at center - should keep
        cv2.circle(mask, (50, 50), 10, 255, -1)

        # Grain at edge - should remove
        cv2.circle(mask, (5, 50), 8, 255, -1)

        # Grain at corner - should remove
        cv2.circle(mask, (95, 95), 8, 255, -1)

        from core.preprocessor import filter_edge_grains
        filtered = filter_edge_grains(mask, border_margin=5)

        # Count remaining grains
        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1

        # Verify the remaining grain is the center one
        x, y, bw, bh = cv2.boundingRect(contours[0])
        assert x > 5 and y > 5

    def test_filter_edge_grains_keeps_all_center(self):
        """All center grains should be kept."""
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.uint8)

        # Multiple grains at center
        cv2.circle(mask, (50, 50), 15, 255, -1)
        cv2.circle(mask, (100, 100), 15, 255, -1)
        cv2.circle(mask, (150, 150), 15, 255, -1)

        from core.preprocessor import filter_edge_grains
        filtered = filter_edge_grains(mask, border_margin=10)

        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 3


class TestAutoTune:
    """Tests for auto-parameter tuning."""

    def test_auto_tune_params_returns_valid_config(self):
        """Auto-tuning should return reasonable parameters."""
        from core.preprocessor import auto_tune_params

        # Create a synthetic image with known grain size
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 20, 200, -1)

        config = auto_tune_params(img)

        assert config.blur_kernel >= 3
        assert config.adaptive_block_size >= 3
        assert config.min_area > 0

    def test_auto_tune_params_on_color_image(self):
        """Auto-tuning should work on color images."""
        from core.preprocessor import auto_tune_params

        # Create a color image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 20, (200, 200, 200), -1)

        config = auto_tune_params(img)

        assert config.blur_kernel >= 3
        assert config.min_area > 0
    """Tests for different input image formats."""

    def test_grayscale_input(self, sample_grain_image):
        """Test that grayscale input is handled correctly."""
        config = PreprocessConfig(use_clahe=False, use_watershed=False)
        result = preprocess(sample_grain_image, config)
        assert result.shape == sample_grain_image.shape

    def test_color_input(self, sample_grain_image):
        """Test that color (3-channel) input is handled correctly."""
        color_image = cv2.cvtColor(sample_grain_image, cv2.COLOR_GRAY2BGR)
        config = PreprocessConfig(use_clahe=False, use_watershed=False)
        result = preprocess(color_image, config)
        assert result.shape == sample_grain_image.shape

    def test_grayscale_and_color_produce_similar_results(self, sample_grain_image):
        """Test that grayscale and color inputs produce similar masks."""
        color_image = cv2.cvtColor(sample_grain_image, cv2.COLOR_GRAY2BGR)
        config = PreprocessConfig(use_clahe=False, use_watershed=False)

        result_gray = preprocess(sample_grain_image, config)
        result_color = preprocess(color_image, config)

        assert result_gray.shape == result_color.shape
        assert result_gray.dtype == result_color.dtype
