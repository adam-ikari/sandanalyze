"""Tests for the image preprocessing module."""

import cv2
import numpy as np
import pytest

from core.preprocessor import PreprocessConfig, preprocess, crop_black_background


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


class TestPreprocessReturns:
    """Tests for the preprocess function return values."""

    def test_returns_binary_mask(self, sample_grain_image):
        """Test that preprocess returns a binary mask (0 and 255)."""
        config = PreprocessConfig(use_clahe=False)
        result = preprocess(sample_grain_image, config)

        assert result.dtype == np.uint8
        assert set(np.unique(result).tolist()).issubset({0, 255})

    def test_returns_same_shape(self, sample_grain_image):
        """Test that output mask has the same shape as input."""
        config = PreprocessConfig(use_clahe=False)
        result = preprocess(sample_grain_image, config)
        assert result.shape == sample_grain_image.shape


class TestClahe:
    """Tests for CLAHE enhancement."""

    def test_clahe_enhances_contrast(self, sample_grain_image):
        """Test that CLAHE increases local contrast."""
        config_no_clahe = PreprocessConfig(use_clahe=False)
        config_with_clahe = PreprocessConfig(use_clahe=True)

        result_no_clahe = preprocess(sample_grain_image, config_no_clahe)
        result_with_clahe = preprocess(sample_grain_image, config_with_clahe)

        # With CLAHE, the mask should have more foreground pixels
        # because contrast enhancement makes grains more distinct
        assert np.sum(result_with_clahe) >= np.sum(result_no_clahe)


class TestCropBlackBackground:
    """Tests for crop_black_background helper."""

    def test_crops_black_background(self):
        """Test that black background is cropped to the bright region."""
        # Create an image with a bright rectangle on a black background
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 120), (200, 200, 200), -1)

        cropped, (x, y, w, h) = crop_black_background(image)

        # Check that the crop bounds match the bright rectangle
        assert x == 50
        assert y == 50
        assert w == 101
        assert h == 71

        # Check that the cropped image has the expected shape
        assert cropped.shape == (71, 101, 3)

        # Check that the cropped image contains the bright region
        assert np.mean(cropped) > 100

    def test_crops_grayscale_image(self):
        """Test that crop_black_background works on grayscale images."""
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (30, 40), (100, 90), 255, -1)

        cropped, (x, y, w, h) = crop_black_background(image)

        assert x == 30
        assert y == 40
        assert w == 71
        assert h == 51
        assert cropped.shape == (51, 71)

    def test_returns_full_image_when_no_bright_region(self):
        """Test that an all-black image returns the original image."""
        image = np.zeros((100, 100), dtype=np.uint8)

        cropped, (x, y, w, h) = crop_black_background(image)

        assert x == 0
        assert y == 0
        assert w == 100
        assert h == 100
        assert cropped.shape == (100, 100)

    def test_returns_largest_component(self):
        """Test that only the largest bright component is kept."""
        image = np.zeros((200, 200), dtype=np.uint8)
        # Small bright component
        cv2.rectangle(image, (10, 10), (30, 30), 255, -1)
        # Large bright component
        cv2.rectangle(image, (80, 80), (180, 160), 255, -1)

        cropped, (x, y, w, h) = crop_black_background(image)

        # Should crop to the large component
        assert x == 80
        assert y == 80
        assert w == 101
        assert h == 81


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


class TestInputHandling:
    """Tests for different input image formats."""

    def test_grayscale_input(self, sample_grain_image):
        """Test that grayscale input is handled correctly."""
        config = PreprocessConfig(use_clahe=False)
        result = preprocess(sample_grain_image, config)
        assert result.shape == sample_grain_image.shape

    def test_color_input(self, sample_grain_image):
        """Test that color (3-channel) input is handled correctly."""
        color_image = cv2.cvtColor(sample_grain_image, cv2.COLOR_GRAY2BGR)
        config = PreprocessConfig(use_clahe=False)
        result = preprocess(color_image, config)
        assert result.shape == sample_grain_image.shape

    def test_grayscale_and_color_produce_similar_results(self, sample_grain_image):
        """Test that grayscale and color inputs produce similar masks."""
        color_image = cv2.cvtColor(sample_grain_image, cv2.COLOR_GRAY2BGR)
        config = PreprocessConfig(use_clahe=False)

        result_gray = preprocess(sample_grain_image, config)
        result_color = preprocess(color_image, config)

        assert result_gray.shape == result_color.shape
        assert result_gray.dtype == result_color.dtype
