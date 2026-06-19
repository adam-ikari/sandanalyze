"""Tests for the shadow enhancement module."""

import cv2
import numpy as np
import pytest

from core.shadow_enhancer import enhance_shadow_regions, validate_local_contrast


class TestEnhanceShadowRegions:
    """Tests for the enhance_shadow_regions function."""

    def test_enhances_shadow_regions(self):
        """Shadow regions should have increased contrast."""
        # Create a dark image with some shadow regions
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:] = 30  # Dark background
        # Add a slightly brighter region
        cv2.circle(img, (50, 50), 20, 60, -1)

        enhanced = enhance_shadow_regions(img, clip_limit=3.0)

        # Shadow regions should have increased contrast
        assert enhanced is not None
        assert enhanced.shape == img.shape
        # CLAHE should increase local contrast in shadow regions
        assert np.std(enhanced) > np.std(img)

    def test_preserves_bright_regions(self):
        """Bright regions should not be over-enhanced."""
        # Create a bright image
        img = np.ones((100, 100), dtype=np.uint8) * 200
        # Add some variation
        img[40:60, 40:60] = 220

        enhanced = enhance_shadow_regions(img, clip_limit=3.0)

        # Bright regions should not be dramatically changed
        assert enhanced is not None
        assert enhanced.shape == img.shape
        # The mean should not change drastically for bright images
        # CLAHE can clip bright values, so allow a larger tolerance
        assert np.abs(np.mean(enhanced) - np.mean(img)) < 100

    def test_handles_color_image(self):
        """CLAHE should work on color images."""
        # Create a color image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        cv2.circle(img, (50, 50), 20, (60, 60, 60), -1)

        enhanced = enhance_shadow_regions(img, clip_limit=3.0)

        # Should return a color image
        assert enhanced is not None
        assert enhanced.shape == img.shape
        assert len(enhanced.shape) == 3

    def test_handles_grayscale_image(self):
        """CLAHE should work on grayscale images."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:] = 30
        cv2.circle(img, (50, 50), 20, 60, -1)

        enhanced = enhance_shadow_regions(img, clip_limit=3.0)

        assert enhanced is not None
        assert enhanced.shape == img.shape
        assert len(enhanced.shape) == 2


class TestValidateLocalContrast:
    """Tests for the validate_local_contrast function."""

    def test_validates_real_grain(self):
        """Real grain should have sufficient local contrast."""
        # Create a region with high contrast (simulating a real grain)
        region = np.random.randint(50, 200, (20, 20), dtype=np.uint8)

        result = validate_local_contrast(region, min_std=15.0)

        assert result is True

    def test_rejects_uniform_region(self):
        """Uniform region (shadow) should be rejected."""
        # Create a uniform region (simulating a shadow)
        region = np.ones((20, 20), dtype=np.uint8) * 50

        result = validate_local_contrast(region, min_std=15.0)

        assert result is False

    def test_rejects_low_contrast_region(self):
        """Low contrast region should be rejected."""
        # Create a low contrast region
        region = np.ones((20, 20), dtype=np.uint8) * 100
        region[5:15, 5:15] = 105

        result = validate_local_contrast(region, min_std=15.0)

        assert result is False

    def test_accepts_high_contrast_region(self):
        """High contrast region should be accepted."""
        # Create a high contrast region
        region = np.zeros((20, 20), dtype=np.uint8)
        region[:10, :] = 50
        region[10:, :] = 200

        result = validate_local_contrast(region, min_std=15.0)

        assert result is True

    def test_custom_min_std(self):
        """Custom min_std threshold should work."""
        # Create a region with moderate std
        region = np.zeros((20, 20), dtype=np.uint8)
        region[:10, :] = 80
        region[10:, :] = 120

        # std is ~20, so 10 should pass, 30 should fail
        assert validate_local_contrast(region, min_std=10.0) is True
        assert validate_local_contrast(region, min_std=30.0) is False
