"""Tests for lightweight false-positive filtering module."""

import numpy as np
import pytest

from core.texture_edge_filter import (
    SimpleValidator,
    ValidationConfig,
)
from core.multiscale_detector import GrainCandidate


class TestSimpleValidator:
    """Tests for SimpleValidator — lightweight false-positive filter."""

    # ------------------------------------------------------------------
    # Initialization tests
    # ------------------------------------------------------------------
    def test_validator_initialization(self):
        """Default config values should be set correctly."""
        validator = SimpleValidator()
        assert validator.config.lens_edge_margin == 0.05
        assert validator.config.lens_edge_circularity == 0.7
        assert validator.config.lens_edge_min_area == 50000
        assert validator.config.noise_max_area == 500
        assert validator.config.min_contrast == 5.0

    def test_validator_with_custom_config(self):
        """Custom config values should override defaults."""
        custom = ValidationConfig(
            lens_edge_margin=0.1,
            lens_edge_circularity=0.8,
            lens_edge_min_area=60000,
            noise_max_area=300,
            min_contrast=10.0,
        )
        validator = SimpleValidator(config=custom)
        assert validator.config.lens_edge_margin == 0.1
        assert validator.config.lens_edge_circularity == 0.8
        assert validator.config.lens_edge_min_area == 60000
        assert validator.config.noise_max_area == 300
        assert validator.config.min_contrast == 10.0

    # ------------------------------------------------------------------
    # Lens edge detection tests
    # ------------------------------------------------------------------
    def test_lens_edge_detection(self):
        """Large circular object near border should be detected as lens edge."""
        h, w = 600, 600
        center = (20, 300)
        radius = 250
        theta = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        contour = np.array([[[int(xi), int(yi)]] for xi, yi in zip(x, y)], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((h, w), dtype=np.uint8),
            area=60000.0,
            perimeter=2 * np.pi * radius,
            circularity=0.85,
            aspect_ratio=1.0,
            major_axis=radius * 2,
            minor_axis=radius * 2,
            convexity=0.95,
            is_flocculation=False,
            border_distance=5.0,
            solidity=0.95,
        )

        validator = SimpleValidator()
        full_image = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        result = validator._is_lens_edge(candidate, full_image)
        assert result is True

    def test_non_lens_edge(self):
        """Small object far from border should NOT be lens edge."""
        contour = np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((300, 300), dtype=np.uint8),
            area=2500.0,
            perimeter=200.0,
            circularity=0.8,
            aspect_ratio=1.0,
            major_axis=50.0,
            minor_axis=50.0,
            convexity=0.9,
            is_flocculation=False,
            border_distance=100.0,
            solidity=0.9,
        )

        validator = SimpleValidator()
        full_image = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        result = validator._is_lens_edge(candidate, full_image)
        assert result is False

    # ------------------------------------------------------------------
    # Noise detection tests
    # ------------------------------------------------------------------
    def test_noise_detection_small_area(self):
        """Small area candidate should be detected as noise."""
        contour = np.array([[[10, 10]], [[15, 10]], [[15, 15]], [[10, 15]]], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((50, 50), dtype=np.uint8),
            area=100.0,
            perimeter=20.0,
            circularity=0.8,
            aspect_ratio=1.0,
            major_axis=5.0,
            minor_axis=5.0,
            convexity=0.9,
            is_flocculation=False,
            border_distance=50.0,
            solidity=0.9,
        )

        validator = SimpleValidator()
        full_image = np.ones((50, 50), dtype=np.uint8) * 128
        result = validator._is_noise(candidate, full_image)
        assert result is True

    def test_non_noise(self):
        """Large area candidate should NOT be noise."""
        contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((300, 300), dtype=np.uint8),
            area=10000.0,
            perimeter=400.0,
            circularity=0.8,
            aspect_ratio=1.0,
            major_axis=100.0,
            minor_axis=100.0,
            convexity=0.9,
            is_flocculation=False,
            border_distance=100.0,
            solidity=0.9,
        )

        validator = SimpleValidator()
        full_image = np.ones((300, 300), dtype=np.uint8) * 128
        result = validator._is_noise(candidate, full_image)
        assert result is False

    # ------------------------------------------------------------------
    # Low contrast detection tests
    # ------------------------------------------------------------------
    def test_low_contrast_rejection(self):
        """Extremely low contrast region should be rejected."""
        contour = np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((200, 200), dtype=np.uint8),
            area=2500.0,
            perimeter=200.0,
            circularity=0.8,
            aspect_ratio=1.0,
            major_axis=50.0,
            minor_axis=50.0,
            convexity=0.9,
            is_flocculation=False,
            border_distance=100.0,
            solidity=0.9,
        )

        validator = SimpleValidator()
        # Uniform image (std = 0)
        full_image = np.ones((200, 200), dtype=np.uint8) * 128
        result = validator._is_low_contrast(candidate, full_image)
        assert result is True

    def test_normal_contrast_acceptance(self):
        """Normal contrast region should be accepted."""
        contour = np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((200, 200), dtype=np.uint8),
            area=2500.0,
            perimeter=200.0,
            circularity=0.8,
            aspect_ratio=1.0,
            major_axis=50.0,
            minor_axis=50.0,
            convexity=0.9,
            is_flocculation=False,
            border_distance=100.0,
            solidity=0.9,
        )

        validator = SimpleValidator()
        # Random image (high std)
        full_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = validator._is_low_contrast(candidate, full_image)
        assert result is False

    # ------------------------------------------------------------------
    # Integration test
    # ------------------------------------------------------------------
    def test_validate_integration(self):
        """Real grain should pass validation."""
        contour = np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((200, 200), dtype=np.uint8),
            area=2500.0,
            perimeter=200.0,
            circularity=0.8,
            aspect_ratio=1.0,
            major_axis=50.0,
            minor_axis=50.0,
            convexity=0.9,
            is_flocculation=False,
            border_distance=100.0,
            solidity=0.9,
        )

        validator = SimpleValidator()
        full_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = validator.validate(candidate, full_image)
        assert result is True
