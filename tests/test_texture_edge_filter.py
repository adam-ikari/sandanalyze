"""Tests for texture and edge filtering module."""

import numpy as np
import pytest

from core.texture_edge_filter import (
    TextureEdgeValidator,
    ValidationConfig,
    extract_lbp_features,
    extract_glcm_features,
    compute_texture_consistency_score,
    compute_edge_strength,
    compute_edge_direction_consistency,
    compute_edge_closure,
    extract_lbp_features_opencv,
    extract_glcm_features_opencv,
)
from core.multiscale_detector import GrainCandidate


class TestTextureEdgeValidator:
    """Comprehensive tests for TextureEdgeValidator and related functions."""

    # ------------------------------------------------------------------
    # Initialization tests
    # ------------------------------------------------------------------
    def test_validator_initialization(self):
        """Default config values should be set correctly."""
        validator = TextureEdgeValidator()
        assert validator.config.texture_score_threshold == 0.15
        assert validator.config.edge_direction_threshold == 0.6
        assert validator.config.edge_closure_threshold == 0.05
        assert validator.config.lens_edge_margin == 0.05
        assert validator.config.lens_edge_circularity == 0.7
        assert validator.config.lens_edge_min_area == 50000
        assert validator.config.noise_max_texture_score == 0.3
        assert validator.config.noise_max_edge_strength == 30.0
        assert validator.config.noise_max_area == 500
        assert isinstance(validator._has_skimage, bool)

    def test_validator_with_custom_config(self):
        """Custom config values should override defaults."""
        custom = ValidationConfig(
            texture_score_threshold=0.5,
            edge_direction_threshold=0.7,
            edge_closure_threshold=0.4,
            lens_edge_margin=0.1,
            lens_edge_circularity=0.8,
            lens_edge_min_area=60000,
            noise_max_texture_score=0.2,
            noise_max_edge_strength=20.0,
            noise_max_area=300,
        )
        validator = TextureEdgeValidator(config=custom)
        assert validator.config.texture_score_threshold == 0.5
        assert validator.config.edge_direction_threshold == 0.7
        assert validator.config.edge_closure_threshold == 0.4
        assert validator.config.lens_edge_margin == 0.1
        assert validator.config.lens_edge_circularity == 0.8
        assert validator.config.lens_edge_min_area == 60000
        assert validator.config.noise_max_texture_score == 0.2
        assert validator.config.noise_max_edge_strength == 20.0
        assert validator.config.noise_max_area == 300

    def test_check_skimage(self):
        """_check_skimage should return a boolean."""
        result = TextureEdgeValidator._check_skimage()
        assert isinstance(result, bool)

    # ------------------------------------------------------------------
    # Texture feature tests
    # ------------------------------------------------------------------
    def test_extract_lbp_features_shape(self):
        """LBP output should be a (10,) array that sums to ~1.0."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        features = extract_lbp_features(img)
        assert features.shape == (10,)
        assert np.isclose(features.sum(), 1.0, atol=1e-6)

    def test_extract_glcm_features_keys(self):
        """GLCM should return a dict with the expected 5 keys."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        features = extract_glcm_features(img)
        expected_keys = {"contrast", "dissimilarity", "homogeneity", "energy", "correlation"}
        assert set(features.keys()) == expected_keys

    def test_compute_texture_consistency_score_range(self):
        """Texture consistency score should be in [0, 1]."""
        lbp = np.random.rand(10)
        lbp /= lbp.sum()
        glcm = {
            "contrast": 0.5,
            "dissimilarity": 0.3,
            "homogeneity": 0.7,
            "energy": 0.4,
            "correlation": 0.6,
        }
        score = compute_texture_consistency_score(lbp, glcm)
        assert 0.0 <= score <= 1.0

    # ------------------------------------------------------------------
    # Edge feature tests
    # ------------------------------------------------------------------
    def test_compute_edge_strength(self):
        """Edge strength should be non-negative."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        strength = compute_edge_strength(img)
        assert strength >= 0.0

    def test_compute_edge_direction_consistency_range(self):
        """Edge direction consistency should be in [0, 1]."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        consistency = compute_edge_direction_consistency(img)
        assert 0.0 <= consistency <= 1.0

    def test_compute_edge_closure_range(self):
        """Edge closure should be in [0, 1]."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)
        closure = compute_edge_closure(contour, img)
        assert 0.0 <= closure <= 1.0

    # ------------------------------------------------------------------
    # Validator logic tests
    # ------------------------------------------------------------------
    def test_lens_edge_detection(self):
        """Large circular object near border should be detected as lens edge."""
        # Create a large circular contour near the image border
        h, w = 600, 600
        center = (20, 300)
        radius = 250
        theta = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        contour = np.array([[[int(xi), int(yi)]] for xi, yi in zip(x, y)], dtype=np.int32)

        # Area > 50000, circularity > 0.7, near border (border_distance small)
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

        validator = TextureEdgeValidator()
        full_image = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        result = validator._is_lens_edge(candidate, full_image)
        assert result is True

    def test_noise_detection_small_area(self):
        """Small area candidate should be detected as noise."""
        # Create a small contour
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

        validator = TextureEdgeValidator()
        # Use a low-texture, low-edge-strength image for the ROI
        full_image = np.ones((50, 50), dtype=np.uint8) * 128
        result = validator._is_noise(candidate, full_image)
        assert result is True

    def test_composite_score_range(self):
        """Composite score should be in [0, 1]."""
        contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)
        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((50, 50), dtype=np.uint8),
            area=1000.0,
            perimeter=40.0,
            circularity=0.8,
            aspect_ratio=1.0,
            major_axis=10.0,
            minor_axis=10.0,
            convexity=0.9,
            is_flocculation=False,
            border_distance=50.0,
            solidity=0.9,
        )

        validator = TextureEdgeValidator()
        full_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        score = validator._compute_composite_score(candidate, full_image)
        assert 0.0 <= score <= 1.0

    # ------------------------------------------------------------------
    # OpenCV fallback tests
    # ------------------------------------------------------------------
    def test_opencv_fallback_lbp(self):
        """OpenCV fallback LBP should output (10,) array summing to ~1.0."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        features = extract_lbp_features_opencv(img)
        assert features.shape == (10,)
        assert np.isclose(features.sum(), 1.0, atol=1e-6)

    def test_opencv_fallback_glcm(self):
        """OpenCV fallback GLCM should return expected 5 keys."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        features = extract_glcm_features_opencv(img)
        expected_keys = {"contrast", "dissimilarity", "homogeneity", "energy", "correlation"}
        assert set(features.keys()) == expected_keys
