"""Tests for the traditional grain detection module."""

import cv2
import numpy as np
import pytest

from core.traditional import detect_grains, GrainContour


class TestDetectGrains:
    """Tests for the detect_grains function."""

    def test_finds_grains_in_sample_image(self):
        """Test that detect_grains finds grains in a sample binary mask."""
        # Create a binary mask with two distinct white blobs on black background
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (30, 30), 10, 255, -1)
        cv2.circle(mask, (70, 70), 15, 255, -1)

        grains = detect_grains(mask)

        assert len(grains) == 2
        # Should be sorted by area descending (larger first)
        assert cv2.contourArea(grains[0].contour) >= cv2.contourArea(grains[1].contour)

    def test_grain_has_contour_and_mask(self):
        """Test that each detected grain has a contour and a mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 10, 255, -1)

        grains = detect_grains(mask)

        assert len(grains) == 1
        grain = grains[0]
        assert isinstance(grain, GrainContour)
        assert grain.contour is not None
        assert grain.contour.ndim == 3  # Nx1x2 shape
        assert grain.contour.shape[1] == 1
        assert grain.contour.shape[2] == 2
        assert grain.mask is not None
        assert grain.mask.shape == mask.shape

    def test_min_area_filter_works(self):
        """Test that min_area filters out small contours."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Small circle (area ~314)
        cv2.circle(mask, (50, 50), 10, 255, -1)
        # Large circle (area ~1256)
        cv2.circle(mask, (150, 150), 20, 255, -1)

        grains_strict = detect_grains(mask, min_area=500)
        grains_loose = detect_grains(mask, min_area=50)

        assert len(grains_strict) == 1
        assert len(grains_loose) == 2

    def test_empty_mask_returns_empty_list(self):
        """Test that an empty (all-black) mask returns an empty list."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        grains = detect_grains(mask)
        assert grains == []

    def test_grain_mask_size_matches_image(self):
        """Test that each grain's mask has the same size as the input image."""
        mask = np.zeros((150, 180), dtype=np.uint8)
        cv2.ellipse(mask, (75, 90), (20, 15), 30, 0, 360, 255, -1)

        grains = detect_grains(mask)

        assert len(grains) == 1
        assert grains[0].mask.shape == mask.shape
