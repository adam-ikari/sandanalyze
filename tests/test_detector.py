"""Tests for the grain detection module."""

import cv2
import numpy as np

from core.detector import detect_flocculation, is_edge_grain, detect_grains, FlocculationConfig


class TestFlocculationDetection:
    """Tests for flocculation detection."""

    def test_large_irregular_is_flocculation(self):
        """Large irregular contour should be detected as flocculation."""
        # Create a large irregular contour with low circularity and convexity
        # Use a very irregular shape with deep indentations
        mask = np.zeros((400, 400), dtype=np.uint8)

        # Draw a very irregular shape - multiple circles connected
        # This simulates a flocculation (cluster of grains)
        cv2.circle(mask, (100, 100), 40, 255, -1)
        cv2.circle(mask, (180, 120), 35, 255, -1)
        cv2.circle(mask, (140, 200), 45, 255, -1)
        cv2.circle(mask, (220, 180), 30, 255, -1)
        cv2.circle(mask, (120, 280), 38, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        # Use a config that will detect this as flocculation
        config = FlocculationConfig(
            max_circularity=0.5,  # Higher threshold for this test
            max_convexity=0.8,    # Higher threshold
        )
        is_floc = detect_flocculation(contour, config)

        assert is_floc is True

    def test_small_round_is_not_flocculation(self):
        """Small round contour should NOT be flocculation."""
        # Create a small round contour
        contour = np.array([
            [0, 10], [5, 5], [10, 0], [15, 5], [20, 10], [15, 15], [10, 20], [5, 15]
        ], dtype=np.int32).reshape(-1, 1, 2)

        config = FlocculationConfig()
        is_floc = detect_flocculation(contour, config)

        assert is_floc is False


class TestEdgeFiltering:
    """Tests for edge filtering."""

    def test_edge_grain_detected(self):
        """Grain at edge should be detected as edge."""
        contour = np.array([
            [0, 0], [10, 0], [10, 10], [0, 10]
        ], dtype=np.int32).reshape(-1, 1, 2)

        is_edges = is_edge_grain(contour, (100, 100), border_margin=5)
        assert is_edges is True

    def test_center_grain_not_edge(self):
        """Grain at center should NOT be edge."""
        contour = np.array([
            [40, 40], [60, 40], [60, 60], [40, 60]
        ], dtype=np.int32).reshape(-1, 1, 2)

        is_edges = is_edge_grain(contour, (100, 100), border_margin=5)
        assert is_edges is False


class TestDetectGrains:
    """Tests for the detect_grains function."""

    def test_detect_grains_from_mask(self):
        """Should detect grains from a binary mask."""
        # Create a mask with two grains
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (30, 30), 10, 255, -1)
        cv2.circle(mask, (70, 70), 10, 255, -1)

        results = detect_grains(mask, (100, 100), min_area=50)

        assert len(results) == 2
        assert all(not r.is_edge for r in results)

    def test_detect_grains_filters_small(self):
        """Should filter out small grains."""
        # Create a mask with one large and one small grain
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (30, 30), 10, 255, -1)  # Large
        cv2.circle(mask, (70, 70), 3, 255, -1)   # Small

        results = detect_grains(mask, (100, 100), min_area=50)

        # Should only detect the large grain
        assert len(results) == 1
        assert results[0].area > 50
