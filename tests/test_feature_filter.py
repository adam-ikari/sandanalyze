"""Tests for multi-feature false positive filtering."""

import numpy as np
import pytest

from core.multiscale_detector import GrainCandidate
from core.feature_filter import (
    filter_edge_false_positives,
    filter_noise,
    filter_filaments,
)


def _make_candidate(area=1000, circularity=0.8, aspect_ratio=1.2,
                    solidity=0.9, border_distance=50.0):
    """Create a GrainCandidate with specified feature values."""
    return GrainCandidate(
        contour=np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]]),
        mask=np.zeros((10, 10), dtype=np.uint8),
        area=area,
        perimeter=100.0,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        major_axis=10.0,
        minor_axis=8.0,
        convexity=0.9,
        is_flocculation=False,
        border_distance=border_distance,
        solidity=solidity,
    )


class TestFilterEdgeFalsePositives:
    """Tests for filter_edge_false_positives."""

    def test_removes_border_touching_small_components(self):
        """Should remove small components near the image border."""
        border_small = _make_candidate(area=1000, border_distance=5.0)
        center_large = _make_candidate(area=3000, border_distance=50.0)

        candidates = [border_small, center_large]
        filtered = filter_edge_false_positives(candidates, edge_margin=10)

        assert len(filtered) == 1
        assert filtered[0].area == 3000

    def test_keeps_center_components(self):
        """Should keep components that are not near the border."""
        center = _make_candidate(area=1000, border_distance=50.0)
        candidates = [center]
        filtered = filter_edge_false_positives(candidates, edge_margin=10)
        assert len(filtered) == 1
        assert filtered[0].area == 1000


class TestFilterNoise:
    """Tests for filter_noise."""

    def test_removes_small_low_circularity(self):
        """Should remove noise-like components (small area + low circularity)."""
        noise = _make_candidate(area=300, circularity=0.1)
        real_grain = _make_candidate(area=1000, circularity=0.8)

        candidates = [noise, real_grain]
        filtered = filter_noise(candidates, min_area=500)

        assert len(filtered) == 1
        assert filtered[0].area == 1000


class TestFilterFilaments:
    """Tests for filter_filaments."""

    def test_removes_high_aspect_ratio_low_solidity(self):
        """Should remove filament-like shapes (high aspect ratio + low solidity)."""
        filament = _make_candidate(area=1000, aspect_ratio=8.0, solidity=0.3)
        real_grain = _make_candidate(area=1000, aspect_ratio=1.5, solidity=0.9)

        candidates = [filament, real_grain]
        filtered = filter_filaments(candidates, max_aspect_ratio=5.0, min_solidity=0.5)

        assert len(filtered) == 1
        assert filtered[0].aspect_ratio == 1.5
