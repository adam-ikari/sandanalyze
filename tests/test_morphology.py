"""Tests for morphology computation module."""

import math

import cv2
import numpy as np
import pytest

from core.morphology import (
    GrainMorphology,
    compute_morphology,
    compute_statistics,
)


def _find_contour(mask: np.ndarray) -> np.ndarray:
    """Extract the first contour from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) > 0, "No contour found in mask"
    return contours[0]


def test_circle_grain():
    """A circle should have high circularity and low aspect ratio."""
    size = 200
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = 50
    cv2.circle(mask, center, radius, 255, -1)

    contour = _find_contour(mask)
    morph = compute_morphology(contour, mask)

    assert morph.circularity > 0.85, f"Expected circularity > 0.85, got {morph.circularity}"
    assert morph.aspect_ratio < 1.2, f"Expected aspect_ratio < 1.2, got {morph.aspect_ratio}"


def test_elongated_grain():
    """An elongated ellipse should have high aspect ratio and low circularity."""
    size = 200
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    axes = (80, 20)  # major=160, minor=40 -> aspect_ratio ~4
    angle = 0
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

    contour = _find_contour(mask)
    morph = compute_morphology(contour, mask)

    assert morph.aspect_ratio > 2.0, f"Expected aspect_ratio > 2.0, got {morph.aspect_ratio}"
    assert morph.circularity < 0.8, f"Expected circularity < 0.8, got {morph.circularity}"


def test_area_matches_mask():
    """Computed area should match the number of pixels in the mask."""
    size = 200
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = 30
    cv2.circle(mask, center, radius, 255, -1)

    expected_area = cv2.countNonZero(mask)
    contour = _find_contour(mask)
    morph = compute_morphology(contour, mask)

    assert morph.area == pytest.approx(expected_area, rel=0.05)


def test_sphericity_is_inverse_of_aspect_ratio():
    """Sphericity should be the reciprocal of aspect_ratio."""
    size = 200
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.ellipse(mask, center, (60, 30), 45, 0, 360, 255, -1)

    contour = _find_contour(mask)
    morph = compute_morphology(contour, mask)

    assert morph.sphericity == pytest.approx(1.0 / morph.aspect_ratio, rel=1e-6)


def test_convexity_for_smooth_shape():
    """A smooth shape (circle/ellipse) should have convexity > 0.9."""
    size = 200
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(mask, center, 40, 255, -1)

    contour = _find_contour(mask)
    morph = compute_morphology(contour, mask)

    assert morph.convexity > 0.9, f"Expected convexity > 0.9, got {morph.convexity}"


def test_solidity_equals_convexity():
    """Solidity should equal convexity since both are area / hull_area."""
    size = 200
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(mask, center, 40, 255, -1)

    contour = _find_contour(mask)
    morph = compute_morphology(contour, mask)

    assert morph.solidity == pytest.approx(morph.convexity, rel=1e-12)


def test_solidity_for_smooth_shape():
    """A smooth shape (circle/ellipse) should have solidity > 0.9."""
    size = 200
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(mask, center, 40, 255, -1)

    contour = _find_contour(mask)
    morph = compute_morphology(contour, mask)

    assert morph.solidity > 0.9, f"Expected solidity > 0.9, got {morph.solidity}"


def test_statistics_from_multiple_grains():
    """Statistics should correctly aggregate multiple grains."""
    size = 200
    masks = []
    morphologies = []

    for i in range(3):
        mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2 + i * 10, size // 2)
        radius = 20 + i * 5
        cv2.circle(mask, center, radius, 255, -1)
        masks.append(mask)

        contour = _find_contour(mask)
        morph = compute_morphology(contour, mask)
        morphologies.append(morph)

    stats = compute_statistics(morphologies)

    assert stats.count == 3
    assert stats.area_mean > 0
    assert stats.area_std >= 0
    assert stats.circularity_mean > 0.85  # circles are very circular
    assert len(stats.d_eq_values) == 3
    assert len(stats.circularity_values) == 3
    assert len(stats.sphericity_values) == 3


def test_zingg_classification():
    """Zingg classification should categorize grains correctly."""
    size = 200
    morphologies = []

    # Sphere-like (球状): aspect_ratio < 1.5
    mask1 = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask1, (size // 2, size // 2), 40, 255, -1)
    contour1 = _find_contour(mask1)
    morph1 = compute_morphology(contour1, mask1)
    morphologies.append(morph1)

    # Rod-like (棒状): 1.5 <= aspect_ratio < 2.5
    mask2 = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask2, (size // 2, size // 2), (50, 25), 0, 0, 360, 255, -1)
    contour2 = _find_contour(mask2)
    morph2 = compute_morphology(contour2, mask2)
    morphologies.append(morph2)

    # Sheet-like (片状): aspect_ratio >= 2.5
    mask3 = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask3, (size // 2, size // 2), (80, 20), 0, 0, 360, 255, -1)
    contour3 = _find_contour(mask3)
    morph3 = compute_morphology(contour3, mask3)
    morphologies.append(morph3)

    stats = compute_statistics(morphologies)

    assert stats.zingg_counts["spherical"] == 1
    assert stats.zingg_counts["rod-like"] == 1
    assert stats.zingg_counts["discoidal"] == 1
