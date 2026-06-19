"""Tests for multi-scale grain detection data structures."""

import cv2
import numpy as np
import pytest

from core.multiscale_detector import GrainCandidate, MultiScaleConfig
from core.preprocessor import PreprocessConfig


def test_grain_candidate_creation():
    """Verify GrainCandidate can be created with all fields."""
    contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    candidate = GrainCandidate(
        contour=contour,
        mask=mask,
        area=100.0,
        perimeter=40.0,
        circularity=0.85,
        aspect_ratio=1.0,
        major_axis=10.0,
        minor_axis=10.0,
        convexity=0.95,
        is_flocculation=False,
        border_distance=25.0,
        solidity=0.95,
    )
    assert candidate.area == 100.0
    assert candidate.circularity == 0.85
    assert candidate.aspect_ratio == 1.0
    assert candidate.solidity == 0.95
    assert candidate.border_distance == 25.0
    assert np.array_equal(candidate.contour, contour)
    assert np.array_equal(candidate.mask, mask)


def test_multiscale_config_creation():
    """Verify MultiScaleConfig can be created with PreprocessConfig instances."""
    large = PreprocessConfig(blur_kernel=7, adaptive_block_size=61, min_area=1500)
    medium = PreprocessConfig(blur_kernel=5, adaptive_block_size=51, min_area=800)
    small = PreprocessConfig(blur_kernel=3, adaptive_block_size=41, min_area=400)
    config = MultiScaleConfig(
        large_scale=large,
        medium_scale=medium,
        small_scale=small,
    )
    assert config.large_scale.blur_kernel == 7
    assert config.medium_scale.adaptive_block_size == 51
    assert config.small_scale.min_area == 400
    assert config.shadow_enhance is True


def test_preprocess_all_scales():
    """Verify all three scales produce binary uint8 masks."""
    from core.multiscale_detector import preprocess_all_scales

    # Create a synthetic image with some texture
    np.random.seed(42)
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    large = PreprocessConfig(blur_kernel=7, adaptive_block_size=61, min_area=1500)
    medium = PreprocessConfig(blur_kernel=5, adaptive_block_size=51, min_area=800)
    small = PreprocessConfig(blur_kernel=3, adaptive_block_size=41, min_area=400)
    config = MultiScaleConfig(
        large_scale=large,
        medium_scale=medium,
        small_scale=small,
    )

    large_mask, medium_mask, small_mask = preprocess_all_scales(image, config)

    # All returned values should be binary uint8 ndarrays
    assert isinstance(large_mask, np.ndarray)
    assert isinstance(medium_mask, np.ndarray)
    assert isinstance(small_mask, np.ndarray)

    assert large_mask.dtype == np.uint8
    assert medium_mask.dtype == np.uint8
    assert small_mask.dtype == np.uint8

    # Should be binary (only 0 and 255)
    assert set(np.unique(large_mask)).issubset({0, 255})
    assert set(np.unique(medium_mask)).issubset({0, 255})
    assert set(np.unique(small_mask)).issubset({0, 255})

    # Should match image dimensions
    assert large_mask.shape == image.shape[:2]
    assert medium_mask.shape == image.shape[:2]
    assert small_mask.shape == image.shape[:2]


def test_large_scale_catches_bigger_components():
    """Verify large scale detects fewer components due to higher min_area."""
    from core.multiscale_detector import preprocess_all_scales, _count_components

    # Create a synthetic image with multiple blobs of varying sizes
    np.random.seed(42)
    image = np.ones((300, 300, 3), dtype=np.uint8) * 180
    # Add some dark blobs (simulating grains)
    for _ in range(20):
        cx, cy = np.random.randint(20, 280), np.random.randint(20, 280)
        radius = np.random.randint(5, 25)
        cv2.circle(image, (cx, cy), radius, (50, 50, 50), -1)

    large = PreprocessConfig(blur_kernel=7, adaptive_block_size=61, min_area=1500)
    medium = PreprocessConfig(blur_kernel=5, adaptive_block_size=51, min_area=800)
    small = PreprocessConfig(blur_kernel=3, adaptive_block_size=41, min_area=400)
    config = MultiScaleConfig(
        large_scale=large,
        medium_scale=medium,
        small_scale=small,
    )

    large_mask, medium_mask, small_mask = preprocess_all_scales(image, config)

    large_count = _count_components(large_mask)
    medium_count = _count_components(medium_mask)
    small_count = _count_components(small_mask)

    # Large scale should catch fewer or equal components than medium
    assert large_count <= medium_count
    # Medium should catch fewer or equal components than small
    assert medium_count <= small_count


def test_merge_removes_duplicates():
    """Overlapping components should be deduplicated, keeping the one with better circularity."""
    from core.multiscale_detector import merge_multiscale_results

    # Create two overlapping circular components on a canvas
    h, w = 100, 100
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)

    # Two overlapping circles: one at (30, 30) radius 10, another at (35, 35) radius 10
    cv2.circle(mask1, (30, 30), 10, 255, -1)
    cv2.circle(mask2, (35, 35), 10, 255, -1)

    masks = [mask1, mask2]
    merged = merge_multiscale_results(masks, iou_threshold=0.5)

    # Should return a single binary mask
    assert isinstance(merged, np.ndarray)
    assert merged.dtype == np.uint8

    # Count connected components in merged result
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(merged, connectivity=8)
    component_count = num_labels - 1  # exclude background

    # With high overlap (IoU > 0.5), only one component should remain
    assert component_count == 1


def test_merge_keeps_distinct_components():
    """Non-overlapping components should be preserved."""
    from core.multiscale_detector import merge_multiscale_results

    h, w = 100, 100
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)

    # Two non-overlapping circles
    cv2.circle(mask1, (20, 20), 8, 255, -1)
    cv2.circle(mask2, (80, 80), 8, 255, -1)

    masks = [mask1, mask2]
    merged = merge_multiscale_results(masks, iou_threshold=0.5)

    # Should return a single binary mask
    assert isinstance(merged, np.ndarray)
    assert merged.dtype == np.uint8

    # Count connected components in merged result
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(merged, connectivity=8)
    component_count = num_labels - 1  # exclude background

    # Non-overlapping components should both be preserved
    assert component_count == 2
