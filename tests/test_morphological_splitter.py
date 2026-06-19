import cv2
import numpy as np
import pytest

from core.morphological_splitter import split_by_watershed, split_by_concave_points


class TestSplitByWatershed:
    def test_split_touching_circles(self):
        """Two touching circles should be split into two components."""
        img_size = 100
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        # Draw two touching circles
        cv2.circle(mask, (35, 50), 20, 255, -1)
        cv2.circle(mask, (65, 50), 20, 255, -1)

        result = split_by_watershed(mask, min_circularity=0.3)

        # Count connected components (excluding background)
        num_labels, _ = cv2.connectedComponents(result, connectivity=8)
        assert num_labels - 1 == 2, f"Expected 2 components, got {num_labels - 1}"

    def test_single_circle_unchanged(self):
        """A single circle should not be split."""
        img_size = 100
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 25, 255, -1)

        result = split_by_watershed(mask, min_circularity=0.3)

        num_labels, _ = cv2.connectedComponents(result, connectivity=8)
        assert num_labels - 1 == 1, f"Expected 1 component, got {num_labels - 1}"


class TestSplitByConcavePoints:
    def test_split_dumbbell_shape(self):
        """A dumbbell shape should be split at concave points into two components."""
        img_size = 100
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        # Draw a dumbbell: two circles connected by a narrow bridge
        cv2.circle(mask, (30, 50), 20, 255, -1)
        cv2.circle(mask, (70, 50), 20, 255, -1)
        # Draw a narrow rectangle connecting them
        cv2.rectangle(mask, (30, 42), (70, 58), 255, -1)

        result = split_by_concave_points(mask, min_concave_depth=5)

        # Count connected components (excluding background)
        num_labels, _ = cv2.connectedComponents(result, connectivity=8)
        assert num_labels - 1 == 2, f"Expected 2 components, got {num_labels - 1}"

    def test_single_component_unchanged(self):
        """A single circle should not be split."""
        img_size = 100
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 25, 255, -1)

        result = split_by_concave_points(mask, min_concave_depth=5)

        num_labels, _ = cv2.connectedComponents(result, connectivity=8)
        assert num_labels - 1 == 1, f"Expected 1 component, got {num_labels - 1}"
