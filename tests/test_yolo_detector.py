"""Tests for the YOLOv8-seg grain detector module."""

import numpy as np
import pytest

from core.yolo_detector import YOLODetector
from core.traditional import GrainContour


class TestYOLODetector:
    """Tests for YOLODetector."""

    def test_init_without_model_creates_detector(self):
        """Test that __init__ creates a detector even when the model is absent."""
        detector = YOLODetector(model_name="nonexistent_model.pt")
        assert detector is not None
        assert detector._model_name == "nonexistent_model.pt"

    def test_detect_without_model_returns_empty_list(self):
        """Test that detect returns an empty list when the model is unavailable."""
        detector = YOLODetector(model_name="nonexistent_model.pt")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(image)
        assert result == []

    def test_detect_returns_grain_contour_list_type(self):
        """Test that detect returns a list of GrainContour objects."""
        detector = YOLODetector(model_name="nonexistent_model.pt")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(image)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, GrainContour)

    def test_is_available_is_bool(self):
        """Test that is_available returns a boolean."""
        detector = YOLODetector(model_name="nonexistent_model.pt")
        assert isinstance(detector.is_available, bool)
