"""Tests for core/yolo_detector.py."""

import numpy as np
import pytest

from core.yolo_detector import YOLODetector, refine_with_yolo


class TestYOLODetector:
    """Test suite for YOLODetector."""

    def test_init_without_model_creates_detector(self) -> None:
        """Detector should initialize even without a model file."""
        detector = YOLODetector(model_name="nonexistent_model.pt")
        assert detector is not None
        assert not detector.is_available

    def test_detect_without_model_returns_empty(self) -> None:
        """Detection without model should return empty list with warning."""
        detector = YOLODetector(model_name="nonexistent_model.pt")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.warns(UserWarning, match="not available"):
            grains = detector.detect(image)
        assert grains == []

    def test_is_available_is_bool(self) -> None:
        """is_available should return a boolean."""
        detector = YOLODetector(model_name="nonexistent_model.pt")
        assert isinstance(detector.is_available, bool)


class TestRefineWithYOLO:
    """Test suite for refine_with_yolo hybrid function."""

    def test_refine_without_yolo_returns_traditional(self) -> None:
        """When YOLO unavailable, should return traditional grains."""
        from core.traditional import GrainContour

        # Create a mock traditional grain
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]])
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2 = pytest.importorskip("cv2")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        traditional = [GrainContour(contour=contour, mask=mask)]

        # Create unavailable YOLO detector
        yolo = YOLODetector(model_name="nonexistent_model.pt")
        assert not yolo.is_available

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = refine_with_yolo(image, traditional, yolo)

        assert result is traditional
