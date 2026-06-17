"""Tests for the PDF report generation module."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from core.morphology import GrainMorphology, GrainStatistics, compute_statistics
from core.report import generate_pdf_report, _create_classification_chart


def _make_sample_morphologies() -> list[GrainMorphology]:
    """Create sample GrainMorphology objects for testing."""
    return [
        GrainMorphology(
            area=100.0, perimeter=40.0, circularity=0.785, d_eq=11.28,
            major_axis=12.0, minor_axis=10.0, aspect_ratio=1.2,
            sphericity=0.833, convexity=0.95, feret_max=13.0, feret_min=9.0,
            shape_class="spherical", is_flocculation=False, confidence=0.95,
        ),
        GrainMorphology(
            area=200.0, perimeter=60.0, circularity=0.698, d_eq=15.95,
            major_axis=20.0, minor_axis=8.0, aspect_ratio=2.5,
            sphericity=0.4, convexity=0.88, feret_max=22.0, feret_min=7.0,
            shape_class="flocculation", is_flocculation=True, confidence=0.88,
        ),
        GrainMorphology(
            area=150.0, perimeter=50.0, circularity=0.75, d_eq=13.82,
            major_axis=16.0, minor_axis=11.0, aspect_ratio=1.45,
            sphericity=0.688, convexity=0.92, feret_max=18.0, feret_min=10.0,
            shape_class="rod-like", is_flocculation=False, confidence=0.91,
        ),
    ]


class TestCreateClassificationChart:
    """Tests for the _create_classification_chart function."""

    def test_creates_chart_image(self):
        """Should create a valid chart image file."""
        counts = {"spherical": 1, "rod-like": 1, "discoidal": 0, "flocculation": 1}
        path = _create_classification_chart(counts)

        try:
            assert os.path.exists(path), "Chart image was not created"
            img = cv2.imread(path)
            assert img is not None, "Created file is not a valid image"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_empty_counts(self):
        """Should handle empty counts gracefully."""
        counts = {"spherical": 0, "rod-like": 0, "discoidal": 0, "flocculation": 0}
        path = _create_classification_chart(counts)

        try:
            assert os.path.exists(path)
            img = cv2.imread(path)
            assert img is not None
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestGeneratePDFReport:
    """Tests for the generate_pdf_report function."""

    def test_creates_pdf_file(self):
        """Should create a valid PDF file."""
        morphologies = _make_sample_morphologies()
        stats = compute_statistics(morphologies)

        # Create a dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_path = tmp.name
        cv2.imwrite(image_path, image)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            output_path = tmp.name

        try:
            generate_pdf_report(image_path, morphologies, stats, output_path)

            assert os.path.exists(output_path), "PDF file was not created"
            assert os.path.getsize(output_path) > 0, "PDF file is empty"
        finally:
            if os.path.exists(image_path):
                os.unlink(image_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_pdf_with_annotated_image(self):
        """Should include annotated image in the PDF."""
        morphologies = _make_sample_morphologies()
        stats = compute_statistics(morphologies)

        # Create dummy images
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_path = tmp.name
        cv2.imwrite(image_path, image)

        annotated = np.ones((100, 100, 3), dtype=np.uint8) * 255
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            annotated_path = tmp.name
        cv2.imwrite(annotated_path, annotated)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            output_path = tmp.name

        try:
            generate_pdf_report(
                image_path, morphologies, stats, output_path,
                annotated_image_path=annotated_path
            )

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            for p in [image_path, annotated_path, output_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_pdf_with_empty_morphologies(self):
        """Should handle empty morphologies list."""
        stats = GrainStatistics(count=0)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_path = tmp.name
        cv2.imwrite(image_path, image)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            output_path = tmp.name

        try:
            generate_pdf_report(image_path, [], stats, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(image_path):
                os.unlink(image_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
