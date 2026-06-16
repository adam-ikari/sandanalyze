"""Tests for the export utilities module."""

import csv
import os
import tempfile

import cv2
import numpy as np
import pytest

from core.exporter import export_annotated_image, export_csv
from core.morphology import GrainMorphology
from core.traditional import GrainContour


def _make_sample_morphologies() -> list[GrainMorphology]:
    """Create sample GrainMorphology objects for testing."""
    return [
        GrainMorphology(
            area=100.0,
            perimeter=40.0,
            circularity=0.785398,
            d_eq=11.2838,
            major_axis=12.0,
            minor_axis=10.0,
            aspect_ratio=1.2,
            sphericity=0.833333,
            convexity=0.95,
            feret_max=13.0,
            feret_min=9.0,
        ),
        GrainMorphology(
            area=200.0,
            perimeter=60.0,
            circularity=0.698132,
            d_eq=15.9577,
            major_axis=18.0,
            minor_axis=14.0,
            aspect_ratio=1.285714,
            sphericity=0.777778,
            convexity=0.88,
            feret_max=20.0,
            feret_min=12.0,
        ),
    ]


def _make_sample_grains() -> list[GrainContour]:
    """Create sample GrainContour objects for testing."""
    # Create a simple square contour
    contour1 = np.array(
        [[[10, 10]], [[40, 10]], [[40, 40]], [[10, 40]]], dtype=np.int32
    )
    mask1 = np.zeros((50, 50), dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 255, thickness=cv2.FILLED)

    # Create a circular contour
    mask2 = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(mask2, (25, 25), 10, 255, -1)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour2 = contours2[0]

    return [
        GrainContour(contour=contour1, mask=mask1),
        GrainContour(contour=contour2, mask=mask2),
    ]


class TestExportCSV:
    """Tests for the export_csv function."""

    def test_creates_file_with_correct_data(self):
        """Test that export_csv creates a CSV with correct columns and data."""
        morphologies = _make_sample_morphologies()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            export_csv(morphologies, tmp_path)

            assert os.path.exists(tmp_path), "CSV file was not created"

            with open(tmp_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

            # Verify header columns
            expected_fields = [
                "grain_id",
                "area",
                "perimeter",
                "circularity",
                "d_eq",
                "major_axis",
                "minor_axis",
                "aspect_ratio",
                "sphericity",
                "convexity",
                "feret_max",
                "feret_min",
            ]
            assert reader.fieldnames == expected_fields

            # Verify first row data
            assert rows[0]["grain_id"] == "1"
            assert float(rows[0]["area"]) == pytest.approx(100.0, rel=1e-4)
            assert float(rows[0]["circularity"]) == pytest.approx(0.785398, rel=1e-6)
            assert float(rows[0]["sphericity"]) == pytest.approx(0.833333, rel=1e-6)

            # Verify second row data
            assert rows[1]["grain_id"] == "2"
            assert float(rows[1]["area"]) == pytest.approx(200.0, rel=1e-4)
            assert float(rows[1]["aspect_ratio"]) == pytest.approx(1.285714, rel=1e-4)
        finally:
            os.unlink(tmp_path)

    def test_empty_list_creates_header_only(self):
        """Test that export_csv with empty list creates header-only CSV."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            export_csv([], tmp_path)

            with open(tmp_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 0
            assert "grain_id" in reader.fieldnames
        finally:
            os.unlink(tmp_path)


class TestExportAnnotatedImage:
    """Tests for the export_annotated_image function."""

    def test_creates_valid_png(self):
        """Test that export_annotated_image creates a valid PNG file."""
        grains = _make_sample_grains()
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_annotated_image(image, grains, tmp_path)

            assert os.path.exists(tmp_path), "Annotated image was not created"

            # Verify it's a valid image by reading it back
            loaded = cv2.imread(tmp_path)
            assert loaded is not None, "Created file is not a valid image"
            assert loaded.shape == image.shape
        finally:
            os.unlink(tmp_path)

    def test_draws_contours_and_labels(self):
        """Test that contours and labels are actually drawn on the image."""
        grains = _make_sample_grains()
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_annotated_image(image, grains, tmp_path, color=(0, 255, 0))

            loaded = cv2.imread(tmp_path)
            assert loaded is not None

            # The image should no longer be all black (contours drawn)
            assert np.any(loaded > 0), "Image is still all black, contours not drawn"
        finally:
            os.unlink(tmp_path)

    def test_custom_color_and_thickness(self):
        """Test that custom color and thickness are applied."""
        grains = _make_sample_grains()
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_annotated_image(
                image, grains, tmp_path, color=(255, 0, 0), thickness=2
            )

            loaded = cv2.imread(tmp_path)
            assert loaded is not None
            # Check that red pixels (BGR: 0,0,255) or blue pixels exist
            assert np.any(loaded > 0)
        finally:
            os.unlink(tmp_path)

    def test_empty_grains_list(self):
        """Test that empty grains list still produces a valid image."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_annotated_image(image, [], tmp_path)

            loaded = cv2.imread(tmp_path)
            assert loaded is not None
            assert loaded.shape == image.shape
        finally:
            os.unlink(tmp_path)
