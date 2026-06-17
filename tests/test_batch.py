"""Tests for the batch processing module."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from core.batch import process_single_image, process_batch, BatchSummary
from core.detector import FlocculationConfig
from core.preprocessor import PreprocessConfig


def _create_test_image(path: str, size: tuple[int, int] = (200, 200)) -> None:
    """Create a test image with some grains."""
    img = np.ones((*size, 3), dtype=np.uint8) * 255
    # Draw some circles as grains
    cv2.circle(img, (50, 50), 15, (0, 0, 0), -1)
    cv2.circle(img, (120, 80), 20, (0, 0, 0), -1)
    cv2.circle(img, (80, 150), 12, (0, 0, 0), -1)
    cv2.imwrite(path, img)


class TestProcessSingleImage:
    """Tests for process_single_image."""

    def test_processes_valid_image(self):
        """Should process a valid image successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.png")
            _create_test_image(image_path)
            output_dir = os.path.join(tmpdir, "output")

            result = process_single_image(
                image_path, output_dir, config=PreprocessConfig(min_area=50)
            )

            assert result.success is True
            assert result.grain_count > 0
            assert os.path.exists(result.output_csv_path)
            assert os.path.exists(result.output_image_path)
            assert os.path.exists(result.output_pdf_path)

    def test_handles_invalid_image(self):
        """Should handle invalid image gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "nonexistent.png")
            output_dir = os.path.join(tmpdir, "output")

            result = process_single_image(image_path, output_dir)

            assert result.success is False
            assert "Failed to load" in result.error_message

    def test_respects_flocculation_config(self):
        """Should use flocculation config when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.png")
            _create_test_image(image_path)
            output_dir = os.path.join(tmpdir, "output")

            floc_config = FlocculationConfig(min_area=100000)  # Too high to trigger
            result = process_single_image(
                image_path, output_dir,
                config=PreprocessConfig(min_area=50),
                floc_config=floc_config,
            )

            assert result.success is True
            # All grains should be non-flocculation
            for m in result.morphologies:
                assert m.is_flocculation is False


class TestProcessBatch:
    """Tests for process_batch."""

    def test_processes_multiple_images(self):
        """Should process all images in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)

            # Create multiple test images
            for i in range(3):
                image_path = os.path.join(input_dir, f"test_{i}.png")
                _create_test_image(image_path)

            summary = process_batch(
                input_dir, output_dir,
                config=PreprocessConfig(min_area=50),
            )

            assert summary.total_images == 3
            assert summary.successful == 3
            assert summary.failed == 0
            assert summary.total_grains > 0
            assert summary.success_rate == 100.0

    def test_empty_directory(self):
        """Should handle empty directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)

            summary = process_batch(input_dir, output_dir)

            assert summary.total_images == 0
            assert summary.successful == 0
            assert summary.failed == 0
            assert summary.total_grains == 0

    def test_skips_non_image_files(self):
        """Should skip non-image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)

            # Create a text file
            with open(os.path.join(input_dir, "readme.txt"), "w") as f:
                f.write("test")

            # Create one valid image
            image_path = os.path.join(input_dir, "test.png")
            _create_test_image(image_path)

            summary = process_batch(
                input_dir, output_dir,
                config=PreprocessConfig(min_area=50),
            )

            assert summary.total_images == 1
            assert summary.successful == 1


class TestBatchSummary:
    """Tests for BatchSummary."""

    def test_success_rate_calculation(self):
        """Should calculate success rate correctly."""
        summary = BatchSummary(total_images=10, successful=7, failed=3)
        assert summary.success_rate == 70.0

    def test_success_rate_zero_images(self):
        """Should handle zero images."""
        summary = BatchSummary()
        assert summary.success_rate == 0.0
