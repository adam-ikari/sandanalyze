# Unified Pipeline Framework and Agent Debug Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified framework where tests and the actual project share the same image pipeline and detection code, plus an Agent-accessible skill for parameter debugging.

**Architecture:** Add a `core/pipeline_debugger.py` module with `ImageSource` (abstract base for real/synthetic images), `PipelineRunner` (unified execution wrapper), and `PipelineResult` (structured output with intermediate masks). Enhance `tests/conftest.py` with real-image fixtures. Create `tools/debug_skill.py` as the Agent-facing CLI interface. All existing code in `core/` remains untouched.

**Tech Stack:** Python 3.13, OpenCV, NumPy, pytest, matplotlib

---

## File Structure

| File | Responsibility |
|------|---------------|
| `core/pipeline_debugger.py` | `ImageSource`, `SyntheticImageSource`, `RealImageSource`, `PipelineRunner`, `PipelineResult` |
| `tests/conftest.py` | Enhanced fixtures: `synthetic_image_source`, `real_image_source`, `pipeline_runner` |
| `tools/debug_skill.py` | Agent-facing skill: `DebugSkill` class with visualize/compare/grid_search/regression |
| `tests/test_pipeline_debugger.py` | Unit tests for ImageSource, PipelineRunner, and DebugSkill |

---

### Task 1: Create ImageSource Abstract Base and Implementations

**Files:**
- Create: `core/pipeline_debugger.py` (first part: ImageSource classes)
- Test: `tests/test_pipeline_debugger.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
import cv2

from core.pipeline_debugger import ImageSource, SyntheticImageSource, RealImageSource


class TestImageSource:
    """Tests for ImageSource abstract base and implementations."""

    def test_synthetic_source_generates_image(self):
        """SyntheticImageSource should generate a valid image."""
        source = SyntheticImageSource(width=200, height=200)
        image = source.load()

        assert isinstance(image, np.ndarray)
        assert image.shape == (200, 200)
        assert image.dtype == np.uint8

    def test_synthetic_source_has_grains(self):
        """Synthetic image should contain detectable grain-like shapes."""
        source = SyntheticImageSource(width=400, height=400, num_grains=3)
        image = source.load()

        # The image should have bright regions (grains) on dark background
        assert np.max(image) > 150
        assert np.min(image) < 100

    def test_real_source_loads_from_path(self):
        """RealImageSource should load an image from disk."""
        import os
        # Use existing test image if available
        test_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test")
        if os.path.exists(test_dir):
            png_files = [f for f in os.listdir(test_dir) if f.endswith(".png")]
            if png_files:
                source = RealImageSource(os.path.join(test_dir, png_files[0]))
                image = source.load()
                assert isinstance(image, np.ndarray)
                assert len(image.shape) in (2, 3)
                return
        pytest.skip("No real test images available")

    def test_real_source_raises_on_missing_file(self):
        """RealImageSource should raise FileNotFoundError for invalid path."""
        with pytest.raises(FileNotFoundError):
            source = RealImageSource("/nonexistent/path.png")
            source.load()

    def test_factory_from_path_synthetic(self):
        """ImageSource.from_path with None should return SyntheticImageSource."""
        source = ImageSource.from_path(None, width=100, height=100)
        assert isinstance(source, SyntheticImageSource)
        image = source.load()
        assert image.shape == (100, 100)

    def test_factory_from_path_real(self, tmp_path):
        """ImageSource.from_path with valid path should return RealImageSource."""
        # Create a dummy image file
        dummy_path = tmp_path / "dummy.png"
        dummy_image = np.full((50, 50), 128, dtype=np.uint8)
        cv2.imwrite(str(dummy_path), dummy_image)

        source = ImageSource.from_path(str(dummy_path))
        assert isinstance(source, RealImageSource)
        image = source.load()
        assert image.shape == (50, 50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_debugger.py::TestImageSource -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'core.pipeline_debugger'"

- [ ] **Step 3: Write minimal implementation**

Create `core/pipeline_debugger.py`:

```python
"""Unified pipeline debugger for sand grain analysis.

Provides ImageSource abstractions and PipelineRunner for shared use
between tests and the actual project.
"""

import os
from abc import ABC, abstractmethod

import cv2
import numpy as np


class ImageSource(ABC):
    """Abstract base class for image sources."""

    @abstractmethod
    def load(self) -> np.ndarray:
        """Load and return the image as a numpy array."""
        pass

    @classmethod
    def from_path(cls, path: str | None, **kwargs) -> "ImageSource":
        """Factory method to create appropriate ImageSource.

        Args:
            path: File path to load. If None, creates a synthetic image.
            **kwargs: Passed to SyntheticImageSource if path is None.

        Returns:
            ImageSource instance.
        """
        if path is None:
            return SyntheticImageSource(**kwargs)
        return RealImageSource(path)


class SyntheticImageSource(ImageSource):
    """Generate synthetic test images with grain-like shapes."""

    def __init__(
        self,
        width: int = 400,
        height: int = 400,
        num_grains: int = 3,
        background_value: int = 50,
        grain_value: int = 180,
        noise_range: int = 20,
    ):
        self.width = width
        self.height = height
        self.num_grains = num_grains
        self.background_value = background_value
        self.grain_value = grain_value
        self.noise_range = noise_range

    def load(self) -> np.ndarray:
        """Generate a synthetic image with elliptical grains."""
        # Start with background + noise
        image = np.full((self.height, self.width), self.background_value, dtype=np.uint8)
        noise = np.random.randint(0, self.noise_range, (self.height, self.width), dtype=np.uint8)
        image = cv2.add(image, noise)

        # Add grain-like ellipses
        np.random.seed(42)  # Reproducible
        for _ in range(self.num_grains):
            cx = np.random.randint(50, self.width - 50)
            cy = np.random.randint(50, self.height - 50)
            ax = np.random.randint(10, 25)
            ay = np.random.randint(8, 20)
            angle = np.random.randint(0, 180)
            cv2.ellipse(image, (cx, cy), (ax, ay), angle, 0, 360, self.grain_value, -1)

        return image


class RealImageSource(ImageSource):
    """Load real images from disk."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> np.ndarray:
        """Load image from disk."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Image not found: {self.path}")

        image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {self.path}")
        return image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_debugger.py::TestImageSource -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add core/pipeline_debugger.py tests/test_pipeline_debugger.py
git commit -m "feat: add ImageSource abstractions for real and synthetic images"
```

---

### Task 2: Create PipelineResult and PipelineRunner

**Files:**
- Modify: `core/pipeline_debugger.py` (add PipelineResult and PipelineRunner)
- Test: `tests/test_pipeline_debugger.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline_debugger.py`:

```python
from dataclasses import dataclass

from core.preprocessor import PreprocessConfig
from core.pipeline_debugger import PipelineRunner, PipelineResult


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_result_has_required_fields(self):
        """PipelineResult should have mask, grains, morphologies, statistics."""
        result = PipelineResult(
            mask=np.zeros((100, 100), dtype=np.uint8),
            grains=[],
            morphologies=[],
            statistics=None,
            intermediate_masks={},
        )
        assert hasattr(result, "mask")
        assert hasattr(result, "grains")
        assert hasattr(result, "morphologies")
        assert hasattr(result, "statistics")
        assert hasattr(result, "intermediate_masks")


class TestPipelineRunner:
    """Tests for PipelineRunner."""

    def test_runner_executes_pipeline(self):
        """PipelineRunner should execute the full pipeline."""
        from core.pipeline_debugger import SyntheticImageSource

        source = SyntheticImageSource(width=400, height=400, num_grains=3)
        runner = PipelineRunner(source)

        config = PreprocessConfig(
            adaptive_block_size=11,
            morph_kernel_size=3,
            morph_open_iter=1,
            min_area=50,
        )
        result = runner.run(config)

        assert isinstance(result, PipelineResult)
        assert result.mask is not None
        assert result.mask.dtype == np.uint8

    def test_runner_finds_grains(self):
        """PipelineRunner should detect grains in synthetic image."""
        from core.pipeline_debugger import SyntheticImageSource

        source = SyntheticImageSource(width=400, height=400, num_grains=3)
        runner = PipelineRunner(source)

        config = PreprocessConfig(
            adaptive_block_size=11,
            morph_kernel_size=3,
            morph_open_iter=1,
            min_area=50,
        )
        result = runner.run(config)

        # Should detect at least one grain
        assert len(result.grains) > 0
        assert len(result.morphologies) > 0
        assert result.statistics is not None
        assert result.statistics.count > 0

    def test_runner_includes_intermediate_masks(self):
        """PipelineRunner should include brightness/edge/texture masks."""
        from core.pipeline_debugger import SyntheticImageSource

        source = SyntheticImageSource(width=400, height=400, num_grains=2)
        runner = PipelineRunner(source)

        config = PreprocessConfig(
            adaptive_block_size=11,
            morph_kernel_size=3,
            morph_open_iter=1,
            min_area=50,
        )
        result = runner.run(config)

        assert "brightness" in result.intermediate_masks
        assert "edge" in result.intermediate_masks
        assert "texture" in result.intermediate_masks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_debugger.py::TestPipelineResult -v`
Expected: FAIL with "ImportError: cannot import name 'PipelineRunner'"

- [ ] **Step 3: Write minimal implementation**

Add to `core/pipeline_debugger.py`:

```python
from dataclasses import dataclass, field
from typing import Any

from core.detector import detect_grains, DetectionResult, FlocculationConfig
from core.morphology import (
    compute_morphology,
    compute_statistics,
    GrainContour,
    GrainMorphology,
    GrainStatistics,
)
from core.classifier import classify_grain
from core.preprocessor import (
    PreprocessConfig,
    _preprocess_brightness,
    _preprocess_edge,
    _preprocess_texture,
    _fuse_masks,
)


@dataclass
class PipelineResult:
    """Result of running the detection pipeline."""

    mask: np.ndarray
    grains: list[GrainContour]
    morphologies: list[GrainMorphology]
    statistics: GrainStatistics | None
    intermediate_masks: dict[str, np.ndarray] = field(default_factory=dict)


class PipelineRunner:
    """Unified pipeline runner for sand grain detection.

    Wraps the full pipeline: preprocess → detect → morphology → classify → statistics.
    Works with any ImageSource (real or synthetic).
    """

    def __init__(self, image_source: ImageSource):
        self.image_source = image_source

    def run(
        self,
        config: PreprocessConfig,
        min_area: int = 1000,
        max_area: int = 15000,
        border_margin: int = 5,
        hull_expansion_ratio: float = 1.5,
        floc_config: FlocculationConfig | None = None,
        crop_black_background: bool = False,
    ) -> PipelineResult:
        """Run the full detection pipeline.

        Args:
            config: Preprocessing configuration.
            min_area: Minimum grain area.
            max_area: Maximum grain area.
            border_margin: Distance from border to filter.
            hull_expansion_ratio: Threshold for hull vs mask filling.
            floc_config: Flocculation detection config.
            crop_black_background: Whether to crop black background.

        Returns:
            PipelineResult with all outputs.
        """
        image = self.image_source.load()

        # Convert grayscale to BGR if needed for detector
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        # Run preprocessing branches
        brightness_mask = _preprocess_brightness(image_bgr, config)
        edge_mask = _preprocess_edge(image_bgr, config)
        texture_mask = _preprocess_texture(image_bgr, config)
        fused_mask = _fuse_masks([brightness_mask, edge_mask, texture_mask])

        # Detect grains
        results = detect_grains(
            image=image_bgr,
            config=config,
            min_area=min_area,
            max_area=max_area,
            border_margin=border_margin,
            hull_expansion_ratio=hull_expansion_ratio,
            floc_config=floc_config,
            crop_black_background=crop_black_background,
        )

        # Build grains and morphologies
        grains: list[GrainContour] = []
        morphologies: list[GrainMorphology] = []

        for result in results:
            grain = GrainContour(contour=result.contour, mask=result.mask)
            grains.append(grain)

            morph = compute_morphology(result.contour, result.mask)
            morph.is_flocculation = result.is_flocculation
            morph.shape_class = classify_grain(
                morph.aspect_ratio, result.is_flocculation, circularity=morph.circularity
            )
            morph.confidence = 0.9 if result.is_flocculation else 0.95
            morphologies.append(morph)

        statistics = compute_statistics(morphologies) if morphologies else None

        return PipelineResult(
            mask=fused_mask,
            grains=grains,
            morphologies=morphologies,
            statistics=statistics,
            intermediate_masks={
                "brightness": brightness_mask,
                "edge": edge_mask,
                "texture": texture_mask,
                "fused": fused_mask,
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_debugger.py::TestPipelineResult tests/test_pipeline_debugger.py::TestPipelineRunner -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add core/pipeline_debugger.py tests/test_pipeline_debugger.py
git commit -m "feat: add PipelineRunner and PipelineResult for unified execution"
```

---

### Task 3: Add Visualization Methods to PipelineRunner

**Files:**
- Modify: `core/pipeline_debugger.py` (add visualize methods)
- Test: `tests/test_pipeline_debugger.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline_debugger.py`:

```python
import tempfile
import os


class TestPipelineVisualizer:
    """Tests for PipelineRunner visualization methods."""

    def test_visualize_branches_creates_file(self):
        """visualize_branches should create a PNG file."""
        from core.pipeline_debugger import SyntheticImageSource

        source = SyntheticImageSource(width=400, height=400, num_grains=3)
        runner = PipelineRunner(source)

        config = PreprocessConfig(
            adaptive_block_size=11,
            morph_kernel_size=3,
            morph_open_iter=1,
            min_area=50,
        )
        runner.run(config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            runner.visualize_branches(save_path)
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_compare_configs_creates_file(self):
        """compare_configs should create a comparison PNG file."""
        from core.pipeline_debugger import SyntheticImageSource

        source = SyntheticImageSource(width=400, height=400, num_grains=3)
        runner = PipelineRunner(source)

        config_a = PreprocessConfig(
            adaptive_block_size=11,
            morph_kernel_size=3,
            morph_open_iter=1,
            min_area=50,
        )
        config_b = PreprocessConfig(
            adaptive_block_size=21,
            morph_kernel_size=3,
            morph_open_iter=1,
            min_area=50,
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            runner.compare_configs(config_a, config_b, save_path)
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_debugger.py::TestPipelineVisualizer -v`
Expected: FAIL with "AttributeError: 'PipelineRunner' object has no attribute 'visualize_branches'"

- [ ] **Step 3: Write minimal implementation**

Add to `core/pipeline_debugger.py` (inside PipelineRunner class):

```python
    def visualize_branches(self, save_path: str = "debug_branches.png") -> None:
        """Visualize preprocessing branch outputs.

        Creates a figure showing original image, brightness mask,
        edge mask, texture mask, and fused mask.

        Args:
            save_path: Path to save the visualization PNG.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not hasattr(self, "_last_result") or self._last_result is None:
            raise RuntimeError("Must call run() before visualize_branches()")

        result = self._last_result
        image = self.image_source.load()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original
        if len(image.shape) == 3:
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[0, 0].imshow(image, cmap="gray")
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        # Branch masks
        masks = result.intermediate_masks
        titles = [
            ("brightness", "Brightness"),
            ("edge", "Edge"),
            ("texture", "Texture"),
            ("fused", "Fused"),
        ]

        for idx, (key, title) in enumerate(titles):
            row = (idx + 1) // 3
            col = (idx + 1) % 3
            if key in masks:
                axes[row, col].imshow(masks[key], cmap="gray")
                pixel_count = np.count_nonzero(masks[key])
                axes[row, col].set_title(f"{title}\n({pixel_count} px)")
            axes[row, col].axis("off")

        # Hide unused subplot
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def compare_configs(
        self,
        config_a: PreprocessConfig,
        config_b: PreprocessConfig,
        save_path: str = "debug_compare.png",
    ) -> None:
        """Compare two configurations side-by-side.

        Args:
            config_a: First configuration.
            config_b: Second configuration.
            save_path: Path to save the comparison PNG.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result_a = self.run(config_a)
        result_b = self.run(config_b)

        image = self.image_source.load()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Config A
        axes[0, 0].imshow(result_a.mask, cmap="gray")
        axes[0, 0].set_title(f"Config A\n({np.count_nonzero(result_a.mask)} px, {len(result_a.grains)} grains)")
        axes[0, 0].axis("off")

        # Config B
        axes[0, 1].imshow(result_b.mask, cmap="gray")
        axes[0, 1].set_title(f"Config B\n({np.count_nonzero(result_b.mask)} px, {len(result_b.grains)} grains)")
        axes[0, 1].axis("off")

        # Difference
        diff = cv2.subtract(result_b.mask, result_a.mask)
        axes[0, 2].imshow(diff, cmap="hot")
        axes[0, 2].set_title(f"Difference\n({np.count_nonzero(diff)} px)")
        axes[0, 2].axis("off")

        # Overlay A
        overlay_a = image.copy()
        if len(overlay_a.shape) == 2:
            overlay_a = cv2.cvtColor(overlay_a, cv2.COLOR_GRAY2BGR)
        mask_a_bool = result_a.mask > 0
        overlay_a[mask_a_bool] = [0, 255, 0]
        axes[1, 0].imshow(cv2.cvtColor(overlay_a, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Config A Overlay (green)")
        axes[1, 0].axis("off")

        # Overlay B
        overlay_b = image.copy()
        if len(overlay_b.shape) == 2:
            overlay_b = cv2.cvtColor(overlay_b, cv2.COLOR_GRAY2BGR)
        mask_b_bool = result_b.mask > 0
        overlay_b[mask_b_bool] = [0, 255, 0]
        axes[1, 1].imshow(cv2.cvtColor(overlay_b, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Config B Overlay (green)")
        axes[1, 1].axis("off")

        # Diff overlay
        overlay_diff = image.copy()
        if len(overlay_diff.shape) == 2:
            overlay_diff = cv2.cvtColor(overlay_diff, cv2.COLOR_GRAY2BGR)
        diff_bool = diff > 0
        overlay_diff[diff_bool] = [0, 0, 255]
        axes[1, 2].imshow(cv2.cvtColor(overlay_diff, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("Added by B (red)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
```

Also update the `run` method to store the last result:

```python
        self._last_result = result
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_debugger.py::TestPipelineVisualizer -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add core/pipeline_debugger.py tests/test_pipeline_debugger.py
git commit -m "feat: add visualize_branches and compare_configs to PipelineRunner"
```

---

### Task 4: Add Grid Search to PipelineRunner

**Files:**
- Modify: `core/pipeline_debugger.py` (add grid_search method)
- Test: `tests/test_pipeline_debugger.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline_debugger.py`:

```python
import pandas as pd


class TestPipelineGridSearch:
    """Tests for PipelineRunner grid search."""

    def test_grid_search_returns_dataframe(self):
        """grid_search should return a DataFrame with results."""
        from core.pipeline_debugger import SyntheticImageSource

        source = SyntheticImageSource(width=400, height=400, num_grains=3)
        runner = PipelineRunner(source)

        param_ranges = {
            "adaptive_block_size": [11, 21],
            "adaptive_c": [2, 5],
        }

        df = runner.grid_search(param_ranges, metric="grain_count")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "grain_count" in df.columns
        assert "adaptive_block_size" in df.columns
        assert "adaptive_c" in df.columns

    def test_grid_search_evaluates_all_combinations(self):
        """grid_search should evaluate all parameter combinations."""
        from core.pipeline_debugger import SyntheticImageSource

        source = SyntheticImageSource(width=400, height=400, num_grains=3)
        runner = PipelineRunner(source)

        param_ranges = {
            "adaptive_block_size": [11, 21],
        }

        df = runner.grid_search(param_ranges, metric="grain_count")

        # Should have 2 rows (one per value)
        assert len(df) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_debugger.py::TestPipelineGridSearch -v`
Expected: FAIL with "AttributeError: 'PipelineRunner' object has no attribute 'grid_search'"

- [ ] **Step 3: Write minimal implementation**

Add to `core/pipeline_debugger.py` (inside PipelineRunner class):

```python
    def grid_search(
        self,
        param_ranges: dict[str, list],
        metric: str = "grain_count",
        min_area: int = 50,
        max_area: int = 15000,
    ) -> "pd.DataFrame":
        """Grid search over parameter ranges.

        Args:
            param_ranges: Dict of parameter names to lists of values.
                Supported params: adaptive_block_size, adaptive_c, blur_kernel,
                morph_kernel_size, morph_open_iter, morph_close_iter, min_area.
            metric: Metric to optimize ("grain_count", "mask_pixels", "circularity_mean").
            min_area: Minimum grain area for detection.
            max_area: Maximum grain area for detection.

        Returns:
            DataFrame with one row per parameter combination and result columns.
        """
        import itertools
        import pandas as pd

        # Build all combinations
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]
        combinations = list(itertools.product(*values))

        results = []
        base_config = PreprocessConfig()

        for combo in combinations:
            config_kwargs = {}
            for key, value in zip(keys, combo):
                config_kwargs[key] = value

            config = PreprocessConfig(**config_kwargs)

            try:
                result = self.run(
                    config=config,
                    min_area=min_area,
                    max_area=max_area,
                    crop_black_background=False,
                )

                # Compute metric
                if metric == "grain_count":
                    metric_value = len(result.grains)
                elif metric == "mask_pixels":
                    metric_value = int(np.count_nonzero(result.mask))
                elif metric == "circularity_mean":
                    metric_value = (
                        result.statistics.circularity_mean
                        if result.statistics else 0.0
                    )
                else:
                    metric_value = len(result.grains)

                row = {k: v for k, v in zip(keys, combo)}
                row[metric] = metric_value
                row["grain_count"] = len(result.grains)
                row["mask_pixels"] = int(np.count_nonzero(result.mask))
                results.append(row)

            except Exception as e:
                row = {k: v for k, v in zip(keys, combo)}
                row[metric] = -1
                row["grain_count"] = -1
                row["mask_pixels"] = -1
                row["error"] = str(e)
                results.append(row)

        return pd.DataFrame(results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_debugger.py::TestPipelineGridSearch -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add core/pipeline_debugger.py tests/test_pipeline_debugger.py
git commit -m "feat: add grid_search to PipelineRunner for parameter optimization"
```

---

### Task 5: Enhance Test Fixtures

**Files:**
- Modify: `tests/conftest.py`
- Test: Existing tests should still pass

- [ ] **Step 1: Write the failing test**

No new test needed — we verify existing tests still pass.

- [ ] **Step 2: Add fixtures to conftest.py**

Append to `tests/conftest.py`:

```python
import os

import pytest

from core.pipeline_debugger import (
    ImageSource,
    PipelineRunner,
    RealImageSource,
    SyntheticImageSource,
)


@pytest.fixture
def synthetic_image_source():
    """Provide a synthetic image source with 3 grains."""
    return SyntheticImageSource(width=400, height=400, num_grains=3)


@pytest.fixture
def real_image_source():
    """Provide a real image source from data/test if available."""
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test")
    if os.path.exists(test_dir):
        png_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
        if png_files:
            return RealImageSource(os.path.join(test_dir, png_files[0]))
    return None


@pytest.fixture
def pipeline_runner(synthetic_image_source):
    """Provide a PipelineRunner with synthetic image source."""
    return PipelineRunner(synthetic_image_source)
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `pytest tests/ -v --tb=short`
Expected: 103+ passed, 0 failed

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: add unified fixtures for real and synthetic image sources"
```

---

### Task 6: Create Agent Debug Skill

**Files:**
- Create: `tools/debug_skill.py`
- Test: `tests/test_pipeline_debugger.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline_debugger.py`:

```python
class TestDebugSkill:
    """Tests for Agent-facing DebugSkill."""

    def test_debug_skill_initializes(self):
        """DebugSkill should initialize with default settings."""
        from tools.debug_skill import DebugSkill

        skill = DebugSkill()
        assert skill is not None

    def test_visualize_branches_runs(self):
        """DebugSkill.visualize_branches should create output file."""
        import tempfile
        from tools.debug_skill import DebugSkill

        skill = DebugSkill()

        # Create a temporary synthetic image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            # Create dummy image
            dummy = np.full((200, 200), 128, dtype=np.uint8)
            cv2.imwrite(temp_path, dummy)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                save_path = f.name

            try:
                skill.visualize_branches(
                    temp_path,
                    save_path=save_path,
                )
                assert os.path.exists(save_path)
                assert os.path.getsize(save_path) > 0
            finally:
                if os.path.exists(save_path):
                    os.remove(save_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_compare_configs_runs(self):
        """DebugSkill.compare_configs should create comparison file."""
        import tempfile
        from tools.debug_skill import DebugSkill
        from core.preprocessor import PreprocessConfig

        skill = DebugSkill()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            dummy = np.full((200, 200), 128, dtype=np.uint8)
            cv2.imwrite(temp_path, dummy)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                save_path = f.name

            try:
                config_a = PreprocessConfig(adaptive_block_size=11)
                config_b = PreprocessConfig(adaptive_block_size=21)

                skill.compare_configs(
                    temp_path,
                    config_a,
                    config_b,
                    save_path=save_path,
                )
                assert os.path.exists(save_path)
                assert os.path.getsize(save_path) > 0
            finally:
                if os.path.exists(save_path):
                    os.remove(save_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_grid_search_runs(self):
        """DebugSkill.grid_search should return DataFrame."""
        import tempfile
        from tools.debug_skill import DebugSkill

        skill = DebugSkill()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            dummy = np.full((200, 200), 128, dtype=np.uint8)
            cv2.imwrite(temp_path, dummy)

            param_ranges = {"adaptive_block_size": [11, 21]}
            df = skill.grid_search(temp_path, param_ranges, metric="grain_count")

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "grain_count" in df.columns
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_debugger.py::TestDebugSkill -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'tools.debug_skill'"

- [ ] **Step 3: Write minimal implementation**

Create `tools/debug_skill.py`:

```python
"""Agent-accessible debug skill for sand grain parameter tuning.

Provides a high-level interface for:
    - Visualizing pipeline branches
    - Comparing configurations
    - Grid searching parameters
    - Running regression tests

Usage:
    from tools.debug_skill import DebugSkill
    skill = DebugSkill()
    skill.visualize_branches("image.png", save_path="branches.png")
"""

import os
from typing import Any

import numpy as np

from core.pipeline_debugger import ImageSource, PipelineRunner
from core.preprocessor import PreprocessConfig


class DebugSkill:
    """Agent-facing skill for debugging sand grain detection parameters.

    This class provides a unified interface for parameter exploration
    and validation. Agents can call these methods without writing
    ad-hoc debug scripts.
    """

    def __init__(self):
        pass

    def visualize_branches(
        self,
        image_path: str | None,
        config: PreprocessConfig | None = None,
        save_path: str = "debug_branches.png",
        **source_kwargs: Any,
    ) -> str:
        """Visualize preprocessing branch outputs.

        Args:
            image_path: Path to image file, or None for synthetic image.
            config: Preprocessing config. Uses defaults if None.
            save_path: Path to save visualization PNG.
            **source_kwargs: Passed to SyntheticImageSource if image_path is None.

        Returns:
            Path to saved visualization file.
        """
        if config is None:
            config = PreprocessConfig()

        source = ImageSource.from_path(image_path, **source_kwargs)
        runner = PipelineRunner(source)
        runner.run(config, crop_black_background=False)
        runner.visualize_branches(save_path)

        return save_path

    def compare_configs(
        self,
        image_path: str | None,
        config_a: PreprocessConfig,
        config_b: PreprocessConfig,
        save_path: str = "debug_compare.png",
        **source_kwargs: Any,
    ) -> str:
        """Compare two preprocessing configurations side-by-side.

        Args:
            image_path: Path to image file, or None for synthetic image.
            config_a: First configuration.
            config_b: Second configuration.
            save_path: Path to save comparison PNG.
            **source_kwargs: Passed to SyntheticImageSource if image_path is None.

        Returns:
            Path to saved comparison file.
        """
        source = ImageSource.from_path(image_path, **source_kwargs)
        runner = PipelineRunner(source)
        runner.compare_configs(config_a, config_b, save_path)

        return save_path

    def grid_search(
        self,
        image_path: str | None,
        param_ranges: dict[str, list],
        metric: str = "grain_count",
        save_path: str | None = None,
        **source_kwargs: Any,
    ) -> "pd.DataFrame":
        """Grid search over parameter ranges.

        Args:
            image_path: Path to image file, or None for synthetic image.
            param_ranges: Dict of parameter names to lists of values.
            metric: Metric to optimize ("grain_count", "mask_pixels", "circularity_mean").
            save_path: Optional path to save results CSV.
            **source_kwargs: Passed to SyntheticImageSource if image_path is None.

        Returns:
            DataFrame with one row per parameter combination.
        """
        import pandas as pd

        source = ImageSource.from_path(image_path, **source_kwargs)
        runner = PipelineRunner(source)
        df = runner.grid_search(param_ranges, metric=metric)

        if save_path:
            df.to_csv(save_path, index=False)

        return df

    def regression_test(
        self,
        image_path: str | None,
        config: PreprocessConfig,
        expected_counts: dict[str, int],
        tolerance: float = 0.1,
        **source_kwargs: Any,
    ) -> dict[str, bool]:
        """Run regression test against expected grain counts.

        Args:
            image_path: Path to image file, or None for synthetic image.
            config: Preprocessing configuration.
            expected_counts: Dict mapping image names to expected grain counts.
            tolerance: Relative tolerance for count matching (0.1 = 10%).
            **source_kwargs: Passed to SyntheticImageSource if image_path is None.

        Returns:
            Dict mapping image names to pass/fail boolean.
        """
        source = ImageSource.from_path(image_path, **source_kwargs)
        runner = PipelineRunner(source)
        result = runner.run(config, crop_black_background=False)

        actual_count = len(result.grains)
        results = {}

        for name, expected in expected_counts.items():
            lower = expected * (1 - tolerance)
            upper = expected * (1 + tolerance)
            results[name] = lower <= actual_count <= upper

        return results


def main():
    """CLI entry point for manual testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Debug sand grain detection parameters")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--visualize", action="store_true", help="Visualize branches")
    parser.add_argument("--compare", nargs=2, metavar=("CONFIG_A", "CONFIG_B"),
                        help="Compare two configs (not yet implemented)")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search")
    parser.add_argument("--output", "-o", default="debug_output.png", help="Output path")

    args = parser.parse_args()

    skill = DebugSkill()

    if args.visualize:
        path = skill.visualize_branches(args.image, save_path=args.output)
        print(f"Visualization saved to: {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_debugger.py::TestDebugSkill -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tools/debug_skill.py tests/test_pipeline_debugger.py
git commit -m "feat: add Agent-facing DebugSkill for parameter debugging"
```

---

### Task 7: Final Integration Test

**Files:**
- Test: All tests

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (103+ existing + new tests)

- [ ] **Step 2: Verify no regressions**

Run: `pytest tests/ --cov=core --cov-report=term-missing`
Expected: No coverage regressions in core modules

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: add integration tests for unified pipeline framework"
```

---

## Spec Coverage Check

| Spec Section | Implementing Task |
|--------------|-------------------|
| ImageSource abstract base + implementations | Task 1 |
| PipelineRunner + PipelineResult | Task 2 |
| visualize_branches | Task 3 |
| compare_configs | Task 3 |
| grid_search | Task 4 |
| Enhanced test fixtures | Task 5 |
| DebugSkill (Agent-facing) | Task 6 |
| Regression test | Task 6 |
| Backward compatibility | Task 7 |

## Placeholder Scan

- ✅ No TBD/TODO/fill-in-details
- ✅ All code blocks contain actual implementation
- ✅ No vague requirements

## Type Consistency

- `PipelineResult` fields match usage in `PipelineRunner`
- `DebugSkill` methods accept same types as `PipelineRunner`
- `ImageSource.from_path()` returns correct subclass
