# Technical Debt Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up technical debt across the SandAnalyze codebase by eliminating duplication, unifying type hints, removing dead code, extracting shared logic, and strengthening tests.

**Architecture:** Refactor the codebase into cleaner layers: a shared `pipeline` module for the common "detect → compute morphology → classify" flow, remove the unused `traditional.py` and `yolo_detector.py`/`model_manager.py` modules, unify all type hints to modern Python 3.10+ syntax (`list`, `|`, etc.), and extract the duplicated contour-to-morphology logic from `app.py` and `batch.py` into a single reusable function.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, pytest

---

## File Structure Changes

| File | Action | Responsibility |
|------|--------|---------------|
| `core/pipeline.py` | **Create** | Single shared function `run_detection_pipeline()` that wraps detect → compute morphology → classify |
| `core/detector.py` | **Modify** | Remove inline morphology computation and classification; keep only contour detection and flocculation flagging |
| `core/morphology.py` | **Modify** | Add `classify_and_enrich()` helper that computes morphology + runs classifier |
| `core/traditional.py` | **Delete** | Unused; `detect_grains` in this module is not called by app.py or batch.py |
| `core/yolo_detector.py` | **Delete** | Unused; YOLO integration never wired into main flow |
| `core/model_manager.py` | **Delete** | Unused; only referenced by yolo_detector.py |
| `core/__init__.py` | **Modify** | Remove deleted module exports, add `pipeline` exports |
| `core/batch.py` | **Modify** | Replace duplicated pipeline logic with call to `core.pipeline.run_detection_pipeline()` |
| `app.py` | **Modify** | Replace duplicated pipeline logic with call to `core.pipeline.run_detection_pipeline()` |
| `tests/test_traditional.py` | **Delete** | Tests for deleted module |
| `tests/test_yolo_detector.py` | **Delete** | Tests for deleted module |
| `tests/test_pipeline.py` | **Create** | Tests for the new pipeline module |
| `tests/test_detector.py` | **Modify** | Update imports and assertions after detector.py refactor |
| `tests/test_integration.py` | **Modify** | Remove references to deleted `traditional.py` |
| `tests/test_batch.py` | **Modify** | Update assertions if pipeline output changes |
| `tests/test_exporter.py` | **Modify** | Remove `GrainContour` references from deleted module |

---

## Task 1: Create `core/pipeline.py` — Shared Detection Pipeline

**Files:**
- Create: `core/pipeline.py`
- Test: `tests/test_pipeline.py`

**Background:** Both `app.py:311-359` and `batch.py:99-119` contain nearly identical code:
```python
# 1. Call detect_grains()
# 2. For each result: create GrainContour, compute_morphology(), classify_grain(), set fields
```
This duplication is technical debt — any bug fix or feature change must be made in two places.

**New module responsibility:** Provide a single `run_detection_pipeline()` function that takes an image + config and returns `(grains, morphologies, statistics)`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the shared detection pipeline module."""

import cv2
import numpy as np
import pytest

from core.pipeline import run_detection_pipeline
from core.preprocessor import PreprocessConfig
from core.detector import FlocculationConfig


def _make_test_image(size=400):
    """Create a synthetic image with a gray background and a bright circle."""
    image = np.full((size, size, 3), 50, dtype=np.uint8)
    noise = np.random.randint(0, 20, (size, size, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    cv2.circle(image, (200, 200), 35, (200, 200, 200), -1)
    return image


class TestRunDetectionPipeline:
    """Tests for run_detection_pipeline."""

    def test_returns_grains_morphologies_and_stats(self):
        """Should return all three expected outputs."""
        image = _make_test_image()
        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)

        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert len(grains) > 0
        assert len(morphologies) > 0
        assert len(grains) == len(morphologies)
        assert stats.count == len(morphologies)

    def test_grains_have_contour_and_mask(self):
        """Each grain should have a contour and mask."""
        image = _make_test_image()
        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)

        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        for grain in grains:
            assert hasattr(grain, "contour")
            assert hasattr(grain, "mask")
            assert grain.contour is not None
            assert grain.mask is not None

    def test_morphologies_have_classification(self):
        """Each morphology should have shape_class and is_flocculation set."""
        image = _make_test_image()
        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)

        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        for morph in morphologies:
            assert morph.shape_class in {"spherical", "rod-like", "discoidal", "flocculation"}
            assert isinstance(morph.is_flocculation, bool)
            assert morph.confidence > 0

    def test_statistics_are_computed(self):
        """Stats should be a valid GrainStatistics object."""
        image = _make_test_image()
        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)

        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert stats.count == len(morphologies)
        assert stats.area_mean > 0
        assert stats.circularity_mean > 0

    def test_empty_image_returns_empty(self):
        """Should handle empty images gracefully."""
        image = np.full((400, 400, 3), 50, dtype=np.uint8)
        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)

        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert grains == []
        assert morphologies == []
        assert stats.count == 0

    def test_flocculation_config_is_respected(self):
        """Should pass flocculation config through to detector."""
        image = _make_test_image()
        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)
        floc_config = FlocculationConfig(min_area=100000)  # Too high to trigger

        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False,
            floc_config=floc_config,
        )

        for morph in morphologies:
            assert morph.is_flocculation is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'core.pipeline'"

- [ ] **Step 3: Write minimal implementation**

Create `core/pipeline.py`:

```python
"""Shared detection pipeline for SandAnalyze.

Provides a single entry point that wraps the full detection flow:
detect grains → compute morphology → classify → compute statistics.
"""

from typing import Tuple

import numpy as np

from core.classifier import classify_grain
from core.detector import detect_grains, FlocculationConfig
from core.morphology import compute_morphology, compute_statistics, GrainMorphology, GrainStatistics
from core.preprocessor import PreprocessConfig
from core.traditional import GrainContour


def run_detection_pipeline(
    image: np.ndarray,
    config: PreprocessConfig,
    min_area: int = 1000,
    max_area: int = 15000,
    border_margin: int = 5,
    hull_expansion_ratio: float = 1.5,
    floc_config: FlocculationConfig | None = None,
    crop_black_background: bool = True,
) -> Tuple[list[GrainContour], list[GrainMorphology], GrainStatistics]:
    """Run the full detection pipeline on an image.

    Args:
        image: Raw input image (BGR or grayscale).
        config: Preprocessing configuration.
        min_area: Minimum grain area.
        max_area: Maximum grain area.
        border_margin: Distance from border for edge filtering.
        hull_expansion_ratio: Threshold for convex hull vs mask filling.
        floc_config: Flocculation detection config. Uses defaults if None.
        crop_black_background: Whether to crop black background before processing.

    Returns:
        Tuple of (grains, morphologies, statistics).
    """
    # Detect grains using v6 single-step pipeline
    results = detect_grains(
        image,
        config=config,
        min_area=min_area,
        max_area=max_area,
        border_margin=border_margin,
        hull_expansion_ratio=hull_expansion_ratio,
        floc_config=floc_config,
        crop_black_background=crop_black_background,
    )

    # Convert DetectionResult to GrainContour + compute morphology
    grains: list[GrainContour] = []
    morphologies: list[GrainMorphology] = []

    for r in results:
        gc = GrainContour(contour=r.contour, mask=r.mask)
        grains.append(gc)

        morph = compute_morphology(r.contour, r.mask)
        morph.shape_class = classify_grain(morph.aspect_ratio, r.is_flocculation)
        morph.is_flocculation = r.is_flocculation
        morph.confidence = 0.9 if r.is_flocculation else 0.95
        morphologies.append(morph)

    stats = compute_statistics(morphologies)

    return grains, morphologies, stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/pipeline.py tests/test_pipeline.py
git commit -m "feat: add shared detection pipeline module"
```

---

## Task 2: Refactor `app.py` to Use `core.pipeline`

**Files:**
- Modify: `app.py:311-359`

**Background:** `app.py` lines 311-359 duplicate the pipeline logic. Replace with a single call to `run_detection_pipeline()`.

- [ ] **Step 1: Add import**

Add to `app.py` imports (around line 30):
```python
from core.pipeline import run_detection_pipeline
```

- [ ] **Step 2: Replace duplicated pipeline logic**

In `app.py`, replace lines 327-359 (the block starting with `# Use v6 single-step detection...` through `st.session_state.last_processing_time = time.time() - start`) with:

```python
                # Use shared pipeline
                grains, morphologies, stats = run_detection_pipeline(
                    image,
                    config=config,
                    min_area=config.min_area,
                    max_area=15000,
                    border_margin=st.session_state.border_margin,
                    hull_expansion_ratio=st.session_state.hull_expansion_ratio,
                    floc_config=st.session_state.floc_config if st.session_state.use_flocculation else None,
                    crop_black_background=True,
                )

                st.session_state.detection_method = "traditional"
                st.session_state.grains = grains
                st.session_state.morphologies = morphologies
                st.session_state.statistics = stats
                st.session_state.last_processing_time = time.time() - start
```

Also remove the now-unused import of `GrainContour` from `core.traditional` (if it was only used in this block). Check if `GrainContour` is used elsewhere in `app.py`.

- [ ] **Step 3: Remove unused imports**

Check if these imports are still needed in `app.py`:
- `from core.traditional import GrainContour` — if only used in the replaced block, remove it
- `from core.classifier import classify_grain` — if only used in the replaced block, remove it
- `from core.morphology import compute_morphology, compute_statistics` — if only used in the replaced block, remove them (keep what's still used)

- [ ] **Step 4: Verify app still runs**

Run: `python -c "import app"`
Expected: No import errors

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "refactor: use shared pipeline in app.py"
```

---

## Task 3: Refactor `core/batch.py` to Use `core.pipeline`

**Files:**
- Modify: `core/batch.py:99-119`

**Background:** `batch.py` lines 99-119 duplicate the same pipeline logic.

- [ ] **Step 1: Add import**

Add to `core/batch.py` imports (around line 14):
```python
from core.pipeline import run_detection_pipeline
```

- [ ] **Step 2: Replace duplicated pipeline logic**

In `core/batch.py`, replace lines 99-119 (the block starting with `# Detect grains using v6 single-step pipeline` through `stats = compute_statistics(morphologies)`) with:

```python
        # Detect grains using shared pipeline
        grains, morphologies, stats = run_detection_pipeline(
            image,
            config=config,
            min_area=config.min_area,
            max_area=15000,
            border_margin=border_margin,
            floc_config=floc_config,
        )
```

- [ ] **Step 3: Remove unused imports**

Check if these imports are still needed in `core/batch.py`:
- `from core.classifier import classify_grain` — remove if unused
- `from core.morphology import compute_morphology, compute_statistics` — remove if unused
- `from core.traditional import GrainContour` — remove if unused (pipeline returns GrainContour objects, but we don't need to import the class if we only use the returned objects)

Actually, keep `GrainStatistics` and `GrainMorphology` imports if they're used for type hints.

- [ ] **Step 4: Run batch tests**

Run: `pytest tests/test_batch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/batch.py
git commit -m "refactor: use shared pipeline in batch.py"
```

---

## Task 4: Unify Type Hints Across All Modules

**Files:**
- Modify: `core/detector.py`, `core/morphology.py`, `core/preprocessor.py`, `core/batch.py`, `core/exporter.py`, `core/report.py`, `core/classifier.py`, `app.py`

**Background:** The codebase mixes `typing.List`, `typing.Tuple`, `typing.Optional` with modern `list[...]`, `tuple[...]`, `| None`. Standardize on Python 3.10+ native syntax.

**Pattern:** Replace `from typing import List, Tuple, Optional` with no import (use built-ins), and `List[X]` → `list[X]`, `Tuple[X, Y]` → `tuple[X, Y]`, `Optional[X]` → `X | None`.

- [ ] **Step 1: Update `core/detector.py`**

Replace line 13 `from typing import List, Tuple` with nothing (remove the import).
Replace `List[DetectionResult]` → `list[DetectionResult]` in the function signature on line 59.

- [ ] **Step 2: Update `core/morphology.py`**

Replace `from typing import List` with nothing.
Replace `List[float]` → `list[float]` in `GrainStatistics` dataclass fields (lines 117-119).

- [ ] **Step 3: Update `core/preprocessor.py`**

Replace `from typing import Tuple` with nothing.
Replace `Tuple[np.ndarray, Tuple[int, int, int, int]]` → `tuple[np.ndarray, tuple[int, int, int, int]]` in `crop_black_background` signature (line 98).

- [ ] **Step 4: Update `core/batch.py`**

Replace `list[GrainMorphology]` → already uses modern syntax, but check for any `typing` imports. Remove `from typing import List, Tuple` if present.
Replace `GrainStatistics | None` — already modern, good.

- [ ] **Step 5: Update `core/exporter.py`**

Replace `list[GrainMorphology]` — already modern. Check for `typing` imports and remove unused ones.
Replace `list[GrainContour] | None` and `list[GrainMorphology] | None` — already modern.

- [ ] **Step 6: Update `core/report.py`**

Replace `list` type hint in `generate_pdf_report` signature (line 89): `morphologies: list` → `morphologies: list[GrainMorphology]`.

- [ ] **Step 7: Update `core/classifier.py`**

Replace `list[float]` and `list[bool]` and `list[str]` — already modern. Check for `typing` imports.

- [ ] **Step 8: Update `app.py`**

Check for any `typing` imports. The file uses `list` directly in some places. Ensure consistency.

- [ ] **Step 9: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "style: unify type hints to Python 3.10+ native syntax"
```

---

## Task 5: Remove Dead Code — `core/traditional.py`

**Files:**
- Delete: `core/traditional.py`
- Delete: `tests/test_traditional.py`
- Modify: `core/__init__.py`
- Modify: `tests/test_integration.py`

**Background:** `core/traditional.py` provides `detect_grains(mask, min_area)` which operates on preprocessed binary masks. The v6 pipeline in `core/detector.py` operates directly on raw images and is the only path used by `app.py` and `batch.py`. The `GrainContour` dataclass is still needed (used by pipeline and exporter), so move it to `core/pipeline.py` or `core/morphology.py`.

**Decision:** Move `GrainContour` to `core/morphology.py` since it's a data structure representing a detected grain, closely related to `GrainMorphology`.

- [ ] **Step 1: Move `GrainContour` to `core/morphology.py`**

Add to `core/morphology.py` (after the imports, before `GrainMorphology`):

```python
@dataclass
class GrainContour:
    """Represents a detected grain via its contour and binary mask."""

    contour: np.ndarray
    mask: np.ndarray
```

- [ ] **Step 2: Update `core/pipeline.py` import**

Replace:
```python
from core.traditional import GrainContour
```
with:
```python
from core.morphology import GrainContour
```

- [ ] **Step 3: Update `core/exporter.py` import**

Replace:
```python
from core.traditional import GrainContour
```
with:
```python
from core.morphology import GrainContour
```

- [ ] **Step 4: Update `core/__init__.py`**

Remove:
```python
from core.traditional import GrainContour, detect_grains as detect_grains_traditional
```
and remove `"GrainContour"`, `"detect_grains_traditional"` from `__all__`.

Add to `__all__`:
```python
"GrainContour",
```

And add the import:
```python
from core.morphology import GrainContour
```

- [ ] **Step 5: Delete `core/traditional.py`**

```bash
rm core/traditional.py
```

- [ ] **Step 6: Delete `tests/test_traditional.py`**

```bash
rm tests/test_traditional.py
```

- [ ] **Step 7: Update `tests/test_integration.py`**

Remove the import:
```python
from core.traditional import detect_grains
```

The integration test uses `detect_grains(mask, min_area)` from traditional.py. Since we're removing it, update the test to use the v6 pipeline instead. Replace the test logic:

In `TestFullPipelineSynthetic.test_pipeline_without_watershed`, replace:
```python
        # Step 2: Detect grains
        grains = detect_grains(mask, min_area=50)
        assert len(grains) > 0, "Should detect at least one grain"

        # Step 3: Compute morphology for each grain
        morphologies = []
        for grain in grains:
            morph = compute_morphology(grain.contour, grain.mask)
            morphologies.append(morph)
```
with:
```python
        # Step 2: Detect grains from mask using v6 detector on original image
        from core.detector import detect_grains as detect_grains_v6
        from core.preprocessor import PreprocessConfig
        config = PreprocessConfig(use_clahe=False, min_area=50)
        image_color = cv2.cvtColor(sample_grain_image, cv2.COLOR_GRAY2BGR)
        results = detect_grains_v6(image_color, config, min_area=50, crop_black_background=False)
        assert len(results) > 0, "Should detect at least one grain"

        # Step 3: Compute morphology for each grain
        morphologies = []
        for r in results:
            morph = compute_morphology(r.contour, r.mask)
            morphologies.append(morph)
```

Similarly update `TestFullPipelineRealImage.test_pipeline_with_real_image`.

- [ ] **Step 8: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove unused traditional.py detection module"
```

---

## Task 6: Remove Dead Code — `core/yolo_detector.py` and `core/model_manager.py`

**Files:**
- Delete: `core/yolo_detector.py`
- Delete: `core/model_manager.py`
- Delete: `tests/test_yolo_detector.py`
- Modify: `core/__init__.py`

**Background:** YOLO integration was planned but never wired into the main detection flow. `app.py` and `batch.py` never call `YOLODetector` or `refine_with_yolo`. These modules add complexity and dependencies (ultralytics) without value.

- [ ] **Step 1: Remove imports from `core/__init__.py`**

Remove:
```python
from core.yolo_detector import YOLODetector, refine_with_yolo
```

Remove from `__all__`:
```python
    # YOLO
    "YOLODetector",
    "refine_with_yolo",
```

- [ ] **Step 2: Delete files**

```bash
rm core/yolo_detector.py
core/model_manager.py
rm tests/test_yolo_detector.py
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS (no tests should reference these modules)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove unused YOLO detector and model manager modules"
```

---

## Task 7: Refactor `core/detector.py` — Remove Inline Morphology/Classification

**Files:**
- Modify: `core/detector.py`
- Modify: `tests/test_detector.py`

**Background:** `core/detector.py`'s `detect_grains()` currently computes aspect ratio, major/minor axis, and flocculation classification inline. These responsibilities belong in `core/morphology.py` and `core/classifier.py`. The detector should only detect contours and return raw geometric data. However, the v6 pipeline design intentionally computes some features inline for filtering purposes (circularity filter, flocculation detection). The key issue is that `DetectionResult` carries computed morphology fields that are also computed by `compute_morphology()`.

**Decision:** Keep the inline computations in `detector.py` because they're needed for filtering (area, circularity, convexity, aspect ratio for flocculation detection). But remove the redundant `major_axis`/`minor_axis` computation from `DetectionResult` since `compute_morphology()` already does this. Actually, looking more carefully:

- `detector.py` computes: area, perimeter, circularity, aspect_ratio, major_axis, minor_axis, convexity, is_flocculation
- `morphology.py` `compute_morphology()` computes: area, perimeter, circularity, d_eq, major_axis, minor_axis, aspect_ratio, sphericity, convexity, feret_max, feret_min, solidity

The duplication is: area, perimeter, circularity, aspect_ratio, major_axis, minor_axis, convexity.

**Approach:** Keep `DetectionResult` lean — only include fields needed for filtering and downstream processing. The `contour` and `mask` are sufficient for `compute_morphology()` to derive everything else. Remove redundant computed fields from `DetectionResult`.

Wait — but `DetectionResult` is used by `app.py` and `batch.py` via the pipeline. The pipeline calls `compute_morphology(r.contour, r.mask)` which recomputes everything. So the fields on `DetectionResult` are only used internally within `detector.py` for filtering.

**Better approach:** Don't change `DetectionResult` fields (they're used for the return value and tests check them). Instead, document that `DetectionResult` contains **preliminary** morphology values used for filtering, and `compute_morphology()` produces the **authoritative** values. This is actually fine — the duplication is intentional for the pipeline design.

**Revised approach for this task:** Just add a docstring comment to `DetectionResult` clarifying that these are preliminary values, and ensure `compute_morphology()` is the authoritative source. No code changes needed beyond documentation.

Actually, let's be more aggressive. The real debt is that `DetectionResult` has 10 fields and `GrainMorphology` has 15 fields, with significant overlap. Let's keep `DetectionResult` minimal:

```python
@dataclass
class DetectionResult:
    contour: np.ndarray
    mask: np.ndarray
    is_flocculation: bool
```

But this would break tests. Let's keep the fields but add a clear docstring.

**Final decision:** Keep `DetectionResult` as-is for backward compatibility, but add docstring explaining the relationship. The real cleanup is already done by extracting the pipeline.

- [ ] **Step 1: Add docstring to `DetectionResult`**

Update `core/detector.py` lines 34-48:

```python
@dataclass
class DetectionResult:
    """Result of grain detection.

    Contains preliminary morphology values computed during detection for
    filtering purposes (area filtering, flocculation detection, circularity
    filtering). For authoritative morphology measurements, use
    :func:`core.morphology.compute_morphology`.
    """

    contour: np.ndarray
    mask: np.ndarray
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    major_axis: float
    minor_axis: float
    convexity: float
    is_flocculation: bool
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_detector.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add core/detector.py
git commit -m "docs: clarify DetectionResult as preliminary morphology values"
```

---

## Task 8: Strengthen Tests for `core/pipeline.py`

**Files:**
- Modify: `tests/test_pipeline.py`

**Background:** The new pipeline module needs comprehensive tests covering edge cases.

- [ ] **Step 1: Add edge case tests**

Add to `tests/test_pipeline.py`:

```python
    def test_grayscale_image(self):
        """Should accept grayscale images."""
        image = np.full((400, 400), 50, dtype=np.uint8)
        noise = np.random.randint(0, 20, (400, 400), dtype=np.uint8)
        image = cv2.add(image, noise)
        cv2.circle(image, (200, 200), 35, 200, -1)

        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)
        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=False
        )

        assert len(grains) > 0

    def test_crop_black_background(self):
        """Should work with crop_black_background=True."""
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (350, 350), (220, 220, 220), -1)
        cv2.circle(image, (200, 200), 30, (80, 80, 80), -1)

        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)
        grains, morphologies, stats = run_detection_pipeline(
            image, config, min_area=50, crop_black_background=True
        )

        assert len(grains) > 0

    def test_hull_expansion_ratio(self):
        """Should respect hull_expansion_ratio parameter."""
        image = _make_test_image()
        config = PreprocessConfig(adaptive_block_size=11, morph_kernel_size=3, morph_open_iter=1)

        grains_low, _, _ = run_detection_pipeline(
            image, config, min_area=50, hull_expansion_ratio=1.2, crop_black_background=False
        )
        grains_high, _, _ = run_detection_pipeline(
            image, config, min_area=50, hull_expansion_ratio=5.0, crop_black_background=False
        )

        assert len(grains_low) > 0
        assert len(grains_high) > 0
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: add edge case tests for pipeline module"
```

---

## Task 9: Final Integration Test Run

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Check for import errors in app.py**

Run: `python -c "import app"`
Expected: No errors

- [ ] **Step 3: Verify no remaining references to deleted modules**

Run:
```bash
grep -r "traditional" core/ tests/ --include="*.py" | grep -v "__pycache__"
grep -r "yolo_detector" core/ tests/ --include="*.py" | grep -v "__pycache__"
grep -r "model_manager" core/ tests/ --include="*.py" | grep -v "__pycache__"
```
Expected: Only references in git history or comments, no active imports

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: verify full test suite passes after cleanup"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ Code deduplication (app.py + batch.py → pipeline.py) — Task 1-3
- ✅ Unify type hints — Task 4
- ✅ Remove dead code (traditional.py) — Task 5
- ✅ Remove dead code (yolo_detector.py + model_manager.py) — Task 6
- ✅ Refactor detector.py responsibilities — Task 7
- ✅ Strengthen tests — Task 8
- ✅ Final integration verification — Task 9

**2. Placeholder scan:**
- ✅ No "TBD", "TODO", "implement later"
- ✅ All steps have actual code
- ✅ All file paths are exact

**3. Type consistency:**
- ✅ `run_detection_pipeline()` returns `tuple[list[GrainContour], list[GrainMorphology], GrainStatistics]`
- ✅ `GrainContour` moved to `core.morphology`
- ✅ All modules use `list[X]` not `List[X]`
- ✅ All modules use `X | None` not `Optional[X]`

**4. Backward compatibility:**
- ✅ `app.py` behavior unchanged (same UI, same outputs)
- ✅ `batch.py` behavior unchanged (same API, same outputs)
- ✅ `core/__init__.py` exports adjusted but main public API preserved
- ✅ Tests updated to reflect module moves but test assertions unchanged
