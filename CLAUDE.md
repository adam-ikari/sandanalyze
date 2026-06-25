# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SandAnalyze is a sand grain morphology analysis system built with traditional OpenCV image processing (no ML inference). It runs as a Streamlit web app with a CLI entry point.

## Common Commands

```bash
# Install dependencies
uv sync --extra dev

# Run the Streamlit web app
uv run python main.py
# Or with custom port:
uv run python main.py --server.port 8080
# Or directly:
streamlit run app.py

# Run tests
pytest
# Run a specific test file
pytest tests/test_detector.py
# Run a specific test class
pytest tests/test_detector.py::TestDetectGrains
# Run a specific test method
pytest tests/test_detector.py::TestDetectGrains::test_detect_grains_from_image
# Run with coverage
pytest --cov=core --cov-report=term-missing
```

## Architecture

### Pipeline Entry Points

There are two main pipeline functions in `core/pipeline.py`:

- **`run_detection_pipeline()`** — Single-scale detection. The default path used by the web app and batch processing. Orchestrates detection → validation → morphology → classification → statistics.
- **`run_multiscale_detection_pipeline()`** — Multi-scale detection that preprocesses at 3 scales (large/medium/small), merges results, applies watershed/concave-point splitting, and multi-feature filtering. Useful for images with grains of widely varying sizes.

Both pipelines share the same `detect_grains()` function from `core/detector.py` for final contour extraction.

### Detection Pipeline (v6)

`core/detector.py::detect_grains()` is the core detection function. It operates in a single pass:

1. Optional black background crop (finds largest bright component)
2. Adaptive threshold in ROI (detects dark grains on light background)
3. Morphological open
4. Connected-components filtering by area + border-touching rejection
5. Contour detection
6. Per-contour feature computation + hull smoothing / mask filling
7. Inline flocculation detection (requires ≥2 of: low circularity, low convexity, high aspect ratio)

Key design note: Area filtering happens at the connected-components level (Step 4), not at contour level. ContourArea / filled mask overestimates for shapes with interior holes, so area re-checking after contour extraction was intentionally removed.

### Preprocessing

`core/preprocessor.py` provides:

- **`preprocess()`** — Full pipeline: grayscale → CLAHE → Gaussian blur → adaptive threshold → morphological open/close → watershed splitting → area filtering.
- **`auto_detect_preset()`** — Chooses from `default`, `macro_sand`, `microscope`, `shadow` based on brightness/contrast/noise analysis.
- **`auto_tune_for_microscope()`** — Fine-tunes blur kernel, adaptive C, block size, and min_area based on image characteristics.
- **`preprocess_shadow_regions()`** — Double-pass CLAHE (6.0 then 3.0) with smaller tiles for shadow areas.

`PreprocessConfig` is a dataclass with sensible defaults. Presets are accessed via `PreprocessConfig.from_preset(name)`.

### Data Flow

```
Raw Image
  → detect_grains() [core/detector.py]
    → Optional: SimpleValidator texture/edge validation [core/texture_edge_filter.py]
  → GrainContour + GrainMorphology [core/morphology.py]
    → classify_grain() [core/classifier.py] (Zingg + flocculation)
  → compute_statistics() [core/morphology.py]
  → Export (CSV, annotated PNG, PDF) [core/exporter.py, core/report.py]
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `core/detector.py` | v6 grain detection with flocculation/edge filtering |
| `core/preprocessor.py` | Image preprocessing, auto-tuning, presets |
| `core/pipeline.py` | Orchestrates full detection → classification → statistics |
| `core/morphology.py` | Morphological parameter computation (area, circularity, Feret, etc.) |
| `core/classifier.py` | Zingg shape classification + flocculation detection |
| `core/texture_edge_filter.py` | Lightweight false-positive validator (SimpleValidator) |
| `core/feature_filter.py` | Multi-feature filters for multiscale pipeline (edge, noise, filament, strict) |
| `core/morphological_splitter.py` | Watershed + concave-point splitting for touching grains |
| `core/multiscale_detector.py` | Multi-scale preprocessing and result merging |
| `core/batch.py` | Batch processing over directories |
| `core/exporter.py` | CSV and annotated image export |
| `core/report.py` | PDF report generation |
| `app.py` | Streamlit web UI |
| `main.py` | CLI entry point (wraps `streamlit run app.py`) |

### Important Design Decisions

- **No CNN/ML**: The project previously had a CNN enhancement module that was fully removed. All detection is traditional OpenCV.
- **Texture validation was simplified**: An earlier complex LBP/GLCM-based validator was refactored down to `SimpleValidator` (filters lens edges, tiny noise, and extremely low-contrast blobs only). The `scikit-image` dependency remains but is unused.
- **Flocculation takes priority over Zingg**: In classification, flocculation is checked first; Zingg categories are only applied to non-flocculation grains.
- **Border-touching components are rejected** in `detect_grains()` to avoid the circular microscope field-of-view being detected as a grain.
- **Hull smoothing vs mask filling**: When `hull_area / area < hull_expansion_ratio`, the convex hull is used as the final contour; otherwise the original contour is kept with a filled mask.

### Testing

Tests use synthetic images (gray background with ellipses/circles) in `tests/conftest.py`. The `sample_grain_image` and `overlapping_grain_image` fixtures are the standard test inputs. Real image testing references `Sand_from_Gobi_Desert.jpg`.

### Dependencies

Managed via `pyproject.toml` with `uv`. Optional extras:
- `dev`: pytest, pytest-cov
- `texture`: scikit-image (currently unused in code but declared)
