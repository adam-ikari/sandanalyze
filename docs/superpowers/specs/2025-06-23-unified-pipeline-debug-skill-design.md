# Unified Pipeline Framework and Agent Debug Skill Design

> **Date:** 2025-06-23
> **Status:** Approved

## Goal

Build a unified framework where tests and the actual project share the same image pipeline and detection code, plus an Agent-accessible skill for parameter debugging.

## Architecture

### 1. Core Pipeline Debugger (`core/pipeline_debugger.py`)

A single module providing three classes:

#### `ImageSource` (Abstract Base)
- `SyntheticImageSource` — generates synthetic test images (ellipses, circles, etc.)
- `RealImageSource` — loads real images from disk
- `ImageSource.from_path(path)` — factory method

#### `PipelineRunner`
- `__init__(image_source: ImageSource)`
- `run(config: PreprocessConfig) -> PipelineResult`
- `visualize_branches(save_path: str)` — generates branch comparison PNG
- `compare_configs(config_a, config_b, save_path: str)` — side-by-side comparison
- `grid_search(param_ranges: dict, metric: str) -> list[GridResult]`

#### `PipelineResult` (dataclass)
- `mask: np.ndarray` — final binary mask
- `grains: list[GrainContour]`
- `morphologies: list[GrainMorphology]`
- `statistics: GrainStatistics`
- `intermediate_masks: dict[str, np.ndarray]` — brightness/edge/texture masks

### 2. Enhanced Test Fixtures (`tests/conftest.py`)

Add new fixtures:
- `real_image_source()` — loads from `data/test/*.png`
- `synthetic_image_source()` — generates synthetic images
- `pipeline_runner(real_image_source)` — pre-configured PipelineRunner

Tests can now run against both real and synthetic images:
```python
def test_detects_grains_on_real_image(pipeline_runner):
    result = pipeline_runner.run(PreprocessConfig())
    assert result.statistics.count > 0
```

### 3. Agent Debug Skill (`tools/debug_skill.py`)

CLI-style interface for Agents:

```python
class DebugSkill:
    """Agent-accessible skill for parameter debugging."""

    def visualize_branches(self, image_path: str, config: PreprocessConfig,
                           save_path: str = "debug_branches.png"):
        """Visualize brightness/edge/texture branches."""

    def compare_configs(self, image_path: str,
                        config_a: PreprocessConfig,
                        config_b: PreprocessConfig,
                        save_path: str = "debug_compare.png"):
        """Compare two configurations side-by-side."""

    def grid_search(self, image_path: str,
                    param_ranges: dict[str, list],
                    metric: str = "grain_count") -> pd.DataFrame:
        """Grid search over parameter ranges."""

    def regression_test(self, config: PreprocessConfig,
                        expected_counts: dict[str, int]) -> dict[str, bool]:
        """Run regression tests against expected grain counts."""
```

### 4. Data Flow

```
ImageSource
    → PipelineRunner.run(config)
        → preprocess() [brightness + edge + texture]
        → detect_grains()
        → compute_morphology()
        → classify_grain()
        → compute_statistics()
    → PipelineResult
        → visualize_branches()
        → compare_configs()
        → grid_search()
```

### 5. File Changes

| File | Action | Description |
|------|--------|-------------|
| `core/pipeline_debugger.py` | Create | PipelineRunner, ImageSource, PipelineResult |
| `tests/conftest.py` | Modify | Add real_image_source, synthetic_image_source, pipeline_runner fixtures |
| `tools/debug_skill.py` | Create | Agent-accessible debug skill |
| `tests/test_pipeline_debugger.py` | Create | Tests for the new framework |

### 6. Testing Strategy

- Unit tests for `ImageSource` (synthetic and real)
- Unit tests for `PipelineRunner` (run, visualize, compare)
- Integration tests using both real and synthetic images
- Regression tests ensuring existing tests still pass

### 7. Backward Compatibility

- All existing tests continue to work unchanged
- Existing `debug_*.py` scripts can be migrated gradually
- `core/preprocessor.py` and `core/detector.py` remain untouched

## Success Criteria

1. Tests can run against real images using the same pipeline as the project
2. Agent can call `DebugSkill.visualize_branches()` without writing ad-hoc scripts
3. Parameter grid search produces a readable report
4. All existing tests pass (103/103)
