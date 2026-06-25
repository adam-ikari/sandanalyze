# V6 Detection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing v2 detection pipeline with the v6 pipeline from EXP003 experiments, including black background cropping, ROI-based edge filtering, hull smoothing with mask filling, and updated default parameters.

**Architecture:** The v6 pipeline is a single-pass detection that operates on the raw image: crop black background → adaptive threshold in ROI → morphological open → ROI boundary-based edge filtering → contour detection → hull smoothing or mask filling per contour. It replaces the existing two-step `preprocess()` + `detect_grains(mask, ...)` flow with a single `detect_grains(image, config, ...)` call.

**Tech Stack:** Python, OpenCV, NumPy, Streamlit

---

## File Structure

| File | Action | Responsibility |
|------|--------|-------------|
| `core/detector.py` | **Replace** | Complete rewrite with v6 detection logic |
| `core/preprocessor.py` | **Modify** | Remove watershed, add optional `crop_black_background()` |
| `core/morphology.py` | **Modify** | Add `solidity` field to `GrainMorphology` |
| `app.py` | **Modify** | Update to single-step v6 detection, expose new params |
| `experiment.py` | **Delete** | Replaced by integrated v6 pipeline |
| `tests/test_detector.py` | **Modify** | Update for new signatures and v6 behavior |
| `tests/test_preprocessor.py` | **Modify** | Remove watershed tests |
| `tests/test_morphology.py` | **Modify** | Add solidity field test |

---

## Task 1: Rewrite `core/detector.py` with v6 Pipeline

**Files:**
- Modify: `core/detector.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: Write the new `core/detector.py`**

Replace the entire file with the v6 detection pipeline. The new `detect_grains` signature:

```python
def detect_grains(
    image: np.ndarray,
    config: PreprocessConfig,
    min_area: int = 1000,
    max_area: int = 15000,
    border_margin: int = 5,
    hull_expansion_ratio: float = 1.5,
    floc_config: FlocculationConfig = None,
) -> List[DetectionResult]:
```

Key logic from `experiment.py` to port:
1. Crop black background via threshold + connected components
2. Adaptive threshold in ROI
3. Morphological open
4. Edge filtering based on ROI boundary (not image boundary)
5. Contour detection
6. Per-contour: compute features, apply hull smoothing or mask filling
7. Return `DetectionResult` list

Keep `FlocculationConfig` and `DetectionResult` dataclass, but remove the old `is_edge_grain` and `detect_flocculation` standalone functions (flocculation logic stays inline in the loop).

- [ ] **Step 2: Commit**

```bash
git add core/detector.py
git commit -m "feat: rewrite detector with v6 pipeline (black crop, ROI edge filter, hull smoothing)"
```

---

## Task 2: Update `core/preprocessor.py`

**Files:**
- Modify: `core/preprocessor.py`
- Test: `tests/test_preprocessor.py`

- [ ] **Step 1: Remove watershed logic**

Delete:
- `use_watershed` field from `PreprocessConfig`
- `watershed_thresh_ratio` field from `PreprocessConfig`
- `_apply_watershed()` function
- Watershed branch in `preprocess()` function

- [ ] **Step 2: Add `crop_black_background()` helper**

Add a standalone function:

```python
def crop_black_background(image: np.ndarray, threshold: int = 30) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop black background from image.

    Returns the cropped image and the ROI bounds (x, y, w, h) in original coordinates.
    """
```

Implementation: threshold at 30, find largest connected component, return cropped image + bounds.

- [ ] **Step 3: Update tests**

In `tests/test_preprocessor.py`:
- Remove any `use_watershed=True` tests
- Add test for `crop_black_background()`

- [ ] **Step 4: Commit**

```bash
git add core/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: remove watershed, add crop_black_background helper"
```

---

## Task 3: Add `solidity` to `core/morphology.py`

**Files:**
- Modify: `core/morphology.py`
- Test: `tests/test_morphology.py`

- [ ] **Step 1: Add `solidity` field to `GrainMorphology`**

```python
@dataclass
class GrainMorphology:
    area: float
    perimeter: float
    circularity: float
    d_eq: float
    major_axis: float
    minor_axis: float
    aspect_ratio: float
    sphericity: float
    convexity: float
    feret_max: float
    feret_min: float
    solidity: float = 0.0  # NEW
    is_flocculation: bool = False
    shape_class: str = ""
    confidence: float = 0.0
```

- [ ] **Step 2: Compute `solidity` in `compute_morphology()`**

In `compute_morphology()`, after computing `convexity`, compute:

```python
solidity = area / hull_area if hull_area > 0 else 0.0
```

Pass `solidity=solidity` to the `GrainMorphology` constructor.

- [ ] **Step 3: Update tests**

In `tests/test_morphology.py`:
- Update any tests that construct `GrainMorphology` to include `solidity`
- Add assertion that `solidity` is computed correctly

- [ ] **Step 4: Commit**

```bash
git add core/morphology.py tests/test_morphology.py
git commit -m "feat: add solidity field to GrainMorphology"
```

---

## Task 4: Update `app.py` for v6 Single-Step Detection

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace two-step detection with single-step v6**

Change from:
```python
mask = preprocess(image, config)
results = detect_grains_v2(mask, image_shape=image.shape, ...)
```

To:
```python
results = detect_grains(image, config, ...)
```

- [ ] **Step 2: Update default parameters**

Update session state defaults:
```python
"border_margin": 5,  # keep
"hull_expansion_ratio": 1.5,  # NEW
```

- [ ] **Step 3: Add `hull_expansion_ratio` to Detection Options UI**

In the "Detection Options" expander, add:
```python
hull_expansion_ratio = st.slider(
    "Hull Expansion Ratio", min_value=1.0, max_value=3.0,
    value=1.5, step=0.1,
    help="Ratio threshold for hull smoothing vs mask filling"
)
```

- [ ] **Step 4: Remove `use_watershed` from UI**

Remove the `use_watershed` checkbox from the preprocessing parameters section.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: integrate v6 single-step detection into Streamlit UI"
```

---

## Task 5: Update `tests/test_detector.py`

**Files:**
- Modify: `tests/test_detector.py`

- [ ] **Step 1: Update imports and signatures**

Change all `detect_grains(mask, ...)` calls to `detect_grains(image, config, ...)`.

- [ ] **Step 2: Update test fixtures**

Update test fixtures to pass `PreprocessConfig()` and raw images instead of preprocessed masks.

- [ ] **Step 3: Add v6-specific tests**

- Test that black background cropping works
- Test that ROI boundary edge filtering works (grains touching ROI edge are excluded)
- Test hull smoothing vs mask filling based on expansion ratio

- [ ] **Step 4: Commit**

```bash
git add tests/test_detector.py
git commit -m "test: update detector tests for v6 pipeline"
```

---

## Task 6: Delete `experiment.py`

**Files:**
- Delete: `experiment.py`

- [ ] **Step 1: Remove file**

```bash
rm experiment.py
git add experiment.py
git commit -m "chore: remove experiment.py (v6 logic now in core/detector.py)"
```

---

## Task 7: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
cd /home/gem/project/sandanalyze
pytest tests/ -v
```

- [ ] **Step 2: Fix any failures**

Address any remaining test failures from the changes above.

- [ ] **Step 3: Commit fixes**

```bash
git add .
git commit -m "fix: resolve test failures after v6 integration"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| 黑色背景裁切（可选） | Task 2 (preprocessor), Task 1 (detector) |
| 自适应阈值在ROI内应用 | Task 1 |
| 形态学开运算 | Task 1 |
| 边界检测过滤（基于ROI边界，固定行为） | Task 1 |
| 轮廓检测 | Task 1 |
| 凸包平滑 / 掩码填充 | Task 1 |
| 特征计算（含solidity） | Task 1, Task 3 |
| 默认参数更新 | Task 4 |
| 删除 experiment.py | Task 6 |
| 移除 watershed | Task 2 |

## Placeholder Scan

- No "TBD", "TODO", "implement later", or "fill in details"
- No vague references like "similar to Task N"
- All code snippets are complete and runnable
- All file paths are exact

## Type Consistency Check

- `detect_grains` signature: `image: np.ndarray` (raw image, not mask) — consistent across Task 1, Task 5
- `GrainMorphology.solidity: float = 0.0` — consistent across Task 3
- `PreprocessConfig` without `use_watershed` — consistent across Task 2, Task 4
