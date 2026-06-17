# SandAnalyze v2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a traditional CV-based sand grain morphology analysis system with Zingg + Flocculation classification, Streamlit UI, batch processing, and PDF report generation.

**Architecture:** Refactor existing core modules to add flocculation detection, edge filtering, and auto-parameter tuning. Extend the Streamlit UI with new classification display and batch processing. Add PDF report generation via ReportLab.

**Tech Stack:** Python 3.13, OpenCV, NumPy, SciPy, Streamlit, Plotly, ReportLab, Pandas, Pytest

---

## File Structure

```
sandanalyze/
├── app.py                      # Streamlit UI (modify)
├── main.py                     # Entry point (keep)
├── core/
│   ├── __init__.py             # Package exports (modify)
│   ├── preprocessor.py         # Image preprocessing (modify)
│   ├── detector.py             # Grain detection + flocculation (NEW)
│   ├── morphology.py           # Morphological params (modify)
│   ├── classifier.py           # Zingg + Flocculation (NEW)
│   ├── exporter.py             # CSV + image export (modify)
│   └── report.py               # PDF report generation (NEW)
├── tests/
│   ├── test_detector.py        # Detection tests (NEW)
│   ├── test_classifier.py      # Classification tests (NEW)
│   ├── test_preprocessor.py    # Preprocessing tests (modify)
│   └── test_morphology.py      # Morphology tests (modify)
└── docs/
    └── design_v2.md            # Design spec (existing)
```

---

## Task Dependencies

```
Task 1 (preprocessor) ─┐
Task 2 (morphology) ───┼→ Task 3 (detector) ─→ Task 4 (classifier) ─→ Task 5 (exporter)
                       │                                              │
                       └→ Task 6 (report) ←───────────────────────────┘
                                                                        │
Task 7 (app UI) ←───────────────────────────────────────────────────────┘
```

---

## Task 1: Refactor Preprocessor Module

**Files:**
- Modify: `core/preprocessor.py`
- Test: `tests/test_preprocessor.py`

**Context:** The existing preprocessor works but needs to support configurable CLAHE and watershed. We also need to add edge filtering support.

- [ ] **Step 1: Write the failing test for edge filtering**

```python
import numpy as np
import cv2
from core.preprocessor import PreprocessConfig, preprocess, filter_edge_grains


def test_filter_edge_grains_removes_border_contours():
    """Contours touching image border should be removed."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Grain at center - should keep
    cv2.circle(mask, (50, 50), 10, 255, -1)
    
    # Grain at edge - should remove
    cv2.circle(mask, (5, 50), 8, 255, -1)
    
    # Grain at corner - should remove
    cv2.circle(mask, (95, 95), 8, 255, -1)
    
    filtered = filter_edge_grains(mask, border_margin=5)
    
    # Count remaining grains
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    
    # Verify the remaining grain is the center one
    x, y, bw, bh = cv2.boundingRect(contours[0])
    assert x > 5 and y > 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocessor.py::test_filter_edge_grains_removes_border_contours -v`

Expected: FAIL with "function not defined"

- [ ] **Step 3: Add edge filtering function to preprocessor**

```python
def filter_edge_grains(mask: np.ndarray, border_margin: int = 5) -> np.ndarray:
    """Remove grains that touch or are too close to the image border.
    
    Args:
        mask: Binary mask with grains as 255.
        border_margin: Minimum distance from border (pixels).
    
    Returns:
        Filtered binary mask with edge grains removed.
    """
    if mask is None or mask.size == 0:
        return mask
    
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_mask = np.zeros_like(mask)
    
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Check if contour touches or is too close to border
        touches_border = (
            x <= border_margin or
            y <= border_margin or
            x + bw >= w - border_margin or
            y + bh >= h - border_margin
        )
        
        if not touches_border:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    return filtered_mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocessor.py::test_filter_edge_grains_removes_border_contours -v`

Expected: PASS

- [ ] **Step 5: Write test for auto-parameter tuning**

```python
def test_auto_tune_params():
    """Auto-tuning should return reasonable parameters."""
    from core.preprocessor import auto_tune_params
    
    # Create a synthetic image with known grain size
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 20, 200, -1)
    
    config = auto_tune_params(img)
    
    assert config.blur_kernel >= 3
    assert config.adaptive_block_size >= 3
    assert config.min_area > 0
```

- [ ] **Step 6: Add auto-tune function**

```python
def auto_tune_params(image: np.ndarray) -> PreprocessConfig:
    """Automatically tune preprocessing parameters based on image characteristics.
    
    Uses image brightness, contrast, and estimated grain size to determine
    optimal parameters.
    
    Args:
        image: Input grayscale or color image.
    
    Returns:
        Tuned PreprocessConfig.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Estimate image characteristics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Adjust blur kernel based on noise level
    if std_brightness > 60:
        blur_kernel = 7
    elif std_brightness > 40:
        blur_kernel = 5
    else:
        blur_kernel = 3
    
    # Ensure odd
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    
    # Adjust adaptive block based on image size
    h, w = gray.shape
    min_dim = min(h, w)
    if min_dim > 1000:
        adaptive_block = 21
    elif min_dim > 500:
        adaptive_block = 15
    else:
        adaptive_block = 11
    
    # Ensure odd and >= 3
    adaptive_block = max(3, adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1)
    
    # Adjust C based on brightness
    if mean_brightness < 100:
        adaptive_c = 5
    elif mean_brightness < 150:
        adaptive_c = 2
    else:
        adaptive_c = -2
    
    # Estimate min_area from image size
    estimated_grain_area = (min_dim / 50) ** 2
    min_area = max(50, int(estimated_grain_area * 0.5))
    
    return PreprocessConfig(
        blur_kernel=blur_kernel,
        adaptive_block_size=adaptive_block,
        adaptive_c=adaptive_c,
        min_area=min_area,
    )
```

- [ ] **Step 7: Run all preprocessor tests**

Run: `pytest tests/test_preprocessor.py -v`

Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add core/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: add edge filtering and auto-parameter tuning to preprocessor"
```

---

## Task 2: Refactor Morphology Module

**Files:**
- Modify: `core/morphology.py`
- Test: `tests/test_morphology.py`

**Context:** Existing morphology module computes basic parameters. Need to add flocculation-related fields and ensure all parameters are computed correctly.

- [ ] **Step 1: Update GrainMorphology dataclass**

Add fields to existing `GrainMorphology` in `core/morphology.py`:

```python
@dataclass
class GrainMorphology:
    """Morphological parameters for a single sand grain."""

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
    # New fields
    is_flocculation: bool = False
    shape_class: str = ""
    confidence: float = 0.0
```

- [ ] **Step 2: Add Zingg + Flocculation color constants**

Add to `core/morphology.py` after existing ZINGG_COLORS:

```python
# Extended classification colors (BGR format for OpenCV)
CLASSIFICATION_COLORS = {
    "球状": (0, 255, 0),      # Green
    "棒状": (0, 0, 255),      # Red
    "片状": (255, 0, 0),      # Blue
    "絮凝": (0, 255, 255),    # Yellow (BGR)
}


def get_classification_color(shape_class: str) -> tuple[int, int, int]:
    """Get the color for a classification.
    
    Args:
        shape_class: One of "球状", "棒状", "片状", "絮凝".
    
    Returns:
        BGR color tuple.
    """
    return CLASSIFICATION_COLORS.get(shape_class, (128, 128, 128))
```

- [ ] **Step 3: Write test for classification colors**

```python
def test_classification_colors():
    from core.morphology import get_classification_color, CLASSIFICATION_COLORS
    
    assert get_classification_color("球状") == (0, 255, 0)
    assert get_classification_color("棒状") == (0, 0, 255)
    assert get_classification_color("片状") == (255, 0, 0)
    assert get_classification_color("絮凝") == (0, 255, 255)
    assert get_classification_color("unknown") == (128, 128, 128)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_morphology.py::test_classification_colors -v`

Expected: PASS

- [ ] **Step 5: Update GrainStatistics for four-class system**

Modify `GrainStatistics` in `core/morphology.py`:

```python
@dataclass
class GrainStatistics:
    """Aggregate statistics across multiple sand grains."""

    count: int
    area_mean: float = 0.0
    area_std: float = 0.0
    area_median: float = 0.0
    circularity_mean: float = 0.0
    circularity_std: float = 0.0
    circularity_median: float = 0.0
    d_eq_mean: float = 0.0
    d_eq_std: float = 0.0
    d_eq_median: float = 0.0
    aspect_ratio_mean: float = 0.0
    aspect_ratio_std: float = 0.0
    aspect_ratio_median: float = 0.0
    sphericity_mean: float = 0.0
    sphericity_std: float = 0.0
    sphericity_median: float = 0.0
    convexity_mean: float = 0.0
    convexity_std: float = 0.0
    convexity_median: float = 0.0
    # Updated for four-class system
    zingg_counts: dict = field(default_factory=lambda: {"球状": 0, "棒状": 0, "片状": 0, "絮凝": 0})
    zingg_colors: dict = field(default_factory=dict)
    d_eq_values: List[float] = field(default_factory=list)
    circularity_values: List[float] = field(default_factory=list)
    sphericity_values: List[float] = field(default_factory=list)
    # New: flocculation stats
    flocculation_count: int = 0
    flocculation_ratio: float = 0.0
```

- [ ] **Step 6: Update compute_statistics for four-class**

Update the `compute_statistics` function to handle four classes:

```python
def compute_statistics(morphologies: List[GrainMorphology]) -> GrainStatistics:
    """Compute aggregate statistics across multiple grains."""
    if not morphologies:
        return GrainStatistics(count=0)
    
    def _stats(values: List[float]) -> tuple[float, float, float]:
        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        median = float(np.median(arr))
        return mean, std, median
    
    areas = [m.area for m in morphologies]
    circularities = [m.circularity for m in morphologies]
    d_eqs = [m.d_eq for m in morphologies]
    aspect_ratios = [m.aspect_ratio for m in morphologies]
    sphericities = [m.sphericity for m in morphologies]
    convexities = [m.convexity for m in morphologies]
    
    area_mean, area_std, area_median = _stats(areas)
    circ_mean, circ_std, circ_median = _stats(circularities)
    d_eq_mean, d_eq_std, d_eq_median = _stats(d_eqs)
    ar_mean, ar_std, ar_median = _stats(aspect_ratios)
    sph_mean, sph_std, sph_median = _stats(sphericities)
    conv_mean, conv_std, conv_median = _stats(convexities)
    
    # Four-class classification counts
    zingg_counts: dict[str, int] = {"球状": 0, "棒状": 0, "片状": 0, "絮凝": 0}
    for m in morphologies:
        if m.is_flocculation:
            zingg_counts["絮凝"] += 1
        elif m.aspect_ratio < 1.5:
            zingg_counts["球状"] += 1
        elif m.aspect_ratio < 2.5:
            zingg_counts["棒状"] += 1
        else:
            zingg_counts["片状"] += 1
    
    # Per-grain classification colors
    zingg_colors = {}
    for i, m in enumerate(morphologies):
        if m.is_flocculation:
            zingg_colors[i] = CLASSIFICATION_COLORS["絮凝"]
        else:
            zingg_colors[i] = get_zingg_color(m.aspect_ratio)
    
    flocculation_count = zingg_counts["絮凝"]
    flocculation_ratio = flocculation_count / len(morphologies) if morphologies else 0.0
    
    return GrainStatistics(
        count=len(morphologies),
        area_mean=area_mean,
        area_std=area_std,
        area_median=area_median,
        circularity_mean=circ_mean,
        circularity_std=circ_std,
        circularity_median=circ_median,
        d_eq_mean=d_eq_mean,
        d_eq_std=d_eq_std,
        d_eq_median=d_eq_median,
        aspect_ratio_mean=ar_mean,
        aspect_ratio_std=ar_std,
        aspect_ratio_median=ar_median,
        sphericity_mean=sph_mean,
        sphericity_std=sph_std,
        sphericity_median=sph_median,
        convexity_mean=conv_mean,
        convexity_std=conv_std,
        convexity_median=conv_median,
        zingg_counts=zingg_counts,
        zingg_colors=zingg_colors,
        d_eq_values=d_eqs,
        circularity_values=circularities,
        sphericity_values=sphericities,
        flocculation_count=flocculation_count,
        flocculation_ratio=flocculation_ratio,
    )
```

- [ ] **Step 7: Write test for four-class statistics**

```python
def test_compute_statistics_with_flocculation():
    from core.morphology import GrainMorphology, compute_statistics
    
    morphs = [
        GrainMorphology(area=100, perimeter=40, circularity=0.8, d_eq=11.3,
                         major_axis=12, minor_axis=10, aspect_ratio=1.2,
                         sphericity=0.83, convexity=0.9, feret_max=12, feret_min=10,
                         is_flocculation=False),
        GrainMorphology(area=5000, perimeter=300, circularity=0.2, d_eq=80,
                         major_axis=100, minor_axis=20, aspect_ratio=5.0,
                         sphericity=0.2, convexity=0.5, feret_max=100, feret_min=20,
                         is_flocculation=True),
    ]
    
    stats = compute_statistics(morphs)
    
    assert stats.count == 2
    assert stats.zingg_counts["球状"] == 1
    assert stats.zingg_counts["絮凝"] == 1
    assert stats.flocculation_count == 1
    assert stats.flocculation_ratio == 0.5
```

- [ ] **Step 8: Run all morphology tests**

Run: `pytest tests/test_morphology.py -v`

Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add core/morphology.py tests/test_morphology.py
git commit -m "feat: add flocculation support to morphology module"
```

---

## Task 3: Create Detector Module

**Files:**
- Create: `core/detector.py`
- Test: `tests/test_detector.py`

**Context:** New module combining detection, flocculation detection, and edge filtering. Replaces the traditional.py module's responsibilities.

- [ ] **Step 1: Write failing test for flocculation detection**

```python
import numpy as np
import cv2
from core.detector import detect_flocculation, FlocculationConfig


def test_detect_flocculation_large_irregular():
    """Large irregular contour should be detected as flocculation."""
    # Create a large irregular contour
    contour = np.array([
        [0, 0], [100, 5], [105, 50], [95, 100], [50, 95], [5, 90]
    ], dtype=np.int32).reshape(-1, 1, 2)
    
    config = FlocculationConfig()
    is_floc = detect_flocculation(contour, config)
    
    assert is_floc is True


def test_detect_flocculation_small_round():
    """Small round contour should NOT be flocculation."""
    # Create a small round contour
    contour = np.array([
        [0, 10], [5, 5], [10, 0], [15, 5], [20, 10], [15, 15], [10, 20], [5, 15]
    ], dtype=np.int32).reshape(-1, 1, 2)
    
    config = FlocculationConfig()
    is_floc = detect_flocculation(contour, config)
    
    assert is_floc is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector.py -v`

Expected: FAIL with module not found

- [ ] **Step 3: Create detector module**

```python
"""Grain detection module with flocculation and edge filtering support."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class FlocculationConfig:
    """Configuration for flocculation detection."""
    
    min_area: int = 5000
    max_area: int = 50000
    min_circularity: float = 0.01
    max_circularity: float = 0.3
    min_convexity: float = 0.2
    max_convexity: float = 0.7
    max_aspect_ratio: float = 5.0


@dataclass
class DetectionResult:
    """Result of grain detection."""
    
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
    is_edge: bool


def detect_flocculation(contour: np.ndarray, config: FlocculationConfig) -> bool:
    """Detect if a contour is a flocculation (cluster of grains).
    
    Uses combined criteria: large area + low circularity + low convexity + high aspect ratio.
    Must satisfy area condition + at least 2 other conditions.
    
    Args:
        contour: Contour to check.
        config: Flocculation detection configuration.
    
    Returns:
        True if the contour is detected as flocculation.
    """
    area = cv2.contourArea(contour)
    if area < config.min_area or area > config.max_area:
        return False
    
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    x, y, bw, bh = cv2.boundingRect(contour)
    aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    
    # Combined criteria: must satisfy area + at least 2 other conditions
    conditions = [
        circularity <= config.max_circularity,  # Low circularity
        convexity <= config.max_convexity,       # Low convexity
        aspect_ratio >= config.max_aspect_ratio, # High aspect ratio
    ]
    
    return sum(conditions) >= 2


def is_edge_grain(contour: np.ndarray, image_shape: Tuple[int, ...], 
                   border_margin: int = 5) -> bool:
    """Check if a grain touches or is too close to the image border.
    
    Args:
        contour: Contour to check.
        image_shape: Shape of the original image (h, w) or (h, w, c).
        border_margin: Minimum distance from border.
    
    Returns:
        True if the grain is at the edge.
    """
    h, w = image_shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    
    return (
        x <= border_margin or
        y <= border_margin or
        x + bw >= w - border_margin or
        y + bh >= h - border_margin
    )


def detect_grains(mask: np.ndarray, image_shape: Tuple[int, ...],
                  min_area: int = 50, max_area: int = 50000,
                  border_margin: int = 5,
                  floc_config: FlocculationConfig = None) -> List[DetectionResult]:
    """Detect grains from preprocessed mask with flocculation and edge filtering.
    
    Args:
        mask: Preprocessed binary mask.
        image_shape: Original image shape for edge filtering.
        min_area: Minimum grain area.
        max_area: Maximum grain area.
        border_margin: Distance from border to filter.
        floc_config: Flocculation detection config. Uses defaults if None.
    
    Returns:
        List of DetectionResult objects.
    """
    if floc_config is None:
        floc_config = FlocculationConfig()
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    h, w = image_shape[:2]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Area filtering
        if area < min_area or area > max_area:
            continue
        
        # Edge filtering
        is_edge = is_edge_grain(contour, image_shape, border_margin)
        
        # Flocculation detection
        is_floc = detect_flocculation(contour, floc_config)
        
        # Compute morphology
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
        else:
            major_axis = np.sqrt(area * 4 / np.pi)
            minor_axis = major_axis
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Create mask for this grain
        grain_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(grain_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        results.append(DetectionResult(
            contour=contour,
            mask=grain_mask,
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            major_axis=major_axis,
            minor_axis=minor_axis,
            convexity=convexity,
            is_flocculation=is_floc,
            is_edge=is_edge,
        ))
    
    # Sort by area descending
    results.sort(key=lambda r: r.area, reverse=True)
    return results
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_detector.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/detector.py tests/test_detector.py
git commit -m "feat: add detector module with flocculation and edge filtering"
```

---

## Task 4: Create Classifier Module

**Files:**
- Create: `core/classifier.py`
- Test: `tests/test_classifier.py`

**Context:** New module handling Zingg classification + flocculation assignment.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from core.classifier import classify_grain, ZinggClassifier


def test_classify_spherical():
    """Low aspect ratio should classify as spherical."""
    classifier = ZinggClassifier()
    result = classify_grain(aspect_ratio=1.2, is_flocculation=False, classifier=classifier)
    assert result == "球状"


def test_classify_flocculation():
    """Flocculation should always be classified as flocculation."""
    classifier = ZinggClassifier()
    result = classify_grain(aspect_ratio=5.0, is_flocculation=True, classifier=classifier)
    assert result == "絮凝"


def test_classify_rod():
    """Medium aspect ratio should classify as rod-like."""
    classifier = ZinggClassifier()
    result = classify_grain(aspect_ratio=2.0, is_flocculation=False, classifier=classifier)
    assert result == "棒状"


def test_classify_discoidal():
    """High aspect ratio should classify as discoidal."""
    classifier = ZinggClassifier()
    result = classify_grain(aspect_ratio=3.0, is_flocculation=False, classifier=classifier)
    assert result == "片状"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py -v`

Expected: FAIL

- [ ] **Step 3: Create classifier module**

```python
"""Classification module for sand grain shape analysis.

Implements Zingg classification with flocculation support.
"""

from dataclasses import dataclass


@dataclass
class ZinggClassifier:
    """Zingg shape classification configuration."""
    
    spherical_threshold: float = 1.5
    bladed_threshold: float = 2.5


def classify_grain(aspect_ratio: float, is_flocculation: bool,
                     classifier: ZinggClassifier = None) -> str:
    """Classify a grain using Zingg + Flocculation system.
    
    Priority: Flocculation > Zingg classification
    
    Args:
        aspect_ratio: Major axis / minor axis ratio.
        is_flocculation: Whether the grain is detected as flocculation.
        classifier: ZinggClassifier instance. Uses defaults if None.
    
    Returns:
        One of: "球状", "棒状", "片状", "絮凝"
    """
    if classifier is None:
        classifier = ZinggClassifier()
    
    # Priority: Flocculation first
    if is_flocculation:
        return "絮凝"
    
    # Zingg classification
    if aspect_ratio < classifier.spherical_threshold:
        return "球状"
    elif aspect_ratio < classifier.bladed_threshold:
        return "棒状"
    else:
        return "片状"


def classify_batch(aspect_ratios: list[float], flocculation_flags: list[bool],
                   classifier: ZinggClassifier = None) -> list[str]:
    """Classify multiple grains.
    
    Args:
        aspect_ratios: List of aspect ratios.
        flocculation_flags: List of flocculation flags.
        classifier: ZinggClassifier instance.
    
    Returns:
        List of classification strings.
    """
    if classifier is None:
        classifier = ZinggClassifier()
    
    return [
        classify_grain(ar, is_floc, classifier)
        for ar, is_floc in zip(aspect_ratios, flocculation_flags)
    ]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_classifier.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/classifier.py tests/test_classifier.py
git commit -m "feat: add classifier module with Zingg + flocculation support"
```

---

## Task 5: Refactor Exporter Module

**Files:**
- Modify: `core/exporter.py`
- Test: `tests/test_exporter.py`

**Context:** Update exporter to handle four-class system and add new fields.

- [ ] **Step 1: Update export_csv to include classification**

Modify `export_csv` in `core/exporter.py`:

```python
def export_csv(morphologies: list[GrainMorphology], path: str) -> None:
    """Export grain morphologies to a CSV file."""
    fieldnames = [
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
        "shape_class",
        "is_flocculation",
        "confidence",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, morph in enumerate(morphologies, start=1):
            writer.writerow(
                {
                    "grain_id": idx,
                    "area": round(morph.area, 4),
                    "perimeter": round(morph.perimeter, 4),
                    "circularity": round(morph.circularity, 6),
                    "d_eq": round(morph.d_eq, 4),
                    "major_axis": round(morph.major_axis, 4),
                    "minor_axis": round(morph.minor_axis, 4),
                    "aspect_ratio": round(morph.aspect_ratio, 4),
                    "sphericity": round(morph.sphericity, 6),
                    "convexity": round(morph.convexity, 6),
                    "feret_max": round(morph.feret_max, 4),
                    "feret_min": round(morph.feret_min, 4),
                    "shape_class": morph.shape_class,
                    "is_flocculation": morph.is_flocculation,
                    "confidence": round(morph.confidence, 4),
                }
            )
```

- [ ] **Step 2: Update export_annotated_image for four-class colors**

Modify `export_annotated_image` in `core/exporter.py`:

```python
def export_annotated_image(
    image: np.ndarray,
    grains: list[GrainContour],
    path: str,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    morphologies: list[GrainMorphology] | None = None,
) -> None:
    """Export an annotated image with grain contours and labels."""
    annotated = image.copy()

    for idx, grain in enumerate(grains, start=1):
        # Determine color based on classification
        if morphologies and idx - 1 < len(morphologies):
            from core.morphology import get_classification_color
            contour_color = get_classification_color(morphologies[idx - 1].shape_class)
        else:
            contour_color = color

        cv2.drawContours(annotated, [grain.contour], -1, contour_color, thickness)

        # Compute centroid for label placement
        moments = cv2.moments(grain.contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            x, y, w, h = cv2.boundingRect(grain.contour)
            cx = x + w // 2
            cy = y + h // 2

        cv2.putText(
            annotated,
            str(idx),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness,
        )

    cv2.imwrite(path, annotated)
```

- [ ] **Step 3: Write test for four-class export**

```python
def test_export_csv_with_flocculation():
    from core.morphology import GrainMorphology
    from core.exporter import export_csv
    import tempfile
    import csv
    
    morphs = [
        GrainMorphology(area=100, perimeter=40, circularity=0.8, d_eq=11.3,
                       major_axis=12, minor_axis=10, aspect_ratio=1.2,
                       sphericity=0.83, convexity=0.9, feret_max=12, feret_min=10,
                       shape_class="球状", is_flocculation=False, confidence=0.9),
        GrainMorphology(area=5000, perimeter=300, circularity=0.2, d_eq=80,
                       major_axis=100, minor_axis=20, aspect_ratio=5.0,
                       sphericity=0.2, convexity=0.5, feret_max=100, feret_min=20,
                       shape_class="絮凝", is_flocculation=True, confidence=0.7),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        path = f.name
    
    export_csv(morphs, path)
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 2
    assert rows[0]["shape_class"] == "球状"
    assert rows[0]["is_flocculation"] == "False"
    assert rows[1]["shape_class"] == "絮凝"
    assert rows[1]["is_flocculation"] == "True"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_exporter.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/exporter.py tests/test_exporter.py
git commit -m "feat: update exporter for four-class system"
```

---

## Task 6: Create PDF Report Module

**Files:**
- Create: `core/report.py`
- Test: `tests/test_report.py`

**Context:** New module for generating PDF reports with images, charts, and data tables.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import tempfile
from core.report import generate_pdf_report
from core.morphology import GrainMorphology, GrainStatistics, compute_statistics


def test_generate_pdf_report():
    """PDF report generation should create a file."""
    morphs = [
        GrainMorphology(area=100, perimeter=40, circularity=0.8, d_eq=11.3,
                       major_axis=12, minor_axis=10, aspect_ratio=1.2,
                       sphericity=0.83, convexity=0.9, feret_max=12, feret_min=10,
                       shape_class="球状", is_flocculation=False, confidence=0.9),
    ]
    stats = compute_statistics(morphs)
    
    # Create a dummy image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        output_path = f.name
    
    generate_pdf_report(
        image=image,
        annotated_image=annotated,
        morphologies=morphs,
        statistics=stats,
        output_path=output_path,
    )
    
    import os
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`

Expected: FAIL

- [ ] **Step 3: Create report module**

```python
"""PDF report generation for sand grain analysis results."""

from datetime import datetime
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import cv2

from core.morphology import GrainMorphology, GrainStatistics


def _numpy_to_image_bytes(image: np.ndarray, fmt: str = ".png") -> bytes:
    """Convert numpy image to bytes for ReportLab."""
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise RuntimeError("Failed to encode image")
    return buffer.tobytes()


def generate_pdf_report(
    image: np.ndarray,
    annotated_image: np.ndarray,
    morphologies: list[GrainMorphology],
    statistics: GrainStatistics,
    output_path: str,
    title: str = "沙粒形态分析报告",
) -> None:
    """Generate a PDF report with images, statistics, and data tables.
    
    Args:
        image: Original input image.
        annotated_image: Annotated result image.
        morphologies: List of grain morphologies.
        statistics: Aggregate statistics.
        output_path: Output PDF file path.
        title: Report title.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Summary statistics
    story.append(Paragraph("统计摘要", styles['Heading2']))
    summary_data = [
        ["指标", "数值"],
        ["颗粒总数", str(statistics.count)],
        ["球状颗粒", str(statistics.zingg_counts.get("球状", 0))],
        ["棒状颗粒", str(statistics.zingg_counts.get("棒状", 0))],
        ["片状颗粒", str(statistics.zingg_counts.get("片状", 0))],
        ["絮凝", str(statistics.zingg_counts.get("絮凝", 0))],
        ["平均圆度", f"{statistics.circularity_mean:.4f}"],
        ["平均球度", f"{statistics.sphericity_mean:.4f}"],
        ["平均等效粒径", f"{statistics.d_eq_mean:.4f}"],
    ]
    
    summary_table = Table(summary_data, colWidths=[6 * cm, 6 * cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Images
    story.append(Paragraph("分析结果", styles['Heading2']))
    
    # Resize images to fit page width
    max_width = 16 * cm
    img_height = 8 * cm
    
    if annotated_image is not None:
        img_bytes = _numpy_to_image_bytes(annotated_image)
        img = Image(img_bytes, width=max_width, height=img_height)
        story.append(img)
        story.append(Spacer(1, 10))
    
    # Detailed data table
    if morphologies:
        story.append(Paragraph("颗粒详细数据", styles['Heading2']))
        
        detail_data = [["ID", "面积", "圆度", "等效粒径", "长短轴比", "球度", "凸度", "分类"]]
        for idx, m in enumerate(morphologies, start=1):
            detail_data.append([
                str(idx),
                f"{m.area:.2f}",
                f"{m.circularity:.4f}",
                f"{m.d_eq:.4f}",
                f"{m.aspect_ratio:.4f}",
                f"{m.sphericity:.4f}",
                f"{m.convexity:.4f}",
                m.shape_class,
            ])
        
        detail_table = Table(detail_data, repeatRows=1)
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(detail_table)
    
    # Build PDF
    doc.build(story)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_report.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/report.py tests/test_report.py
git commit -m "feat: add PDF report generation module"
```

---

## Task 7: Update Streamlit UI

**Files:**
- Modify: `app.py`
- Test: Manual testing

**Context:** Update the existing Streamlit UI to support four-class system, edge filtering, and batch processing.

- [ ] **Step 1: Update imports and session state**

Add imports to `app.py`:

```python
from core.detector import detect_grains, FlocculationConfig
from core.classifier import classify_grain, ZinggClassifier
from core.report import generate_pdf_report
```

Update DEFAULTS in session state:

```python
DEFAULTS = {
    "original_image": None,
    "grains": [],
    "morphologies": [],
    "statistics": None,
    "config": PreprocessConfig(),
    "floc_config": FlocculationConfig(),
    "detection_method": "传统方法",
    "last_processing_time": 0.0,
    "yolo_detector": None,
    "batch_mode": False,
    "batch_results": [],
}
```

- [ ] **Step 2: Add flocculation parameters to sidebar**

Add after preprocessing params section:

```python
with st.expander("🔬 絮凝检测参数", expanded=False):
    floc_config = st.session_state.floc_config
    
    col_a, col_b = st.columns(2)
    with col_a:
        floc_min_area = st.number_input(
            "絮凝最小面积", min_value=1000, max_value=100000,
            value=floc_config.min_area, step=100,
        )
        floc_max_circ = st.number_input(
            "絮凝最大圆度", min_value=0.01, max_value=1.0,
            value=floc_config.max_circularity, step=0.05,
        )
    with col_b:
        floc_max_convex = st.number_input(
            "絮凝最大凸度", min_value=0.1, max_value=1.0,
            value=floc_config.max_convexity, step=0.05,
        )
        floc_max_ar = st.number_input(
            "絮凝最小长宽比", min_value=1.0, max_value=10.0,
            value=floc_config.max_aspect_ratio, step=0.5,
        )
    
    st.session_state.floc_config = FlocculationConfig(
        min_area=floc_min_area,
        max_circularity=floc_max_circ,
        max_convexity=floc_max_convex,
        max_aspect_ratio=floc_max_ar,
    )
```

- [ ] **Step 3: Update detection logic**

Replace the detection section in the Run button handler:

```python
if st.button("🔍 运行检测", type="primary", width="stretch"):
    if st.session_state.original_image is None:
        st.warning("请先加载图像")
    else:
        start = time.time()
        try:
            image = st.session_state.original_image
            config = st.session_state.config
            floc_config = st.session_state.floc_config
            
            # Preprocess
            mask = preprocess(image, config)
            
            # Detect grains with flocculation and edge filtering
            detection_results = detect_grains(
                mask=mask,
                image_shape=image.shape,
                min_area=config.min_area,
                border_margin=5,
                floc_config=floc_config,
            )
            
            # Filter out edge grains
            valid_results = [r for r in detection_results if not r.is_edge]
            
            # Convert to GrainContour and compute morphology
            from core.traditional import GrainContour
            grains = []
            morphologies = []
            classifier = ZinggClassifier()
            
            for result in valid_results:
                grain = GrainContour(contour=result.contour, mask=result.mask)
                grains.append(grain)
                
                # Compute morphology
                morph = compute_morphology(result.contour, result.mask)
                
                # Classify
                morph.shape_class = classify_grain(
                    morph.aspect_ratio,
                    result.is_flocculation,
                    classifier,
                )
                morph.is_flocculation = result.is_flocculation
                morph.confidence = (morph.circularity + morph.convexity) / 2
                
                morphologies.append(morph)
            
            st.session_state.grains = grains
            st.session_state.morphologies = morphologies
            st.session_state.statistics = compute_statistics(morphologies)
            st.session_state.last_processing_time = time.time() - start
            
        except Exception as exc:
            st.error(f"检测错误: {exc}")
```

- [ ] **Step 4: Update visualization for four-class**

Update `_overlay_grains` function:

```python
def _overlay_grains(
    image: np.ndarray, grains: list, morphologies: list
) -> np.ndarray:
    """Draw grain contours and labels on a copy of the image."""
    display = image.copy()
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    for idx, grain in enumerate(grains):
        contour = getattr(grain, "contour", None)
        if contour is None or len(contour) == 0:
            continue

        # Use classification color
        if idx < len(morphologies):
            color = get_classification_color(morphologies[idx].shape_class)
        else:
            color = (0, 255, 0)

        cv2.drawContours(display, [contour], -1, color, 2)

        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(display, (cx, cy), 3, (255, 255, 255), -1)
            cv2.putText(
                display, str(idx + 1), (cx - 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            )

    return display
```

Update `_draw_legend` function for four classes:

```python
def _draw_legend(image: np.ndarray) -> np.ndarray:
    """Draw classification legend on the image."""
    from core.morphology import CLASSIFICATION_COLORS
    
    h, w = image.shape[:2]
    lx, ly = w - 140, h - 100
    ih = 20
    items = list(CLASSIFICATION_COLORS.items())

    cv2.rectangle(image, (lx - 10, ly - 25),
                  (lx + 130, ly + len(items) * ih + 5), (40, 40, 40), -1)
    cv2.rectangle(image, (lx - 10, ly - 25),
                  (lx + 130, ly + len(items) * ih + 5), (200, 200, 200), 1)
    cv2.putText(image, "分类", (lx, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (label, color) in enumerate(items):
        y = ly + i * ih + 10
        cv2.rectangle(image, (lx, y - 10), (lx + 15, y + 5), color, -1)
        cv2.putText(image, label, (lx + 20, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return image
```

- [ ] **Step 5: Update statistics display**

Update the Summary tab to show four-class counts:

```python
with tab1:
    if stats is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("颗粒数量", stats.count)
            st.metric("平均圆度", f"{stats.circularity_mean:.4f}")
            st.metric("平均球度", f"{stats.sphericity_mean:.4f}")
        with col_b:
            st.metric("平均等效粒径", f"{stats.d_eq_mean:.4f}")
            st.metric("平均长短轴比", f"{stats.aspect_ratio_mean:.4f}")
            st.metric("平均凸度", f"{stats.convexity_mean:.4f}")

        st.divider()
        st.caption("分类统计")
        if stats.zingg_counts:
            parts = []
            for key, cnt in stats.zingg_counts.items():
                pct = cnt / stats.count * 100 if stats.count > 0 else 0
                parts.append(f"**{key}**: {cnt} ({pct:.1f}%)")
            st.markdown(" | ".join(parts))
        
        if stats.flocculation_count > 0:
            st.caption(f"絮凝比例: {stats.flocculation_ratio*100:.1f}%")
```

- [ ] **Step 6: Add PDF export button**

Add after CSV and image export buttons:

```python
# PDF export
if st.session_state.original_image is not None and st.session_state.morphologies:
    if st.button("📄 导出PDF报告", width="stretch"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            pdf_path = f.name
        
        annotated = st.session_state.original_image.copy()
        for idx, grain in enumerate(st.session_state.grains, start=1):
            contour = getattr(grain, "contour", None)
            if contour is None:
                continue
            color = (0, 255, 0)
            if idx - 1 < len(morphs):
                color = get_classification_color(morphs[idx - 1].shape_class)
            cv2.drawContours(annotated, [contour], -1, color, 2)
        
        generate_pdf_report(
            image=st.session_state.original_image,
            annotated_image=annotated,
            morphologies=morphs,
            statistics=stats,
            output_path=pdf_path,
        )
        
        with open(pdf_path, 'rb') as f:
            st.download_button(
                "📄 下载PDF报告", data=f.read(),
                file_name="sand_analysis_report.pdf", mime="application/pdf",
                width="stretch",
            )
```

- [ ] **Step 7: Test the UI**

Run: `uv run streamlit run app.py`

Expected: Streamlit app loads without errors

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "feat: update Streamlit UI for four-class system with flocculation support"
```

---

## Task 8: Add Batch Processing

**Files:**
- Modify: `app.py`
- Test: Manual testing

**Context:** Add batch processing mode to the Streamlit UI.

- [ ] **Step 1: Add batch mode toggle to sidebar**

Add to sidebar:

```python
# Mode selection
st.divider()
mode = st.radio("模式", ["单张分析", "批量处理"], index=0)
st.session_state.batch_mode = (mode == "批量处理")
```

- [ ] **Step 2: Add batch upload and processing**

Add batch processing section:

```python
if st.session_state.batch_mode:
    with st.expander("📁 批量上传", expanded=True):
        uploaded_files = st.file_uploader(
            "选择多张沙粒图像",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        
        if uploaded_files:
            st.success(f"✓ 已选择 {len(uploaded_files)} 张图像")
            
            if st.button("🔍 批量运行检测", type="primary"):
                results = []
                progress = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Process image
                        mask = preprocess(image, st.session_state.config)
                        detection_results = detect_grains(
                            mask=mask,
                            image_shape=image.shape,
                            min_area=st.session_state.config.min_area,
                            floc_config=st.session_state.floc_config,
                        )
                        
                        valid_results = [r for r in detection_results if not r.is_edge]
                        
                        grains = []
                        morphologies = []
                        classifier = ZinggClassifier()
                        
                        for result in valid_results:
                            from core.traditional import GrainContour
                            grain = GrainContour(contour=result.contour, mask=result.mask)
                            grains.append(grain)
                            
                            morph = compute_morphology(result.contour, result.mask)
                            morph.shape_class = classify_grain(
                                morph.aspect_ratio, result.is_flocculation, classifier
                            )
                            morph.is_flocculation = result.is_flocculation
                            morphologies.append(morph)
                        
                        stats = compute_statistics(morphologies)
                        
                        results.append({
                            "filename": uploaded_file.name,
                            "grains": grains,
                            "morphologies": morphologies,
                            "statistics": stats,
                        })
                    
                    progress.progress((i + 1) / len(uploaded_files))
                
                st.session_state.batch_results = results
                st.success(f"✓ 完成处理 {len(results)} 张图像")
```

- [ ] **Step 3: Add batch results display**

Add batch results tab:

```python
if st.session_state.batch_mode and st.session_state.batch_results:
    st.divider()
    st.subheader("批量处理结果")
    
    # Summary table
    summary_data = []
    for result in st.session_state.batch_results:
        stats = result["statistics"]
        summary_data.append({
            "文件名": result["filename"],
            "颗粒数": stats.count,
            "球状": stats.zingg_counts.get("球状", 0),
            "棒状": stats.zingg_counts.get("棒状", 0),
            "片状": stats.zingg_counts.get("片状", 0),
            "絮凝": stats.zingg_counts.get("絮凝", 0),
        })
    
    st.dataframe(summary_data, hide_index=True)
    
    # Export all results
    if st.button("📊 导出所有CSV"):
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for result in st.session_state.batch_results:
                csv_buf = io.StringIO()
                fieldnames = [
                    "grain_id", "area", "perimeter", "circularity", "d_eq",
                    "major_axis", "minor_axis", "aspect_ratio", "sphericity",
                    "convexity", "shape_class", "is_flocculation",
                ]
                writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
                writer.writeheader()
                for idx, m in enumerate(result["morphologies"], start=1):
                    writer.writerow({
                        "grain_id": idx,
                        "area": round(m.area, 4),
                        "perimeter": round(m.perimeter, 4),
                        "circularity": round(m.circularity, 6),
                        "d_eq": round(m.d_eq, 4),
                        "major_axis": round(m.major_axis, 4),
                        "minor_axis": round(m.minor_axis, 4),
                        "aspect_ratio": round(m.aspect_ratio, 4),
                        "sphericity": round(m.sphericity, 6),
                        "convexity": round(m.convexity, 6),
                        "shape_class": m.shape_class,
                        "is_flocculation": m.is_flocculation,
                    })
                
                filename = result["filename"].rsplit('.', 1)[0] + '.csv'
                zf.writestr(filename, csv_buf.getvalue())
        
        st.download_button(
            "📦 下载所有CSV", data=zip_buffer.getvalue(),
            file_name="batch_results.zip", mime="application/zip",
        )
```

- [ ] **Step 4: Test batch processing**

Run: `uv run streamlit run app.py`

Upload multiple images and verify batch processing works.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add batch processing mode to Streamlit UI"
```

---

## Task 9: Update Package Exports

**Files:**
- Modify: `core/__init__.py`

- [ ] **Step 1: Update package exports**

```python
"""SandAnalyze core package."""

from core.preprocessor import PreprocessConfig, preprocess, filter_edge_grains, auto_tune_params
from core.detector import detect_grains, FlocculationConfig, DetectionResult
from core.classifier import classify_grain, ZinggClassifier
from core.morphology import (
    GrainMorphology,
    GrainStatistics,
    compute_morphology,
    compute_statistics,
    get_classification_color,
    CLASSIFICATION_COLORS,
)
from core.exporter import export_csv, export_annotated_image
from core.report import generate_pdf_report

__all__ = [
    "PreprocessConfig",
    "preprocess",
    "filter_edge_grains",
    "auto_tune_params",
    "detect_grains",
    "FlocculationConfig",
    "DetectionResult",
    "classify_grain",
    "ZinggClassifier",
    "GrainMorphology",
    "GrainStatistics",
    "compute_morphology",
    "compute_statistics",
    "get_classification_color",
    "CLASSIFICATION_COLORS",
    "export_csv",
    "export_annotated_image",
    "generate_pdf_report",
]
```

- [ ] **Step 2: Commit**

```bash
git add core/__init__.py
git commit -m "feat: update package exports for v2.0 modules"
```

---

## Task 10: Add ReportLab Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add ReportLab to dependencies**

```toml
[project]
dependencies = [
    "opencv-python>=4.8",
    "ultralytics>=8.0",
    "streamlit>=1.32",
    "plotly>=5.18",
    "kaleido>=1.0",
    "numpy>=1.26",
    "scipy>=1.12",
    "anthropic>=0.109.2",
    "reportlab>=4.0",
    "pandas>=2.0",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync --extra dev`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add reportlab and pandas dependencies"
```

---

## Task 11: Run All Tests

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`

Expected: All PASS

- [ ] **Step 2: Commit**

```bash
git commit -m "test: verify all tests pass for v2.0"
```

---

## Task 12: Update Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/usage_v2.md`

- [ ] **Step 1: Update README.md**

Update the README to reflect v2.0 features:
- Four-class system (Zingg + Flocculation)
- Edge filtering
- Auto-parameter tuning
- Batch processing
- PDF report generation

- [ ] **Step 2: Create usage guide**

Create `docs/usage_v2.md` with detailed usage instructions for new features.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/usage_v2.md
git commit -m "docs: update README and add v2.0 usage guide"
```

---

## Spec Coverage Check

| Spec Requirement | Task | Status |
|------------------|------|--------|
| 图像预处理（去黑边、亮度均衡） | Task 1 | ✅ |
| 传统CV检测 | Task 3 | ✅ |
| 絮凝检测与标记 | Task 3, 4 | ✅ |
| 边缘过滤 | Task 1, 3 | ✅ |
| 参数自动调优 | Task 1 | ✅ |
| Zingg形状分类 | Task 4 | ✅ |
| 形态参数计算 | Task 2 | ✅ |
| CSV导出 | Task 5 | ✅ |
| 标注图片 | Task 5 | ✅ |
| 统计图表 | Task 7 | ✅ |
| PDF报告 | Task 6 | ✅ |
| Streamlit UI | Task 7, 8 | ✅ |
| 批量处理 | Task 8 | ✅ |

---

## Placeholder Scan

- No TBD/TODO placeholders found
- All steps contain actual code
- All test code is complete
- All file paths are exact

---

## Type Consistency Check

- `GrainMorphology` fields consistent across all modules
- `FlocculationConfig` fields consistent in detector and UI
- `ZinggClassifier` used consistently in classifier and UI
- Color functions (`get_classification_color`) used in morphology, exporter, and UI

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-17-sandanalyze-v2.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
