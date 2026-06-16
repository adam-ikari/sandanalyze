# SandAnalyze 沙粒形态分析系统 - 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个桌面 GUI 应用，对光学显微镜拍摄的沙粒图像进行颗粒识别、多维度形态参数计算和数量统计。

**Architecture:** 传统 OpenCV 图像处理（轮廓检测 + 分水岭分割）作为默认检测方法，YOLOv8-seg 作为可选对比。PyQt6 桌面 GUI 展示标注图、统计图表和参数表格。核心处理逻辑与 GUI 解耦，core/ 模块可独立测试。

**Tech Stack:** Python 3.13, OpenCV, ultralytics (YOLOv8-seg), PyQt6, matplotlib, numpy, scipy

---

## File Structure

```
sandanalyze/
├── core/
│   ├── __init__.py              # 导出核心类
│   ├── preprocessor.py          # PreprocessConfig dataclass + preprocess() 函数
│   ├── traditional.py           # detect_grains() 传统轮廓检测
│   ├── yolo_detector.py         # YOLODetector 类
│   ├── morphology.py            # GrainMorphology dataclass + compute_morphology() + compute_statistics()
│   └── exporter.py              # export_csv() + export_annotated_image() + export_pdf_report()
├── gui/
│   ├── __init__.py
│   ├── app.py                   # SandAnalyzeApp 主窗口
│   ├── image_panel.py           # ImagePanel 图像显示与交互
│   ├── result_panel.py          # ResultPanel 统计结果 + 图表
│   └── settings_panel.py        # SettingsPanel 参数设置
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # 共享 fixtures
│   ├── test_preprocessor.py
│   ├── test_traditional.py
│   ├── test_morphology.py
│   ├── test_exporter.py
│   └── test_integration.py
├── models/                      # YOLO 模型文件（.gitkeep）
├── data/                        # 示例图像（.gitkeep）
├── main.py                      # 应用入口
├── pyproject.toml               # 依赖配置
└── README.md
```

---

### Task 1: 项目依赖与基础结构

**Files:**
- Modify: `pyproject.toml`
- Create: `core/__init__.py`
- Create: `core/preprocessor.py`
- Create: `models/.gitkeep`
- Create: `data/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 更新 pyproject.toml 添加依赖**

```toml
[project]
name = "sandanalyze"
version = "0.1.0"
description = "沙粒形态分析系统 - 基于OpenCV和YOLO的沙粒识别与统计"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "opencv-python>=4.8",
    "ultralytics>=8.0",
    "PyQt6>=6.6",
    "matplotlib>=3.8",
    "numpy>=1.26",
    "scipy>=1.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
]

[project.scripts]
sandanalyze = "main:main"
```

- [ ] **Step 2: 安装依赖**

Run: `cd /Users/dadi/Project/sandanalyze && uv sync --extra dev`
Expected: 依赖安装成功

- [ ] **Step 3: 创建目录结构和空文件**

```bash
mkdir -p core gui models data tests
touch core/__init__.py gui/__init__.py tests/__init__.py
touch models/.gitkeep data/.gitkeep
```

- [ ] **Step 4: 创建 tests/conftest.py 共享 fixtures**

```python
"""Shared test fixtures for sandanalyze."""
import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_grain_image():
    """生成一张模拟沙粒图像：深灰背景上多个亮色椭圆颗粒。"""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[:] = 40  # 深灰背景

    # 画3个不同大小的椭圆模拟沙粒
    img = cv2.ellipse(img, (50, 50), (15, 10), 30, 0, 360, 180, -1)
    img = cv2.ellipse(img, (130, 60), (20, 12), -20, 0, 360, 180, -1)
    img = cv2.ellipse(img, (80, 140), (12, 12), 0, 0, 360, 180, -1)
    return img


@pytest.fixture
def overlapping_grain_image():
    """生成一张有粘连颗粒的图像，用于测试分水岭。"""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[:] = 40
    # 两个靠近的椭圆模拟粘连颗粒
    img = cv2.ellipse(img, (60, 100), (20, 12), 0, 0, 360, 180, -1)
    img = cv2.ellipse(img, (100, 100), (20, 12), 0, 0, 360, 180, -1)
    return img


@pytest.fixture
def real_sand_image_path():
    """返回项目中的真实沙粒图像路径。"""
    return "Sand_from_Gobi_Desert.jpg"
```

- [ ] **Step 5: 创建 core/__init__.py**

```python
"""SandAnalyze core processing modules."""
```

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml core/ gui/ models/ data/ tests/ .gitignore
git commit -m "feat: project structure and dependencies"
```

---

### Task 2: 图像预处理模块 (core/preprocessor.py)

**Files:**
- Create: `core/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: 编写预处理器测试**

```python
"""Tests for core/preprocessor.py."""
import numpy as np
import pytest
from core.preprocessor import PreprocessConfig, preprocess


class TestPreprocessConfig:
    def test_default_values(self):
        cfg = PreprocessConfig()
        assert cfg.blur_kernel == 5
        assert cfg.adaptive_block_size == 11
        assert cfg.adaptive_c == 2
        assert cfg.morph_kernel_size == 3
        assert cfg.min_area == 50
        assert cfg.use_clahe is False
        assert cfg.use_watershed is True

    def test_custom_values(self):
        cfg = PreprocessConfig(blur_kernel=7, min_area=100)
        assert cfg.blur_kernel == 7
        assert cfg.min_area == 100


class TestPreprocess:
    def test_output_is_binary(self, sample_grain_image):
        cfg = PreprocessConfig(use_watershed=False)
        result = preprocess(sample_grain_image, cfg)
        assert result.dtype == np.uint8
        unique = np.unique(result)
        assert set(unique).issubset({0, 1})
        # 应该有前景和背景
        assert 0 in unique
        assert 1 in unique

    def test_clahe_enhancement(self, sample_grain_image):
        cfg = PreprocessConfig(use_clahe=True, use_watershed=False)
        result = preprocess(sample_grain_image, cfg)
        assert result.dtype == np.uint8
        assert 1 in np.unique(result)

    def test_watershed_separates_touching_grains(self, overlapping_grain_image):
        cfg = PreprocessConfig(use_watershed=True)
        result = preprocess(overlapping_grain_image, cfg)
        assert result.dtype == np.uint8
        # 分水岭后应该有不同标签（>1 个前景标签）
        labels = np.unique(result)
        foreground_labels = labels[labels > 0]
        assert len(foreground_labels) >= 2

    def test_grayscale_input(self):
        """灰度图直接处理。"""
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[30:70, 30:70] = 200
        result = preprocess(gray, PreprocessConfig(use_watershed=False))
        assert 1 in np.unique(result)

    def test_color_input_converted(self):
        """彩色图自动转灰度。"""
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        color[30:70, 30:70] = [200, 200, 200]
        result = preprocess(color, PreprocessConfig(use_watershed=False))
        assert 1 in np.unique(result)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_preprocessor.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现预处理模块**

```python
"""Image preprocessing for sand grain analysis.

Pipeline: grayscale → CLAHE (optional) → blur → adaptive threshold
→ morphological operations → watershed (optional) → binary mask.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Configuration for image preprocessing pipeline."""

    blur_kernel: int = 5
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    morph_kernel_size: int = 3
    morph_open_iter: int = 2
    morph_close_iter: int = 2
    min_area: int = 50
    use_clahe: bool = False
    use_watershed: bool = True
    watershed_thresh_ratio: float = 0.5


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_threshold(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Apply adaptive thresholding to get binary image."""
    blurred = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0)
    block = cfg.adaptive_block_size
    if block % 2 == 0:
        block += 1
    if block < 3:
        block = 3
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, cfg.adaptive_c,
    )
    return binary


def _morphological_cleanup(binary: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Apply morphological open + close to clean up binary image."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.morph_kernel_size, cfg.morph_kernel_size),
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_open_iter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_close_iter)
    return closed


def _apply_watershed(binary: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Apply watershed algorithm to separate touching grains.

    Returns a label image where 0=background, 1..N=grain labels.
    """
    # Distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Find sure foreground peaks
    _, sure_fg = cv2.threshold(
        dist, cfg_thresh_factor * dist.max(), 255, cv2.THRESH_BINARY,
    )
    # We use a ratio of the max distance as threshold
    # This is handled inside the function via a local ratio
    thresh_val = dist.max() * 0.5
    _, sure_fg = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    sure_bg = cv2.dilate(binary, None, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label connected components
    num_labels, markers = cv2.connectedComponents(sure_fg)
    # Shift labels so background is 0 (not -1 for watershed later)
    # But we use markers directly: 0=unknown, 1..N=grains
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed needs a 3-channel image
    if len(gray.shape) == 2:
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        color = gray.copy()
    markers = cv2.watershed(color, markers)

    # Build output: 0=background, positive=grain labels
    result = np.zeros(gray.shape[:2], dtype=np.uint8)
    for label_id in range(2, markers.max() + 1):
        result[markers == label_id] = label_id - 1

    return result


def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Run the full preprocessing pipeline.

    Args:
        image: Input image (grayscale or BGR color).
        config: Preprocessing configuration. Uses defaults if None.

    Returns:
        Binary mask (0=background, 1..N=grain labels) if watershed enabled,
        otherwise binary mask (0=background, 1=foreground).
    """
    if config is None:
        config = PreprocessConfig()

    gray = _to_grayscale(image)

    if config.use_clahe:
        gray = _apply_clahe(gray)

    binary = _adaptive_threshold(gray, config)
    binary = _morphological_cleanup(binary, config)

    if config.use_watershed:
        return _apply_watershed(binary, gray)

    # Non-watershed: return simple binary (0 or 1)
    return (binary > 0).astype(np.uint8)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_preprocessor.py -v`
Expected: All tests PASS

- [ ] **Step 5: 提交**

```bash
git add core/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: image preprocessing module with adaptive threshold and watershed"
```

---

### Task 3: 传统检测模块 (core/traditional.py)

**Files:**
- Create: `core/traditional.py`
- Create: `tests/test_traditional.py`

- [ ] **Step 1: 编写传统检测测试**

```python
"""Tests for core/traditional.py."""
import numpy as np
import pytest
from core.preprocessor import PreprocessConfig, preprocess
from core.traditional import detect_grains, GrainContour


class TestDetectGrains:
    def test_finds_grains_in_sample_image(self, sample_grain_image):
        cfg = PreprocessConfig(use_watershed=True)
        mask = preprocess(sample_grain_image, cfg)
        grains = detect_grains(mask, min_area=50)
        assert len(grains) >= 2

    def test_grain_has_contour_and_mask(self, sample_grain_image):
        cfg = PreprocessConfig(use_watershed=True)
        mask = preprocess(sample_grain_image, cfg)
        grains = detect_grains(mask, min_area=50)
        for g in grains:
            assert g.contour is not None
            assert g.mask is not None
            assert g.contour.shape[1] == 2  # Nx2 points
            assert g.mask.dtype == np.uint8
            assert np.any(g.mask > 0)

    def test_min_area_filter(self, sample_grain_image):
        cfg = PreprocessConfig(use_watershed=True)
        mask = preprocess(sample_grain_image, cfg)
        grains_all = detect_grains(mask, min_area=10)
        grains_large = detect_grains(mask, min_area=500)
        assert len(grains_large) <= len(grains_all)

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        grains = detect_grains(mask, min_area=50)
        assert len(grains) == 0

    def test_grain_mask_size_matches_image(self, sample_grain_image):
        cfg = PreprocessConfig(use_watershed=False)
        mask = preprocess(sample_grain_image, cfg)
        grains = detect_grains(mask, min_area=50)
        for g in grains:
            assert g.mask.shape == sample_grain_image.shape[:2]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_traditional.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现传统检测模块**

```python
"""Traditional OpenCV-based grain detection using contour analysis.

Works with the binary mask output from preprocessor.
Handles both watershed label masks and simple binary masks.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GrainContour:
    """Detected grain with its contour and mask."""

    contour: np.ndarray  # Nx1x2 contour points from cv2.findContours
    mask: np.ndarray     # Binary mask (same size as input image)


def detect_grains(mask: np.ndarray, min_area: int = 50) -> list[GrainContour]:
    """Detect grain contours from a preprocessed mask.

    For watershed masks (labels 1..N), extracts contours per label.
    For binary masks (0 and 1), extracts all contours at once.

    Args:
        mask: Preprocessed mask from preprocess(). Watershed labels or binary.
        min_area: Minimum contour area in pixels to keep.

    Returns:
        List of GrainContour objects, sorted by area descending.
    """
    h, w = mask.shape[:2]
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # skip background

    grains: list[GrainContour] = []

    if len(unique_labels) > 1:
        # Watershed mask: process each label separately
        for label_id in unique_labels:
            label_mask = (mask == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    grain_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(grain_mask, [cnt], -1, 255, -1)
                    grains.append(GrainContour(contour=cnt, mask=grain_mask))
    else:
        # Simple binary mask
        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                grain_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(grain_mask, [cnt], -1, 255, -1)
                grains.append(GrainContour(contour=cnt, mask=grain_mask))

    # Sort by area descending
    grains.sort(key=lambda g: cv2.contourArea(g.contour), reverse=True)
    return grains
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_traditional.py -v`
Expected: All tests PASS

- [ ] **Step 5: 提交**

```bash
git add core/traditional.py tests/test_traditional.py
git commit -m "feat: traditional grain detection module with contour extraction"
```

---

### Task 4: 形态参数计算模块 (core/morphology.py)

**Files:**
- Create: `core/morphology.py`
- Create: `tests/test_morphology.py`

- [ ] **Step 1: 编写形态参数测试**

```python
"""Tests for core/morphology.py."""
import math

import numpy as np
import pytest
from core.morphology import (
    GrainMorphology,
    GrainStatistics,
    compute_morphology,
    compute_statistics,
)


class TestComputeMorphology:
    def test_circle_grain(self):
        """圆形颗粒应有圆度接近1。"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask = cv2_ellipse(mask, (50, 50), (20, 20), 0)
        contour = _mask_to_contour(mask)
        m = compute_morphology(contour, mask)
        assert m.circularity > 0.85
        assert m.aspect_ratio < 1.2

    def test_elongated_grain(self):
        """细长颗粒应有较大长短轴比。"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask = cv2_ellipse(mask, (50, 50), (30, 8), 0)
        contour = _mask_to_contour(mask)
        m = compute_morphology(contour, mask)
        assert m.aspect_ratio > 2.0
        assert m.circularity < 0.8

    def test_area_matches_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask = cv2_ellipse(mask, (50, 50), (15, 10), 0)
        contour = _mask_to_contour(mask)
        m = compute_morphology(contour, mask)
        expected_area = np.count_nonzero(mask)
        assert abs(m.area - expected_area) < expected_area * 0.05

    def test_sphericity_is_inverse_of_aspect_ratio(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask = cv2_ellipse(mask, (50, 50), (20, 10), 0)
        contour = _mask_to_contour(mask)
        m = compute_morphology(contour, mask)
        assert abs(m.sphericity - 1.0 / m.aspect_ratio) < 0.05

    def test_convexity_for_smooth_shape(self):
        """光滑形状凸度应接近1。"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask = cv2_ellipse(mask, (50, 50), (15, 15), 0)
        contour = _mask_to_contour(mask)
        m = compute_morphology(contour, mask)
        assert m.convexity > 0.9


class TestComputeStatistics:
    def test_statistics_from_multiple_grains(self):
        morphologies = []
        for r in [(15, 15), (20, 10), (10, 8)]:
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask = cv2_ellipse(mask, (50, 50), r, 0)
            contour = _mask_to_contour(mask)
            morphologies.append(compute_morphology(contour, mask))

        stats = compute_statistics(morphologies)
        assert stats.count == 3
        assert stats.circularity_mean > 0
        assert stats.circularity_std >= 0
        assert stats.d_eq_median > 0

    def test_zingg_classification(self):
        """测试 Zingg 分类计数。"""
        morphologies = []
        # 一个接近圆的（球状）
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask = cv2_ellipse(mask, (50, 50), (15, 15), 0)
        contour = _mask_to_contour(mask)
        morphologies.append(compute_morphology(contour, mask))

        # 一个细长的（棒状）
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = cv2_ellipse(mask2, (50, 50), (30, 6), 0)
        contour2 = _mask_to_contour(mask2)
        morphologies.append(compute_morphology(contour2, mask2))

        stats = compute_statistics(morphologies)
        total = sum(stats.zingg_counts.values())
        assert total == 2


# --- Test helpers ---

def cv2_ellipse(img, center, axes, angle):
    """Draw filled ellipse on image, return modified image."""
    import cv2
    return cv2.ellipse(img, center, axes, angle, 0, 360, 255, -1)


def _mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Extract the largest external contour from a binary mask."""
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_morphology.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现形态参数模块**

```python
"""Morphological parameter computation for sand grains.

Computes multi-dimensional shape descriptors from grain contours and masks.
Supports geological analysis including Zingg classification.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class GrainMorphology:
    """Morphological parameters for a single grain."""

    area: float                    # Pixel area
    perimeter: float               # Pixel perimeter length
    circularity: float             # 4πA/P², 1=perfect circle
    d_eq: float                    # Equivalent diameter √(4A/π)
    major_axis: float              # Major axis of min enclosing ellipse
    minor_axis: float              # Minor axis of min enclosing ellipse
    aspect_ratio: float            # major/minor
    sphericity: float              # minor/major (Sneed & Folk)
    convexity: float               # area / convex_hull_area
    feret_max: float               # Maximum Feret diameter
    feret_min: float               # Minimum Feret diameter


@dataclass
class GrainStatistics:
    """Aggregate statistics across all detected grains."""

    count: int
    # Per-parameter statistics
    area_mean: float
    area_std: float
    area_median: float
    circularity_mean: float
    circularity_std: float
    circularity_median: float
    d_eq_mean: float
    d_eq_std: float
    d_eq_median: float
    aspect_ratio_mean: float
    aspect_ratio_std: float
    aspect_ratio_median: float
    sphericity_mean: float
    sphericity_std: float
    sphericity_median: float
    convexity_mean: float
    convexity_std: float
    convexity_median: float
    # Zingg classification counts
    zingg_counts: dict = field(default_factory=dict)
    # Raw data for plotting
    d_eq_values: list = field(default_factory=list)
    circularity_values: list = field(default_factory=list)
    sphericity_values: list = field(default_factory=list)


def compute_morphology(contour: np.ndarray, mask: np.ndarray) -> GrainMorphology:
    """Compute morphological parameters for a single grain.

    Args:
        contour: Nx1x2 contour array from cv2.findContours.
        mask: Binary mask of the grain (same size as original image).

    Returns:
        GrainMorphology with all shape descriptors.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Circularity: 4πA/P²
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # Equivalent diameter
    d_eq = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    # Min enclosing ellipse
    ellipse = cv2.fitEllipse(contour)
    major_axis = max(ellipse[1]) / 2.0   # ellipse[1] = (width, height), semi-axes
    minor_axis = min(ellipse[1]) / 2.0

    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0
    sphericity = minor_axis / major_axis if major_axis > 0 else 0.0

    # Convexity
    hull = cv2.convexHull(contour)
    convex_area = cv2.contourArea(hull)
    convexity = area / convex_area if convex_area > 0 else 0.0

    # Feret diameters via rotating calipers approximation
    feret_max, feret_min = _compute_feret(contour)

    return GrainMorphology(
        area=area,
        perimeter=perimeter,
        circularity=min(circularity, 1.0),
        d_eq=d_eq,
        major_axis=major_axis,
        minor_axis=minor_axis,
        aspect_ratio=aspect_ratio,
        sphericity=sphericity,
        convexity=convexity,
        feret_max=feret_max,
        feret_min=feret_min,
    )


def _compute_feret(contour: np.ndarray) -> tuple[float, float]:
    """Compute max and min Feret diameters by rotating calipers approximation.

    Samples rotations at 1-degree intervals.
    """
    contour_squeezed = contour.squeeze()
    if contour_squeezed.ndim < 2 or len(contour_squeezed) < 3:
        return 0.0, 0.0

    max_feret = 0.0
    min_feret = float("inf")

    for angle in range(0, 180):
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotated = contour_squeezed @ np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        width = rotated[:, 0].max() - rotated[:, 0].min()
        height = rotated[:, 1].max() - rotated[:, 1].min()
        feret = max(width, height)
        feret_perp = min(width, height)
        if feret > max_feret:
            max_feret = feret
        if feret_perp < min_feret and feret_perp > 0:
            min_feret = feret_perp

    if min_feret == float("inf"):
        min_feret = 0.0
    return max_feret, min_feret


def _zingg_classify(morphology: GrainMorphology) -> str:
    """Classify grain using Zingg (1935) shape classification.

    Based on aspect ratio (disc/blade threshold = 2/3, blade/roller = 2/3).
    Simplified 2D version using aspect ratio only:
    - 球状 (Spherical): AR < 1.5
    - 圆盘状 (Disc): AR < 1.5 would be disc if we had flatness; simplified as spherical
    - 棒状 (Blade/Roller): 1.5 <= AR < 2.5
    - 片状 (Flat): AR >= 2.5
    """
    ar = morphology.aspect_ratio
    if ar < 1.5:
        return "球状"
    elif ar < 2.5:
        return "棒状"
    else:
        return "片状"


def compute_statistics(morphologies: list[GrainMorphology]) -> GrainStatistics:
    """Compute aggregate statistics across all grains.

    Args:
        morphologies: List of GrainMorphology objects.

    Returns:
        GrainStatistics with aggregate measures and Zingg classification.
    """
    if not morphologies:
        return GrainStatistics(
            count=0,
            area_mean=0, area_std=0, area_median=0,
            circularity_mean=0, circularity_std=0, circularity_median=0,
            d_eq_mean=0, d_eq_std=0, d_eq_median=0,
            aspect_ratio_mean=0, aspect_ratio_std=0, aspect_ratio_median=0,
            sphericity_mean=0, sphericity_std=0, sphericity_median=0,
            convexity_mean=0, convexity_std=0, convexity_median=0,
        )

    def _stats(values: list[float]) -> tuple[float, float, float]:
        arr = np.array(values)
        return float(arr.mean()), float(arr.std()), float(np.median(arr))

    area_m, area_s, area_med = _stats([m.area for m in morphologies])
    circ_m, circ_s, circ_med = _stats([m.circularity for m in morphologies])
    deq_m, deq_s, deq_med = _stats([m.d_eq for m in morphologies])
    ar_m, ar_s, ar_med = _stats([m.aspect_ratio for m in morphologies])
    sp_m, sp_s, sp_med = _stats([m.sphericity for m in morphologies])
    cvx_m, cvx_s, cvx_med = _stats([m.convexity for m in morphologies])

    zingg_counts: dict[str, int] = {}
    for m in morphologies:
        cls = _zingg_classify(m)
        zingg_counts[cls] = zingg_counts.get(cls, 0) + 1

    return GrainStatistics(
        count=len(morphologies),
        area_mean=area_m, area_std=area_s, area_median=area_med,
        circularity_mean=circ_m, circularity_std=circ_s, circularity_median=circ_med,
        d_eq_mean=deq_m, d_eq_std=deq_s, d_eq_median=deq_med,
        aspect_ratio_mean=ar_m, aspect_ratio_std=ar_s, aspect_ratio_median=ar_med,
        sphericity_mean=sp_m, sphericity_std=sp_s, sphericity_median=sp_med,
        convexity_mean=cvx_m, convexity_std=cvx_s, convexity_median=cvx_med,
        zingg_counts=zingg_counts,
        d_eq_values=[m.d_eq for m in morphologies],
        circularity_values=[m.circularity for m in morphologies],
        sphericity_values=[m.sphericity for m in morphologies],
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_morphology.py -v`
Expected: All tests PASS

- [ ] **Step 5: 提交**

```bash
git add core/morphology.py tests/test_morphology.py
git commit -m "feat: morphology computation with Zingg classification"
```

---

### Task 5: YOLO 检测模块 (core/yolo_detector.py)

**Files:**
- Create: `core/yolo_detector.py`
- Create: `tests/test_yolo_detector.py`

- [ ] **Step 1: 编写 YOLO 检测测试**

```python
"""Tests for core/yolo_detector.py."""
import numpy as np
import pytest
from core.traditional import GrainContour
from core.yolo_detector import YOLODetector


class TestYOLODetector:
    def test_init_without_model_creates_detector(self):
        """即使没有下载模型，初始化也不应崩溃。"""
        detector = YOLODetector()
        assert detector is not None

    def test_detect_without_model_returns_empty(self):
        """没有模型时应返回空列表而非崩溃。"""
        detector = YOLODetector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        grains = detector.detect(img)
        assert isinstance(grains, list)

    def test_detect_returns_grain_contours(self):
        """检测返回类型应为 GrainContour 列表。"""
        detector = YOLODetector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        grains = detector.detect(img)
        for g in grains:
            assert isinstance(g, GrainContour)

    def test_is_available_without_model(self):
        detector = YOLODetector()
        # 模型未下载时 is_available 应为 False 或 True（如果自动下载了）
        assert isinstance(detector.is_available, bool)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_yolo_detector.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现 YOLO 检测模块**

```python
"""YOLOv8-seg based grain detection.

Uses ultralytics YOLOv8 segmentation model as an optional
alternative to traditional contour detection.
"""
import logging

import cv2
import numpy as np

from core.traditional import GrainContour

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLOv8-seg grain detector."""

    def __init__(self, model_name: str = "yolov8n-seg.pt"):
        self._model = None
        self._model_name = model_name
        self._load_attempted = False

    @property
    def is_available(self) -> bool:
        """Check if YOLO model is loaded and available."""
        if self._model is not None:
            return True
        if not self._load_attempted:
            self._try_load()
        return self._model is not None

    def _try_load(self) -> None:
        """Attempt to load the YOLO model."""
        self._load_attempted = True
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_name)
            logger.info("YOLO model %s loaded successfully", self._model_name)
        except Exception as e:
            logger.warning("Failed to load YOLO model %s: %s", self._model_name, e)
            self._model = None

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        min_area: int = 50,
    ) -> list[GrainContour]:
        """Detect grains using YOLOv8-seg.

        Args:
            image: Input BGR image.
            conf: Confidence threshold.
            min_area: Minimum mask area in pixels.

        Returns:
            List of GrainContour objects from YOLO segmentation.
        """
        if not self.is_available:
            logger.warning("YOLO model not available, returning empty results")
            return []

        results = self._model(image, conf=conf, verbose=False)
        if not results:
            return []

        h, w = image.shape[:2]
        grains: list[GrainContour] = []

        for result in results:
            if result.masks is None:
                continue
            masks = result.masks.data.cpu().numpy()
            for mask_data in masks:
                # Resize mask to image size
                binary_mask = cv2.resize(
                    mask_data.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                if np.count_nonzero(binary_mask) < min_area:
                    continue

                contours, _ = cv2.findContours(
                    binary_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                for cnt in contours:
                    if cv2.contourArea(cnt) >= min_area:
                        grain_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(grain_mask, [cnt], -1, 255, -1)
                        grains.append(GrainContour(contour=cnt, mask=grain_mask))

        grains.sort(key=lambda g: cv2.contourArea(g.contour), reverse=True)
        return grains
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_yolo_detector.py -v`
Expected: All tests PASS

- [ ] **Step 5: 提交**

```bash
git add core/yolo_detector.py tests/test_yolo_detector.py
git commit -m "feat: YOLOv8-seg detector module with graceful fallback"
```

---

### Task 6: 导出模块 (core/exporter.py)

**Files:**
- Create: `core/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: 编写导出模块测试**

```python
"""Tests for core/exporter.py."""
import csv
import os
import tempfile

import numpy as np
import pytest
from core.exporter import export_csv, export_annotated_image
from core.morphology import GrainMorphology, GrainStatistics


def _make_morphologies(n: int = 3) -> list[GrainMorphology]:
    return [
        GrainMorphology(
            area=100 + i * 50, perimeter=40 + i * 10,
            circularity=0.7 + i * 0.05, d_eq=11.3 + i,
            major_axis=8 + i, minor_axis=6 + i * 0.5,
            aspect_ratio=1.3 + i * 0.2, sphericity=0.75 + i * 0.02,
            convexity=0.9 + i * 0.01, feret_max=18 + i, feret_min=10 + i,
        )
        for i in range(n)
    ]


class TestExportCSV:
    def test_creates_csv_file(self):
        morphologies = _make_morphologies()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_csv(morphologies, path)
            assert os.path.exists(path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 3
            assert "area" in rows[0]
            assert "circularity" in rows[0]
            assert float(rows[0]["area"]) == 100
        finally:
            os.unlink(path)


class TestExportAnnotatedImage:
    def test_creates_annotated_image(self):
        from core.traditional import GrainContour
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create a simple contour
        mask = np.zeros((100, 100), dtype=np.uint8)
        import cv2
        mask = cv2.ellipse(mask, (50, 50), (15, 10), 0, 0, 360, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grain = GrainContour(contour=contours[0], mask=mask)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            export_annotated_image(img, [grain], path)
            assert os.path.exists(path)
            result = cv2.imread(path)
            assert result is not None
        finally:
            os.unlink(path)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_exporter.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现导出模块**

```python
"""Export utilities for sand grain analysis results.

Supports CSV export of morphological data and annotated image export.
"""
import csv
import logging

import cv2
import numpy as np

from core.morphology import GrainMorphology
from core.traditional import GrainContour

logger = logging.getLogger(__name__)

# CSV column order matching GrainMorphology fields
CSV_COLUMNS = [
    "grain_id", "area", "perimeter", "circularity", "d_eq",
    "major_axis", "minor_axis", "aspect_ratio", "sphericity",
    "convexity", "feret_max", "feret_min",
]


def export_csv(morphologies: list[GrainMorphology], path: str) -> None:
    """Export grain morphologies to CSV file.

    Args:
        morphologies: List of GrainMorphology objects.
        path: Output CSV file path.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, m in enumerate(morphologies, 1):
            writer.writerow({
                "grain_id": i,
                "area": m.area,
                "perimeter": m.perimeter,
                "circularity": round(m.circularity, 4),
                "d_eq": round(m.d_eq, 2),
                "major_axis": round(m.major_axis, 2),
                "minor_axis": round(m.minor_axis, 2),
                "aspect_ratio": round(m.aspect_ratio, 4),
                "sphericity": round(m.sphericity, 4),
                "convexity": round(m.convexity, 4),
                "feret_max": round(m.feret_max, 2),
                "feret_min": round(m.feret_min, 2),
            })
    logger.info("Exported %d grains to %s", len(morphologies), path)


def export_annotated_image(
    image: np.ndarray,
    grains: list[GrainContour],
    path: str,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
) -> None:
    """Export annotated image with grain contours overlaid.

    Args:
        image: Original BGR image.
        grains: List of GrainContour objects.
        path: Output image file path.
        color: Contour color (B, G, R).
        thickness: Contour line thickness.
    """
    annotated = image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(annotated, [g.contour for g in grains], -1, color, thickness)

    # Add grain ID labels
    for i, grain in enumerate(grains, 1):
        M = cv2.moments(grain.contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                annotated, str(i), (cx - 5, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1,
            )

    cv2.imwrite(path, annotated)
    logger.info("Exported annotated image to %s", path)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/test_exporter.py -v`
Expected: All tests PASS

- [ ] **Step 5: 提交**

```bash
git add core/exporter.py tests/test_exporter.py
git commit -m "feat: CSV and annotated image export"
```

---

### Task 7: 集成测试 (tests/test_integration.py)

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 编写集成测试**

```python
"""Integration tests: full pipeline from image to morphologies."""
import cv2
import numpy as np
import pytest

from core.morphology import compute_morphology, compute_statistics
from core.preprocessor import PreprocessConfig, preprocess
from core.traditional import detect_grains


class TestFullPipeline:
    def test_pipeline_with_synthetic_image(self, sample_grain_image):
        """完整流程：预处理 → 检测 → 形态计算 → 统计。"""
        cfg = PreprocessConfig(use_watershed=True, min_area=50)
        mask = preprocess(sample_grain_image, cfg)
        grains = detect_grains(mask, min_area=cfg.min_area)
        assert len(grains) >= 2

        morphologies = [
            compute_morphology(g.contour, g.mask) for g in grains
        ]
        assert len(morphologies) >= 2

        for m in morphologies:
            assert m.area > 0
            assert 0 < m.circularity <= 1.0
            assert m.d_eq > 0
            assert m.aspect_ratio >= 1.0
            assert 0 < m.sphericity <= 1.0

        stats = compute_statistics(morphologies)
        assert stats.count >= 2
        assert stats.circularity_mean > 0
        assert len(stats.d_eq_values) == stats.count

    def test_pipeline_with_real_image(self, real_sand_image_path):
        """真实图像端到端测试。"""
        import os
        if not os.path.exists(real_sand_image_path):
            pytest.skip("Real sand image not available")

        img = cv2.imread(real_sand_image_path)
        assert img is not None

        cfg = PreprocessConfig(use_watershed=True, min_area=30)
        mask = preprocess(img, cfg)
        grains = detect_grains(mask, min_area=cfg.min_area)
        assert len(grains) > 0

        morphologies = [
            compute_morphology(g.contour, g.mask) for g in grains
        ]
        stats = compute_statistics(morphologies)
        assert stats.count > 0
        assert 0 < stats.circularity_mean <= 1.0

    def test_pipeline_no_watershed(self, sample_grain_image):
        """不使用分水岭的简化流程。"""
        cfg = PreprocessConfig(use_watershed=False, min_area=50)
        mask = preprocess(sample_grain_image, cfg)
        grains = detect_grains(mask, min_area=cfg.min_area)
        morphologies = [
            compute_morphology(g.contour, g.mask) for g in grains
        ]
        assert len(morphologies) >= 1
```

- [ ] **Step 2: 运行全部测试**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: integration tests for full pipeline"
```

---

### Task 8: GUI 主窗口 (gui/app.py)

**Files:**
- Create: `gui/app.py`
- Modify: `main.py`

- [ ] **Step 1: 实现 GUI 主窗口**

```python
"""Main application window for SandAnalyze."""
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QWidget,
)

from core.morphology import GrainMorphology, GrainStatistics, compute_morphology, compute_statistics
from core.preprocessor import PreprocessConfig, preprocess
from core.traditional import GrainContour, detect_grains
from core.yolo_detector import YOLODetector
from gui.image_panel import ImagePanel
from gui.result_panel import ResultPanel
from gui.settings_panel import SettingsPanel


class SandAnalyzeApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SandAnalyze - 沙粒形态分析系统")
        self.setMinimumSize(1200, 800)

        # State
        self._original_image: np.ndarray | None = None
        self._grains: list[GrainContour] = []
        self._morphologies: list[GrainMorphology] = []
        self._statistics: GrainStatistics | None = None
        self._config = PreprocessConfig()
        self._yolo_detector = YOLODetector()

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

    def _setup_ui(self) -> None:
        """Set up the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: image display
        self.image_panel = ImagePanel()
        self.image_panel.grain_clicked.connect(self._on_grain_clicked)
        splitter.addWidget(self.image_panel)

        # Right: results + settings
        right_panel = QWidget()
        right_layout = QHBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.result_panel = ResultPanel()
        self.settings_panel = SettingsPanel()
        self.settings_panel.config_changed.connect(self._on_config_changed)
        self.settings_panel.detect_requested.connect(self._run_detection)
        right_splitter.addWidget(self.result_panel)
        right_splitter.addWidget(self.settings_panel)

        right_layout.addWidget(right_splitter)
        splitter.addWidget(right_panel)

        splitter.setSizes([800, 400])
        layout.addWidget(splitter)

    def _setup_menu(self) -> None:
        """Set up menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("文件(&F)")
        open_action = QAction("打开图像(&O)", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)

        file_menu.addSeparator()
        export_csv_action = QAction("导出 CSV(&C)", self)
        export_csv_action.triggered.connect(self._export_csv)
        file_menu.addAction(export_csv_action)

        export_img_action = QAction("导出标注图(&I)", self)
        export_img_action.triggered.connect(self._export_annotated_image)
        file_menu.addAction(export_img_action)

        file_menu.addSeparator()
        quit_action = QAction("退出(&Q)", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Detect menu
        detect_menu = menu_bar.addMenu("检测(&D)")
        detect_action = QAction("运行检测(&R)", self)
        detect_action.setShortcut(QKeySequence("Ctrl+R"))
        detect_action.triggered.connect(self._run_detection)
        detect_menu.addAction(detect_action)

        # Help menu
        help_menu = menu_bar.addMenu("帮助(&H)")
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self) -> None:
        """Set up toolbar."""
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)

        open_action = QAction("打开", self)
        open_action.triggered.connect(self._open_image)
        toolbar.addAction(open_action)

        detect_action = QAction("检测", self)
        detect_action.triggered.connect(self._run_detection)
        toolbar.addAction(detect_action)

    def _setup_statusbar(self) -> None:
        """Set up status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("就绪 - 请打开图像文件")

    # --- Actions ---

    def _open_image(self) -> None:
        """Open an image file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开沙粒图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)",
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "错误", "无法读取图像文件")
            return
        self._original_image = image
        self._grains = []
        self._morphologies = []
        self._statistics = None
        self.image_panel.set_image(image)
        self.result_panel.clear()
        self._statusbar.showMessage(f"已加载图像: {path}")

    def _run_detection(self) -> None:
        """Run grain detection on the loaded image."""
        if self._original_image is None:
            QMessageBox.warning(self, "提示", "请先打开图像文件")
            return

        import time
        start = time.time()

        # Preprocess
        mask = preprocess(self._original_image, self._config)

        # Detect
        self._grains = detect_grains(mask, min_area=self._config.min_area)

        # Compute morphologies
        self._morphologies = [
            compute_morphology(g.contour, g.mask) for g in self._grains
        ]
        self._statistics = compute_statistics(self._morphologies)

        elapsed = time.time() - start

        # Update UI
        self.image_panel.set_grains(self._grains)
        self.result_panel.set_results(self._morphologies, self._statistics)
        self._statusbar.showMessage(
            f"检测完成: {self._statistics.count} 个颗粒 | "
            f"传统方法 | 耗时 {elapsed:.2f}s",
        )

    def _on_config_changed(self, config: PreprocessConfig) -> None:
        """Handle config changes from settings panel."""
        self._config = config

    def _on_grain_clicked(self, grain_id: int) -> None:
        """Handle grain click in image panel."""
        if 0 <= grain_id < len(self._morphologies):
            self.result_panel.highlight_grain(grain_id)

    def _export_csv(self) -> None:
        """Export morphological data to CSV."""
        if not self._morphologies:
            QMessageBox.warning(self, "提示", "请先运行检测")
            return
        from core.exporter import export_csv
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", "sand_analysis.csv", "CSV 文件 (*.csv)",
        )
        if path:
            export_csv(self._morphologies, path)
            self._statusbar.showMessage(f"已导出 CSV: {path}")

    def _export_annotated_image(self) -> None:
        """Export annotated image."""
        if self._original_image is None or not self._grains:
            QMessageBox.warning(self, "提示", "请先运行检测")
            return
        from core.exporter import export_annotated_image
        path, _ = QFileDialog.getSaveFileName(
            self, "导出标注图", "sand_annotated.png", "PNG 图像 (*.png)",
        )
        if path:
            export_annotated_image(self._original_image, self._grains, path)
            self._statusbar.showMessage(f"已导出标注图: {path}")

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self, "关于 SandAnalyze",
            "SandAnalyze v0.1.0\n\n"
            "沙粒形态分析系统\n"
            "基于 OpenCV 和 YOLO 的沙粒识别与统计\n\n"
            "用于地质/沉积学研究中\n"
            "光学显微镜沙粒图像的分析",
        )
```

- [ ] **Step 2: 更新 main.py 入口**

```python
"""SandAnalyze - 沙粒形态分析系统入口。"""
import sys

from PyQt6.QtWidgets import QApplication

from gui.app import SandAnalyzeApp


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SandAnalyze")
    window = SandAnalyzeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证应用能启动**

Run: `cd /Users/dadi/Project/sandanalyze && uv run python main.py`
Expected: 应用窗口启动（手动关闭确认无报错）

- [ ] **Step 4: 提交**

```bash
git add gui/app.py main.py
git commit -m "feat: main application window with menu and toolbar"
```

---

### Task 9: 图像显示面板 (gui/image_panel.py)

**Files:**
- Create: `gui/image_panel.py`

- [ ] **Step 1: 实现图像面板**

```python
"""Image display panel with zoom, pan, and grain interaction."""
import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class ImagePanel(QWidget):
    """Image display panel with contour overlay and grain click support."""

    grain_clicked = pyqtSignal(int)  # Emits grain index (0-based)

    def __init__(self):
        super().__init__()
        self._grains: list = []
        self._display_mode = "annotated"  # "original" or "annotated"
        self._zoom = 1.0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel("请打开图像文件")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(400, 300)
        self._label.setStyleSheet("background-color: #2b2b2b; color: #aaa;")

        self._scroll.setWidget(self._label)
        layout.addWidget(self._scroll)

    def set_image(self, image: np.ndarray) -> None:
        """Set the original image for display."""
        self._original_image = image.copy()
        self._show_pixmap(self._numpy_to_pixmap(image))

    def set_grains(self, grains: list) -> None:
        """Set detected grains and overlay contours on image."""
        self._grains = grains
        self._update_display()

    def clear(self) -> None:
        """Clear the display."""
        self._label.clear()
        self._label.setText("请打开图像文件")
        self._grains = []

    def _update_display(self) -> None:
        """Redraw image with contour overlay."""
        if not hasattr(self, "_original_image") or self._original_image is None:
            return
        if not self._grains:
            self._show_pixmap(self._numpy_to_pixmap(self._original_image))
            return

        annotated = self._original_image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        # Draw contours in green
        for i, grain in enumerate(self._grains):
            color = (0, 255, 0)
            cv2.drawContours(annotated, [grain.contour], -1, color, 1)
            # Label
            M = cv2.moments(grain.contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    annotated, str(i + 1), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
                )

        self._show_pixmap(self._numpy_to_pixmap(annotated))

    def _show_pixmap(self, pixmap: QPixmap) -> None:
        """Display a pixmap with current zoom level."""
        if self._zoom != 1.0:
            size = pixmap.size()
            scaled = pixmap.scaled(
                int(size.width() * self._zoom),
                int(size.height() * self._zoom),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._label.setPixmap(scaled)
        else:
            self._label.setPixmap(pixmap)
        self._label.adjustSize()

    def _numpy_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """Convert numpy BGR image to QPixmap."""
        if len(image.shape) == 2:
            h, w = image.shape
            qimg = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = image.shape
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom with mouse wheel."""
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(self._zoom * 1.2, 5.0)
        else:
            self._zoom = max(self._zoom / 1.2, 0.1)
        if hasattr(self, "_original_image") and self._original_image is not None:
            self._update_display()

    def mousePressEvent(self, event) -> None:
        """Handle click to select grain."""
        if event.button() == Qt.MouseButton.LeftButton and self._grains:
            # Map click position to image coordinates
            pos = event.position().toPoint()
            label_pos = self._label.mapFromParent(pos)
            # Account for zoom
            img_x = int(label_pos.x() / self._zoom)
            img_y = int(label_pos.y() / self._zoom)

            # Find which grain was clicked
            if hasattr(self, "_original_image") and self._original_image is not None:
                h, w = self._original_image.shape[:2]
                if 0 <= img_x < w and 0 <= img_y < h:
                    for i, grain in enumerate(self._grains):
                        if grain.mask[img_y, img_x] > 0:
                            self.grain_clicked.emit(i)
                            break
```

- [ ] **Step 2: 提交**

```bash
git add gui/image_panel.py
git commit -m "feat: image panel with zoom, pan, and grain click"
```

---

### Task 10: 结果面板 (gui/result_panel.py)

**Files:**
- Create: `gui/result_panel.py`

- [ ] **Step 1: 实现结果面板**

```python
"""Result panel showing statistics, tables, and charts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.morphology import GrainMorphology, GrainStatistics

# Set Chinese-compatible font for matplotlib
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class ResultPanel(QWidget):
    """Panel for displaying grain analysis results."""

    def __init__(self):
        super().__init__()
        self._morphologies: list[GrainMorphology] = []
        self._statistics: GrainStatistics | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._tabs = QTabWidget()

        # Summary tab
        self._summary_widget = QWidget()
        self._summary_layout = QVBoxLayout(self._summary_widget)
        self._summary_labels: dict[str, QLabel] = {}
        self._init_summary()
        self._tabs.addTab(self._summary_widget, "统计摘要")

        # Table tab
        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "ID", "面积", "周长", "圆度", "等效粒径", "长短轴比", "球度", "凸度",
        ])
        self._table.setAlternatingRowColors(True)
        self._tabs.addTab(self._table, "颗粒数据")

        # Charts tab
        self._charts_widget = QWidget()
        self._charts_layout = QVBoxLayout(self._charts_widget)
        self._tabs.addTab(self._charts_widget, "图表")

        layout.addWidget(self._tabs)

    def _init_summary(self) -> None:
        """Initialize summary labels."""
        items = [
            ("颗粒总数", "count"),
            ("平均圆度", "circularity_mean"),
            ("平均球度", "sphericity_mean"),
            ("平均等效粒径", "d_eq_mean"),
            ("平均长短轴比", "aspect_ratio_mean"),
            ("平均凸度", "convexity_mean"),
        ]
        for display_name, key in items:
            row = QHBoxLayout()
            name_label = QLabel(f"{display_name}:")
            name_label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("-")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(name_label)
            row.addWidget(value_label)
            self._summary_layout.addLayout(row)
            self._summary_labels[key] = value_label

        # Zingg classification
        zingg_group = QGroupBox("Zingg 分类")
        self._zingg_layout = QVBoxLayout(zingg_group)
        self._zingg_label = QLabel("-")
        self._zingg_layout.addWidget(self._zingg_label)
        self._summary_layout.addWidget(zingg_group)
        self._summary_layout.addStretch()

    def set_results(
        self,
        morphologies: list[GrainMorphology],
        statistics: GrainStatistics,
    ) -> None:
        """Update panel with detection results."""
        self._morphologies = morphologies
        self._statistics = statistics
        self._update_summary()
        self._update_table()
        self._update_charts()

    def clear(self) -> None:
        """Clear all results."""
        self._morphologies = []
        self._statistics = None
        for label in self._summary_labels.values():
            label.setText("-")
        self._table.setRowCount(0)
        self._zingg_label.setText("-")
        # Clear charts
        while self._charts_layout.count():
            child = self._charts_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def highlight_grain(self, grain_id: int) -> None:
        """Highlight a specific grain in the table."""
        if 0 <= grain_id < self._table.rowCount():
            self._table.selectRow(grain_id)
            self._tabs.setCurrentIndex(1)  # Switch to table tab

    def _update_summary(self) -> None:
        """Update summary statistics."""
        if not self._statistics:
            return
        s = self._statistics
        self._summary_labels["count"].setText(str(s.count))
        self._summary_labels["circularity_mean"].setText(f"{s.circularity_mean:.4f}")
        self._summary_labels["sphericity_mean"].setText(f"{s.sphericity_mean:.4f}")
        self._summary_labels["d_eq_mean"].setText(f"{s.d_eq_mean:.2f} px")
        self._summary_labels["aspect_ratio_mean"].setText(f"{s.aspect_ratio_mean:.4f}")
        self._summary_labels["convexity_mean"].setText(f"{s.convexity_mean:.4f}")

        # Zingg
        parts = [f"{k}: {v}" for k, v in s.zingg_counts.items()]
        self._zingg_label.setText(" | ".join(parts) if parts else "-")

    def _update_table(self) -> None:
        """Update grain data table."""
        self._table.setRowCount(len(self._morphologies))
        for i, m in enumerate(self._morphologies):
            values = [
                str(i + 1),
                f"{m.area:.0f}",
                f"{m.perimeter:.1f}",
                f"{m.circularity:.4f}",
                f"{m.d_eq:.2f}",
                f"{m.aspect_ratio:.4f}",
                f"{m.sphericity:.4f}",
                f"{m.convexity:.4f}",
            ]
            for j, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(i, j, item)
        self._table.resizeColumnsToContents()

    def _update_charts(self) -> None:
        """Update charts with current data."""
        # Clear existing charts
        while self._charts_layout.count():
            child = self._charts_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not self._statistics or self._statistics.count == 0:
            return

        s = self._statistics

        # 1. Particle size distribution histogram
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.hist(s.d_eq_values, bins=20, edgecolor="black", alpha=0.7, color="#4CAF50")
        ax1.set_xlabel("等效粒径 (px)")
        ax1.set_ylabel("数量")
        ax1.set_title("粒径分布")
        fig1.tight_layout()
        canvas1 = FigureCanvas(fig1)
        self._charts_layout.addWidget(canvas1)

        # 2. Circularity vs Sphericity scatter
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.scatter(s.circularity_values, s.sphericity_values, alpha=0.6, s=15, c="#2196F3")
        ax2.set_xlabel("圆度")
        ax2.set_ylabel("球度")
        ax2.set_title("圆度-球度散点图")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        fig2.tight_layout()
        canvas2 = FigureCanvas(fig2)
        self._charts_layout.addWidget(canvas2)

        # 3. Zingg classification pie chart
        if s.zingg_counts:
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            labels = list(s.zingg_counts.keys())
            sizes = list(s.zingg_counts.values())
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
            ax3.pie(
                sizes, labels=labels, colors=colors[:len(labels)],
                autopct="%1.1f%%", startangle=90,
            )
            ax3.set_title("Zingg 分类")
            fig3.tight_layout()
            canvas3 = FigureCanvas(fig3)
            self._charts_layout.addWidget(canvas3)
```

- [ ] **Step 2: 提交**

```bash
git add gui/result_panel.py
git commit -m "feat: result panel with summary, table, and charts"
```

---

### Task 11: 设置面板 (gui/settings_panel.py)

**Files:**
- Create: `gui/settings_panel.py`

- [ ] **Step 1: 实现设置面板**

```python
"""Settings panel for adjusting preprocessing and detection parameters."""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.preprocessor import PreprocessConfig


class SettingsPanel(QWidget):
    """Panel for adjusting preprocessing and detection parameters."""

    config_changed = pyqtSignal(object)  # Emits PreprocessConfig
    detect_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._config = PreprocessConfig()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Preprocessing group
        preprocess_group = QGroupBox("预处理参数")
        pg_layout = QVBoxLayout()

        # Blur kernel
        row = QHBoxLayout()
        row.addWidget(QLabel("模糊核大小:"))
        self._blur_spin = QSpinBox()
        self._blur_spin.setRange(1, 31)
        self._blur_spin.setSingleStep(2)
        self._blur_spin.setValue(self._config.blur_kernel)
        self._blur_spin.valueChanged.connect(self._on_param_changed)
        row.addWidget(self._blur_spin)
        pg_layout.addLayout(row)

        # Adaptive block size
        row = QHBoxLayout()
        row.addWidget(QLabel("自适应块大小:"))
        self._block_spin = QSpinBox()
        self._block_spin.setRange(3, 99)
        self._block_spin.setSingleStep(2)
        self._block_spin.setValue(self._config.adaptive_block_size)
        self._block_spin.valueChanged.connect(self._on_param_changed)
        row.addWidget(self._block_spin)
        pg_layout.addLayout(row)

        # Adaptive C
        row = QHBoxLayout()
        row.addWidget(QLabel("自适应 C 值:"))
        self._c_spin = QSpinBox()
        self._c_spin.setRange(-10, 20)
        self._c_spin.setValue(self._config.adaptive_c)
        self._c_spin.valueChanged.connect(self._on_param_changed)
        row.addWidget(self._c_spin)
        pg_layout.addLayout(row)

        # Morph kernel
        row = QHBoxLayout()
        row.addWidget(QLabel("形态学核大小:"))
        self._morph_spin = QSpinBox()
        self._morph_spin.setRange(1, 21)
        self._morph_spin.setSingleStep(2)
        self._morph_spin.setValue(self._config.morph_kernel_size)
        self._morph_spin.valueChanged.connect(self._on_param_changed)
        row.addWidget(self._morph_spin)
        pg_layout.addLayout(row)

        # Min area
        row = QHBoxLayout()
        row.addWidget(QLabel("最小面积:"))
        self._area_spin = QSpinBox()
        self._area_spin.setRange(1, 10000)
        self._area_spin.setValue(self._config.min_area)
        self._area_spin.valueChanged.connect(self._on_param_changed)
        row.addWidget(self._area_spin)
        pg_layout.addLayout(row)

        # CLAHE checkbox
        self._clahe_check = QCheckBox("使用 CLAHE 增强")
        self._clahe_check.setChecked(self._config.use_clahe)
        self._clahe_check.stateChanged.connect(self._on_param_changed)
        pg_layout.addWidget(self._clahe_check)

        # Watershed checkbox
        self._watershed_check = QCheckBox("使用分水岭分割")
        self._watershed_check.setChecked(self._config.use_watershed)
        self._watershed_check.stateChanged.connect(self._on_param_changed)
        pg_layout.addWidget(self._watershed_check)

        preprocess_group.setLayout(pg_layout)
        layout.addWidget(preprocess_group)

        # Detect button
        self._detect_btn = QPushButton("🔍 运行检测")
        self._detect_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 8px; font-size: 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self._detect_btn.clicked.connect(self.detect_requested.emit)
        layout.addWidget(self._detect_btn)

        layout.addStretch()

    def _on_param_changed(self) -> None:
        """Collect current parameter values and emit config."""
        self._config = PreprocessConfig(
            blur_kernel=self._blur_spin.value(),
            adaptive_block_size=self._block_spin.value(),
            adaptive_c=self._c_spin.value(),
            morph_kernel_size=self._morph_spin.value(),
            min_area=self._area_spin.value(),
            use_clahe=self._clahe_check.isChecked(),
            use_watershed=self._watershed_check.isChecked(),
        )
        self.config_changed.emit(self._config)

    def get_config(self) -> PreprocessConfig:
        """Return current configuration."""
        return self._config
```

- [ ] **Step 2: 提交**

```bash
git add gui/settings_panel.py
git commit -m "feat: settings panel for preprocessing parameter adjustment"
```

---

### Task 12: 更新 core/__init__.py 和最终集成

**Files:**
- Modify: `core/__init__.py`
- Modify: `gui/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: 更新 core/__init__.py**

```python
"""SandAnalyze core processing modules.

Provides grain detection, morphology computation, and export utilities.
"""
from core.exporter import export_annotated_image, export_csv
from core.morphology import (
    GrainMorphology,
    GrainStatistics,
    compute_morphology,
    compute_statistics,
)
from core.preprocessor import PreprocessConfig, preprocess
from core.traditional import GrainContour, detect_grains
from core.yolo_detector import YOLODetector

__all__ = [
    "PreprocessConfig",
    "preprocess",
    "GrainContour",
    "detect_grains",
    "YOLODetector",
    "GrainMorphology",
    "GrainStatistics",
    "compute_morphology",
    "compute_statistics",
    "export_csv",
    "export_annotated_image",
]
```

- [ ] **Step 2: 更新 gui/__init__.py**

```python
"""SandAnalyze GUI modules."""
from gui.app import SandAnalyzeApp

__all__ = ["SandAnalyzeApp"]
```

- [ ] **Step 3: 更新 README.md**

```markdown
# SandAnalyze - 沙粒形态分析系统

基于 OpenCV 传统图像处理和 YOLOv8-seg 的沙粒识别、形状分析和数量统计工具。

## 功能

- 🔬 **沙粒检测**：传统 OpenCV 轮廓检测 + YOLOv8-seg 对比
- 📐 **多维度形态参数**：圆度、球度、长短轴比、凸度、Feret 直径等
- 📊 **统计分析**：粒径分布、圆度-球度散点图、Zingg 分类
- 🖥️ **桌面 GUI**：PyQt6 交互界面，支持缩放、点击查看、参数调节
- 📤 **导出**：CSV 数据 + 标注图 PNG

## 安装

```bash
uv sync --extra dev
```

## 使用

```bash
uv run python main.py
```

1. 打开沙粒图像（菜单或拖拽）
2. 调整预处理参数（右侧面板）
3. 点击"运行检测"
4. 查看统计结果、图表和颗粒数据
5. 导出 CSV 或标注图

## 形态参数

| 参数 | 计算方法 | 地质意义 |
|------|----------|----------|
| 面积 (A) | 掩码像素数 | 粒径基础 |
| 周长 (P) | 轮廓长度 | 磨蚀程度 |
| 圆度 (Circularity) | 4πA/P² | 越接近1越圆 |
| 等效粒径 (d_eq) | √(4A/π) | 等效圆直径 |
| 长短轴比 (AR) | 长/短轴 | 扁平程度 |
| 球度 (Sphericity) | 短/长轴 | 三维形状推断 |
| 凸度 (Convexity) | 面积/凸包面积 | 表面凹凸程度 |

## 技术栈

- Python 3.13
- OpenCV（图像处理）
- ultralytics / YOLOv8-seg（深度学习检测）
- PyQt6（GUI）
- matplotlib（图表）
- numpy / scipy（数值计算）
```

- [ ] **Step 4: 运行全部测试**

Run: `cd /Users/dadi/Project/sandanalyze && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: 提交**

```bash
git add core/__init__.py gui/__init__.py README.md
git commit -m "feat: finalize project with exports, README, and integration"
```

---

## Self-Review Checklist

- **Spec coverage:**
  - ✅ 图像预处理 → Task 2
  - ✅ 传统检测 → Task 3
  - ✅ YOLO 检测 → Task 5
  - ✅ 形态参数计算 → Task 4
  - ✅ 统计汇总（均值/标准差/中位数/直方图/散点图/Zingg） → Task 4 + Task 10
  - ✅ GUI 主窗口（菜单/工具栏/状态栏） → Task 8
  - ✅ 图像面板（显示/缩放/交互） → Task 9
  - ✅ 结果面板（摘要/表格/图表） → Task 10
  - ✅ 设置面板（参数调节） → Task 11
  - ✅ 导出 CSV/PNG → Task 6
  - ✅ 导出 PDF 报告 → 未实现（YAGNI，可在后续迭代添加）

- **Placeholder scan:** 无 TBD/TODO，所有步骤包含完整代码

- **Type consistency:**
  - `GrainContour` 定义在 `core/traditional.py`，被 `core/yolo_detector.py` 和 `core/exporter.py` 使用 ✅
  - `GrainMorphology` / `GrainStatistics` 定义在 `core/morphology.py`，被 GUI 和 exporter 使用 ✅
  - `PreprocessConfig` 定义在 `core/preprocessor.py`，被 GUI settings 使用 ✅
