# 纹理与边缘特征过滤方法实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现基于纹理（LBP/GLCM）和边缘特征（Sobel/方向一致性/闭合度）的沙砾检测过滤模块，解决大块沙砾漏检、镜头边缘误检、噪声误识别三个问题。

**Architecture:** 在现有 `detect_grains` → `filter_strict` 流程中插入 `TextureEdgeValidator` 验证层。纹理特征使用 skimage（优先）或纯 OpenCV 回退方案。验证器与现有过滤器互补，前置在 `filter_strict` 之前。

**Tech Stack:** Python, OpenCV, NumPy, scikit-image (optional)

---

## 文件结构

| 文件 | 职责 | 操作 |
|------|------|------|
| `core/texture_edge_filter.py` | 纹理/边缘验证器核心模块 | **新建** |
| `core/pipeline.py` | 集成验证器到检测流程 | **修改** |
| `tests/test_texture_edge_filter.py` | 纹理/边缘过滤单元测试 | **新建** |

---

## Task 1: 创建纹理/边缘过滤核心模块

**Files:**
- Create: `core/texture_edge_filter.py`
- Test: `tests/test_texture_edge_filter.py`

### Step 1.1: 编写骨架代码和接口定义

创建 `core/texture_edge_filter.py`，包含所有接口和数据类：

```python
"""Texture and edge feature based filtering for grain detection.

Provides TextureEdgeValidator that uses LBP/GLCM texture features and
Sobel/edge-direction/closure edge features to distinguish real grains
from lens-edge artifacts and noise.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from core.multiscale_detector import GrainCandidate


@dataclass
class ValidationConfig:
    """Configuration for texture/edge validation."""
    texture_score_threshold: float = 0.4
    edge_direction_threshold: float = 0.6
    edge_closure_threshold: float = 0.3
    lens_edge_margin: float = 0.05
    lens_edge_circularity: float = 0.7
    lens_edge_min_area: float = 50000
    noise_max_texture_score: float = 0.3
    noise_max_edge_strength: float = 30.0
    noise_max_area: int = 500


class TextureEdgeValidator:
    """Validate grain candidates using texture and edge features."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()
        self._has_skimage = self._check_skimage()

    def _check_skimage(self) -> bool:
        """Check if scikit-image is available."""
        try:
            import skimage.feature  # noqa: F401
            return True
        except ImportError:
            return False

    def validate(self, candidate: GrainCandidate, full_image: np.ndarray) -> bool:
        """Return True if candidate is a valid grain."""
        raise NotImplementedError("Implement in Step 1.3")

    def _is_lens_edge(self, candidate: GrainCandidate, full_image: np.ndarray) -> bool:
        """Detect lens-edge artifacts."""
        raise NotImplementedError("Implement in Step 1.3")

    def _is_noise(self, candidate: GrainCandidate, full_image: np.ndarray) -> bool:
        """Detect noise false positives."""
        raise NotImplementedError("Implement in Step 1.3")

    def _compute_composite_score(self, candidate: GrainCandidate, full_image: np.ndarray) -> float:
        """Compute composite texture/edge score."""
        raise NotImplementedError("Implement in Step 1.3")

    def _extract_roi(self, candidate: GrainCandidate, full_image: np.ndarray) -> np.ndarray | None:
        """Extract ROI around candidate from full image."""
        raise NotImplementedError("Implement in Step 1.3")
```

### Step 1.2: 编写骨架测试

创建 `tests/test_texture_edge_filter.py`：

```python
"""Tests for texture_edge_filter module."""

import cv2
import numpy as np
import pytest

from core.texture_edge_filter import TextureEdgeValidator, ValidationConfig
from core.multiscale_detector import GrainCandidate


class TestTextureEdgeValidator:
    """Tests for TextureEdgeValidator."""

    def test_validator_initialization(self):
        """Test validator can be initialized with default config."""
        validator = TextureEdgeValidator()
        assert validator.config.texture_score_threshold == 0.4

    def test_validator_with_custom_config(self):
        """Test validator accepts custom config."""
        config = ValidationConfig(texture_score_threshold=0.6)
        validator = TextureEdgeValidator(config)
        assert validator.config.texture_score_threshold == 0.6

    def test_check_skimage(self):
        """Test skimage availability check."""
        validator = TextureEdgeValidator()
        # Should be a boolean
        assert isinstance(validator._has_skimage, bool)
```

运行测试确认失败：

```bash
pytest tests/test_texture_edge_filter.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.texture_edge_filter'`

### Step 1.3: 实现核心验证逻辑

在 `core/texture_edge_filter.py` 中实现所有方法：

```python
    def validate(self, candidate: GrainCandidate, full_image: np.ndarray) -> bool:
        """Return True if candidate is a valid grain."""
        # 1. Lens edge detection (highest priority)
        if self._is_lens_edge(candidate, full_image):
            return False

        # 2. Noise detection
        if self._is_noise(candidate, full_image):
            return False

        # 3. Composite score
        score = self._compute_composite_score(candidate, full_image)
        if score < self.config.texture_score_threshold:
            return False

        # 4. Edge closure check
        roi = self._extract_roi(candidate, full_image)
        if roi is not None:
            edge_closure = compute_edge_closure(candidate.contour, roi)
            if edge_closure < self.config.edge_closure_threshold:
                return False

        return True

    def _is_lens_edge(self, candidate: GrainCandidate, full_image: np.ndarray) -> bool:
        """Detect lens-edge artifacts: large circular objects near image border."""
        h, w = full_image.shape[:2]
        margin = self.config.lens_edge_margin

        x, y, bw, bh = cv2.boundingRect(candidate.contour)
        near_edge = (
            x < w * margin
            or y < h * margin
            or x + bw > w * (1 - margin)
            or y + bh > h * (1 - margin)
        )

        is_large = candidate.area > self.config.lens_edge_min_area
        is_circular = candidate.circularity > self.config.lens_edge_circularity

        return near_edge and is_large and is_circular

    def _is_noise(self, candidate: GrainCandidate, full_image: np.ndarray) -> bool:
        """Detect noise: very small area with low texture and weak edges."""
        if candidate.area > self.config.noise_max_area:
            return False

        roi = self._extract_roi(candidate, full_image)
        if roi is None:
            return True

        texture_score = self._compute_texture_score(roi)
        if texture_score < self.config.noise_max_texture_score:
            return True

        edge_strength = compute_edge_strength(roi)
        if edge_strength < self.config.noise_max_edge_strength:
            return True

        return False

    def _compute_composite_score(self, candidate: GrainCandidate, full_image: np.ndarray) -> float:
        """Compute composite texture/edge score in [0, 1]."""
        roi = self._extract_roi(candidate, full_image)
        if roi is None:
            return 0.0

        texture_score = self._compute_texture_score(roi)
        edge_consistency = compute_edge_direction_consistency(roi)
        edge_closure = compute_edge_closure(candidate.contour, roi)

        score = (
            texture_score * 0.5
            + (1 - edge_consistency) * 0.25
            + edge_closure * 0.25
        )
        return score

    def _compute_texture_score(self, roi_gray: np.ndarray) -> float:
        """Compute texture consistency score."""
        if self._has_skimage:
            lbp = extract_lbp_features(roi_gray)
            glcm = extract_glcm_features(roi_gray)
        else:
            lbp = extract_lbp_features_opencv(roi_gray)
            glcm = extract_glcm_features_opencv(roi_gray)
        return compute_texture_consistency_score(lbp, glcm)

    def _extract_roi(self, candidate: GrainCandidate, full_image: np.ndarray) -> np.ndarray | None:
        """Extract grayscale ROI around candidate with padding."""
        x, y, w, h = cv2.boundingRect(candidate.contour)
        pad = 10
        h_img, w_img = full_image.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        if len(full_image.shape) == 3:
            roi = full_image[y1:y2, x1:x2]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi = full_image[y1:y2, x1:x2]

        return roi if roi.size > 0 else None
```

### Step 1.4: 实现纹理特征提取函数

在 `core/texture_edge_filter.py` 末尾添加：

```python
# ── Texture feature extraction ───────────────────────────────────────────────


def extract_lbp_features(roi_gray: np.ndarray) -> np.ndarray:
    """Extract uniform LBP histogram (10 bins).

    Requires scikit-image.
    """
    from skimage.feature import local_binary_pattern

    lbp = local_binary_pattern(roi_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-7
    return hist


def extract_glcm_features(roi_gray: np.ndarray) -> dict[str, float]:
    """Extract GLCM texture features.

    Requires scikit-image.
    """
    from skimage.feature import graycomatrix, graycoprops

    roi_32 = (roi_gray / 255 * 31).astype(np.uint8)
    glcm = graycomatrix(
        roi_32,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    )

    features = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        features[prop] = float(graycoprops(glcm, prop).mean())
    return features


def compute_texture_consistency_score(
    lbp_features: np.ndarray, glcm_features: dict[str, float]
) -> float:
    """Compute texture consistency score in [0, 1].

    Higher = more likely a real grain.
    """
    # LBP concentration (inverse of entropy)
    lbp_entropy = -np.sum(lbp_features * np.log(lbp_features + 1e-7))
    lbp_concentration = 1.0 / (1.0 + lbp_entropy)

    contrast = glcm_features["contrast"]
    homogeneity = glcm_features["homogeneity"]
    energy = glcm_features["energy"]

    # Contrast should be in moderate range
    contrast_score = 1.0 - abs(contrast - 0.3) / 0.3
    contrast_score = max(0.0, contrast_score)

    # Homogeneity
    homo_score = min(homogeneity / 0.3, 1.0)

    # Energy
    energy_score = 1.0 - abs(energy - 0.15) / 0.15
    energy_score = max(0.0, energy_score)

    score = (
        lbp_concentration * 0.3
        + contrast_score * 0.25
        + homo_score * 0.25
        + energy_score * 0.2
    )
    return min(score, 1.0)
```

### Step 1.5: 实现边缘特征分析函数

在 `core/texture_edge_filter.py` 末尾添加：

```python
# ── Edge feature analysis ────────────────────────────────────────────────────


def compute_edge_strength(roi_gray: np.ndarray) -> float:
    """Compute average Sobel edge magnitude."""
    sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    return float(np.mean(sobel_mag))


def compute_edge_direction_consistency(roi_gray: np.ndarray) -> float:
    """Compute edge direction consistency in [0, 1].

    Higher = more consistent direction (suspicious for lens edge).
    """
    sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)

    orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
    mag = np.sqrt(sobelx**2 + sobely**2)
    strong_edges = mag > np.percentile(mag, 75)

    if np.sum(strong_edges) < 10:
        return 0.0

    orientations = orientation[strong_edges].flatten()
    hist, _ = np.histogram(orientations, bins=36, range=(-180, 180))
    hist = hist / (hist.sum() + 1e-7)
    consistency = hist.max()
    return float(consistency)


def compute_edge_closure(contour: np.ndarray, roi_gray: np.ndarray) -> float:
    """Compute edge closure ratio in [0, 1].

    Ratio of contour pixels that overlap with Canny edges.
    """
    edges = cv2.Canny(roi_gray, 50, 150)

    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [contour], -1, 255, thickness=2)

    contour_edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, mask))
    contour_length = cv2.arcLength(contour, True)

    if contour_length == 0:
        return 0.0

    closure_ratio = min(contour_edge_pixels / contour_length, 1.0)
    return closure_ratio
```

### Step 1.6: 实现纯 OpenCV 回退方案

在 `core/texture_edge_filter.py` 末尾添加：

```python
# ── OpenCV fallback (no scikit-image) ─────────────────────────────────────────


def extract_lbp_features_opencv(roi_gray: np.ndarray) -> np.ndarray:
    """Simplified LBP using local standard deviation."""
    kernel_size = 5
    local_mean = cv2.blur(roi_gray.astype(np.float32), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(
        (roi_gray.astype(np.float32) ** 2), (kernel_size, kernel_size)
    )
    local_std = np.sqrt(np.abs(local_sq_mean - local_mean**2))

    hist, _ = np.histogram(local_std.ravel(), bins=10, range=(0, 50))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-7
    return hist


def extract_glcm_features_opencv(roi_gray: np.ndarray) -> dict[str, float]:
    """Simplified GLCM using OpenCV operators."""
    roi_16 = (roi_gray / 255 * 15).astype(np.uint8)

    laplacian = cv2.Laplacian(roi_16, cv2.CV_64F)
    contrast = float(np.std(laplacian))

    sobelx = cv2.Sobel(roi_16, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi_16, cv2.CV_64F, 0, 1, ksize=3)
    energy = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))

    local_var = np.var(roi_gray.astype(np.float32))
    homogeneity = 1.0 / (1.0 + local_var / 100.0)

    return {
        "contrast": contrast,
        "dissimilarity": contrast / 2.0,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": 0.5,
    }
```

### Step 1.7: 运行测试确认通过

```bash
pytest tests/test_texture_edge_filter.py -v
```

Expected: PASS

### Step 1.8: Commit

```bash
git add core/texture_edge_filter.py tests/test_texture_edge_filter.py
git commit -m "feat: add texture/edge filtering core module

- TextureEdgeValidator with LBP/GLCM texture features
- Edge direction consistency and closure analysis
- Lens-edge and noise detection
- OpenCV fallback when scikit-image unavailable

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: 添加单元测试

**Files:**
- Modify: `tests/test_texture_edge_filter.py`

### Step 2.1: 添加纹理特征提取测试

```python
    def test_extract_lbp_features_shape(self):
        """Test LBP feature output shape."""
        from core.texture_edge_filter import extract_lbp_features

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        features = extract_lbp_features(roi)
        assert features.shape == (10,)
        assert np.isclose(features.sum(), 1.0, atol=1e-6)

    def test_extract_glcm_features_keys(self):
        """Test GLCM feature output keys."""
        from core.texture_edge_filter import extract_glcm_features

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        features = extract_glcm_features(roi)
        expected_keys = {"contrast", "dissimilarity", "homogeneity", "energy", "correlation"}
        assert set(features.keys()) == expected_keys

    def test_compute_texture_consistency_score_range(self):
        """Test texture score is in [0, 1]."""
        from core.texture_edge_filter import (
            compute_texture_consistency_score,
            extract_lbp_features,
            extract_glcm_features,
        )

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        lbp = extract_lbp_features(roi)
        glcm = extract_glcm_features(roi)
        score = compute_texture_consistency_score(lbp, glcm)
        assert 0.0 <= score <= 1.0
```

### Step 2.2: 添加边缘特征测试

```python
    def test_compute_edge_strength(self):
        """Test edge strength is non-negative."""
        from core.texture_edge_filter import compute_edge_strength

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        strength = compute_edge_strength(roi)
        assert strength >= 0.0

    def test_compute_edge_direction_consistency_range(self):
        """Test edge direction consistency is in [0, 1]."""
        from core.texture_edge_filter import compute_edge_direction_consistency

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        consistency = compute_edge_direction_consistency(roi)
        assert 0.0 <= consistency <= 1.0

    def test_compute_edge_closure_range(self):
        """Test edge closure is in [0, 1]."""
        from core.texture_edge_filter import compute_edge_closure

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        contour = np.array([[[10, 10]], [[40, 10]], [[40, 40]], [[10, 40]]], dtype=np.int32)
        closure = compute_edge_closure(contour, roi)
        assert 0.0 <= closure <= 1.0
```

### Step 2.3: 添加验证器逻辑测试

```python
    def test_lens_edge_detection(self):
        """Test lens-edge artifact filtering."""
        validator = TextureEdgeValidator()

        # Create a large circular contour near the border
        h, w = 1000, 1000
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # Large circle near top-left corner
        center = (50, 50)
        radius = 100
        contour = np.array([
            [[center[0] + int(radius * np.cos(theta)), center[1] + int(radius * np.sin(theta))]]
            for theta in np.linspace(0, 2 * np.pi, 50)
        ], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((h, w), dtype=np.uint8),
            area=60000,
            perimeter=2 * np.pi * radius,
            circularity=0.85,
            aspect_ratio=1.0,
            major_axis=radius * 2,
            minor_axis=radius * 2,
            convexity=0.9,
            is_flocculation=False,
            border_distance=0.0,
            solidity=0.9,
        )

        assert validator._is_lens_edge(candidate, image) is True

    def test_noise_detection_small_area(self):
        """Test noise filtering for small area."""
        validator = TextureEdgeValidator()

        h, w = 500, 500
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Small contour
        contour = np.array([[[100, 100]], [[110, 100]], [[110, 110]], [[100, 110]]], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((h, w), dtype=np.uint8),
            area=100,
            perimeter=40,
            circularity=0.8,
            aspect_ratio=1.0,
            major_axis=10,
            minor_axis=10,
            convexity=0.9,
            is_flocculation=False,
            border_distance=50.0,
            solidity=0.9,
        )

        assert validator._is_noise(candidate, image) is True

    def test_composite_score_range(self):
        """Test composite score is in [0, 1]."""
        validator = TextureEdgeValidator()

        h, w = 500, 500
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        contour = np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]], dtype=np.int32)

        candidate = GrainCandidate(
            contour=contour,
            mask=np.zeros((h, w), dtype=np.uint8),
            area=2500,
            perimeter=200,
            circularity=0.8,
            aspect_ratio=1.0,
            major_axis=50,
            minor_axis=50,
            convexity=0.9,
            is_flocculation=False,
            border_distance=50.0,
            solidity=0.9,
        )

        score = validator._compute_composite_score(candidate, image)
        assert 0.0 <= score <= 1.0
```

### Step 2.4: 添加 OpenCV 回退测试

```python
    def test_opencv_fallback_lbp(self):
        """Test OpenCV fallback LBP extraction."""
        from core.texture_edge_filter import extract_lbp_features_opencv

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        features = extract_lbp_features_opencv(roi)
        assert features.shape == (10,)
        assert np.isclose(features.sum(), 1.0, atol=1e-6)

    def test_opencv_fallback_glcm(self):
        """Test OpenCV fallback GLCM extraction."""
        from core.texture_edge_filter import extract_glcm_features_opencv

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        features = extract_glcm_features_opencv(roi)
        expected_keys = {"contrast", "dissimilarity", "homogeneity", "energy", "correlation"}
        assert set(features.keys()) == expected_keys
```

### Step 2.5: 运行全部测试

```bash
pytest tests/test_texture_edge_filter.py -v
```

Expected: ALL PASS

### Step 2.6: Commit

```bash
git add tests/test_texture_edge_filter.py
git commit -m "test: add texture/edge filter unit tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 集成到检测 Pipeline

**Files:**
- Modify: `core/pipeline.py`

### Step 3.1: 添加导入语句

在 `core/pipeline.py` 的导入区域添加：

```python
from core.texture_edge_filter import (
    TextureEdgeValidator,
    ValidationConfig,
)
```

### Step 3.2: 修改 `run_detection_pipeline` 函数

在 `run_detection_pipeline` 中，在 `detect_grains` 调用之后、遍历 `results` 之前，插入纹理/边缘验证：

```python
    results = detect_grains(
        image=image,
        config=config,
        min_area=min_area,
        max_area=max_area,
        border_margin=border_margin,
        hull_expansion_ratio=hull_expansion_ratio,
        floc_config=floc_config,
        crop_black_background=crop_black_background,
    )

    # 【新增】Texture/edge validation
    validator = TextureEdgeValidator(ValidationConfig())
    filtered_results = []
    for result in results:
        # Convert DetectionResult to GrainCandidate for validation
        candidate = _detection_result_to_candidate(result, image)
        if validator.validate(candidate, image):
            filtered_results.append(result)
    results = filtered_results
```

### Step 3.3: 添加转换辅助函数

在 `core/pipeline.py` 末尾（`_mask_to_candidates` 之后）添加：

```python
def _detection_result_to_candidate(
    result: DetectionResult, image: np.ndarray
) -> GrainCandidate:
    """Convert DetectionResult to GrainCandidate for texture/edge validation."""
    from core.multiscale_detector import GrainCandidate

    x, y, w, h = cv2.boundingRect(result.contour)
    h_img, w_img = image.shape[:2]

    # Compute border distance
    border_distance = min(x, y, w_img - (x + w), h_img - (y + h))

    # Compute solidity
    hull = cv2.convexHull(result.contour)
    hull_area = cv2.contourArea(hull)
    solidity = result.area / hull_area if hull_area > 0 else 0.0

    return GrainCandidate(
        contour=result.contour,
        mask=result.mask,
        area=result.area,
        perimeter=result.perimeter,
        circularity=result.circularity,
        aspect_ratio=result.aspect_ratio,
        major_axis=result.major_axis,
        minor_axis=result.minor_axis,
        convexity=result.convexity,
        is_flocculation=result.is_flocculation,
        border_distance=float(border_distance),
        solidity=solidity,
    )
```

### Step 3.4: 运行现有测试确保无回归

```bash
pytest tests/ -v --tb=short
```

Expected: All existing tests pass (or same failures as before)

### Step 3.5: Commit

```bash
git add core/pipeline.py
git commit -m "feat: integrate texture/edge validator into detection pipeline

- Add TextureEdgeValidator after detect_grains
- Add _detection_result_to_candidate helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 添加可选依赖和文档

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

### Step 4.1: 添加可选依赖

在 `pyproject.toml` 的 `[project.optional-dependencies]` 下添加 `texture` 组：

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
]
texture = [
    "scikit-image>=0.22",
]
```

### Step 4.2: 在 requirements.txt 中添加注释

在 `requirements.txt` 末尾添加：

```
# Optional: for advanced texture features (LBP/GLCM)
# scikit-image>=0.22
```

### Step 4.3: Commit

```bash
git add pyproject.toml requirements.txt
git commit -m "chore: add optional scikit-image dependency for texture features

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 自审检查清单

### Spec 覆盖检查

| Spec 章节 | 实现任务 | 状态 |
|-----------|----------|------|
| 4.1.1 LBP 特征 | Task 1.4 | ✅ |
| 4.1.2 GLCM 特征 | Task 1.4 | ✅ |
| 4.1.3 纹理一致性评分 | Task 1.4 | ✅ |
| 4.1.4 OpenCV 回退 | Task 1.6 | ✅ |
| 4.2.1 Sobel 边缘强度 | Task 1.5 | ✅ |
| 4.2.2 边缘方向一致性 | Task 1.5 | ✅ |
| 4.2.3 边缘闭合度 | Task 1.5 | ✅ |
| 4.3 TextureEdgeValidator | Task 1.3 | ✅ |
| 5.1 局部对比度增强 | 未实现（YAGNI，后续需要时添加） | ⏸️ |
| 6.1 Pipeline 集成 | Task 3 | ✅ |
| 7.1 单元测试 | Task 2 | ✅ |

### Placeholder 扫描

- [x] 无 "TBD" / "TODO" / "implement later"
- [x] 无 "Add appropriate error handling" 等模糊描述
- [x] 无 "Similar to Task N" 引用
- [x] 每个步骤都有完整代码或命令

### 类型一致性

- [x] `ValidationConfig` 字段名与 spec 一致
- [x] `TextureEdgeValidator.validate()` 签名与 spec 一致
- [x] `GrainCandidate` 字段名与现有代码一致
- [x] `_detection_result_to_candidate` 返回类型正确

---

## 执行选项

**Plan complete and saved to `docs/superpowers/plans/2025-06-22-texture-edge-filtering-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
