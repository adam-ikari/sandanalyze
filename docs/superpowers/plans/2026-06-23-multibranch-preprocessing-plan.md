# 多分支预处理融合检测实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将预处理从单一流水线重构为多分支（亮度/边缘/纹理）融合检测，并移除纹理过滤中的低对比度过滤。

**Architecture:** 三个独立预处理分支各自生成二值掩码，通过并集融合，再经形态学清理后进入检测流程。`SimpleValidator`移除低对比度过滤。

**Tech Stack:** Python 3.10+, OpenCV, NumPy, pytest

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `core/preprocessor.py` | 修改 | 扩展 `PreprocessConfig`，重构 `preprocess()`，新增三个分支函数和融合函数 |
| `core/texture_edge_filter.py` | 修改 | 移除 `ValidationConfig.min_contrast`，移除 `_is_low_contrast` 调用 |
| `tests/test_preprocessor.py` | 修改 | 新增多分支预处理测试 |
| `tests/test_texture_edge_filter.py` | 修改 | 更新测试，移除低对比度相关测试 |

---

## Task 1: 扩展 PreprocessConfig

**Files:**
- Modify: `core/preprocessor.py:14-35`

**目标：** 在 `PreprocessConfig` 中新增多分支参数。

- [ ] **Step 1: 修改 PreprocessConfig，添加边缘和纹理分支参数**

```python
@dataclass
class PreprocessConfig:
    """Configuration for the image preprocessing pipeline."""

    # 亮度分支参数（现有）
    blur_kernel: int = 5
    adaptive_block_size: int = 51
    adaptive_c: int = 5

    # 边缘分支参数（新增）
    edge_clip_limit: float = 4.0
    edge_blur_kernel: int = 3
    edge_adaptive_block_size: int = 21
    edge_adaptive_c: int = 2

    # 纹理分支参数（新增）
    texture_window: int = 11
    texture_std_threshold: float = 10.0
    texture_diff_threshold: float = 15.0

    # 通用参数（现有）
    morph_kernel_size: int = 3
    morph_open_iter: int = 1
    morph_close_iter: int = 1
    min_area: int = 800
    use_clahe: bool = True
```

- [ ] **Step 2: 运行现有测试，确保未破坏**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add core/preprocessor.py
git commit -m "feat: extend PreprocessConfig with multibranch params"
```

---

## Task 2: 实现亮度分支函数

**Files:**
- Modify: `core/preprocessor.py`

**目标：** 将现有 `preprocess()` 的核心逻辑提取为 `_preprocess_brightness()`。

- [ ] **Step 1: 在 `preprocess()` 之前添加 `_preprocess_brightness()` 函数**

```python
def _preprocess_brightness(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """亮度分支：高对比度沙粒检测。

    基本保持现有 preprocess() 的核心逻辑（不含 watershed 和 area filtering）。

    Args:
        image: Input image (grayscale or color).
        config: Preprocessing configuration.

    Returns:
        Binary mask (uint8) with foreground grains as 255.
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Optional CLAHE
    if config.use_clahe:
        clip_limit = 4.0 if config.adaptive_block_size >= 91 else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 3. Gaussian blur
    blurred = cv2.GaussianBlur(gray, (config.blur_kernel, config.blur_kernel), 0)

    # 4. Adaptive threshold
    adaptive_block_size = config.adaptive_block_size
    if adaptive_block_size >= 91:
        adaptive_block_size = 21

    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        config.adaptive_c,
    )

    # 5. Morphological open
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
    )
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iter)

    return opened
```

- [ ] **Step 2: Commit**

```bash
git add core/preprocessor.py
git commit -m "feat: extract brightness branch preprocessing"
```

---

## Task 3: 实现边缘分支函数

**Files:**
- Modify: `core/preprocessor.py`

**目标：** 实现边缘分支，使用更强的 CLAHE、更小的模糊、梯度增强和更敏感的阈值。

- [ ] **Step 1: 在 `_preprocess_brightness()` 之后添加 `_preprocess_edge()` 函数**

```python
def _preprocess_edge(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """边缘分支：模糊边界沙粒检测。

    使用更强的 CLAHE、梯度增强和更敏感的阈值来捕捉模糊边界。

    Args:
        image: Input image (grayscale or color).
        config: Preprocessing configuration.

    Returns:
        Binary mask (uint8) with foreground grains as 255.
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Strong CLAHE
    if config.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=config.edge_clip_limit, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 3. Smaller Gaussian blur (preserve edges)
    blur_kernel = config.edge_blur_kernel
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # 4. Gradient enhancement (Unsharp Mask)
    blurred_for_sharpen = cv2.GaussianBlur(blurred, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(blurred, 1.5, blurred_for_sharpen, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 5. Adaptive threshold with more sensitive parameters
    edge_block_size = config.edge_adaptive_block_size
    if edge_block_size % 2 == 0:
        edge_block_size += 1

    thresh = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        edge_block_size,
        config.edge_adaptive_c,
    )

    # 6. Morphological open
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
    )
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iter)

    return opened
```

- [ ] **Step 2: Commit**

```bash
git add core/preprocessor.py
git commit -m "feat: add edge branch preprocessing for blurry boundaries"
```

---

## Task 4: 实现纹理分支函数

**Files:**
- Modify: `core/preprocessor.py`

**目标：** 实现纹理分支，使用局部方差和亮度差检测纯色均匀区域。

- [ ] **Step 1: 在 `_preprocess_edge()` 之后添加 `_preprocess_texture()` 函数**

```python
def _preprocess_texture(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """纹理分支：纯色黑块检测。

    通过局部方差和亮度差来定位内部均匀、与背景有微弱差异的区域。

    Args:
        image: Input image (grayscale or color).
        config: Preprocessing configuration.

    Returns:
        Binary mask (uint8) with foreground grains as 255.
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Gaussian blur (noise reduction)
    blurred = cv2.GaussianBlur(gray, (config.blur_kernel, config.blur_kernel), 0)

    # 3. Local mean (background estimation)
    window = config.texture_window
    if window % 2 == 0:
        window += 1

    gray_f = blurred.astype(np.float32)
    local_mean = cv2.blur(gray_f, (window, window))

    # 4. Local variance
    local_sq_mean = cv2.blur(gray_f ** 2, (window, window))
    local_var = np.abs(local_sq_mean - local_mean ** 2)
    local_std = np.sqrt(local_var)

    # 5. Brightness difference from local mean
    diff = np.abs(gray_f - local_mean)

    # 6. Detect dark uniform regions (potential solid black grains)
    # Condition: low variance AND significant brightness difference
    is_uniform = local_std < config.texture_std_threshold
    is_different = diff > config.texture_diff_threshold
    dark_uniform = is_uniform & is_different

    # 7. Convert to binary mask
    mask = (dark_uniform.astype(np.uint8)) * 255

    # 8. Morphological close to connect fragments
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size + 2, config.morph_kernel_size + 2)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iter + 1)

    return closed
```

- [ ] **Step 2: Commit**

```bash
git add core/preprocessor.py
git commit -m "feat: add texture branch preprocessing for uniform dark grains"
```

---

## Task 5: 实现融合函数

**Files:**
- Modify: `core/preprocessor.py`

**目标：** 实现掩码融合函数，使用并集操作。

- [ ] **Step 1: 在 `_preprocess_texture()` 之后添加 `_fuse_masks()` 函数**

```python
def _fuse_masks(masks: list[np.ndarray]) -> np.ndarray:
    """融合多个二值掩码（并集操作）。

    Args:
        masks: List of binary masks (uint8).

    Returns:
        Binary mask (uint8) with foreground as 255.
    """
    if not masks:
        raise ValueError("At least one mask is required")

    fused = np.zeros_like(masks[0])
    for mask in masks:
        fused = cv2.bitwise_or(fused, mask)
    return fused
```

- [ ] **Step 2: Commit**

```bash
git add core/preprocessor.py
git commit -m "feat: add mask fusion function (union)"
```

---

## Task 6: 重构主 preprocess() 函数

**Files:**
- Modify: `core/preprocessor.py:298-394`

**目标：** 将 `preprocess()` 重构为调用三个分支并融合结果。

- [ ] **Step 1: 重写 `preprocess()` 函数**

替换现有 `preprocess()` 函数（从第298行开始）：

```python
def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Run the full preprocessing pipeline on a sand image.

    Pipeline:
        1. Brightness branch: detect high-contrast grains.
        2. Edge branch: detect blurry-boundary grains.
        3. Texture branch: detect uniform dark grains.
        4. Fuse masks (union).
        5. Morphological open/close.
        6. Watershed splitting.
        7. Area filtering.

    Args:
        image: Input image (grayscale or color).
        config: Preprocessing configuration. Uses defaults if None.

    Returns:
        Binary mask (uint8) with foreground grains as 255 and background as 0.
    """
    if config is None:
        config = PreprocessConfig()

    # Run three branches
    brightness_mask = _preprocess_brightness(image, config)
    edge_mask = _preprocess_edge(image, config)
    texture_mask = _preprocess_texture(image, config)

    # Fuse masks (union)
    fused = _fuse_masks([brightness_mask, edge_mask, texture_mask])

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
    )
    opened = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iter)

    # Watershed splitting to separate touching grains
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Threshold to find sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # Find unknown region
    sure_bg = cv2.dilate(closed, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    color_img = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)

    # Create mask from watershed result
    watershed_mask = np.zeros_like(closed)
    watershed_mask[markers > 1] = 255

    # Area filtering
    mask = _filter_by_area(watershed_mask, config.min_area)

    return mask
```

- [ ] **Step 2: 运行现有测试**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: PASS (或根据测试调整)

- [ ] **Step 3: Commit**

```bash
git add core/preprocessor.py
git commit -m "refactor: preprocess() uses multibranch fusion"
```

---

## Task 7: 修改 SimpleValidator 移除低对比度过滤

**Files:**
- Modify: `core/texture_edge_filter.py:19-35`

**目标：** 移除 `ValidationConfig.min_contrast`，在 `validate()` 中移除 `_is_low_contrast` 调用。

- [ ] **Step 1: 修改 `ValidationConfig`，移除 `min_contrast`**

```python
@dataclass
class ValidationConfig:
    """Configuration for lightweight false-positive filtering."""

    lens_edge_margin: float = 0.05
    lens_edge_circularity: float = 0.7
    lens_edge_min_area: float = 50000
    noise_max_area: int = 500
    # Removed: min_contrast — solid black grains should not be filtered
```

- [ ] **Step 2: 修改 `validate()`，移除低对比度检查**

```python
    def validate(self, candidate: object, full_image: np.ndarray) -> bool:
        """Validate a grain candidate.

        Only rejects obvious false positives. Real grains are preserved.

        Args:
            candidate: GrainCandidate instance (must have area, circularity,
                contour, border_distance attributes).
            full_image: Original input image (grayscale or BGR).

        Returns:
            True if candidate is likely a real grain.
        """
        # 1. Reject lens-edge artifacts
        if self._is_lens_edge(candidate, full_image):
            return False

        # 2. Reject noise/fragments
        if self._is_noise(candidate, full_image):
            return False

        # Removed: low contrast check — solid black grains are valid

        return True
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_texture_edge_filter.py -v
```

Expected: 部分测试可能失败，需要更新测试

- [ ] **Step 4: Commit**

```bash
git add core/texture_edge_filter.py
git commit -m "feat: remove low-contrast filtering from SimpleValidator"
```

---

## Task 8: 更新 texture_edge_filter 测试

**Files:**
- Modify: `tests/test_texture_edge_filter.py`

**目标：** 更新测试以反映 `ValidationConfig` 的变更，移除低对比度相关测试。

- [ ] **Step 1: 更新初始化测试**

将 `test_validator_initialization` 和 `test_validator_with_custom_config` 中的 `min_contrast` 断言移除。

- [ ] **Step 2: 移除低对比度检测测试**

删除或注释掉：
- `test_low_contrast_rejection`
- `test_normal_contrast_acceptance`

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_texture_edge_filter.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_texture_edge_filter.py
git commit -m "test: update texture_edge_filter tests for removed low-contrast filter"
```

---

## Task 9: 新增多分支预处理测试

**Files:**
- Modify: `tests/test_preprocessor.py`

**目标：** 测试三个分支和融合逻辑。

- [ ] **Step 1: 添加分支函数导入**

在文件顶部添加：

```python
from core.preprocessor import (
    PreprocessConfig,
    preprocess,
    _preprocess_brightness,
    _preprocess_edge,
    _preprocess_texture,
    _fuse_masks,
)
```

- [ ] **Step 2: 添加亮度分支测试**

```python
class TestBrightnessBranch:
    """Tests for the brightness preprocessing branch."""

    def test_brightness_branch_returns_binary_mask(self, sample_grain_image):
        """Brightness branch should return a binary mask."""
        config = PreprocessConfig()
        result = _preprocess_brightness(sample_grain_image, config)

        assert result.dtype == np.uint8
        assert set(np.unique(result).tolist()).issubset({0, 255})
        assert result.shape == sample_grain_image.shape
```

- [ ] **Step 3: 添加边缘分支测试**

```python
class TestEdgeBranch:
    """Tests for the edge preprocessing branch."""

    def test_edge_branch_returns_binary_mask(self, sample_grain_image):
        """Edge branch should return a binary mask."""
        config = PreprocessConfig()
        result = _preprocess_edge(sample_grain_image, config)

        assert result.dtype == np.uint8
        assert set(np.unique(result).tolist()).issubset({0, 255})
        assert result.shape == sample_grain_image.shape
```

- [ ] **Step 4: 添加纹理分支测试**

```python
class TestTextureBranch:
    """Tests for the texture preprocessing branch."""

    def test_texture_branch_returns_binary_mask(self, sample_grain_image):
        """Texture branch should return a binary mask."""
        config = PreprocessConfig()
        result = _preprocess_texture(sample_grain_image, config)

        assert result.dtype == np.uint8
        assert set(np.unique(result).tolist()).issubset({0, 255})
        assert result.shape == sample_grain_image.shape
```

- [ ] **Step 5: 添加融合测试**

```python
class TestMaskFusion:
    """Tests for mask fusion."""

    def test_fuse_masks_union(self):
        """Fusing masks should return the union of all masks."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask3 = np.zeros((100, 100), dtype=np.uint8)

        # Different regions in each mask
        cv2.circle(mask1, (30, 30), 10, 255, -1)
        cv2.circle(mask2, (60, 60), 10, 255, -1)
        cv2.circle(mask3, (90, 90), 10, 255, -1)

        fused = _fuse_masks([mask1, mask2, mask3])

        # All three regions should be present
        assert fused[30, 30] == 255
        assert fused[60, 60] == 255
        assert fused[90, 90] == 255
```

- [ ] **Step 6: 运行测试**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_preprocessor.py
git commit -m "test: add multibranch preprocessing tests"
```

---

## Task 10: 端到端测试

**Files：**
- 运行现有测试

**目标：** 确保整个流程未被破坏。

- [ ] **Step 1: 运行所有测试**

```bash
pytest tests/ -v --tb=short
```

Expected: 大部分 PASS，检查是否有回归

- [ ] **Step 2: 运行检测器测试**

```bash
pytest tests/test_detector.py -v
```

Expected: PASS

- [ ] **Step 3: 运行流水线测试**

```bash
pytest tests/test_pipeline.py -v
```

Expected: PASS

- [ ] **Step 4: Commit（如有修复）**

```bash
git commit -m "test: verify multibranch preprocessing integration"
```

---

## Spec 覆盖检查

| Spec 章节 | 实现任务 | 状态 |
|-----------|---------|------|
| 4.1 亮度分支 | Task 2 | ✅ |
| 4.2 边缘分支 | Task 3 | ✅ |
| 4.3 纹理分支 | Task 4 | ✅ |
| 5.1 融合逻辑 | Task 5 | ✅ |
| 5.2 形态学清理 | Task 6 | ✅ |
| 5.4 过滤修改 | Task 7 | ✅ |
| 6.1 PreprocessConfig 扩展 | Task 1 | ✅ |
| 6.3 preprocess() 重构 | Task 6 | ✅ |
| 6.6 SimpleValidator 修改 | Task 7 | ✅ |
| 测试 | Task 8, 9, 10 | ✅ |

---

**计划完成。** 保存到 `docs/superpowers/plans/2026-06-23-multibranch-preprocessing-plan.md`。

**执行选项：**

1. **Subagent-Driven（推荐）** - 每个 Task 由独立子代理执行，我在中间审查
2. **Inline Execution** - 在当前会话中批量执行

**请选择执行方式。**
