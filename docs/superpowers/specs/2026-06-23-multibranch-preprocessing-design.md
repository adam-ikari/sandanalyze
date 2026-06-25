# 多分支预处理融合检测设计文档

**日期**: 2026-06-23  
**作者**: Claude Code  
**状态**: 已批准  
**相关**: 沙粒形态分析系统 (SandAnalyze)

---

## 1. 背景与问题

当前 SandAnalyze 的检测流程使用单一的自适应阈值预处理路径：

```
图像 → CLAHE → 高斯模糊 → 自适应阈值 → 形态学 → 轮廓检测
```

该流程假设沙粒与背景有足够的亮度对比，阈值化后能得到清晰的闭合边界。但实际场景中存在以下问题：

1. **沙粒可能是几乎全黑的色块**：与背景对比度极低，自适应阈值无法切到边界。
2. **边界模糊**：对焦、光照不均或遮挡导致边界断断续续，无法形成闭合轮廓。
3. **纯色沙粒被误过滤**：`SimpleValidator` 的低对比度过滤（std < 5.0）会将纯色均匀区域当作背景过滤掉。

## 2. 设计目标

1. **预处理阶段**：多种处理融合，生成多通道特征，放大低对比度沙粒的边缘特征。
2. **纹理过滤**：移除低对比度过滤，纯色黑块不再被过滤。
3. **形态分割和过滤**：在融合后的统一掩码上继续优化。

## 3. 架构设计

### 3.1 核心思想

将预处理从"单一流水线"改为"多分支独立检测 + 结果融合"。每个分支是一个独立的"证据源"，各自用不同的物理特征来检测沙粒的存在。

### 3.2 整体架构

```
原始图像
  ├── 分支 1（亮度分支）：CLAHE + 自适应阈值 → 高对比度沙粒
  ├── 分支 2（边缘分支）：梯度增强 + 边缘检测 → 模糊边界沙粒
  ├── 分支 3（纹理分支）：局部方差 + 均匀性检测 → 纯色黑块
  └── 融合：并集 → 形态学清理 → 分割 → 过滤 → 轮廓提取
```

### 3.3 融合策略

三个分支独立运行，结果取**并集**（OR 操作）。

**为什么用并集而非交集？**
- 交集要求所有分支都检测到同一个沙粒，过于严格，会漏检。
- 并集确保：只要有一个分支检测到，就保留。
- 误检（假阳性）由后续的形态学过滤和验证步骤处理。

## 4. 分支设计

### 4.1 分支 1 — 亮度分支（Brightness Branch）

**目的**：检测与背景有明显亮度差异的沙粒。这是现有流程的核心能力。

**处理流程**：

```
原始图像
  → CLAHE (clipLimit=2.0, tileGridSize=8x8)
  → 高斯模糊 (kernel=5x5)
  → 自适应阈值 (block=51, C=5, THRESH_BINARY_INV)
  → 形态学开运算 (kernel=3x3, 1 iteration)
  → 二值掩码输出
```

**参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `blur_kernel` | 5 | 标准模糊 |
| `adaptive_block_size` | 51 | 默认块大小 |
| `adaptive_c` | 5 | 默认阈值偏移 |
| `morph_kernel_size` | 3 | 形态学核 |
| `morph_open_iter` | 1 | 开运算次数 |

**适用场景**：沙粒与背景有明显亮度差异，边界相对清晰。

**实现**：基本保持现有 `preprocess()` 的亮度处理逻辑。

### 4.2 分支 2 — 边缘分支（Edge Branch）

**目的**：检测边界模糊、与背景对比度低的沙粒。通过梯度增强和边缘检测来捕捉微弱的亮度变化。

**处理流程**：

```
原始图像
  → CLAHE (clipLimit=4.0, 更强的对比度增强)
  → 高斯模糊 (kernel=3x3, 更小的模糊保留边缘)
  → 梯度增强 (Unsharp Mask 或 Laplacian 锐化)
  → 自适应阈值 (更小的 block_size, 更低的 C)
  → 形态学开运算
  → 二值掩码输出
```

**关键步骤详解**：

**1. 梯度增强（核心）**

```python
# Unsharp Mask 锐化
blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

# 或者 Laplacian 锐化
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpened = gray - 0.5 * laplacian
```

**2. 边缘检测辅助（可选）**

```python
# Canny 边缘检测
edges = cv2.Canny(sharpened, low_threshold=30, high_threshold=100)

# 或者 Sobel 梯度幅值
sobelx = cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
```

**3. 自适应阈值（更敏感的参数）**

```python
# 更小的 block_size → 更局部的阈值
# 更低的 C → 更容易检测到暗区域
thresh = cv2.adaptiveThreshold(
    sharpened, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=21,  # 比默认 51 更小
    C=2            # 比默认 5 更低
)
```

**参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `edge_clip_limit` | 4.0 | 更强的 CLAHE |
| `edge_blur_kernel` | 3 | 更小的模糊，保留边缘 |
| `edge_adaptive_block_size` | 21 | 更局部的阈值 |
| `edge_adaptive_c` | 2-3 | 更低的阈值，更敏感 |
| `morph_kernel_size` | 3 | 形态学核 |

**适用场景**：沙粒与背景对比度低，边界模糊（对焦问题、光照不均）。

### 4.3 分支 3 — 纹理分支（Texture Branch）

**目的**：检测内部均匀、与背景有微弱亮度差异的纯色沙粒。通过局部方差和均匀性检测来定位"内部一致但与背景不同"的区域。

**处理流程**：

```
原始图像
  → 高斯模糊 (kernel=5x5, 降噪)
  → 局部均值滤波 (kernel=11x11, 计算局部背景)
  → 局部方差计算 (kernel=11x11, 检测均匀区域)
  → 方差阈值分割 (低方差 = 均匀区域)
  → 亮度差分割 (与局部背景有明显差异)
  → 形态学闭运算 (连接碎片)
  → 二值掩码输出
```

**关键步骤详解**：

**1. 局部均值（估计背景）**

```python
local_mean = cv2.blur(gray.astype(np.float32), (11, 11))
```

**2. 局部方差（检测均匀区域）**

```python
local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (11, 11))
local_var = local_sq_mean - local_mean ** 2
local_std = np.sqrt(np.abs(local_var))

# 低方差 = 均匀区域（可能是沙粒内部或纯背景）
uniform_mask = (local_std < std_threshold).astype(np.uint8) * 255
```

**3. 亮度差分割（区分沙粒和背景）**

```python
# 计算每个像素与局部均值的差异
diff = np.abs(gray.astype(np.float32) - local_mean)

# 暗区域（比背景暗）且均匀 = 可能是沙粒
dark_uniform = (diff > diff_threshold) & (local_std < std_threshold)
```

**4. 形态学闭运算**

```python
# 连接被噪声打断的均匀区域
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(
    dark_uniform.astype(np.uint8) * 255,
    cv2.MORPH_CLOSE,
    kernel
)
```

**参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `texture_window` | 11 | 局部统计窗口 |
| `texture_std_threshold` | 5-10 | 低于此值视为均匀区域 |
| `texture_diff_threshold` | 10-20 | 与局部背景的亮度差异 |
| `morph_kernel_size` | 5 | 闭运算连接碎片 |
| `morph_close_iter` | 2 | 闭运算次数 |

**适用场景**：沙粒是几乎全黑的纯色块，内部均匀，与背景有微弱但一致的亮度差异。

**关键设计点**：
- 不依赖"边缘"概念，而是依赖"区域均匀性"。
- 纯色黑块在局部方差图上表现为低方差区域。
- 通过与局部背景的亮度差来区分"暗的沙粒"和"暗的背景"。

## 5. 融合与后处理

### 5.1 融合逻辑

```python
def fuse_masks(masks: list[np.ndarray]) -> np.ndarray:
    """融合多个二值掩码（并集）。"""
    fused = np.zeros_like(masks[0])
    for mask in masks:
        fused = cv2.bitwise_or(fused, mask)
    return fused
```

### 5.2 形态学清理

融合后的掩码可能有噪声点、粘连区域、空洞：

```python
# 形态学开运算：去除小噪声
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel, iterations=1)

# 形态学闭运算：填充小空洞
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
```

### 5.3 分割（复用现有实现）

```python
# 分水岭分割
split_mask = split_by_watershed(closed, min_circularity=0.3)

# 凹点分割
split_mask = split_by_concave_points(split_mask, min_concave_depth=5)
```

### 5.4 过滤（修改后的验证器）

修改 `SimpleValidator`，移除低对比度过滤：

```python
@dataclass
class ValidationConfig:
    lens_edge_margin: float = 0.05
    lens_edge_circularity: float = 0.7
    lens_edge_min_area: float = 50000
    noise_max_area: int = 500
    # 移除 min_contrast，不再过滤低对比度
```

保留：
- `_is_lens_edge`：大圆形边界伪影
- `_is_noise`：极小区域

移除：
- `_is_low_contrast`：纯色黑块不再被过滤

### 5.5 轮廓提取与后续流程

融合后的掩码进入标准的 `detect_grains()` 后续流程：轮廓检测 → 特征计算 → 分类 → 统计。

## 6. API 设计

### 6.1 `PreprocessConfig` 扩展

```python
@dataclass
class PreprocessConfig:
    # 亮度分支参数
    blur_kernel: int = 5
    adaptive_block_size: int = 51
    adaptive_c: int = 5
    
    # 边缘分支参数
    edge_clip_limit: float = 4.0
    edge_blur_kernel: int = 3
    edge_adaptive_block_size: int = 21
    edge_adaptive_c: int = 2
    
    # 纹理分支参数
    texture_window: int = 11
    texture_std_threshold: float = 10.0
    texture_diff_threshold: float = 15.0
    
    # 通用参数
    morph_kernel_size: int = 3
    morph_open_iter: int = 1
    morph_close_iter: int = 1
    min_area: int = 800
    use_clahe: bool = True
```

### 6.2 新增预处理函数

```python
# core/preprocessor.py

def _preprocess_brightness(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """亮度分支：高对比度沙粒检测。"""
    ...

def _preprocess_edge(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """边缘分支：模糊边界沙粒检测。"""
    ...

def _preprocess_texture(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """纹理分支：纯色黑块检测。"""
    ...

def _fuse_masks(masks: list[np.ndarray]) -> np.ndarray:
    """融合多个二值掩码（并集）。"""
    ...
```

### 6.3 `preprocess()` 函数重构

```python
def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """多分支预处理融合。
    
    生成三个分支的掩码，融合后返回统一的二值掩码。
    """
    if config is None:
        config = PreprocessConfig()
    
    # 三个分支处理
    brightness_mask = _preprocess_brightness(image, config)
    edge_mask = _preprocess_edge(image, config)
    texture_mask = _preprocess_texture(image, config)
    
    # 融合
    fused = _fuse_masks([brightness_mask, edge_mask, texture_mask])
    
    # 形态学清理
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
    )
    opened = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iter)
    
    return closed
```

### 6.4 `detect_grains()` 调用不变

`detect_grains()` 内部调用 `preprocess()`，而 `preprocess()` 已经返回融合后的掩码。`detect_grains()` 的接口无需修改。

### 6.5 `SimpleValidator` 修改

```python
@dataclass
class ValidationConfig:
    lens_edge_margin: float = 0.05
    lens_edge_circularity: float = 0.7
    lens_edge_min_area: float = 50000
    noise_max_area: int = 500
    # 移除 min_contrast，不再过滤低对比度
```

### 6.6 `app.py` 参数调整

- 移除 `texture_score_threshold` 相关参数（纹理过滤已简化）。
- 添加多分支参数调节（可选）。

## 7. 参数调试策略

由于三个分支的参数需要协同调试，建议以下策略：

1. **独立调试**：先分别调试每个分支，确保各自能检测到目标类型的沙粒。
2. **两两融合**：先融合亮度+边缘，调试；再加入纹理分支。
3. **假阳性分析**：统计每个分支的误检率，调整参数或考虑分支权重。
4. **交叉验证**：使用标注数据集验证融合效果。

## 8. 测试策略

1. **单元测试**：每个分支独立测试。
2. **融合测试**：验证并集逻辑正确。
3. **端到端测试**：完整流程测试。
4. **回归测试**：确保现有高对比度沙粒检测不受影响。

## 9. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 融合后假阳性增加 | 保留形态学过滤和验证器，仅移除低对比度过滤 |
| 参数调优复杂 | 分阶段调试，先独立后融合 |
| 性能下降（三分支计算） | 分支计算可并行化（OpenCV 操作已优化） |
| 回归（现有检测受影响） | 亮度分支保持现有参数，确保高对比度场景不受影响 |

## 10. 实现计划

1. **Phase 1**：实现三个分支的独立预处理函数。
2. **Phase 2**：实现融合逻辑和形态学清理。
3. **Phase 3**：修改 `SimpleValidator`，移除低对比度过滤。
4. **Phase 4**：更新 `app.py` 参数和 UI。
5. **Phase 5**：测试和参数调试。

---

**批准状态**：待用户确认后进入实现阶段。
