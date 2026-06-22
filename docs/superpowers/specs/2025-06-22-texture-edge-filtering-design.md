# 纹理与边缘特征过滤方法设计文档

**日期**: 2025-06-22
**主题**: 基于纹理和边缘特征的沙砾检测过滤方法
**状态**: 待审批

---

## 1. 背景与问题定义

当前沙砾检测系统基于自适应阈值 + 形态学操作 + 多特征过滤（面积、圆度、长宽比、凸性）。在显微镜40倍照片场景下，存在三个核心问题：

1. **大块沙砾漏检**：大颗粒与背景对比度低，自适应阈值无法有效分割
2. **镜头边缘误检**：显微镜圆形视野边界被当成沙砾
3. **噪声误识别为小沙砾**：显微镜40倍下不存在极小颗粒，噪声被当成真实沙砾

## 2. 设计目标

- **减少漏检**：通过纹理特征捕获低对比度大颗粒
- **消除边缘伪影**：通过边缘梯度+位置联合过滤剔除镜头边界
- **抑制噪声**：通过纹理一致性+尺寸联合阈值过滤噪声

## 3. 架构设计

在现有 `detect_grains` → `filter_strict` 流程中，增加一个**纹理/边缘验证层**（`TextureEdgeValidator`），形成三层过滤体系：

```
┌─────────────────────────────────────────────────────────────┐
│                    检测流程 (detect_grains)                    │
├─────────────────────────────────────────────────────────────┤
│  1. 自适应阈值 + 形态学操作                                   │
│  2. 连通区域提取 → 候选列表                                   │
│  3. 基础形态过滤 (面积、圆度、长宽比)                         │
├─────────────────────────────────────────────────────────────┤
│  4. 【新增】纹理/边缘验证层 (TextureEdgeValidator)            │
│     ├─ 纹理特征提取 (LBP + GLCM)                            │
│     ├─ 边缘梯度分析 (Sobel + 边缘方向一致性)                  │
│     └─ 综合评分 → 保留/剔除                                  │
├─────────────────────────────────────────────────────────────┤
│  5. 最终形态过滤 (filter_strict)                            │
└─────────────────────────────────────────────────────────────┘
```

## 4. 核心模块设计

### 4.1 纹理特征提取 (TextureFeatureExtractor)

#### 4.1.1 局部二值模式 (LBP)

对每个候选区域的 ROI 计算均匀 LBP 直方图：

- **实现**: `cv2` 无内置 LBP，使用 `skimage.feature.local_binary_pattern`
- **参数**: `P=8` (邻域点数), `R=1` (半径), `method='uniform'`
- **输出**: 10 维特征向量 (uniform patterns: 0-9)

```python
from skimage.feature import local_binary_pattern

def extract_lbp_features(roi_gray: np.ndarray) -> np.ndarray:
    """提取 LBP 特征向量."""
    lbp = local_binary_pattern(roi_gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist
```

#### 4.1.2 灰度共生矩阵 (GLCM)

计算 ROI 的 GLCM 并提取纹理特征：

- **实现**: `skimage.feature.graycomatrix` + `skimage.feature.graycoprops`
- **参数**: `distances=[1]`, `angles=[0, π/4, π/2, 3π/4]`
- **输出**: 对比度、相异性、同质性、能量、相关性

```python
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(roi_gray: np.ndarray) -> dict[str, float]:
    """提取 GLCM 纹理特征."""
    # 量化到 32 级
    roi_32 = (roi_gray / 255 * 31).astype(np.uint8)
    glcm = graycomatrix(roi_32, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    
    features = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        features[prop] = graycoprops(glcm, prop).mean()
    return features
```

#### 4.1.3 纹理一致性评分

综合 LBP 和 GLCM 特征，计算纹理一致性评分：

```python
def compute_texture_consistency_score(lbp_features: np.ndarray, 
                                       glcm_features: dict) -> float:
    """计算纹理一致性评分 (0-1, 越高越像真实沙砾).
    
    真实沙砾特征：
    - LBP 直方图集中（非均匀分布）
    - GLCM 对比度适中（0.1-0.5）
    - GLCM 同质性较高（>0.3）
    - GLCM 能量适中（0.05-0.3）
    """
    # LBP 集中度 (熵的反面)
    lbp_entropy = -np.sum(lbp_features * np.log(lbp_features + 1e-7))
    lbp_concentration = 1.0 / (1.0 + lbp_entropy)
    
    # GLCM 评分
    contrast = glcm_features['contrast']
    homogeneity = glcm_features['homogeneity']
    energy = glcm_features['energy']
    
    # 对比度应在合理范围内
    contrast_score = 1.0 - abs(contrast - 0.3) / 0.3
    contrast_score = max(0, contrast_score)
    
    # 同质性
    homo_score = min(homogeneity / 0.3, 1.0)
    
    # 能量
    energy_score = 1.0 - abs(energy - 0.15) / 0.15
    energy_score = max(0, energy_score)
    
    # 综合评分
    score = (lbp_concentration * 0.3 + 
             contrast_score * 0.25 + 
             homo_score * 0.25 + 
             energy_score * 0.2)
    
    return min(score, 1.0)
```

### 4.2 边缘梯度分析 (EdgeGradientAnalyzer)

#### 4.2.1 Sobel 边缘强度

```python
def compute_edge_strength(roi_gray: np.ndarray) -> float:
    """计算 ROI 的平均边缘强度."""
    sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    return float(np.mean(sobel_mag))
```

#### 4.2.2 边缘方向一致性

用于区分镜头边缘（方向一致）和真实沙砾（方向多样）：

```python
def compute_edge_direction_consistency(roi_gray: np.ndarray) -> float:
    """计算边缘方向一致性 (0-1, 越高越一致).
    
    镜头边缘：方向高度一致（沿圆周）
    真实沙砾：方向多样
    """
    sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算边缘方向角
    orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
    
    # 只考虑强边缘像素
    mag = np.sqrt(sobelx**2 + sobely**2)
    strong_edges = mag > np.percentile(mag, 75)
    
    if np.sum(strong_edges) < 10:
        return 0.0
    
    # 计算方向分布的集中度
    orientations = orientation[strong_edges].flatten()
    hist, _ = np.histogram(orientations, bins=36, range=(-180, 180))
    hist = hist / hist.sum()
    
    # 集中度 (最大 bin 占比)
    consistency = hist.max()
    return float(consistency)
```

#### 4.2.3 边缘闭合度

真实沙砾有闭合轮廓，噪声/边缘伪影边缘 fragmented：

```python
def compute_edge_closure(contour: np.ndarray, roi_gray: np.ndarray) -> float:
    """计算边缘闭合度 (0-1).
    
    使用 Canny 边缘检测后，检查轮廓附近的边缘像素闭合程度。
    """
    edges = cv2.Canny(roi_gray, 50, 150)
    
    # 创建轮廓 mask
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [contour], -1, 255, thickness=2)
    
    # 计算轮廓上的边缘像素比例
    contour_edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, mask))
    contour_length = cv2.arcLength(contour, True)
    
    if contour_length == 0:
        return 0.0
    
    # 注意: contour_edge_pixels 可能大于 contour_length (边缘线宽 >1)
    # 使用 min 截断到 [0, 1]，表示轮廓上被边缘检测覆盖的比例
    closure_ratio = min(contour_edge_pixels / contour_length, 1.0)
    return closure_ratio
```

### 4.1.4 纯 OpenCV 回退方案 (无 skimage 时)

如果环境中未安装 `scikit-image`，提供纯 OpenCV 实现的简化纹理特征：

```python
def extract_lbp_features_opencv(roi_gray: np.ndarray) -> np.ndarray:
    """使用 OpenCV 实现简化版 LBP 特征.
    
    使用局部标准差作为纹理粗糙度指标，替代 LBP。
    """
    # 计算局部标准差
    kernel_size = 5
    local_mean = cv2.blur(roi_gray.astype(np.float32), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((roi_gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
    local_std = np.sqrt(np.abs(local_sq_mean - local_mean ** 2))
    
    # 直方图
    hist, _ = np.histogram(local_std.ravel(), bins=10, range=(0, 50))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_glcm_features_opencv(roi_gray: np.ndarray) -> dict[str, float]:
    """使用 OpenCV 实现简化版 GLCM 特征.
    
    使用局部对比度和能量作为替代指标。
    """
    # 量化到 16 级
    roi_16 = (roi_gray / 255 * 15).astype(np.uint8)
    
    # 计算局部对比度 (使用拉普拉斯算子)
    laplacian = cv2.Laplacian(roi_16, cv2.CV_64F)
    contrast = float(np.std(laplacian))
    
    # 计算局部能量 (梯度幅值的均值)
    sobelx = cv2.Sobel(roi_16, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi_16, cv2.CV_64F, 0, 1, ksize=3)
    energy = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    
    # 计算同质性 (局部方差的倒数)
    local_var = np.var(roi_gray.astype(np.float32))
    homogeneity = 1.0 / (1.0 + local_var / 100.0)
    
    return {
        'contrast': contrast,
        'dissimilarity': contrast / 2.0,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': 0.5,  # 简化值
    }
```
```

### 4.3 综合验证器 (TextureEdgeValidator)

```python
@dataclass
class ValidationConfig:
    """纹理/边缘验证配置."""
    # 纹理评分阈值
    texture_score_threshold: float = 0.4
    
    # 边缘方向一致性阈值 (超过此值认为是边缘伪影)
    edge_direction_threshold: float = 0.6
    
    # 边缘闭合度阈值
    edge_closure_threshold: float = 0.3
    
    # 镜头边缘过滤参数
    lens_edge_margin: float = 0.05  # 图像宽高的 5%
    lens_edge_circularity: float = 0.7
    lens_edge_min_area: float = 50000
    
    # 噪声过滤参数
    noise_max_texture_score: float = 0.3
    noise_max_edge_strength: float = 30.0
    noise_max_area: int = 500  # 40倍显微镜下不可能有这么小的沙砾


class TextureEdgeValidator:
    """纹理/边缘特征验证器.
    
    对候选区域进行纹理和边缘验证，区分：
    - 真实沙砾 (高纹理评分、低边缘一致性、高边缘闭合度)
    - 镜头边缘伪影 (低纹理评分、高边缘一致性、大圆形)
    - 噪声 (低纹理评分、低边缘闭合度、面积极小)
    """
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
    
    def validate(self, candidate: GrainCandidate, full_image: np.ndarray) -> bool:
        """验证候选区域是否为真实沙砾.
        
        Returns:
            True if valid grain, False if should be filtered out.
        """
        # 1. 镜头边缘检测 (最高优先级)
        if self._is_lens_edge(candidate, full_image):
            return False
        
        # 2. 噪声检测
        if self._is_noise(candidate, full_image):
            return False
        
        # 3. 纹理/边缘综合评分
        score = self._compute_composite_score(candidate, full_image)
        
        # 4. 边缘闭合度单独检查 (防止 fragmented 噪声通过)
        roi = self._extract_roi(candidate, full_image)
        if roi is not None:
            edge_closure = compute_edge_closure(candidate.contour, roi)
            if edge_closure < self.config.edge_closure_threshold:
                return False
        
        return score >= self.config.texture_score_threshold
    
    def _is_lens_edge(self, candidate: GrainCandidate, full_image: np.ndarray) -> bool:
        """检测是否为镜头边缘伪影."""
        h, w = full_image.shape[:2]
        margin = self.config.lens_edge_margin
        
        # 检查是否在图像边缘附近
        x, y, bw, bh = cv2.boundingRect(candidate.contour)
        near_edge = (
            x < w * margin or 
            y < h * margin or 
            x + bw > w * (1 - margin) or 
            y + bh > h * (1 - margin)
        )
        
        # 检查是否为大圆形
        is_large = candidate.area > self.config.lens_edge_min_area
        is_circular = candidate.circularity > self.config.lens_edge_circularity
        
        return near_edge and is_large and is_circular
    
    def _is_noise(self, candidate: GrainCandidate, full_image: np.ndarray) -> bool:
        """检测是否为噪声."""
        # 面积极小
        if candidate.area > self.config.noise_max_area:
            return False
        
        # 提取 ROI
        roi = self._extract_roi(candidate, full_image)
        if roi is None:
            return True
        
        # 纹理评分
        texture_score = self._compute_texture_score(roi)
        if texture_score < self.config.noise_max_texture_score:
            return True
        
        # 边缘强度
        edge_strength = compute_edge_strength(roi)
        if edge_strength < self.config.noise_max_edge_strength:
            return True
        
        return False
    
    def _compute_composite_score(self, candidate: GrainCandidate, 
                                  full_image: np.ndarray) -> float:
        """计算综合评分."""
        roi = self._extract_roi(candidate, full_image)
        if roi is None:
            return 0.0
        
        # 纹理评分
        texture_score = self._compute_texture_score(roi)
        
        # 边缘方向一致性 (越低越好)
        edge_consistency = compute_edge_direction_consistency(roi)
        
        # 边缘闭合度
        edge_closure = compute_edge_closure(candidate.contour, roi)
        
        # 综合评分
        score = (texture_score * 0.5 + 
                 (1 - edge_consistency) * 0.25 + 
                 edge_closure * 0.25)
        
        return score
    
    def _compute_texture_score(self, roi_gray: np.ndarray) -> float:
        """计算纹理评分."""
        lbp = extract_lbp_features(roi_gray)
        glcm = extract_glcm_features(roi_gray)
        return compute_texture_consistency_score(lbp, glcm)
    
    def _extract_roi(self, candidate: GrainCandidate, 
                     full_image: np.ndarray) -> np.ndarray | None:
        """从全图中提取候选区域的 ROI."""
        x, y, w, h = cv2.boundingRect(candidate.contour)
        
        # 添加 padding
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

## 5. 阈值调整策略

### 5.1 自适应阈值调整

针对大块沙砾漏检问题，在预处理阶段增加**局部对比度自适应增强**：

```python
def adaptive_local_enhancement(image: np.ndarray, 
                               candidate: GrainCandidate) -> np.ndarray:
    """对候选区域进行局部对比度增强.
    
    针对大颗粒低对比度问题，在候选区域周围应用更强的 CLAHE。
    """
    x, y, w, h = cv2.boundingRect(candidate.contour)
    
    # 扩展 ROI
    pad = 20
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    
    roi = image[y1:y2, x1:x2]
    
    # 强 CLAHE
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    return enhanced
```

### 5.2 阈值参数表

| 参数 | 默认值 | 说明 | 调整方向 |
|------|--------|------|----------|
| `texture_score_threshold` | 0.4 | 纹理评分阈值 | 降低可减少漏检，提高可减少误检 |
| `edge_direction_threshold` | 0.6 | 边缘方向一致性阈值 | 提高可更严格过滤边缘伪影 |
| `edge_closure_threshold` | 0.3 | 边缘闭合度阈值 | 降低可减少漏检 |
| `lens_edge_margin` | 0.05 | 镜头边缘判定边距 | 根据显微镜视野调整 |
| `lens_edge_circularity` | 0.7 | 镜头边缘圆度阈值 | 提高可更严格过滤 |
| `lens_edge_min_area` | 50000 | 镜头边缘最小面积 | 根据图像分辨率调整 |
| `noise_max_area` | 500 | 噪声最大面积 | 40倍显微镜下固定为500 |
| `noise_max_texture_score` | 0.3 | 噪声纹理评分上限 | 降低可更严格过滤噪声 |
| `noise_max_edge_strength` | 30.0 | 噪声边缘强度上限 | 降低可更严格过滤 |

## 6. 集成方案

### 6.1 与现有 Pipeline 集成

在 `core/pipeline.py` 的 `run_detection_pipeline` 中，在 `detect_grains` 之后、`compute_morphology` 之前插入纹理/边缘验证。

同时，在预处理阶段可选择性启用**局部对比度自适应增强**（§5.1），对检测到的候选区域进行局部 CLAHE 增强，以捕获低对比度大颗粒：

```python
# 现有代码
results = detect_grains(...)

# 【可选】局部对比度增强 (针对低对比度大颗粒)
# 对面积较大但对比度低的候选区域进行局部增强后重新检测
enhanced_results = []
for result in results:
    if result.area > 5000:  # 大颗粒阈值
        enhanced_roi = adaptive_local_enhancement(image, result)
        # 在增强后的 ROI 上重新运行检测
        # ... (具体实现见实现阶段)

# 【新增】纹理/边缘验证
from core.texture_edge_filter import TextureEdgeValidator, ValidationConfig
validator = TextureEdgeValidator(ValidationConfig())
filtered_results = []
for result in results:
    if validator.validate(result, image):
        filtered_results.append(result)
results = filtered_results

# 后续代码不变
for result in results:
    grain = GrainContour(...)
    ...
```

### 6.2 与现有过滤器的关系

```
检测流程:
  detect_grains()
    → 基础形态过滤 (面积、圆度、长宽比)
    → 【新增】纹理/边缘验证 (TextureEdgeValidator)
    → filter_strict (严格过滤)
    → 最终输出
```

纹理/边缘验证层与现有过滤器的关系：
- **互补**：基础过滤处理明显不符合的候选，纹理验证处理边界情况
- **前置**：纹理验证在 `filter_strict` 之前，避免严格过滤误删真实沙砾
- **可配置**：可通过配置单独启用/禁用纹理验证

## 7. 测试策略

### 7.1 单元测试

- **LBP 特征提取**：验证直方图维度和归一化
- **GLCM 特征提取**：验证特征值范围
- **纹理评分**：验证评分在 [0, 1] 范围内
- **镜头边缘检测**：验证大圆形边界区域被正确过滤
- **噪声检测**：验证小面积低纹理区域被正确过滤

### 7.2 集成测试

- **端到端检测**：对已知标注的显微镜图像进行测试
- **漏检率统计**：对比添加纹理验证前后的漏检数量
- **误检率统计**：对比添加纹理验证前后的误检数量

### 7.3 性能测试

- **处理时间**：确保纹理验证不会显著增加处理时间
- **内存占用**：确保特征提取不会导致内存溢出

## 8. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| skimage 依赖增加 | 中 | 可选依赖，未安装时跳过纹理验证；同时提供纯 OpenCV 回退方案（使用局部方差替代 LBP，使用灰度共生矩阵的简化版） |
| 计算量增加 | 中 | LBP/GLCM 计算量可控，可缓存结果 |
| 阈值调参困难 | 低 | 提供默认参数，支持配置化调整 |
| 与现有代码冲突 | 低 | 独立模块，不影响现有功能 |

## 9. 后续优化方向

1. **机器学习增强**：收集标注数据后，用 SVM/RandomForest 替代手动阈值
2. **多尺度纹理分析**：对不同大小的沙砾使用不同尺度的纹理特征
3. **GPU 加速**：使用 CUDA 加速 LBP/GLCM 计算
4. **在线学习**：根据用户反馈自动调整阈值

## 10. 审批记录

- **设计审批**: __待审批__
- **实现审批**: __待审批__
- **测试审批**: __待审批__
