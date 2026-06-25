# 多维度砂砾检测与形态学分割系统设计

**日期**: 2025-06-19
**作者**: Claude Opus 4.8
**状态**: 已批准

## 1. 背景与目标

当前砂砾检测管道存在以下5类问题：
1. **边缘误识别** — 图像边界处的假阳性
2. **噪点误识别** — 小颗粒噪声被误认为砂砾
3. **细丝状误识别** — 细长形状被误认为砂砾
4. **大块阴影识别不出** — 大颗粒因阴影/对比度不足而漏检
5. **过度合并** — 多个砂砾靠近被识别成一个絮凝砂砾

本设计通过**多尺度检测 + 形态学分割 + 多特征过滤**的组合方案解决上述问题。

## 2. 架构设计

```
输入图像
  ├── 多尺度预处理（3个尺度并行）
  │     ├── 大尺度：大自适应块 → 检测大颗粒/阴影区域
  │     ├── 中尺度：中自适应块 → 检测中等颗粒
  │     └── 小尺度：小自适应块 → 检测小颗粒
  │
  ├── 多尺度结果融合（去重 + 合并）
  │
  ├── 形态学分割（解决过度合并）
  │     ├── 距离变换 + 分水岭（圆形颗粒优先）
  │     └── 凹点检测分割（不规则形状回退）
  │
  └── 多特征过滤层（去除误识别）
        ├── 边缘过滤（边界假阳性）
        ├── 噪声过滤（小面积 + 低圆度）
        ├── 细丝过滤（高长宽比 + 低密实度）
        └── 阴影增强过滤（CLAHE + 局部对比度验证）
```

## 3. 核心组件

### 3.1 MultiScalePreprocessor

运行3个不同参数的自适应阈值，分别捕获不同尺寸的砂砾：

| 尺度 | adaptive_block_size | adaptive_c | blur_kernel | 用途 |
|------|---------------------|------------|-------------|------|
| 大   | 101                 | 2          | 7           | 捕获阴影下的大颗粒 |
| 中   | 51                  | 5          | 5           | 默认中等颗粒 |
| 小   | 21                  | 8          | 3           | 捕获小颗粒，减少噪声 |

**融合策略**：
- 对每个候选组件计算其特征（面积、圆度、长宽比）
- 小尺度优先：如果小尺度检测到组件且特征合理，优先使用小尺度结果
- 大尺度补充：如果大尺度检测到组件但中小尺度未检测到，验证其阴影特征后保留
- 去重：IOU > 0.5 的组件合并为同一个，保留特征最优的

### 3.2 MorphologicalSplitter

对检测到的每个连通组件进行形态学分割：

**算法流程**：
1. 计算组件的距离变换
2. 使用分水岭算法进行初步分割
3. 验证分割结果：
   - 如果分割后的子组件圆度均 > 0.3，接受分割
   - 如果分割线穿过原组件的质心附近，拒绝分割（避免过度分割）
4. 对拒绝分水岭的案例，尝试凹点检测：
   - 寻找轮廓上的凹点（曲率突变点）
   - 如果找到2个及以上凹点，在凹点处分割
   - 如果凹点检测也失败，保持原组件不分割

**关键约束**：宁可欠分割（保持合并），不可过度分割（避免切开单个砂砾）。

### 3.3 MultiFeatureFilter

对每个候选颗粒计算多维度特征并过滤：

**几何特征**：
- `area`: 面积（像素数）
- `circularity`: 圆度 = 4π × 面积 / 周长²
- `aspect_ratio`: 长宽比
- `solidity`: 密实度 = 面积 / 凸包面积
- `convexity`: 凸度 = 凸包面积 / 凸包周长²

**纹理特征**：
- `local_std`: 局部灰度标准差（反映纹理均匀性）
- `gradient_consistency`: 边缘梯度方向一致性（真实砂砾边缘梯度方向更一致）

**上下文特征**：
- `border_distance`: 到图像边界的距离
- `neighborhood_density`: 邻域内其他候选颗粒的密度

**过滤规则**：
- 边缘误识别：`border_distance < 10` 且 `area < 2000` → 移除
- 噪点误识别：`area < 500` 且 `circularity < 0.2` → 移除
- 细丝状误识别：`aspect_ratio > 5` 且 `solidity < 0.5` → 移除
- 大块阴影：如果 `area > 10000` 且 `local_std < 20`，验证其是否为真实砂砾（检查边缘梯度）

### 3.4 ShadowAwareEnhancer

在预处理阶段对大尺度检测通道应用增强：

1. **CLAHE增强**：对原始图像应用CLAHE（clipLimit=3.0, tileGridSize=(8,8)）
2. **局部对比度验证**：对检测到的区域，计算其内部灰度标准差
   - 如果 `local_std < 15`，可能是阴影区域，降低其置信度
   - 如果 `local_std >= 15`，确认是真实砂砾

## 4. 数据流

```
输入图像
  │
  ├─→ [大尺度预处理] ─┐
  ├─→ [中尺度预处理] ─┼→ [多尺度融合] → [形态学分割] → [多特征过滤] → 最终砂砾列表
  └─→ [小尺度预处理] ─┘
```

## 5. 接口设计

```python
@dataclass
class MultiScaleConfig:
    """多尺度检测配置"""
    large_scale: PreprocessConfig  # 大尺度参数
    medium_scale: PreprocessConfig  # 中尺度参数
    small_scale: PreprocessConfig   # 小尺度参数
    shadow_enhance: bool = True     # 是否启用阴影增强

@dataclass
class GrainCandidate:
    """候选砂砾"""
    contour: np.ndarray
    mask: np.ndarray
    area: float
    circularity: float
    aspect_ratio: float
    solidity: float
    local_std: float
    gradient_consistency: float
    border_distance: float
    confidence: float  # 综合置信度

def detect_grains_multiscale(
    image: np.ndarray,
    config: MultiScaleConfig,
) -> list[GrainCandidate]:
    """多尺度砂砾检测主入口"""
    ...

def split_overmerged_components(
    candidates: list[GrainCandidate],
    min_circularity: float = 0.3,
) -> list[GrainCandidate]:
    """形态学分割过度合并的组件"""
    ...

def filter_false_positives(
    candidates: list[GrainCandidate],
    edge_margin: int = 10,
    min_area: int = 500,
    max_aspect_ratio: float = 5.0,
    min_solidity: float = 0.5,
) -> list[GrainCandidate]:
    """多特征过滤假阳性"""
    ...
```

## 6. 测试策略

1. **合成数据测试**：生成包含已知砂砾的合成图像，验证检测率和误检率
2. **边界案例测试**：专门测试边缘、阴影、密集区域的图像
3. **形态学分割测试**：验证过度合并的砂砾能被正确分割，单个砂砾不会被错误切开
4. **性能测试**：验证多尺度检测的耗时在可接受范围内（约3倍单尺度）

## 7. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 多尺度检测耗时增加 | 中 | 使用并行处理；提供进度反馈 |
| 形态学分割过度分割 | 高 | 严格的分割验证；宁可欠分割 |
| 阴影区域误检 | 中 | 局部对比度验证；降低阴影区域置信度 |
| 小颗粒漏检 | 中 | 小尺度通道专门检测；调整面积阈值 |

## 8. 后续优化方向

1. 基于机器学习的特征分类器（替代手工规则）
2. 自适应参数调整（根据图像内容自动选择最佳参数）
3. GPU加速（使用CUDA加速多尺度检测）
