# SandAnalyze 沙粒形态分析系统设计方案

> **日期**: 2026-06-17
> **版本**: v2.0
> **分支**: feat/streamlit-ui

---

## 1. 项目概述

SandAnalyze 是一个基于传统计算机视觉方法的沙粒形态分析系统，用于从显微镜照片中自动识别、分类和统计沙粒的形态参数。

### 1.1 核心目标

- 自动检测显微镜照片中的沙粒
- 计算形态参数（面积、周长、圆度、长短轴比等）
- 按 Zingg 分类法对沙粒进行形状分类
- 检测并标记絮凝（粘连沙粒团）
- 生成统计数据、图表和报告

### 1.2 输入输出

**输入：**
- 单张显微镜照片（PNG/JPG/TIFF）
- 批量图片文件夹

**输出：**
- CSV 数据文件（每个沙粒的形态参数）
- 标注图片（带轮廓和分类颜色）
- 统计图表（粒径分布、形状分类饼图等）
- PDF 综合报告

---

## 2. 分类体系

### 2.1 Zingg 分类 + 絮凝

系统采用四分类体系：

| 分类 | 中文 | 判断标准 | 颜色 |
|------|------|----------|------|
| 球状 | Spherical | 长短轴比 < 1.5 | 绿色 |
| 棒状 | Rod-like/Bladed | 1.5 ≤ 长短轴比 < 2.5 | 红色 |
| 片状 | Discoidal/Flat | 长短轴比 ≥ 2.5 | 蓝色 |
| 絮凝 | Flocculation | 综合判断（见下方） | 黄色 |

### 2.2 絮凝检测标准

絮凝通过**综合判断**检测，满足以下多个条件时判定为絮凝：

1. **大面积**: 面积 > 设定阈值（默认 5000 px²）
2. **低圆度**: 圆度 < 0.3（轮廓不规则）
3. **凸度缺陷**: 存在多个凹陷点（凸度 < 0.7）
4. **高长宽比**: 长短轴比 > 3.0（不规则形状）

**判定逻辑：**
- 满足条件 1（大面积）+ 至少 2 个其他条件 → 判定为絮凝
- 不满足絮凝条件的颗粒按 Zingg 分类（球状/棒状/片状）

---

## 3. 系统架构

### 3.1 模块结构

```
sandanalyze/
├── app.py                      # Streamlit 主应用
├── main.py                     # 入口文件
├── core/
│   ├── preprocessor.py         # 图像预处理
│   ├── detector.py             # 颗粒检测（含絮凝检测）
│   ├── morphology.py           # 形态参数计算
│   ├── classifier.py          # Zingg + 絮凝分类
│   ├── exporter.py             # 数据导出
│   └── report.py               # PDF报告生成
├── tests/                      # 测试
└── docs/                       # 文档
```

### 3.2 处理流程

```
输入图像
  ↓
[预处理] → 灰度化 → 高斯模糊 → 自适应阈值 → 形态学操作
  ↓
[轮廓检测] → cv2.findContours
  ↓
[过滤] → 面积过滤 → 边缘过滤
  ↓
[形态计算] → 面积、周长、圆度、长短轴、凸度
  ↓
[分类]
  ├── 絮凝？ → 是 → 标记为絮凝
  └── 否 → Zingg 分类（球状/棒状/片状）
  ↓
[输出] → CSV + 标注图 + 统计图表 + PDF报告
```

---

## 4. 核心模块设计

### 4.1 预处理模块 (preprocessor.py)

**功能：**
- 灰度化（如果输入为彩色图像）
- 高斯模糊（降噪）
- 自适应阈值（分离前景背景）
- 形态学操作（开运算/闭运算，去除噪点）

**参数：**
- `blur_kernel`: 高斯模糊核大小（默认 5）
- `adaptive_block`: 自适应阈值块大小（默认 11）
- `adaptive_c`: 自适应阈值常数（默认 2）
- `morph_kernel`: 形态学核大小（默认 3）

### 4.2 检测模块 (detector.py)

**功能：**
- 轮廓检测（cv2.findContours）
- 面积过滤（去除过小/过大的噪声）
- 边缘过滤（排除图像边缘的不完整颗粒）
- 絮凝检测（综合判断）

**絮凝检测算法：**
```python
def is_flocculation(contour, config):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * π * area) / (perimeter ** 2)
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h)
    
    # 综合判断
    conditions = [
        area > config.floc_area_threshold,      # 大面积
        circularity < config.floc_circularity,    # 低圆度
        convexity < config.floc_convexity,        # 低凸度
        aspect_ratio > config.floc_aspect_ratio,  # 高长宽比
    ]
    
    # 满足大面积 + 至少2个其他条件
    return conditions[0] and sum(conditions[1:]) >= 2
```

### 4.3 分类模块 (classifier.py)

**Zingg 分类：**
```python
def zingg_classify(aspect_ratio):
    if aspect_ratio < 1.5:
        return "球状"
    elif aspect_ratio < 2.5:
        return "棒状"
    else:
        return "片状"
```

**完整分类：**
```python
def classify(contour, morphology):
    if is_flocculation(contour):
        return "絮凝"
    else:
        return zingg_classify(morphology.aspect_ratio)
```

### 4.4 形态参数计算 (morphology.py)

**计算参数：**
- 面积（Area）
- 周长（Perimeter）
- 圆度（Circularity）= 4πA / P²
- 等效粒径（d_eq）= √(4A/π)
- 长轴 / 短轴（Major/Minor Axis）
- 长短轴比（Aspect Ratio）
- 球度（Sphericity）= 短轴/长轴
- 凸度（Convexity）= 面积/凸包面积
- Feret 直径（最大/最小）

### 4.5 导出模块 (exporter.py)

**CSV 导出：**
- grain_id, area, perimeter, circularity, d_eq, major_axis, minor_axis, aspect_ratio, sphericity, convexity, feret_max, feret_min, shape_class, is_flocculation

**标注图片：**
- 轮廓线（按分类着色）
- 颗粒编号
- 图例

### 4.6 报告模块 (report.py)

**PDF 报告内容：**
- 输入图像
- 标注结果图
- 统计摘要（总数、各类别数量、平均参数）
- 粒径分布直方图
- 形状分类饼图
- 详细数据表

---

## 5. UI 设计

### 5.1 Streamlit 界面布局

```
┌─────────────────────────────────────┐
│  SandAnalyze - 沙粒形态分析系统      │
├──────────┬──────────────────────────┤
│          │                          │
│  侧边栏   │      主显示区            │
│          │                          │
│ [上传图片]│    [标注结果图]          │
│          │                          │
│ [参数调整]│    [统计图表]            │
│          │                          │
│ [运行检测]│    [数据表格]            │
│          │                          │
│ [导出]   │                          │
│          │                          │
└──────────┴──────────────────────────┘
```

### 5.2 侧边栏功能

- **图像上传**：支持 PNG/JPG/TIFF
- **预处理参数**：
  - 模糊核大小
  - 自适应块大小
  - 自适应常数 C
  - 形态学核大小
  - 最小面积阈值
- **絮凝检测参数**：
  - 面积阈值
  - 圆度阈值
  - 凸度阈值
  - 长宽比阈值
- **运行按钮**：开始检测
- **导出按钮**：导出 CSV / 标注图 / PDF 报告

### 5.3 主显示区

**标签页 1 - 图像与标注：**
- 原始图像（左）
- 标注结果（右，带轮廓和分类颜色）

**标签页 2 - 统计：**
- 颗粒数量
- 平均圆度、球度、长短轴比
- Zingg + 絮凝分类统计

**标签页 3 - 图表：**
- 粒径分布直方图
- 圆度 vs 球度散点图
- 形状分类饼图（四类）

**标签页 4 - 数据：**
- 颗粒数据表格（可排序、筛选）

---

## 6. 批量处理

### 6.1 批量处理流程

```
输入文件夹
  ↓
遍历所有图片
  ↓
对每张图片执行检测
  ↓
汇总统计数据
  ↓
生成批量报告
```

### 6.2 批量输出

- 每张图片的标注结果
- 汇总 CSV（所有图片的数据）
- 汇总统计图表
- PDF 综合报告

---

## 7. 技术栈

- **Python 3.13**
- **OpenCV** - 图像处理和轮廓检测
- **NumPy/SciPy** - 数值计算
- **Streamlit** - Web UI
- **Plotly** - 交互式图表
- **ReportLab** - PDF 报告生成
- **Pandas** - 数据处理

---

## 8. 测试计划

### 8.1 单元测试

- 预处理模块测试
- 轮廓检测测试
- 形态参数计算测试
- 分类器测试

### 8.2 集成测试

- 完整流程测试
- 批量处理测试
- 导出功能测试

### 8.3 性能测试

- 单张图片处理时间
- 批量处理吞吐量

---

## 9. 部署

### 9.1 本地运行

```bash
# 安装依赖
uv sync --extra dev

# 启动 Streamlit
uv run python main.py

# 或
streamlit run app.py
```

### 9.2 打包

```bash
# PyInstaller 打包
pyinstaller sandanalyze.spec
```

---

## 10. 待办事项

- [ ] 实现预处理模块
- [ ] 实现检测模块（含絮凝检测）
- [ ] 实现分类模块（Zingg + 絮凝）
- [ ] 实现形态参数计算
- [ ] 实现导出模块
- [ ] 实现 PDF 报告生成
- [ ] 实现 Streamlit UI
- [ ] 实现批量处理
- [ ] 编写测试
- [ ] 编写文档
