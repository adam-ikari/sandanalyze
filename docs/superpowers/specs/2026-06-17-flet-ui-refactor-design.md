# Flet UI 重构设计文档

## 概述

将 SandAnalyze 的 GUI 层从 PyQt6 迁移到 Flet 框架，一套代码同时支持 Web 模式和桌面模式。核心处理逻辑（`core/`）保持不变。

## 背景

当前项目使用 PyQt6 作为桌面 GUI 框架，仅支持本地桌面运行。用户希望通过 Flet 重构，使应用既能作为桌面应用运行，也能通过浏览器访问。

## 启动方式

```bash
# 桌面模式（默认）
uv run python main.py

# Web 模式
uv run python main.py --web

# Web 模式指定端口
uv run python main.py --web --port 8080
```

`main.py` 使用 `argparse` 解析参数：
- `--web`：启用 Web 模式，Flet 启动 HTTP 服务器
- `--port`：指定 Web 服务器端口（默认 8000）
- 无参数：Flet 桌面模式（内嵌 WebView 窗口）

## 架构变更

```
sandanalyze
├── core/                    # 不变 — 所有处理逻辑保持原样
│   ├── __init__.py
│   ├── preprocessor.py
│   ├── traditional.py
│   ├── yolo_detector.py
│   ├── morphology.py
│   ├── exporter.py
│   └── model_manager.py
├── gui/                     # 删除 — 原 PyQt6 代码
├── ui/                      # 新 Flet UI 层
│   ├── __init__.py
│   ├── app.py               # Flet 应用入口，页面布局 + 事件处理
│   ├── image_view.py        # 图像显示 + 颗粒标注叠加 + 缩放/点击
│   ├── settings_panel.py    # 预处理参数设置面板
│   ├── result_panel.py      # 统计摘要 + 颗粒数据表 + Plotly 图表
│   └── charts.py            # Plotly 图表生成函数
├── models/                  # 不变
├── data/                    # 不变
├── main.py                  # 修改 — 添加 argparse，分发 Web/桌面模式
├── pyproject.toml           # 修改 — 移除 PyQt6/matplotlib，添加 flet/plotly
└── tests/                   # 不变
```

## UI 布局

```
┌─────────────────────────────────────────────────────────┐
│  AppBar: SandAnalyze - 沙粒形态分析系统                  │
│  [打开图像] [运行检测] [导出CSV] [导出标注图] [YOLO开关] │
├──────────────────────────┬──────────────────────────────┤
│                          │  Tabs: 统计摘要 | 颗粒数据     │
│   图像显示区域            │  ──────────────────────────  │
│   (InteractiveViewer      │  统计数值 (Column)           │
│    支持鼠标滚轮缩放       │                              │
│    支持拖拽平移)          │  Plotly 图表 (Tab 内):       │
│                          │    粒径分布直方图             │
│   - 颗粒轮廓叠加          │    圆度-球度散点图            │
│   - Zingg 颜色标注        │    Zingg 分类饼图            │
│   - 颗粒编号标签          │                              │
│   - 点击颗粒高亮          │  预处理参数面板               │
│   - Zingg 图例            │    模糊核 / 块大小 / C值     │
│                          │    形态学核大小 / 最小面积    │
│                          │    CLAHE 开关 / 分水岭 开关   │
├──────────────────────────┴──────────────────────────────┤
│  状态栏: 检测方法 | 颗粒数 | 处理时间                     │
└─────────────────────────────────────────────────────────┘
```

## 组件映射

| PyQt6 组件 | Flet 替代 |
|---|---|
| `QMainWindow` + `QMenuBar` | `ft.AppBar` + `ft.PopupMenuButton` |
| `QToolBar` | `ft.Row` of `ft.IconButton` in AppBar `actions` |
| `QSplitter` (水平/垂直) | `ft.Row` / `ft.Column` with `expand=True` |
| `QScrollArea` (图像缩放) | `ft.InteractiveViewer` |
| `QTabWidget` | `ft.Tabs` |
| `QTableWidget` | `ft.DataTable` |
| `QSpinBox` | `ft.TextField` with `input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]")` |
| `QCheckBox` | `ft.Switch` 或 `ft.Checkbox` |
| `QPushButton` | `ft.ElevatedButton` / `ft.FilledButton` |
| `QLabel` | `ft.Text` |
| `QGroupBox` | `ft.Container` + `ft.Text` 标题 |
| `matplotlib FigureCanvas` | `ft.PlotlyChart` |
| `QStatusBar` | `ft.Text` in bottom `ft.Container` |
| `QFileDialog` (打开/保存) | `ft.FilePicker` |
| `QMessageBox` | `ft.AlertDialog` |
| `pyqtSignal` | 回调函数 + `page.update()` |
| `QWheelEvent` (缩放) | `ft.InteractiveViewer` 内置支持 |
| `QMouseEvent` (点击) | `ft.GestureDetector.on_tap` |

## 图像显示方案

Flet 不能直接操作 numpy 数组显示图像。需要将 numpy 数组编码为 base64 PNG 后通过 `ft.Image(src_base64=...)` 显示。

图像标注叠加流程：
1. 在 numpy 数组上使用 OpenCV 绘制轮廓、标签、图例
2. 将绘制后的 numpy 数组编码为 PNG base64
3. 设置 `ft.Image.src_base64` 触发更新

缩放/平移：使用 `ft.InteractiveViewer` 包裹 `ft.Image`，内置支持鼠标滚轮缩放和拖拽平移。

颗粒点击：在 `ft.GestureDetector.on_tap` 中获取点击坐标，映射回原始图像坐标，通过 mask/contour 判断点击了哪个颗粒。

## 图表方案

使用 Plotly 替换 matplotlib：

- **粒径分布直方图**：`plotly.express.histogram`
- **圆度-球度散点图**：`plotly.express.scatter`
- **Zingg 分类饼图**：`plotly.express.pie`

Plotly 图表通过 `ft.PlotlyChart` 组件嵌入，支持悬停查看数值、缩放、平移等交互。

## 依赖变更

```toml
# pyproject.toml
[project]
dependencies = [
    "opencv-python>=4.8",
    "ultralytics>=8.0",
    "flet>=0.27.0",
    "plotly>=5.22.0",
    "numpy>=1.26",
    "scipy>=1.12",
]
```

移除：`PyQt6>=6.6`、`matplotlib>=3.8`

## 文件变更清单

### 新增文件
- `ui/__init__.py`
- `ui/app.py` — 主页面布局、事件处理、检测流程编排
- `ui/image_view.py` — 图像显示、标注叠加、缩放、点击交互
- `ui/settings_panel.py` — 预处理参数控件
- `ui/result_panel.py` — 统计摘要、数据表格、图表容器
- `ui/charts.py` — Plotly 图表生成函数

### 修改文件
- `main.py` — 添加 argparse，Flet 启动逻辑
- `pyproject.toml` — 依赖更新

### 删除文件
- `gui/__init__.py`
- `gui/app.py`
- `gui/image_panel.py`
- `gui/result_panel.py`
- `gui/settings_panel.py`

### 不变文件
- `core/` 全部文件
- `models/` 全部文件
- `data/` 全部文件
- `tests/` 全部文件
- `sandanalyze.spec`（后续更新 PyInstaller 配置）
- `scripts/build.sh`、`scripts/build.bat`（后续更新）
- `.github/workflows/`（后续更新依赖）

## 功能保持清单

- [x] 图像加载（FilePicker）
- [x] 预处理参数实时调节
- [x] 传统检测 + YOLO 混合检测
- [x] 颗粒轮廓叠加显示（Zingg 颜色分类）
- [x] 颗粒编号标签
- [x] Zingg 图例
- [x] 图像缩放（鼠标滚轮）
- [x] 图像平移（拖拽）
- [x] 点击颗粒高亮 + 结果面板联动
- [x] 统计摘要显示
- [x] 颗粒数据表格
- [x] 粒径分布直方图
- [x] 圆度-球度散点图
- [x] Zingg 分类饼图
- [x] CSV 导出
- [x] 标注图 PNG 导出
- [x] 状态栏（检测方法、颗粒数、处理时间）
- [x] YOLO 精细分割开关
- [x] 关于对话框

## 不包含的变更

- 不修改任何 `core/` 模块
- 不修改测试
- 不修改 YOLO 模型管理逻辑
- 不添加新功能
- 不修改 CI/CD（后续单独更新）
