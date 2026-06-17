# Flet UI 重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 SandAnalyze 的 GUI 层从 PyQt6 迁移到 Flet，一套代码同时支持 Web 和桌面模式。

**Architecture:** 新建 `ui/` 包替代 `gui/`，使用 Flet 组件重写所有 UI。核心处理逻辑（`core/`）零改动。`main.py` 添加 argparse 以支持 `--web` 和 `--port` 参数。Plotly 替换 matplotlib 生成交互式图表。

**Tech Stack:** Python 3.13, Flet 0.27+, Plotly 5.22+, OpenCV, numpy, scipy

---

### Task 1: 创建分支并更新依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 创建新分支**

```bash
git checkout -b feat/flet-ui
```

- [ ] **Step 2: 更新 pyproject.toml 依赖**

将 `pyproject.toml` 的 dependencies 替换为：

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
    "flet>=0.27.0",
    "plotly>=5.22.0",
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

- [ ] **Step 3: 安装新依赖**

```bash
uv sync --extra dev
```

预期：flet 和 plotly 安装成功，PyQt6 和 matplotlib 不再在锁定文件中。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: replace PyQt6/matplotlib with Flet/Plotly dependencies

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 创建 ui/__init__.py 和 ui/charts.py

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/charts.py`

- [ ] **Step 1: 创建 ui/__init__.py**

```python
"""SandAnalyze Flet UI layer.

Provides the main Flet application page, image display, settings panel,
result panel, and Plotly chart generation.
"""
```

- [ ] **Step 2: 创建 ui/charts.py**

```python
"""Plotly chart generation for sand grain analysis results."""

import plotly.express as px
import plotly.graph_objects as go

from core.morphology import GrainStatistics


def create_size_histogram(stats: GrainStatistics) -> go.Figure:
    """Create an equivalent diameter distribution histogram.

    Args:
        stats: Aggregate grain statistics with d_eq_values populated.

    Returns:
        Plotly Figure object.
    """
    fig = px.histogram(
        x=stats.d_eq_values,
        nbins=20,
        title="粒径分布",
        labels={"x": "等效粒径 (d_eq)", "y": "频数"},
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def create_circularity_sphericity_scatter(stats: GrainStatistics) -> go.Figure:
    """Create a circularity vs sphericity scatter plot.

    Args:
        stats: Aggregate grain statistics with circularity_values
               and sphericity_values populated.

    Returns:
        Plotly Figure object.
    """
    fig = px.scatter(
        x=stats.circularity_values,
        y=stats.sphericity_values,
        title="圆度 vs 球度",
        labels={"x": "圆度", "y": "球度"},
        opacity=0.6,
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="black")))
    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def create_zingg_pie_chart(stats: GrainStatistics) -> go.Figure:
    """Create a Zingg classification pie chart.

    Args:
        stats: Aggregate grain statistics with zingg_counts populated.

    Returns:
        Plotly Figure object.
    """
    labels = list(stats.zingg_counts.keys())
    values = list(stats.zingg_counts.values())
    colors = ["#99ff99", "#66b3ff", "#ff9999"]  # 球状(green), 棒状(blue), 片状(red)

    fig = px.pie(
        names=labels,
        values=values,
        title="Zingg分类",
        color_discrete_sequence=colors[:len(labels)],
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
    )
    return fig
```

- [ ] **Step 3: 验证 charts 模块可导入**

```bash
uv run python -c "from ui.charts import create_size_histogram, create_circularity_sphericity_scatter, create_zingg_pie_chart; print('OK')"
```

预期：OK

- [ ] **Step 4: Commit**

```bash
git add ui/__init__.py ui/charts.py
git commit -m "feat: add Plotly chart generation module

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 创建 ui/settings_panel.py

**Files:**
- Create: `ui/settings_panel.py`

- [ ] **Step 1: 创建 ui/settings_panel.py**

```python
"""Settings panel for preprocessing parameter adjustment (Flet version)."""

import flet as ft

from core.preprocessor import PreprocessConfig


class SettingsPanel:
    """Preprocessing parameter settings panel.

    Provides spin-box-like controls for all PreprocessConfig fields
    plus CLAHE/watershed toggles and a Run Detection button.

    Attributes:
        control: The root Flet Control for this panel.
        on_config_changed: Callback(config) when any parameter changes.
        on_detect_requested: Callback() when Run Detection is clicked.
    """

    def __init__(
        self,
        on_config_changed: callable = None,
        on_detect_requested: callable = None,
    ) -> None:
        self.on_config_changed = on_config_changed
        self.on_detect_requested = on_detect_requested

        # Spin-box-like text fields
        self._blur_kernel = ft.TextField(
            label="模糊核大小",
            value="5",
            width=160,
            height=50,
            text_size=13,
            input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]"),
            on_change=self._on_param_changed,
        )
        self._adaptive_block = ft.TextField(
            label="自适应块大小",
            value="11",
            width=160,
            height=50,
            text_size=13,
            input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]"),
            on_change=self._on_param_changed,
        )
        self._adaptive_c = ft.TextField(
            label="自适应常数 C",
            value="2",
            width=160,
            height=50,
            text_size=13,
            input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\-]"),
            on_change=self._on_param_changed,
        )
        self._morph_kernel = ft.TextField(
            label="形态学核大小",
            value="3",
            width=160,
            height=50,
            text_size=13,
            input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]"),
            on_change=self._on_param_changed,
        )
        self._min_area = ft.TextField(
            label="最小面积",
            value="50",
            width=160,
            height=50,
            text_size=13,
            input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]"),
            on_change=self._on_param_changed,
        )

        self._use_clahe = ft.Switch(
            label="CLAHE 增强",
            value=True,
            on_change=self._on_param_changed,
        )
        self._use_watershed = ft.Switch(
            label="分水岭分割",
            value=True,
            on_change=self._on_param_changed,
        )

        self._detect_button = ft.FilledButton(
            text="运行检测",
            on_click=lambda e: self._on_detect_clicked(),
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.GREEN_600,
                color=ft.Colors.WHITE,
            ),
        )

        self.control = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("预处理参数", weight=ft.FontWeight.BOLD, size=14),
                    ft.Row(
                        controls=[self._blur_kernel, self._adaptive_block],
                        wrap=True,
                    ),
                    ft.Row(
                        controls=[self._adaptive_c, self._morph_kernel],
                        wrap=True,
                    ),
                    ft.Row(
                        controls=[self._min_area],
                        wrap=True,
                    ),
                    ft.Divider(height=1),
                    self._use_clahe,
                    self._use_watershed,
                    ft.Divider(height=1),
                    self._detect_button,
                ],
                spacing=6,
            ),
            padding=ft.padding.all(10),
            border=ft.border.all(1, ft.Colors.OUTLINE),
            border_radius=8,
        )

    def _on_param_changed(self, e) -> None:
        if self.on_config_changed:
            self.on_config_changed(self.get_config())

    def _on_detect_clicked(self) -> None:
        if self.on_detect_requested:
            self.on_detect_requested()

    def get_config(self) -> PreprocessConfig:
        """Return current PreprocessConfig from control values."""
        return PreprocessConfig(
            blur_kernel=int(self._blur_kernel.value or "5"),
            adaptive_block_size=int(self._adaptive_block.value or "11"),
            adaptive_c=int(self._adaptive_c.value or "2"),
            morph_kernel_size=int(self._morph_kernel.value or "3"),
            min_area=int(self._min_area.value or "50"),
            use_clahe=self._use_clahe.value,
            use_watershed=self._use_watershed.value,
        )
```

- [ ] **Step 2: 验证模块可导入**

```bash
uv run python -c "from ui.settings_panel import SettingsPanel; print('OK')"
```

预期：OK

- [ ] **Step 3: Commit**

```bash
git add ui/settings_panel.py
git commit -m "feat: add Flet settings panel component

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 创建 ui/image_view.py

**Files:**
- Create: `ui/image_view.py`

- [ ] **Step 1: 创建 ui/image_view.py**

```python
"""Image display with grain overlay, zoom, and click interaction (Flet version)."""

import base64

import cv2
import flet as ft
import numpy as np


class ImageView:
    """Displays sand grain image with contour overlays and click interaction.

    Renders grain contours directly onto the numpy image using OpenCV,
    encodes as PNG base64, and displays via ft.Image inside an
    InteractiveViewer for zoom/pan support.

    Attributes:
        control: The root Flet Control (InteractiveViewer > GestureDetector > Image).
        on_grain_clicked: Callback(grain_index) when a grain is clicked.
    """

    def __init__(self, on_grain_clicked: callable = None) -> None:
        self.on_grain_clicked = on_grain_clicked

        self._original_image: np.ndarray | None = None
        self._grains: list = []
        self._morphologies: list = []

        self._image_widget = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            visible=False,
        )

        self._gesture = ft.GestureDetector(
            content=self._image_widget,
            on_tap=self._handle_tap,
        )

        self._viewer = ft.InteractiveViewer(
            content=self._gesture,
            min_scale=0.1,
            max_scale=5.0,
            boundary_margin=ft.margin.all(200),
        )

        self.control = ft.Container(
            content=self._viewer,
            expand=True,
            bgcolor=ft.Colors.GREY_900,
            alignment=ft.alignment.center,
        )

    def set_image(self, image: np.ndarray) -> None:
        """Set and display the original image.

        Args:
            image: Numpy array in BGR (color) or grayscale format.
        """
        self._original_image = image.copy() if image is not None else None
        self._grains = []
        self._morphologies = []
        self._update_display()

    def set_grains(self, grains: list, morphologies: list = None) -> None:
        """Set detected grains and overlay their contours on the image.

        Args:
            grains: List of GrainContour objects.
            morphologies: Optional list of GrainMorphology objects for Zingg coloring.
        """
        self._grains = grains if grains is not None else []
        self._morphologies = morphologies if morphologies is not None else []
        self._update_display()

    def clear(self) -> None:
        """Clear the display."""
        self._original_image = None
        self._grains = []
        self._morphologies = []
        self._image_widget.visible = False
        self._image_widget.src_base64 = ""
        self._image_widget.update()

    def _update_display(self) -> None:
        """Render the image with grain overlays and update the widget."""
        if self._original_image is None:
            self._image_widget.visible = False
            self._image_widget.src_base64 = ""
            self._image_widget.update()
            return

        display = self._original_image.copy()

        # Ensure color for overlay drawing
        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Draw grain contours and labels
        if self._grains:
            for idx, grain in enumerate(self._grains):
                contour = getattr(grain, "contour", None)
                if contour is None or len(contour) == 0:
                    continue

                # Determine color from Zingg classification
                color = (0, 255, 0)  # Default green
                if idx < len(self._morphologies):
                    from core.morphology import get_zingg_color

                    color = get_zingg_color(self._morphologies[idx].aspect_ratio)

                cv2.drawContours(display, [contour], -1, color, 2)

                # Draw label at centroid
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cv2.circle(display, (cx, cy), 3, (255, 255, 255), -1)
                    cv2.putText(
                        display,
                        str(idx + 1),
                        (cx - 8, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )

            # Draw Zingg legend if morphologies available
            if self._morphologies:
                self._draw_legend(display)

        # Encode to PNG base64
        _, buffer = cv2.imencode(".png", display)
        b64 = base64.b64encode(buffer).decode("utf-8")

        self._image_widget.src_base64 = b64
        self._image_widget.visible = True
        self._image_widget.update()

    def _draw_legend(self, image: np.ndarray) -> None:
        """Draw Zingg classification legend on the image (in-place)."""
        from core.morphology import ZINGG_COLORS

        h, w = image.shape[:2]
        legend_x = w - 140
        legend_y = h - 80
        item_height = 20

        cv2.rectangle(
            image,
            (legend_x - 10, legend_y - 25),
            (legend_x + 130, legend_y + len(ZINGG_COLORS) * item_height + 5),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            image,
            (legend_x - 10, legend_y - 25),
            (legend_x + 130, legend_y + len(ZINGG_COLORS) * item_height + 5),
            (200, 200, 200),
            1,
        )

        cv2.putText(
            image,
            "Zingg分类",
            (legend_x, legend_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        for i, (label, color) in enumerate(ZINGG_COLORS.items()):
            y = legend_y + i * item_height + 10
            cv2.rectangle(image, (legend_x, y - 10), (legend_x + 15, y + 5), color, -1)
            cv2.putText(
                image,
                label,
                (legend_x + 20, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    def _handle_tap(self, e: ft.TapEvent) -> None:
        """Handle tap on the image to detect grain selection."""
        if self._original_image is None or not self._grains:
            return

        # GestureDetector gives local coordinates relative to the Image widget.
        # InteractiveViewer applies a transform — we approximate by using the
        # image display size vs original image size.
        local_x = e.local_x
        local_y = e.local_y

        if local_x is None or local_y is None:
            return

        # Map to original image coordinates based on displayed size
        img_h, img_w = self._original_image.shape[:2]

        # Get the current displayed size from the image widget
        # Since we use ImageFit.CONTAIN, we need to approximate
        # We use the original image proportions as a fallback
        orig_x = int(local_x * img_w / max(img_w, 1))
        orig_y = int(local_y * img_h / max(img_h, 1))

        orig_x = max(0, min(img_w - 1, orig_x))
        orig_y = max(0, min(img_h - 1, orig_y))

        # Find which grain was clicked
        for idx, grain in enumerate(self._grains):
            mask = getattr(grain, "mask", None)
            if mask is not None and mask.size > 0:
                if 0 <= orig_y < mask.shape[0] and 0 <= orig_x < mask.shape[1]:
                    if mask[orig_y, orig_x] > 0:
                        if self.on_grain_clicked:
                            self.on_grain_clicked(idx)
                        return

            contour = getattr(grain, "contour", None)
            if contour is not None and len(contour) > 0:
                dist = cv2.pointPolygonTest(
                    contour, (float(orig_x), float(orig_y)), False
                )
                if dist >= 0:
                    if self.on_grain_clicked:
                        self.on_grain_clicked(idx)
                    return
```

- [ ] **Step 2: 验证模块可导入**

```bash
uv run python -c "from ui.image_view import ImageView; print('OK')"
```

预期：OK

- [ ] **Step 3: Commit**

```bash
git add ui/image_view.py
git commit -m "feat: add Flet image view with grain overlay and click

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 创建 ui/result_panel.py

**Files:**
- Create: `ui/result_panel.py`

- [ ] **Step 1: 创建 ui/result_panel.py**

```python
"""Result panel with statistics, table, and charts (Flet version)."""

import flet as ft
import plotly.graph_objects as go

from core.morphology import GrainMorphology, GrainStatistics
from ui.charts import (
    create_circularity_sphericity_scatter,
    create_size_histogram,
    create_zingg_pie_chart,
)


class ResultPanel:
    """Result panel displaying statistics, data table, and Plotly charts.

    Attributes:
        control: The root Flet Control (Tabs widget).
    """

    def __init__(self) -> None:
        self._morphologies: list[GrainMorphology] = []
        self._statistics: GrainStatistics | None = None

        # Summary tab content
        self._summary_texts: dict[str, ft.Text] = {}
        self._summary_column = ft.Column(spacing=6)
        self._zingg_text = ft.Text("Zingg分类: --", size=13)

        # Table tab content
        self._table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("ID")),
                ft.DataColumn(ft.Text("面积")),
                ft.DataColumn(ft.Text("周长")),
                ft.DataColumn(ft.Text("圆度")),
                ft.DataColumn(ft.Text("等效粒径")),
                ft.DataColumn(ft.Text("长短轴比")),
                ft.DataColumn(ft.Text("球度")),
                ft.DataColumn(ft.Text("凸度")),
            ],
            rows=[],
            border=ft.border.all(1, ft.Colors.OUTLINE),
            heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            data_row_min_height=32,
        )
        self._table_container = ft.Column(
            controls=[self._table],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

        # Charts tab content (Plotly)
        self._hist_chart = ft.PlotlyChart(
            expand=True,
        )
        self._scatter_chart = ft.PlotlyChart(
            expand=True,
        )
        self._pie_chart = ft.PlotlyChart(
            expand=True,
        )

        # Build tabs
        self._tabs = ft.Tabs(
            selected_index=0,
            animation_duration=200,
            tabs=[
                ft.Tab(
                    text="统计摘要",
                    content=ft.Container(
                        content=self._summary_column,
                        padding=10,
                    ),
                ),
                ft.Tab(
                    text="颗粒数据",
                    content=self._table_container,
                ),
                ft.Tab(
                    text="图表",
                    content=ft.Column(
                        controls=[
                            self._hist_chart,
                            ft.Divider(height=1),
                            self._scatter_chart,
                            ft.Divider(height=1),
                            self._pie_chart,
                        ],
                        scroll=ft.ScrollMode.AUTO,
                        expand=True,
                    ),
                ),
            ],
            expand=True,
        )

        self.control = self._tabs

        # Initialize summary labels
        self._init_summary()

    def _init_summary(self) -> None:
        """Create the initial summary labels."""
        labels = [
            "颗粒数量",
            "平均圆度",
            "平均球度",
            "平均等效粒径",
            "平均长短轴比",
            "平均凸度",
        ]
        self._summary_column.controls.clear()
        for label_text in labels:
            t = ft.Text(f"{label_text}: --", size=13)
            self._summary_texts[label_text] = t
            self._summary_column.controls.append(t)
        self._summary_column.controls.append(ft.Divider(height=1))
        self._summary_column.controls.append(self._zingg_text)

    def set_results(
        self,
        morphologies: list[GrainMorphology],
        statistics: GrainStatistics,
    ) -> None:
        """Update all tabs with new data."""
        self._morphologies = morphologies
        self._statistics = statistics

        self._update_summary()
        self._update_table()
        self._update_charts()

    def _update_summary(self) -> None:
        stats = self._statistics
        if stats is None:
            return

        values = {
            "颗粒数量": str(stats.count),
            "平均圆度": f"{stats.circularity_mean:.4f}",
            "平均球度": f"{stats.sphericity_mean:.4f}",
            "平均等效粒径": f"{stats.d_eq_mean:.4f}",
            "平均长短轴比": f"{stats.aspect_ratio_mean:.4f}",
            "平均凸度": f"{stats.convexity_mean:.4f}",
        }
        for label_text, value in values.items():
            if label_text in self._summary_texts:
                self._summary_texts[label_text].value = f"{label_text}: {value}"

        if stats.zingg_counts:
            parts = []
            for key, count in stats.zingg_counts.items():
                pct = count / stats.count * 100 if stats.count > 0 else 0
                parts.append(f"{key}: {count} ({pct:.1f}%)")
            self._zingg_text.value = "Zingg分类: " + ", ".join(parts)
        else:
            self._zingg_text.value = "Zingg分类: --"

        self._summary_column.update()

    def _update_table(self) -> None:
        self._table.rows.clear()
        if not self._morphologies:
            self._table.update()
            return

        for i, grain in enumerate(self._morphologies):
            self._table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(i + 1))),
                        ft.DataCell(ft.Text(f"{grain.area:.2f}")),
                        ft.DataCell(ft.Text(f"{grain.perimeter:.2f}")),
                        ft.DataCell(ft.Text(f"{grain.circularity:.4f}")),
                        ft.DataCell(ft.Text(f"{grain.d_eq:.4f}")),
                        ft.DataCell(ft.Text(f"{grain.aspect_ratio:.4f}")),
                        ft.DataCell(ft.Text(f"{grain.sphericity:.4f}")),
                        ft.DataCell(ft.Text(f"{grain.convexity:.4f}")),
                    ]
                )
            )
        self._table.update()

    def _update_charts(self) -> None:
        stats = self._statistics
        if stats is None or not self._morphologies:
            return

        # Size histogram
        hist_fig = create_size_histogram(stats)
        self._hist_chart.figure = hist_fig

        # Circularity vs Sphericity scatter
        scatter_fig = create_circularity_sphericity_scatter(stats)
        self._scatter_chart.figure = scatter_fig

        # Zingg pie
        pie_fig = create_zingg_pie_chart(stats)
        self._pie_chart.figure = pie_fig

        self._hist_chart.update()
        self._scatter_chart.update()
        self._pie_chart.update()

    def clear(self) -> None:
        """Clear all results."""
        self._morphologies = []
        self._statistics = None

        for t in self._summary_texts.values():
            t.value = t.value.split(":")[0] + ": --"
        self._zingg_text.value = "Zingg分类: --"
        self._summary_column.update()

        self._table.rows.clear()
        self._table.update()

        self._hist_chart.figure = None
        self._scatter_chart.figure = None
        self._pie_chart.figure = None
        self._hist_chart.update()
        self._scatter_chart.update()
        self._pie_chart.update()

    def highlight_grain(self, grain_index: int) -> None:
        """Highlight a row in the table and switch to the data tab."""
        if not self._morphologies or grain_index < 0 or grain_index >= len(self._morphologies):
            return

        # Switch to table tab
        self._tabs.selected_index = 1
        self._tabs.update()
```

- [ ] **Step 2: 验证模块可导入**

```bash
uv run python -c "from ui.result_panel import ResultPanel; print('OK')"
```

预期：OK

- [ ] **Step 3: Commit**

```bash
git add ui/result_panel.py
git commit -m "feat: add Flet result panel with Plotly charts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 创建 ui/app.py（主 Flet 页面）

**Files:**
- Create: `ui/app.py`

- [ ] **Step 1: 创建 ui/app.py**

```python
"""Main Flet application page for SandAnalyze."""

import time

import cv2
import flet as ft
import numpy as np

from core.exporter import export_annotated_image as _export_annotated_image
from core.exporter import export_csv as _export_csv
from core.morphology import (
    GrainMorphology,
    GrainStatistics,
    compute_morphology,
    compute_statistics,
)
from core.preprocessor import PreprocessConfig, preprocess
from core.traditional import GrainContour, detect_grains
from core.yolo_detector import YOLODetector, refine_with_yolo
from ui.image_view import ImageView
from ui.result_panel import ResultPanel
from ui.settings_panel import SettingsPanel


class SandAnalyzePage:
    """Main Flet page for the SandAnalyze application.

    Layout:
        - AppBar with action buttons
        - Body: Row with ImageView (left) and Tabs (right)
        - SettingsPanel below the right-side tabs
        - Status bar at the bottom
    """

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.page.title = "SandAnalyze - 沙粒形态分析系统"
        self.page.window.width = 1200
        self.page.window.height = 800
        self.page.theme_mode = ft.ThemeMode.LIGHT

        # Internal state
        self._original_image: np.ndarray | None = None
        self._grains: list[GrainContour] = []
        self._morphologies: list[GrainMorphology] = []
        self._statistics: GrainStatistics | None = None
        self._config = PreprocessConfig()
        self._yolo_detector = YOLODetector()
        self._detection_method = "传统方法"
        self._last_processing_time = 0.0

        # File pickers
        self._open_picker = ft.FilePicker(on_result=self._on_image_picked)
        self._export_csv_picker = ft.FilePicker(on_result=self._on_csv_export_path)
        self._export_img_picker = ft.FilePicker(on_result=self._on_img_export_path)
        self.page.overlay.extend([
            self._open_picker,
            self._export_csv_picker,
            self._export_img_picker,
        ])

        # --- Build UI components ---

        # Image view (left)
        self._image_view = ImageView(on_grain_clicked=self._on_grain_clicked)

        # Result panel (right, top)
        self._result_panel = ResultPanel()

        # Settings panel (right, bottom)
        self._settings_panel = SettingsPanel(
            on_config_changed=self._on_config_changed,
            on_detect_requested=self._run_detection,
        )

        # YOLO toggle
        self._yolo_switch = ft.Switch(
            label="YOLO精细分割",
            value=True,
            disabled=not self._yolo_detector.is_available,
            label_position=ft.LabelPosition.LEFT,
        )
        if not self._yolo_detector.is_available:
            self._yolo_switch.label = "YOLO精细分割 (模型不可用)"

        # Status bar
        self._status_text = ft.Text(
            "方法: 传统方法 | 颗粒数: 0 | 请打开图像并运行检测",
            size=12,
            italic=True,
        )

        # --- Layout ---

        # Right side: result tabs + settings
        right_column = ft.Column(
            controls=[
                self._result_panel.control,
                ft.Container(
                    content=ft.Column(
                        controls=[
                            self._yolo_switch,
                            self._settings_panel.control,
                        ],
                        spacing=8,
                    ),
                    padding=ft.padding.only(top=4),
                ),
            ],
            expand=True,
            spacing=0,
        )

        # Main body
        body = ft.Row(
            controls=[
                ft.Container(
                    content=self._image_view.control,
                    expand=2,
                    border=ft.border.all(1, ft.Colors.OUTLINE),
                    border_radius=8,
                ),
                ft.Container(
                    content=right_column,
                    expand=1,
                    padding=ft.padding.only(left=4),
                ),
            ],
            expand=True,
        )

        # AppBar
        self.page.appbar = ft.AppBar(
            title=ft.Text("SandAnalyze - 沙粒形态分析系统", size=16),
            actions=[
                ft.IconButton(
                    icon=ft.Icons.FOLDER_OPEN,
                    tooltip="打开图像",
                    on_click=lambda e: self._open_picker.pick_files(
                        allowed_extensions=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                    ),
                ),
                ft.IconButton(
                    icon=ft.Icons.PLAY_ARROW,
                    tooltip="运行检测",
                    on_click=lambda e: self._run_detection(),
                ),
                ft.IconButton(
                    icon=ft.Icons.TABLE_CHART,
                    tooltip="导出 CSV",
                    on_click=lambda e: self._export_csv(),
                ),
                ft.IconButton(
                    icon=ft.Icons.IMAGE,
                    tooltip="导出标注图",
                    on_click=lambda e: self._export_image(),
                ),
                ft.PopupMenuButton(
                    items=[
                        ft.PopupMenuItem(
                            text="关于 SandAnalyze",
                            on_click=lambda e: self._show_about(),
                        ),
                    ],
                ),
            ],
        )

        # Full page layout
        self.page.add(
            ft.Column(
                controls=[
                    body,
                    ft.Container(
                        content=self._status_text,
                        padding=ft.padding.only(top=8, left=4),
                    ),
                ],
                expand=True,
                spacing=0,
            )
        )

    # --- Event handlers ---

    def _on_image_picked(self, e: ft.FilePickerResultEvent) -> None:
        if e.files is None or len(e.files) == 0:
            return

        path = e.files[0].path
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            self._show_error(f"无法读取图像文件: {path}")
            return

        self._original_image = image
        self._grains = []
        self._morphologies = []
        self._statistics = None
        self._detection_method = "传统方法"

        self._image_view.set_image(image)
        self._result_panel.clear()
        self._update_status()
        self.page.update()

    def _on_config_changed(self, config: PreprocessConfig) -> None:
        self._config = config

    def _on_grain_clicked(self, grain_index: int) -> None:
        if 0 <= grain_index < len(self._morphologies):
            self._result_panel.highlight_grain(grain_index)

    def _run_detection(self) -> None:
        if self._original_image is None:
            self._show_info("请先打开图像文件。")
            return

        start_time = time.time()

        try:
            mask = preprocess(self._original_image, self._config)
            traditional_grains = detect_grains(mask, min_area=self._config.min_area)
            self._detection_method = "传统方法"

            if self._yolo_detector.is_available and self._yolo_switch.value:
                self._grains = refine_with_yolo(
                    self._original_image,
                    traditional_grains,
                    self._yolo_detector,
                    min_area=self._config.min_area,
                )
                if self._grains is not traditional_grains:
                    self._detection_method = "混合方法(传统+YOLO)"
            else:
                self._grains = traditional_grains

            self._morphologies = [
                compute_morphology(g.contour, g.mask) for g in self._grains
            ]
            self._statistics = compute_statistics(self._morphologies)

        except Exception as exc:
            self._show_error(f"检测过程中发生错误:\n{exc}")
            return

        elapsed = time.time() - start_time
        self._last_processing_time = elapsed

        self._image_view.set_grains(self._grains, self._morphologies)
        self._result_panel.set_results(self._morphologies, self._statistics)
        self._update_status()
        self.page.update()

    def _export_csv(self) -> None:
        if not self._morphologies:
            self._show_info("请先运行检测。")
            return

        self._export_csv_picker.save_file(
            file_name="sand_analysis.csv",
            allowed_extensions=["csv"],
        )

    def _on_csv_export_path(self, e: ft.FilePickerResultEvent) -> None:
        if e.path is None:
            return

        try:
            _export_csv(self._morphologies, e.path)
            self._status_text.value = f"已导出 CSV: {e.path}"
            self._status_text.update()
        except Exception as exc:
            self._show_error(f"导出 CSV 失败:\n{exc}")

    def _export_image(self) -> None:
        if self._original_image is None or not self._grains:
            self._show_info("请先运行检测。")
            return

        self._export_img_picker.save_file(
            file_name="sand_annotated.png",
            allowed_extensions=["png"],
        )

    def _on_img_export_path(self, e: ft.FilePickerResultEvent) -> None:
        if e.path is None:
            return

        try:
            _export_annotated_image(
                self._original_image,
                self._grains,
                e.path,
                morphologies=self._morphologies,
            )
            self._status_text.value = f"已导出标注图: {e.path}"
            self._status_text.update()
        except Exception as exc:
            self._show_error(f"导出标注图失败:\n{exc}")

    def _update_status(self) -> None:
        if self._statistics is not None:
            self._status_text.value = (
                f"方法: {self._detection_method} | "
                f"颗粒数: {self._statistics.count} | "
                f"处理时间: {self._last_processing_time:.2f}s"
            )
        else:
            self._status_text.value = (
                f"方法: {self._detection_method} | 颗粒数: 0 | 请打开图像并运行检测"
            )
        self._status_text.update()

    def _show_error(self, message: str) -> None:
        self.page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("错误"),
                content=ft.Text(message),
                actions=[ft.TextButton("确定", on_click=lambda e: self.page.close_dialog())],
            )
        )

    def _show_info(self, message: str) -> None:
        self.page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("提示"),
                content=ft.Text(message),
                actions=[ft.TextButton("确定", on_click=lambda e: self.page.close_dialog())],
            )
        )

    def _show_about(self) -> None:
        self.page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("关于 SandAnalyze"),
                content=ft.Text(
                    "SandAnalyze - 沙粒形态分析系统\n\n"
                    "版本: 0.1.0\n\n"
                    "基于 OpenCV 传统图像处理和 YOLOv8-seg 深度学习，"
                    "对光学显微镜拍摄的沙粒图像进行颗粒识别、形状分析和数量统计。\n\n"
                    "技术栈: Python 3.13, OpenCV, Flet, Plotly, numpy"
                ),
                actions=[ft.TextButton("确定", on_click=lambda e: self.page.close_dialog())],
            )
        )
```

- [ ] **Step 2: 验证模块可导入**

```bash
uv run python -c "from ui.app import SandAnalyzePage; print('OK')"
```

预期：OK

- [ ] **Step 3: Commit**

```bash
git add ui/app.py
git commit -m "feat: add main Flet application page

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: 修改 main.py 支持 --web 和 --port 参数

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 重写 main.py**

```python
"""SandAnalyze - 沙粒形态分析系统入口。

Supports:
    - Desktop mode (default): Flet window with native WebView
    - Web mode (--web): Flet HTTP server, accessible via browser

Usage:
    uv run python main.py              # Desktop mode
    uv run python main.py --web        # Web mode (port 8000)
    uv run python main.py --web --port 8080  # Web mode on port 8080
"""

import argparse
import sys

import flet as ft


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SandAnalyze - 沙粒形态分析系统",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="在 Web 模式下启动（默认桌面模式）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web 模式下的 HTTP 端口（默认: 8000）",
    )
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = parse_args()

    from ui.app import SandAnalyzePage

    if args.web:
        # Web mode: start HTTP server
        ft.app(
            target=SandAnalyzePage,
            view=ft.AppView.WEB_BROWSER,
            port=args.port,
        )
    else:
        # Desktop mode: native window
        ft.app(target=SandAnalyzePage)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证参数解析**

```bash
uv run python main.py --help
```

预期：显示 argparse 帮助文本，包含 `--web` 和 `--port` 选项。

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add --web and --port CLI arguments for Flet modes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: 删除 gui/ 目录

**Files:**
- Delete: `gui/__init__.py`
- Delete: `gui/app.py`
- Delete: `gui/image_panel.py`
- Delete: `gui/result_panel.py`
- Delete: `gui/settings_panel.py`

- [ ] **Step 1: 删除 gui/ 目录**

```bash
git rm -r gui/
```

- [ ] **Step 2: 验证项目仍可导入核心模块**

```bash
uv run python -c "from core.preprocessor import PreprocessConfig, preprocess; from core.traditional import detect_grains; from core.morphology import compute_morphology, compute_statistics; from core.exporter import export_csv, export_annotated_image; from core.yolo_detector import YOLODetector; print('All core modules OK')"
```

预期：All core modules OK

- [ ] **Step 3: 验证 Flet UI 模块可导入**

```bash
uv run python -c "from ui.app import SandAnalyzePage; from ui.charts import create_size_histogram; from ui.image_view import ImageView; from ui.result_panel import ResultPanel; from ui.settings_panel import SettingsPanel; print('All UI modules OK')"
```

预期：All UI modules OK

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: remove PyQt6 gui/ directory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: 验证测试仍然通过

**Files:**
- (无变更，仅验证)

- [ ] **Step 1: 运行全部测试**

```bash
uv run pytest tests/ -v --tb=short
```

预期：所有测试 PASS（核心逻辑测试不受 UI 重构影响）。

- [ ] **Step 2: 确认没有 gui/ 引用残留**

```bash
uv run python -c "import core; print(dir(core))"
```

预期：输出 core 模块的公开 API，无 gui 相关引用。

- [ ] **Step 3: Commit（如有必要修复后提交）**

如果有测试失败，修复后：
```bash
git add <fixed files>
git commit -m "fix: update tests after Flet UI migration"
```

如果没有问题，跳过此 commit。
```

---

### Task 10: 更新 README.md 和 sandanalyze.spec

**Files:**
- Modify: `README.md`
- Modify: `sandanalyze.spec`

- [ ] **Step 1: 更新 README.md 技术栈部分**

将第45-50行的技术栈部分替换为：

```markdown
## 技术栈

- Python 3.13
- OpenCV（图像处理）
- ultralytics / YOLOv8-seg（深度学习检测）
- Flet（GUI / Web）
- Plotly（图表）
- numpy / scipy（数值计算）
```

- [ ] **Step 2: 更新 README.md 使用部分**

在第22行后添加 Web 模式说明：

```markdown
Web 模式启动：
```bash
uv run python main.py --web
```
然后在浏览器中打开 http://localhost:8000。
```

- [ ] **Step 3: 更新 sandanalyze.spec**

替换 spec 文件中的 datas 和 hiddenimports：

```python
# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('core', 'core'), ('ui', 'ui'), ('models', 'models')],
    hiddenimports=['flet', 'plotly', 'cv2', 'numpy', 'scipy', 'ultralytics'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='sandanalyze',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

- [ ] **Step 4: Commit**

```bash
git add README.md sandanalyze.spec
git commit -m "docs: update README and spec for Flet migration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
