"""Result panel with statistics, table, and charts for sand grain analysis."""

from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.morphology import GrainMorphology, GrainStatistics

matplotlib.use("Qt5Agg")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class ResultPanel(QWidget):
    """Result panel displaying statistics, table, and charts for sand grain analysis."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._morphologies: List[GrainMorphology] = []
        self._statistics: Optional[GrainStatistics] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)

        self._tabs = QTabWidget(self)

        # Summary tab
        self._summary_tab = QWidget()
        self._summary_layout = QGridLayout(self._summary_tab)
        self._summary_labels: dict[str, QLabel] = {}
        self._summary_tab.setLayout(self._summary_layout)
        self._tabs.addTab(self._summary_tab, "统计摘要")

        # Table tab
        self._table_widget = QTableWidget()
        self._table_widget.setColumnCount(9)
        self._table_widget.setHorizontalHeaderLabels(
            [
                "ID",
                "面积",
                "周长",
                "圆度",
                "等效粒径",
                "长短轴比",
                "球度",
                "凸度",
            ]
        )
        self._tabs.addTab(self._table_widget, "颗粒数据")

        # Charts tab
        self._charts_tab = QWidget()
        self._charts_layout = QVBoxLayout(self._charts_tab)
        self._charts_tab.setLayout(self._charts_layout)
        self._tabs.addTab(self._charts_tab, "图表")

        self._layout.addWidget(self._tabs)
        self.setLayout(self._layout)

        self._init_summary_tab()
        self._init_charts_tab()

    def _init_summary_tab(self) -> None:
        labels = [
            "颗粒数量",
            "平均圆度",
            "平均球度",
            "平均等效粒径",
            "平均长短轴比",
            "平均凸度",
        ]
        row, col = 0, 0
        for label_text in labels:
            label = QLabel(f"{label_text}: --")
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._summary_layout.addWidget(label, row, col)
            self._summary_labels[label_text] = label
            col += 1
            if col >= 3:
                col = 0
                row += 1

        self._zingg_label = QLabel("Zingg分类: --")
        self._zingg_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._summary_layout.addWidget(self._zingg_label, row, 0, 1, 3)

    def _init_charts_tab(self) -> None:
        self._charts_layout.setSpacing(10)
        self._charts_layout.setContentsMargins(10, 10, 10, 10)

        # Top row: histogram and pie chart
        self._top_row = QHBoxLayout()
        self._charts_layout.addLayout(self._top_row)

        # Particle size distribution histogram
        self._fig_hist, self._ax_hist = plt.subplots(figsize=(5, 4))
        self._canvas_hist = FigureCanvas(self._fig_hist)
        self._top_row.addWidget(self._canvas_hist)

        # Zingg classification pie chart
        self._fig_pie, self._ax_pie = plt.subplots(figsize=(5, 4))
        self._canvas_pie = FigureCanvas(self._fig_pie)
        self._top_row.addWidget(self._canvas_pie)

        # Bottom row: scatter plot
        self._fig_scatter, self._ax_scatter = plt.subplots(figsize=(10, 4))
        self._canvas_scatter = FigureCanvas(self._fig_scatter)
        self._charts_layout.addWidget(self._canvas_scatter)

    def set_results(
        self, morphologies: List[GrainMorphology], statistics: GrainStatistics
    ) -> None:
        """Update all tabs with new data."""
        self._morphologies = morphologies
        self._statistics = statistics

        self._update_summary_tab()
        self._update_table_tab()
        self._update_charts_tab()

    def _update_summary_tab(self) -> None:
        if self._statistics is None:
            return

        stats = self._statistics
        values = {
            "颗粒数量": str(stats.count),
            "平均圆度": f"{stats.circularity_mean:.4f}",
            "平均球度": f"{stats.sphericity_mean:.4f}",
            "平均等效粒径": f"{stats.d_eq_mean:.4f}",
            "平均长短轴比": f"{stats.aspect_ratio_mean:.4f}",
            "平均凸度": f"{stats.convexity_mean:.4f}",
        }

        for label_text, value in values.items():
            if label_text in self._summary_labels:
                self._summary_labels[label_text].setText(f"{label_text}: {value}")

        if stats.zingg_counts:
            zingg_parts = []
            for key, count in stats.zingg_counts.items():
                percentage = (
                    count / stats.count * 100 if stats.count > 0 else 0
                )
                zingg_parts.append(f"{key}: {count} ({percentage:.1f}%)")
            self._zingg_label.setText("Zingg分类: " + ", ".join(zingg_parts))
        else:
            self._zingg_label.setText("Zingg分类: --")

    def _update_table_tab(self) -> None:
        if not self._morphologies:
            self._table_widget.setRowCount(0)
            return

        self._table_widget.setRowCount(len(self._morphologies))
        for i, grain in enumerate(self._morphologies):
            self._table_widget.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._table_widget.setItem(
                i, 1, QTableWidgetItem(f"{grain.area:.2f}")
            )
            self._table_widget.setItem(
                i, 2, QTableWidgetItem(f"{grain.perimeter:.2f}")
            )
            self._table_widget.setItem(
                i, 3, QTableWidgetItem(f"{grain.circularity:.4f}")
            )
            self._table_widget.setItem(
                i, 4, QTableWidgetItem(f"{grain.d_eq:.4f}")
            )
            self._table_widget.setItem(
                i, 5, QTableWidgetItem(f"{grain.aspect_ratio:.4f}")
            )
            self._table_widget.setItem(
                i, 6, QTableWidgetItem(f"{grain.sphericity:.4f}")
            )
            self._table_widget.setItem(
                i, 7, QTableWidgetItem(f"{grain.convexity:.4f}")
            )
            self._table_widget.setItem(
                i, 8, QTableWidgetItem(f"{grain.feret_max:.4f}")
            )

    def _update_charts_tab(self) -> None:
        if self._statistics is None or not self._morphologies:
            return

        stats = self._statistics

        # Particle size distribution histogram
        self._ax_hist.clear()
        if stats.d_eq_values:
            self._ax_hist.hist(stats.d_eq_values, bins=20, edgecolor="black")
            self._ax_hist.set_xlabel("等效粒径 (d_eq)")
            self._ax_hist.set_ylabel("频数")
            self._ax_hist.set_title("粒径分布")
        self._fig_hist.tight_layout()
        self._canvas_hist.draw()

        # Circularity vs Sphericity scatter plot
        self._ax_scatter.clear()
        if stats.circularity_values and stats.sphericity_values:
            self._ax_scatter.scatter(
                stats.circularity_values,
                stats.sphericity_values,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
            )
            self._ax_scatter.set_xlabel("圆度")
            self._ax_scatter.set_ylabel("球度")
            self._ax_scatter.set_title("圆度 vs 球度")
        self._fig_scatter.tight_layout()
        self._canvas_scatter.draw()

        # Zingg classification pie chart
        self._ax_pie.clear()
        if stats.zingg_counts:
            labels = list(stats.zingg_counts.keys())
            sizes = list(stats.zingg_counts.values())
            colors = ["#ff9999", "#66b3ff", "#99ff99"]
            self._ax_pie.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors[: len(labels)],
                startangle=90,
            )
            self._ax_pie.set_title("Zingg分类")
        self._fig_pie.tight_layout()
        self._canvas_pie.draw()

    def clear(self) -> None:
        """Clear all results."""
        self._morphologies = []
        self._statistics = None

        # Clear summary
        for label_text, label in self._summary_labels.items():
            label.setText(f"{label_text}: --")
        self._zingg_label.setText("Zingg分类: --")

        # Clear table
        self._table_widget.setRowCount(0)

        # Clear charts
        self._ax_hist.clear()
        self._fig_hist.tight_layout()
        self._canvas_hist.draw()

        self._ax_scatter.clear()
        self._fig_scatter.tight_layout()
        self._canvas_scatter.draw()

        self._ax_pie.clear()
        self._fig_pie.tight_layout()
        self._canvas_pie.draw()

    def highlight_grain(self, grain_id: int) -> None:
        """Highlight a specific grain in the table."""
        if not self._morphologies or grain_id < 1 or grain_id > len(self._morphologies):
            return

        row = grain_id - 1
        self._table_widget.selectRow(row)
        self._table_widget.scrollToItem(self._table_widget.item(row, 0))
