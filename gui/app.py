"""Main application window for SandAnalyze."""

import time
from typing import Optional

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
    QVBoxLayout,
    QWidget,
)

from core.exporter import export_annotated_image as _export_annotated_image
from core.exporter import export_csv as _export_csv
from core.morphology import GrainMorphology, GrainStatistics, compute_morphology, compute_statistics
from core.preprocessor import PreprocessConfig, preprocess
from core.traditional import GrainContour, detect_grains
from core.yolo_detector import YOLODetector
from gui.image_panel import ImagePanel
from gui.result_panel import ResultPanel
from gui.settings_panel import SettingsPanel


class SandAnalyzeApp(QMainWindow):
    """Main application window for the SandAnalyze sand morphology analysis system.

    Layout:
        - Left side: ImagePanel for displaying images and grain overlays.
        - Right side: ResultPanel (top) + SettingsPanel (bottom) in a vertical splitter.

    Features:
        - Menu bar: File (Open, Export CSV, Export Image, Quit),
                    Detect (Run Detection),
                    Help (About)
        - Toolbar: Open, Detect buttons
        - Status bar: Shows detection method, grain count, processing time
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SandAnalyze - 沙粒形态分析系统")
        self.setMinimumSize(1200, 800)

        # Internal state
        self._original_image: Optional[np.ndarray] = None
        self._grains: list[GrainContour] = []
        self._morphologies: list[GrainMorphology] = []
        self._statistics: Optional[GrainStatistics] = None
        self._config = PreprocessConfig()
        self._yolo_detector = YOLODetector()
        self._detection_method = "传统方法"

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

    def _setup_ui(self) -> None:
        """Set up the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: ImagePanel
        self.image_panel = ImagePanel()
        self.image_panel.grain_clicked.connect(self._on_grain_clicked)
        main_splitter.addWidget(self.image_panel)

        # Right: ResultPanel (top) + SettingsPanel (bottom)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.result_panel = ResultPanel()
        self.settings_panel = SettingsPanel()
        self.settings_panel.config_changed.connect(self._on_config_changed)
        self.settings_panel.detect_requested.connect(self._run_detection)

        right_splitter.addWidget(self.result_panel)
        right_splitter.addWidget(self.settings_panel)
        right_splitter.setSizes([400, 300])

        right_layout.addWidget(right_splitter)
        main_splitter.addWidget(right_widget)

        main_splitter.setSizes([800, 400])
        layout.addWidget(main_splitter)

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("文件(&F)")

        open_action = QAction("打开图像(&O)...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_csv_action = QAction("导出 CSV(&C)...", self)
        export_csv_action.triggered.connect(self._export_csv)
        file_menu.addAction(export_csv_action)

        export_img_action = QAction("导出标注图(&I)...", self)
        export_img_action.triggered.connect(self._export_annotated_image)
        file_menu.addAction(export_img_action)

        file_menu.addSeparator()

        quit_action = QAction("退出(&Q)", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
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
        about_action = QAction("关于(&A)...", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self) -> None:
        """Set up the main toolbar."""
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)

        open_action = QAction("打开", self)
        open_action.triggered.connect(self._open_image)
        toolbar.addAction(open_action)

        detect_action = QAction("检测", self)
        detect_action.triggered.connect(self._run_detection)
        toolbar.addAction(detect_action)

    def _setup_statusbar(self) -> None:
        """Set up the status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._update_statusbar()

    def _update_statusbar(self) -> None:
        """Update status bar text based on current state."""
        if self._statistics is not None:
            self._statusbar.showMessage(
                f"方法: {self._detection_method} | "
                f"颗粒数: {self._statistics.count} | "
                f"处理时间: {getattr(self, '_last_processing_time', 0.0):.2f}s"
            )
        else:
            self._statusbar.showMessage(
                f"方法: {self._detection_method} | 颗粒数: 0 | 请打开图像并运行检测"
            )

    # --- Actions ---

    def _open_image(self) -> None:
        """Open an image file dialog and load the selected image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开沙粒图像",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)",
        )
        if not path:
            return

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "错误", f"无法读取图像文件:\n{path}")
            return

        self._original_image = image
        self._grains = []
        self._morphologies = []
        self._statistics = None
        self._detection_method = "传统方法"

        self.image_panel.set_image(image)
        self.result_panel.clear()
        self._update_statusbar()

    def _run_detection(self) -> None:
        """Run grain detection on the currently loaded image."""
        if self._original_image is None:
            QMessageBox.information(self, "提示", "请先打开图像文件。")
            return

        start_time = time.time()

        try:
            # Preprocess
            mask = preprocess(self._original_image, self._config)

            # Detect grains using traditional method
            self._grains = detect_grains(mask, min_area=self._config.min_area)
            self._detection_method = "传统方法"

            # Compute morphologies
            self._morphologies = [
                compute_morphology(g.contour, g.mask) for g in self._grains
            ]
            self._statistics = compute_statistics(self._morphologies)

        except Exception as exc:
            QMessageBox.critical(self, "检测错误", f"检测过程中发生错误:\n{exc}")
            return

        elapsed = time.time() - start_time
        self._last_processing_time = elapsed

        # Update UI
        self.image_panel.set_grains(self._grains, self._morphologies)
        self.result_panel.set_results(self._morphologies, self._statistics)
        self._update_statusbar()

    def _on_config_changed(self, config: PreprocessConfig) -> None:
        """Handle preprocessing config changes from the settings panel."""
        self._config = config

    def _on_grain_clicked(self, grain_id: int) -> None:
        """Handle grain click from the image panel to highlight in result panel."""
        if 0 <= grain_id < len(self._morphologies):
            self.result_panel.highlight_grain(grain_id)

    def _export_csv(self) -> None:
        """Export morphological data to a CSV file."""
        if not self._morphologies:
            QMessageBox.information(self, "提示", "请先运行检测。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 CSV",
            "sand_analysis.csv",
            "CSV 文件 (*.csv)",
        )
        if path:
            try:
                _export_csv(self._morphologies, path)
                self._statusbar.showMessage(f"已导出 CSV: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "导出错误", f"导出 CSV 失败:\n{exc}")

    def _export_annotated_image(self) -> None:
        """Export an annotated image with grain contours overlaid."""
        if self._original_image is None or not self._grains:
            QMessageBox.information(self, "提示", "请先运行检测。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出标注图",
            "sand_annotated.png",
            "PNG 图像 (*.png)",
        )
        if path:
            try:
                _export_annotated_image(
                    self._original_image, self._grains, path,
                    morphologies=self._morphologies
                )
                self._statusbar.showMessage(f"已导出标注图: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "导出错误", f"导出标注图失败:\n{exc}")

    def _show_about(self) -> None:
        """Show the About dialog."""
        QMessageBox.about(
            self,
            "关于 SandAnalyze",
            "<h2>SandAnalyze - 沙粒形态分析系统</h2>"
            "<p>版本: 0.1.0</p>"
            "<p>基于 OpenCV 传统图像处理和 YOLOv8-seg 深度学习，"
            "对光学显微镜拍摄的沙粒图像进行颗粒识别、形状分析和数量统计。</p>"
            "<p>技术栈: Python 3.13, OpenCV, PyQt6, matplotlib, numpy</p>",
        )
