"""Settings panel for preprocessing parameter adjustment."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.preprocessor import PreprocessConfig


class SettingsPanel(QWidget):
    """Settings panel for preprocessing parameters."""

    config_changed = pyqtSignal(object)
    detect_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        group = QGroupBox("预处理参数")
        group_layout = QVBoxLayout(group)

        # blur_kernel
        self.blur_kernel_spin = self._create_spin_box(1, 31, 5, 2)
        group_layout.addLayout(self._create_param_row("模糊核大小:", self.blur_kernel_spin))

        # adaptive_block_size
        self.adaptive_block_size_spin = self._create_spin_box(3, 99, 11, 2)
        group_layout.addLayout(
            self._create_param_row("自适应块大小:", self.adaptive_block_size_spin)
        )

        # adaptive_c
        self.adaptive_c_spin = self._create_spin_box(-10, 20, 2)
        group_layout.addLayout(self._create_param_row("自适应常数 C:", self.adaptive_c_spin))

        # morph_kernel_size
        self.morph_kernel_size_spin = self._create_spin_box(1, 21, 3, 2)
        group_layout.addLayout(
            self._create_param_row("形态学核大小:", self.morph_kernel_size_spin)
        )

        # min_area
        self.min_area_spin = self._create_spin_box(1, 10000, 50)
        group_layout.addLayout(self._create_param_row("最小面积:", self.min_area_spin))

        # Checkboxes
        self.use_clahe_check = QCheckBox("CLAHE 增强")
        self.use_clahe_check.setChecked(True)
        group_layout.addWidget(self.use_clahe_check)

        self.use_watershed_check = QCheckBox("分水岭分割")
        self.use_watershed_check.setChecked(True)
        group_layout.addWidget(self.use_watershed_check)

        layout.addWidget(group)

        # Detect button
        self.detect_button = QPushButton("运行检测")
        self.detect_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #4CAF50;"
            "  color: white;"
            "  font-weight: bold;"
            "  padding: 8px 16px;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #45a049;"
            "}"
        )
        self.detect_button.clicked.connect(self.detect_requested.emit)
        layout.addWidget(self.detect_button)

        layout.addStretch()

        # Connect signals
        for spin in (
            self.blur_kernel_spin,
            self.adaptive_block_size_spin,
            self.adaptive_c_spin,
            self.morph_kernel_size_spin,
            self.min_area_spin,
        ):
            spin.valueChanged.connect(self._on_param_changed)

        self.use_clahe_check.stateChanged.connect(self._on_param_changed)
        self.use_watershed_check.stateChanged.connect(self._on_param_changed)

    def _create_spin_box(
        self, min_val: int, max_val: int, default: int, step: int = 1
    ) -> QSpinBox:
        """Create a QSpinBox with the given range and default value."""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(default)
        return spin

    def _create_param_row(self, label_text: str, spin_box: QSpinBox) -> QHBoxLayout:
        """Create a horizontal layout with a label and spin box."""
        layout = QHBoxLayout()
        label = QLabel(label_text)
        layout.addWidget(label)
        layout.addWidget(spin_box)
        layout.addStretch()
        return layout

    def _on_param_changed(self) -> None:
        """Collect all parameter values, create PreprocessConfig, emit config_changed."""
        config = self.get_config()
        self.config_changed.emit(config)

    def get_config(self) -> PreprocessConfig:
        """Return current PreprocessConfig."""
        return PreprocessConfig(
            blur_kernel=self.blur_kernel_spin.value(),
            adaptive_block_size=self.adaptive_block_size_spin.value(),
            adaptive_c=self.adaptive_c_spin.value(),
            morph_kernel_size=self.morph_kernel_size_spin.value(),
            min_area=self.min_area_spin.value(),
            use_clahe=self.use_clahe_check.isChecked(),
            use_watershed=self.use_watershed_check.isChecked(),
        )
