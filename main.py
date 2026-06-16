"""SandAnalyze - 沙粒形态分析系统入口。"""

import sys

from PyQt6.QtWidgets import QApplication

from gui.app import SandAnalyzeApp


def main() -> None:
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("SandAnalyze")
    app.setApplicationDisplayName("SandAnalyze - 沙粒形态分析系统")

    window = SandAnalyzeApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
