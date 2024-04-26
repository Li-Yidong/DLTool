"""app/init.py"""
import sys
from PyQt6.QtWidgets import QApplication
from .ui.main_window import MainWindow
from .utils.config import AppConfig
from .ui.heatmap_view_window import HeatmapViewerWindow

def run() -> int:
    """
    Initializes the application and runs it.

    Returns:
        int: the exit status code
    """
    app: QApplication = QApplication(sys.argv)
    AppConfig.initialize()

    window: MainWindow = MainWindow()
    window.show()

    return sys.exit(app.exec())
