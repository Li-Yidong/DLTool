import os
from PyQt6.QtWidgets import QApplication
from app.ui.heatmap_view_window import HeatmapViewerWindow

def launch_heatmap_viewer(config: dict) -> None:

    heatmap_viewer = HeatmapViewerWindow(config)
    heatmap_viewer.show()

