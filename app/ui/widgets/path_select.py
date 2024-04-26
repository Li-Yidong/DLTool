"""app/ui/widgets/path_select.py"""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
)
from PyQt6.QtGui import QFontMetrics
from typing import List

from ...utils.config import AppConfig


class PathSelect(QWidget):
    directoryEntered: pyqtSignal = pyqtSignal()
    WARNING_MESSAGE: List[str] = ["No path selected", "No path selected"]

    def __init__(self, text: str, path: str, target: str = "Folder") -> None:
        super().__init__()
        self.layout_widget: QHBoxLayout = QHBoxLayout(self)
        self.layout_widget.setContentsMargins(0, 0, 0, 0)
        self.layout_widget.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.target = target
        self.file_dialog: QFileDialog = QFileDialog(self, text)
        self.file_dialog.setFileMode(
            QFileDialog.FileMode.Directory
            if target == "Folder"
            else QFileDialog.FileMode.ExistingFile
        )

        self.label_none: str = self.WARNING_MESSAGE[AppConfig.CUR_LANG]
        self.label: QLabel = QLabel(path if path else self.label_none)
        self.button: QPushButton = QPushButton(text, self)
        metrics: QFontMetrics = self.button.fontMetrics()
        self.button.setFixedWidth(metrics.horizontalAdvance(text) + 20)
        self.button.clicked.connect(self.show_dialog)
        self.folder_path = path

        self.layout_widget.addWidget(self.button)
        self.layout_widget.addWidget(self.label)
        self.setFixedSize(self.layout_widget.sizeHint())

    def show_dialog(self) -> None:
        self.folder_path = (
            self.file_dialog.getExistingDirectory(
                None,
                ["Select Directory", "日本語"][AppConfig.CUR_LANG],
            )
            if self.target == "Folder"
            else self.file_dialog.getOpenFileName(
                None,
                ["Select File", "日本語"][AppConfig.CUR_LANG],
            )[0]
        )
        if self.folder_path:
            self.folder_path = f"{self.folder_path}{'/' if self.target == 'Folder' else ''}"
            self.label.setText(self.folder_path)
            self.directoryEntered.emit()
            self.setFixedSize(self.layout_widget.sizeHint())
        else:
            self.label.setText(self.label_none)

    def update_val(self, new_val: str) -> None:
        self.folder_path = new_val
        self.label.setText(new_val if new_val else self.label_none)

    def update_language(self, new_name: str) -> None:
        self.label_none = self.WARNING_MESSAGE[AppConfig.CUR_LANG]
        self.label.setText(self.folder_path if self.folder_path else self.label_none)
        self.button.setText(new_name)
        self.button.setFixedSize(self.button.sizeHint())
        self.setFixedSize(self.layout_widget.sizeHint())
