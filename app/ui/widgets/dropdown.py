"""app/ui/widgets/dropdown.py"""
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QComboBox,
    QHBoxLayout,
)
from typing import List

from ...utils.config import AppConfig


class Dropdown(QWidget):
    currentTextChanged: pyqtSignal = pyqtSignal()

    def __init__(self, text: str, items: List[str], default: str = None) -> None:
        super().__init__()
        self.layout_widget: QHBoxLayout = QHBoxLayout(self)
        self.layout_widget.setContentsMargins(0, 0, 0, 0)
        self.layout_widget.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.label: QLabel = QLabel(f"{text}:")  # todo
        metrics: QFontMetrics = self.label.fontMetrics()
        self.label.setFixedWidth(metrics.horizontalAdvance(text) + 10)

        self.combo_box: QComboBox = QComboBox()
        self.combo_box.addItems(items)
        self.combo_box.setCurrentText(default if default else items[0])
        self.combo_box.currentTextChanged.connect(self.on_current_changed)
        self.combo_box.setSizeAdjustPolicy(
            self.combo_box.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.current_text = self.combo_box.currentText()

        max_width, font_metrics = 0, self.combo_box.fontMetrics()
        for i in range(self.combo_box.count()):
            max_width = max(
                max_width, font_metrics.horizontalAdvance(self.combo_box.itemText(i))
            )
        self.combo_box.setFixedWidth(max_width + 30)

        self.layout_widget.addWidget(self.label)
        self.layout_widget.addWidget(self.combo_box)
        self.setFixedSize(self.layout_widget.sizeHint())

    def on_current_changed(self) -> None:
        self.current_text = self.combo_box.currentText()
        self.currentTextChanged.emit()

    def update_val(self, new_val: str) -> None:
        self.current_text = new_val
        self.combo_box.setCurrentText(new_val)

    def update_language(self, new_name: str):
        self.label.setText(f"{new_name}:")
        metrics: QFontMetrics = self.label.fontMetrics()
        self.label.setFixedWidth(metrics.horizontalAdvance(new_name) + 10)
        self.setFixedSize(self.layout_widget.sizeHint())
