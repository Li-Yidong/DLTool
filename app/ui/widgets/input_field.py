"""app/ui/widgets/input_field.py"""
from PyQt6.QtCore import Qt, QRegularExpression, QSize, QLocale, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QLineEdit,
    QHBoxLayout,
)
from PyQt6.QtGui import (
    QValidator,
    QFontMetrics,
    QIntValidator,
    QDoubleValidator,
    QRegularExpressionValidator,
)
from typing import Dict, List, Union, Callable

from ...utils.config import AppConfig
from ...utils.helper import InputTypes


class InputField(QWidget):
    textChanged: pyqtSignal = pyqtSignal()
    WARNING_MESSAGE: List[str] = ["Must not be empty", "日本語"]
    VALIDATORS: Dict[str, QValidator] = {
        InputTypes.STR_INPUT.value: None,
        InputTypes.INT_INPUT.value: QRegularExpressionValidator(
            QRegularExpression("\\d+")
        ),
        InputTypes.FLT_INPUT.value: QRegularExpressionValidator(
            QRegularExpression("\\d+\\.?\\d+")
        ),
        InputTypes.SHP_INPUT.value: QRegularExpressionValidator(
            QRegularExpression("\\d+(,\\s*\\d+){3}")
        ),
        InputTypes.SIZE_INPUT.value: QRegularExpressionValidator(
            QRegularExpression("\\d+,\\s*\\d+")
        ),
        InputTypes.LIST_INPUT.value: QRegularExpressionValidator(
            QRegularExpression("\\w+(,\\s*\\w+)+")
        ),
    }
    LIST_LENGTH_VALIDATORS: Dict[str, Callable] = {
        InputTypes.SHP_INPUT.value: lambda x: len(x) == 4,
        InputTypes.SIZE_INPUT.value: lambda x: len(x) == 2,
        InputTypes.LIST_INPUT.value: lambda x: len(x) > 1,
    }
    WARNING_MESSAGE_LIST: Dict[str, List[str]] = {
        InputTypes.SHP_INPUT.value: ["Shape must have 4 elements", "日本語"],
        InputTypes.SIZE_INPUT.value: ["Size must have 2 elements", "日本語"],
        InputTypes.LIST_INPUT.value: [
            "Element > 1 and no trailing commas",
            "日本語",
        ],
    }

    def __init__(
        self,
        text: str,
        placeholder: Union[str, int, float, list],
        width: int = 250,
        input_type="StrInput",
    ) -> None:
        super().__init__()
        if input_type not in self.VALIDATORS:  # guard for unrecognized input types
            raise Exception(f"InputType: {input_type} not recognized")
        self.layout_widget: QHBoxLayout = QHBoxLayout(self)
        self.layout_widget.setContentsMargins(0, 0, 0, 0)
        self.layout_widget.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.label: QLabel = QLabel(f"{text}:")
        metrics: QFontMetrics = self.label.fontMetrics()
        self.label.setFixedWidth(metrics.horizontalAdvance(text) + 10)

        self.custom_input_type = input_type
        self.input_type = type(placeholder)
        self.default_val = placeholder
        self.cur_val: Union[str, int, float] = placeholder
        self.line_edit: QLineEdit = QLineEdit()
        # if self.VALIDATORS[input_type]: self.VALIDATORS[input_type].setLocale(QLocale("en_US"))
        self.line_edit.setValidator(self.VALIDATORS[input_type])
        self.line_edit.setFixedWidth(width)
        placeholder_text: str = (
            str(placeholder)
            if self.input_type is not list
            else ", ".join(map(str, placeholder))
        )
        self.line_edit.setText(placeholder_text)
        self.line_edit.setPlaceholderText(placeholder_text)
        self.line_edit.textChanged.connect(
            lambda: self.on_text_changed(self.line_edit.text())
        )

        self.label_warning: QLabel = QLabel()
        self.label_warning.setText(self.WARNING_MESSAGE[AppConfig.CUR_LANG])
        self.label_warning.setStyleSheet("QLabel {color: red;}")

        self.layout_widget.addWidget(self.label)
        self.layout_widget.addWidget(self.line_edit)
        self.layout_widget.addWidget(self.label_warning)
        if self.input_type is list:
            self.list_warning: QLabel = QLabel()
            self.list_warning.setText(
                self.WARNING_MESSAGE_LIST[input_type][AppConfig.CUR_LANG]
            )
            self.list_warning.setStyleSheet("QLabel {color: red;}")
            self.layout_widget.addWidget(self.list_warning)
            self.LIST_LENGTH_VALIDATORS = self.LIST_LENGTH_VALIDATORS[input_type]
        self.setFixedSize(self.layout_widget.sizeHint())
        self.label_warning.setHidden(len(placeholder_text))
        if hasattr(self, "list_warning"):
            self.list_warning.setHidden(self.LIST_LENGTH_VALIDATORS(placeholder))

    def on_text_changed(self, text: str) -> None:
        self.label_warning.setHidden(len(text))
        if self.input_type is list:
            temp = [num.strip() for num in text.split(",")]
            # if len(temp) == 4 and all(list(map(lambda x: bool(x), temp))): # ! alternative
            if self.LIST_LENGTH_VALIDATORS(temp) and not any(
                list(map(lambda x: not bool(x), temp))
            ):
                self.cur_val = [
                    int(el.strip())
                    if self.custom_input_type != InputTypes.LIST_INPUT.value
                    else str(el.strip())
                    for el in text.split(",")
                ]
                self.list_warning.setHidden(True)
                return self.textChanged.emit()
            else:
                self.list_warning.setHidden(False)
        else:
            self.cur_val = self.input_type(text or self.default_val)
            return self.textChanged.emit()

    def update_val(self, new_val: Union[str, list, float, int]) -> None:
        self.cur_val = new_val
        self.line_edit.setText(
            str(new_val)
            if not isinstance(new_val, list)
            else ", ".join(map(str, new_val))
        )

    def update_language(self, new_name: str) -> None:
        size_hints: List[int] = []
        self.label.setText(f"{new_name}:")
        self.label_warning.setText(self.WARNING_MESSAGE[AppConfig.CUR_LANG])
        if self.input_type is list:
            self.list_warning.setText(
                self.WARNING_MESSAGE_LIST[self.custom_input_type][AppConfig.CUR_LANG]
            )
            self.list_warning.setFixedSize(self.list_warning.sizeHint())
            size_hints.append(self.list_warning.size().width())
        self.label_warning.setFixedSize(self.label_warning.sizeHint())
        size_hints.append(self.label_warning.size().width())
        metrics: QFontMetrics = self.label.fontMetrics()
        self.label.setFixedWidth(metrics.horizontalAdvance(new_name) + 10)
        self.setFixedSize(
            self.layout_widget.sizeHint() + QSize(max(size_hints), 0)
        )
