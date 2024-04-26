"""app/ui/widgets/switch.py"""
from PyQt6.QtCore import (
    Qt,
    QSize,
    QObject,
    QPointF,
    QEasingCurve,
    QPropertyAnimation,
    pyqtSlot,
    pyqtSignal,
    pyqtProperty,
)
from PyQt6.QtGui import QPainter, QPalette, QLinearGradient, QGradient
from PyQt6.QtWidgets import QAbstractButton, QApplication, QWidget, QLabel, QHBoxLayout
from typing import List


class SwitchPrivate(QObject):
    def __init__(self, q, parent=None):
        QObject.__init__(self, parent=parent)
        self.mPointer = q
        self.mPosition = 0.0
        self.mGradient = QLinearGradient()
        self.mGradient.setSpread(QGradient.Spread.PadSpread)

        self.animation = QPropertyAnimation(self)
        self.animation.setTargetObject(self)
        self.animation.setPropertyName(b"position")
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutExpo)

        self.animation.finished.connect(self.mPointer.update)

    @pyqtProperty(float)
    def position(self):
        return self.mPosition

    @position.setter
    def position(self, value):
        self.mPosition = value
        self.mPointer.update()

    def draw(self, painter):
        r = self.mPointer.rect()
        margin = r.height() // 10
        shadow = self.mPointer.palette().color(QPalette.ColorRole.Dark)
        light = self.mPointer.palette().color(QPalette.ColorRole.Light)
        button = self.mPointer.palette().color(QPalette.ColorRole.Button)
        painter.setPen(Qt.PenStyle.NoPen)

        self.mGradient.setColorAt(0, shadow.darker(130))
        self.mGradient.setColorAt(1, light.darker(130))
        self.mGradient.setStart(0, r.height())
        self.mGradient.setFinalStop(0, 0)
        painter.setBrush(self.mGradient)
        painter.drawRoundedRect(r, r.height() / 2, r.height() / 2)

        self.mGradient.setColorAt(0, shadow.darker(140))
        self.mGradient.setColorAt(1, light.darker(160))
        self.mGradient.setStart(0, 0)
        self.mGradient.setFinalStop(0, r.height())
        painter.setBrush(self.mGradient)
        painter.drawRoundedRect(
            r.adjusted(margin, margin, -margin, -margin), r.height() / 2, r.height() / 2
        )

        self.mGradient.setColorAt(0, button.darker(130))
        self.mGradient.setColorAt(1, button)

        painter.setBrush(self.mGradient)

        x = r.height() / 2.0 + self.mPosition * (r.width() - r.height())
        painter.drawEllipse(
            QPointF(x, r.height() / 2), r.height() / 2 - margin, r.height() / 2 - margin
        )

    @pyqtSlot(bool, name="animate")
    def animate(self, checked):
        self.animation.setDirection(
            QPropertyAnimation.Direction.Forward
            if checked
            else QPropertyAnimation.Direction.Backward
        )
        self.animation.start()

    def __del__(self):
        del self.animation


class Switch(QAbstractButton):
    def __init__(self, parent=None):
        QAbstractButton.__init__(self, parent=parent)
        self.dPtr = SwitchPrivate(self)
        self.setCheckable(True)
        self.clicked.connect(self.dPtr.animate)

    def sizeHint(self):
        return QSize(84, 42)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.dPtr.draw(painter)

    def resizeEvent(self, event):
        self.update()

    def __del__(self):
        del self.dPtr


class SwitchLang(QWidget):
    switchClicked: pyqtSignal = pyqtSignal()

    def __init__(  # ! lang_options can be inconsistent here
        self,
        lang_options: List[str] = ["EN", "JP"],
        checked: bool = False,
        size: QSize = QSize(50, 25),
    ) -> None:
        super().__init__()
        self.layout_widget: QHBoxLayout = QHBoxLayout(self)
        self.layout_widget.setContentsMargins(0, 0, 0, 0)
        self.layout_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.switch: Switch = Switch()
        self.switch.setFixedSize(size)
        if checked:
            self.switch.click()
        self.switch.clicked.connect(self.on_switch_clicked)

        self.lang_options: List[str] = lang_options
        self.label_lang_1: QLabel = QLabel(lang_options[0])  # unchecked state
        self.label_lang_2: QLabel = QLabel(lang_options[-1])  # checked state

        self.layout_widget.addWidget(self.label_lang_1)
        self.layout_widget.addWidget(self.switch)
        self.layout_widget.addWidget(self.label_lang_2)

        self.setFixedSize(self.layout_widget.sizeHint())

    def on_switch_clicked(self) -> None:
        self.switchClicked.emit()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = Switch()
    w.show()
    sys.exit(app.exec())
