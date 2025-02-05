import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout,
                               QSpinBox, QCheckBox, QLabel)
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtCore import Qt


# Виджет для отрисовки графиков
class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.x_min = -10
        self.x_max = 10
        self.step = 1
        self.functions = {
            "x^2": True,
            "sin(x) * exp(-0.1x^2)": True,
            "1/x": True
        }

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width, height = self.width(), self.height()
        center_x, center_y = width // 2, height // 2
        grid_step = min(width, height) / ((self.x_max - self.x_min) / self.step * 2)

        painter.fillRect(self.rect(), QBrush(Qt.white))
        pen = QPen(Qt.lightGray, 1, Qt.SolidLine)
        painter.setPen(pen)

        for i in range(self.x_min, self.x_max + 1, self.step):
            x = center_x + i * grid_step
            painter.drawLine(x, 0, x, height)
            painter.drawLine(0, center_y - i * grid_step, width, center_y - i * grid_step)

        pen.setColor(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(0, center_y, width, center_y)
        painter.drawLine(center_x, 0, center_x, height)

        pen.setWidth(1)
        painter.setPen(pen)
        for i in range(self.x_min, self.x_max + 1, self.step):
            x = center_x + i * grid_step
            y = center_y - i * grid_step
            painter.drawLine(x, center_y - 5, x, center_y + 5)
            painter.drawLine(center_x - 5, y, center_x + 5, y)
            if i != 0:
                painter.drawText(x - 10, center_y + 20, str(i))
                painter.drawText(center_x + 10, y + 5, str(-i))

        pen.setWidth(2)
        if self.functions["x^2"]:
            pen.setColor(Qt.blue)
            painter.setPen(pen)
            self.draw_function(painter, lambda x: x ** 2, center_x, center_y, grid_step)

        if self.functions["sin(x) * exp(-0.1x^2)"]:
            pen.setColor(Qt.green)
            painter.setPen(pen)
            self.draw_function(painter, lambda x: np.sin(x) * np.exp(-0.1 * x ** 2), center_x, center_y, grid_step)

        if self.functions["1/x"]:
            pen.setColor(Qt.red)
            painter.setPen(pen)
            self.draw_function(painter, lambda x: 1 / x if x != 0 else None, center_x, center_y, grid_step)

    def draw_function(self, painter, func, cx, cy, scale, step=0.05):
        prev_point = None
        x = self.x_min
        while x <= self.x_max:
            y = func(x)
            if y is not None and abs(y) < 10:
                px, py = cx + x * scale, cy - y * scale
                if prev_point:
                    painter.drawLine(prev_point[0], prev_point[1], px, py)
                prev_point = (px, py)
            else:
                prev_point = None
            x += step

    def update_settings(self, x_min, x_max, step, functions):
        self.x_min = x_min
        self.x_max = x_max
        self.step = step
        self.functions = functions
        self.update()


# Главное окно приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Графики функций")
        self.setGeometry(100, 100, 900, 700)

        layout = QVBoxLayout()
        control_layout = QHBoxLayout()

        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)

        self.x_min_spin = QSpinBox()
        self.x_min_spin.setRange(-100, 0)
        self.x_min_spin.setValue(-10)
        self.x_min_spin.valueChanged.connect(self.update_plot)

        self.x_max_spin = QSpinBox()
        self.x_max_spin.setRange(0, 100)
        self.x_max_spin.setValue(10)
        self.x_max_spin.valueChanged.connect(self.update_plot)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 10)
        self.step_spin.setValue(1)
        self.step_spin.valueChanged.connect(self.update_plot)

        self.check_x2 = QCheckBox("x^2")
        self.check_x2.setChecked(True)
        self.check_x2.stateChanged.connect(self.update_plot)

        self.check_sin_exp = QCheckBox("sin(x) * exp(-0.1x^2)")
        self.check_sin_exp.setChecked(True)
        self.check_sin_exp.stateChanged.connect(self.update_plot)

        self.check_1x = QCheckBox("1/x")
        self.check_1x.setChecked(True)
        self.check_1x.stateChanged.connect(self.update_plot)

        control_layout.addWidget(QLabel("X min:"))
        control_layout.addWidget(self.x_min_spin)
        control_layout.addWidget(QLabel("X max:"))
        control_layout.addWidget(self.x_max_spin)
        control_layout.addWidget(QLabel("Step:"))
        control_layout.addWidget(self.step_spin)
        control_layout.addWidget(self.check_x2)
        control_layout.addWidget(self.check_sin_exp)
        control_layout.addWidget(self.check_1x)

        layout.addLayout(control_layout)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def update_plot(self):
        functions = {
            "x^2": self.check_x2.isChecked(),
            "sin(x) * exp(-0.1x^2)": self.check_sin_exp.isChecked(),
            "1/x": self.check_1x.isChecked()
        }
        self.plot_widget.update_settings(self.x_min_spin.value(), self.x_max_spin.value(), self.step_spin.value(),
                                         functions)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
