import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout,
                               QDoubleSpinBox, QCheckBox, QLabel)
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtCore import Qt


class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.x_min = -10.0
        self.x_max = 10.0
        self.step = 1.0
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

        # Рисуем пунктирные линии сетки
        pen = QPen(Qt.lightGray, 1, Qt.DashLine)
        painter.setPen(pen)

        i = self.x_min
        while i <= self.x_max:
            x = center_x + i * grid_step
            painter.drawLine(x, 0, x, height)
            painter.drawLine(0, center_y - i * grid_step, width, center_y - i * grid_step)
            i += self.step

        # Рисуем оси жирными сплошными линиями
        pen.setColor(Qt.black)
        pen.setWidth(2)
        pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)
        painter.drawLine(0, center_y, width, center_y)
        painter.drawLine(center_x, 0, center_x, height)

        # Подписи осей
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(width - 30, center_y - 5, "X")
        painter.drawText(center_x + 10, 15, "Y")

        # Подписи меток на осях с отступами
        font.setPointSize(10)
        painter.setFont(font)
        i = self.x_min
        while i <= self.x_max:
            x = center_x + i * grid_step
            y = center_y - i * grid_step
            if i != 0:
                painter.drawText(x - 15, center_y + 20, f"{i:.2f}")  # Подписи оси X
                painter.drawText(center_x - 40, y + 5, f"{i:.2f}")  # Подписи оси Y слева
            i += self.step

        # Отрисовка функций
        pen.setWidth(2)
        for func_name, enabled in self.functions.items():
            if enabled:
                if func_name == "x^2":
                    pen.setColor(Qt.blue)
                    painter.setPen(pen)
                    self.draw_function(painter, lambda x: x ** 2, center_x, center_y, grid_step)
                elif func_name == "sin(x) * exp(-0.1x^2)":
                    pen.setColor(Qt.green)
                    painter.setPen(pen)
                    self.draw_function(painter, lambda x: np.sin(x) * np.exp(-0.1 * x ** 2), center_x, center_y,
                                       grid_step)
                elif func_name == "1/x":
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Графики функций")
        self.setGeometry(100, 100, 900, 700)

        layout = QVBoxLayout()
        control_layout = QHBoxLayout()

        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)

        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(-1000.0, 1000.0)
        self.x_min_spin.setSingleStep(0.1)
        self.x_min_spin.setValue(-10.0)
        self.x_min_spin.valueChanged.connect(self.update_plot)

        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(-1000.0, 1000.0)
        self.x_max_spin.setSingleStep(0.1)
        self.x_max_spin.setValue(10.0)
        self.x_max_spin.valueChanged.connect(self.update_plot)

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 10.0)
        self.step_spin.setSingleStep(0.1)
        self.step_spin.setValue(1.0)
        self.step_spin.valueChanged.connect(self.update_plot)

        self.function_checkboxes = {}
        for func_name in self.plot_widget.functions:
            checkbox = QCheckBox(func_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_plot)
            control_layout.addWidget(checkbox)
            self.function_checkboxes[func_name] = checkbox

        control_layout.addWidget(QLabel("X min:"))
        control_layout.addWidget(self.x_min_spin)
        control_layout.addWidget(QLabel("X max:"))
        control_layout.addWidget(self.x_max_spin)
        control_layout.addWidget(QLabel("Step:"))
        control_layout.addWidget(self.step_spin)

        layout.addLayout(control_layout)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def update_plot(self):
        functions = {name: checkbox.isChecked() for name, checkbox in self.function_checkboxes.items()}
        self.plot_widget.update_settings(self.x_min_spin.value(), self.x_max_spin.value(), self.step_spin.value(), functions)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
