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
        self.y_min = -10.0
        self.y_max = 10.0
        self.step = 1.0
        self.functions = {
            "x^2": True,
            "sin(x) * exp(-0.1x^2)": True,
            "1/x": True
        }
        self.colors = {
            "x^2": Qt.blue,
            "sin(x) * exp(-0.1x^2)": Qt.green,
            "1/x": Qt.red
        }

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width, height = self.width(), self.height()
        center_x = width / 2
        center_y = height / 2

        # Определяем шаг сетки
        scale_x = width / (self.x_max - self.x_min)
        scale_y = height / (self.y_max - self.y_min)
        grid_step_x = scale_x * self.step
        grid_step_y = scale_y * self.step

        painter.fillRect(self.rect(), QBrush(Qt.white))

        # Рисуем пунктирные линии сетки
        pen = QPen(Qt.lightGray, 1, Qt.DashLine)
        painter.setPen(pen)

        # Вертикальные линии сетки
        i = self.x_min
        while i <= self.x_max:
            x = center_x + i * scale_x
            painter.drawLine(x, 0, x, height)
            i += self.step

        # Горизонтальные линии сетки
        i = self.y_min
        while i <= self.y_max:
            y = center_y - i * scale_y
            painter.drawLine(0, y, width, y)
            i += self.step

        # Рисуем оси
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

        # Подписи меток на осях
        font.setPointSize(10)
        painter.setFont(font)
        i = self.x_min
        while i <= self.x_max:
            x = center_x + i * scale_x
            if i != 0:
                painter.drawText(x - 15, center_y + 20, f"{i:.2f}")
            i += self.step

        i = self.y_min
        while i <= self.y_max:
            y = center_y - i * scale_y
            if i != 0:
                painter.drawText(center_x - 40, y + 5, f"{i:.2f}")
            i += self.step

        # Отрисовка функций
        pen.setWidth(2)
        for func_name, enabled in self.functions.items():
            if enabled:
                pen.setColor(self.colors[func_name])
                painter.setPen(pen)
                try:
                    if func_name == "x^2":
                        self.draw_function(painter, lambda x: x ** 2, center_x, center_y, scale_x, scale_y)
                    elif func_name == "sin(x) * exp(-0.1x^2)":
                        self.draw_function(painter, lambda x: np.sin(x) * np.exp(-0.1 * x ** 2), center_x, center_y, scale_x, scale_y)
                    elif func_name == "1/x":
                        self.draw_function(painter, lambda x: 1 / x if x != 0 else None, center_x, center_y, scale_x, scale_y, exclude_zero=True)
                except Exception as e:
                    print(f"Error drawing function {func_name}: {e}")

        # Отрисовка легенды
        self.draw_legend(painter, 10, 10)

    def draw_function(self, painter, func, cx, cy, scale_x, scale_y, step=0.05, exclude_zero=False):
        prev_point = None
        x = self.x_min
        while x <= self.x_max:
            if exclude_zero and -step < x < step:
                prev_point = None
                x += step
                continue
            try:
                y = func(x)
                if y is not None and self.y_min <= y <= self.y_max:
                    px, py = cx + x * scale_x, cy - y * scale_y
                    if prev_point:
                        painter.drawLine(prev_point[0], prev_point[1], px, py)
                    prev_point = (px, py)
                else:
                    prev_point = None
            except ZeroDivisionError:
                prev_point = None
            x += step

    def draw_legend(self, painter, x, y):
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)

        box_size = 15
        spacing = 5
        for func_name, color in self.colors.items():
            if self.functions[func_name]:
                painter.setBrush(QBrush(color))
                painter.drawRect(x, y, box_size, box_size)
                painter.drawText(x + box_size + spacing, y + box_size - 2, func_name)
                y += box_size + spacing

    def update_settings(self, x_min, x_max, y_min, y_max, step, functions):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
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
        self.x_min_spin.setValue(-10.0)
        self.x_min_spin.valueChanged.connect(self.update_plot)

        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(-1000.0, 1000.0)
        self.x_max_spin.setValue(10.0)
        self.x_max_spin.valueChanged.connect(self.update_plot)

        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-1000.0, 1000.0)
        self.y_min_spin.setValue(-10.0)
        self.y_min_spin.valueChanged.connect(self.update_plot)

        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-1000.0, 1000.0)
        self.y_max_spin.setValue(10.0)
        self.y_max_spin.valueChanged.connect(self.update_plot)

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 10.0)
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
        control_layout.addWidget(QLabel("Y min:"))
        control_layout.addWidget(self.y_min_spin)
        control_layout.addWidget(QLabel("Y max:"))
        control_layout.addWidget(self.y_max_spin)
        control_layout.addWidget(QLabel("Step:"))
        control_layout.addWidget(self.step_spin)

        layout.addLayout(control_layout)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def update_plot(self):
        functions = {name: checkbox.isChecked() for name, checkbox in self.function_checkboxes.items()}
        self.plot_widget.update_settings(self.x_min_spin.value(), self.x_max_spin.value(), self.y_min_spin.value(), self.y_max_spin.value(), self.step_spin.value(), functions)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())