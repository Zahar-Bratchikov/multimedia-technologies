import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout,
                               QDoubleSpinBox, QCheckBox, QLabel)
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtCore import Qt


def func_square(x):
    return x ** 2


def func_sin_exp(x):
    return np.sin(x) * np.exp(-0.1 * x ** 2)


def func_inverse(x):
    if x == 0:
        return None
    return 1 / x


class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        # Границы области построения функций (для графика)
        self.x_min = -10.0
        self.x_max = 10.0
        self.y_min = -10.0
        self.y_max = 10.0
        self.step = 1.0
        # Функции для графика (ключ: имя функции, значение: включена ли отрисовка)
        self.functions = {
            "x^2": True,
            "sin(x) * exp(-0.1*x^2)": True,
            "1/x": True
        }
        self.colors = {
            "x^2": Qt.blue,
            "sin(x) * exp(-0.1*x^2)": Qt.green,
            "1/x": Qt.red
        }

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()

        # Расширенные границы для сетки и подписей (чтобы был шаг на 1 единицу с каждой стороны)
        x_min_grid = self.x_min - 1
        x_max_grid = self.x_max + 1
        y_min_grid = self.y_min - 1
        y_max_grid = self.y_max + 1

        # Масштабы для преобразования координат в пиксели (для всей расширенной области)
        scale_x = width / (x_max_grid - x_min_grid)
        scale_y = height / (y_max_grid - y_min_grid)

        # Функции для преобразования математических координат в пиксели
        def to_pixel_x(x):
            return int((x - x_min_grid) * scale_x)

        def to_pixel_y(y):
            # В графических координатах ось y направлена вниз, поэтому инвертируем
            return int(height - (y - y_min_grid) * scale_y)

        painter.fillRect(self.rect(), QBrush(Qt.white))

        # Отрисовка пунктирной сетки
        grid_pen = QPen(Qt.lightGray, 1, Qt.DashLine)
        painter.setPen(grid_pen)

        # Вертикальные линии сетки
        x = x_min_grid
        while x <= x_max_grid + 1e-9:
            painter.drawLine(to_pixel_x(x), 0, to_pixel_x(x), height)
            x += self.step

        # Горизонтальные линии сетки
        y = y_min_grid
        while y <= y_max_grid + 1e-9:
            painter.drawLine(0, to_pixel_y(y), width, to_pixel_y(y))
            y += self.step

        # Отрисовка осей координат (ось X: y=0, ось Y: x=0)
        axis_pen = QPen(Qt.black, 2, Qt.SolidLine)
        painter.setPen(axis_pen)
        painter.drawLine(0, to_pixel_y(0), width, to_pixel_y(0))
        painter.drawLine(to_pixel_x(0), 0, to_pixel_x(0), height)

        # Подписи осей
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(width - 30, to_pixel_y(0) - 5, "X")
        painter.drawText(to_pixel_x(0) + 10, 15, "Y")

        # Подписи меток по осям
        tick_font = QFont()
        tick_font.setPointSize(10)
        painter.setFont(tick_font)

        # Тики по оси X
        x = x_min_grid
        while x <= x_max_grid + 1e-9:
            if abs(x) > 1e-6:
                painter.drawText(to_pixel_x(x) - 15, to_pixel_y(0) + 20, f"{x:.2f}")
            x += self.step

        # Тики по оси Y
        y = y_min_grid
        while y <= y_max_grid + 1e-9:
            if abs(y) > 1e-6:
                painter.drawText(to_pixel_x(0) - 40, to_pixel_y(y) + 5, f"{y:.2f}")
            y += self.step

        # Отрисовка функций на диапазоне [self.x_min, self.x_max]
        func_pen = QPen()
        func_pen.setWidth(2)
        painter.setPen(func_pen)
        for func_name, enabled in self.functions.items():
            if enabled:
                func_pen.setColor(self.colors[func_name])
                painter.setPen(func_pen)
                try:
                    if func_name == "x^2":
                        self.draw_function(painter, func_square, to_pixel_x, to_pixel_y, self.x_min, self.x_max)
                    elif func_name == "sin(x) * exp(-0.1*x^2)":
                        self.draw_function(painter, func_sin_exp, to_pixel_x, to_pixel_y, self.x_min, self.x_max)
                    elif func_name == "1/x":
                        self.draw_function(painter, func_inverse, to_pixel_x, to_pixel_y, self.x_min, self.x_max,
                                           exclude_zero=True)
                except Exception as e:
                    print(f"Error drawing function {func_name}: {e}")

        # Отрисовка легенды
        self.draw_legend(painter, 10, 10)

    def draw_function(self, painter, func, to_pixel_x, to_pixel_y, x_start, x_end, step=0.05, exclude_zero=False):
        """
        Рисует функцию, соединяя отрезками последовательные точки.
        Если отрезок выходит за пределы по y, он клиппируется так, чтобы оставаться в диапазоне [y_min, y_max].
        """

        def clip_segment(p0, p1, y_min, y_max):
            (x0, y0), (x1, y1) = p0, p1
            # Если оба конца отрезка находятся вне диапазона по одной стороне, не отрисовываем
            if (y0 < y_min and y1 < y_min) or (y0 > y_max and y1 > y_max):
                return None
            new_p0, new_p1 = p0, p1
            if y0 < y_min:
                t = (y_min - y0) / (y1 - y0)
                new_p0 = (x0 + t * (x1 - x0), y_min)
            elif y0 > y_max:
                t = (y_max - y0) / (y1 - y0)
                new_p0 = (x0 + t * (x1 - x0), y_max)
            if y1 < y_min:
                t = (y_min - y0) / (y1 - y0)
                new_p1 = (x0 + t * (x1 - x0), y_min)
            elif y1 > y_max:
                t = (y_max - y0) / (y1 - y0)
                new_p1 = (x0 + t * (x1 - x0), y_max)
            return new_p0, new_p1

        prev_math = None  # предыдущая точка в математической системе координат
        x = x_start
        while x <= x_end + 1e-9:
            if exclude_zero and abs(x) < step:
                prev_math = None
                x += step
                continue
            try:
                y = func(x)
            except ZeroDivisionError:
                prev_math = None
                x += step
                continue
            if y is None:
                prev_math = None
                x += step
                continue
            curr_math = (x, y)
            if prev_math is not None:
                segment = clip_segment(prev_math, curr_math, self.y_min, self.y_max)
                if segment is not None:
                    p0 = (to_pixel_x(segment[0][0]), to_pixel_y(segment[0][1]))
                    p1 = (to_pixel_x(segment[1][0]), to_pixel_y(segment[1][1]))
                    painter.drawLine(p0[0], p0[1], p1[0], p1[1])
            prev_math = curr_math
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

        # Контролы для изменения настроек
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
        self.plot_widget.update_settings(
            self.x_min_spin.value(),
            self.x_max_spin.value(),
            self.y_min_spin.value(),
            self.y_max_spin.value(),
            self.step_spin.value(),
            functions
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
