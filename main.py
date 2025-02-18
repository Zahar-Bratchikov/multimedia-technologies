import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QLabel
)
from PySide6.QtGui import QPainter, QPen, QBrush, QFont, QPolygon
from PySide6.QtCore import Qt, QPoint
from functions import get_function_data

# Цвета для функций: функция 1 – синий, функция 2 – зеленый, функция 3 – красный.
FUNCTION_COLORS = {
    1: Qt.blue,
    2: Qt.green,
    3: Qt.red
}

class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        # Данные для графика: список словарей с ключами: x, y, label, color.
        self.data = []
        # Границы по оси Y рассчитываются автоматически.
        self.y_min = -10
        self.y_max = 10
        # Значения оси X задаются пользователем.
        self.user_x_start = -10
        self.user_x_end = 10

    def setData(self, data_list):
        """
        Принимает список словарей с данными для построения графиков,
        рассчитывает границы по оси Y (с отступом) и обновляет изображение.
        Границы оси X остаются заданными пользователем.
        """
        self.data = data_list
        if self.data:
            # Расчёт границ по оси Y по всем данным (исключая NaN).
            all_y = np.concatenate([d["y"][~np.isnan(d["y"])] for d in self.data])
            margin_y = (all_y.max() - all_y.min()) * 0.1
            self.y_min = all_y.min() - margin_y
            self.y_max = all_y.max() + margin_y
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()

        # Расширяем диапазон оси X, добавляя по одной клетке с каждой стороны.
        # Если user_x_start и user_x_end равны, это гарантирует, что диапазон не нулевой.
        grid_x_min = self.user_x_start - 1
        grid_x_max = self.user_x_end + 1
        x_range = grid_x_max - grid_x_min
        if x_range == 0:
            x_range = 1e-6  # предотвращаем деление на ноль

        # Функция для преобразования x в пиксели.
        def to_pixel_x(x):
            return int((x - grid_x_min) / x_range * width)

        # Для оси Y: если диапазон равен нулю, то аналогично.
        y_range = self.y_max - self.y_min
        if y_range == 0:
            y_range = 1e-6

        def to_pixel_y(y):
            return int(height - (y - self.y_min) / y_range * height)

        # Заливка фона.
        painter.fillRect(self.rect(), QBrush(Qt.white))

        # Отрисовка сетки.
        grid_pen = QPen(Qt.lightGray, 1, Qt.DashLine)
        painter.setPen(grid_pen)

        # Отрисовка вертикальных линий и подписей по оси X с шагом 1.
        # Используем расширенный диапазон: от (user_x_start - 1) до (user_x_end + 1)
        start_label = int(np.floor(grid_x_min))
        end_label = int(np.ceil(grid_x_max))
        for x_val in range(start_label, end_label + 1):
            x_pixel = to_pixel_x(x_val)
            painter.drawLine(x_pixel, 0, x_pixel, height)
            painter.setPen(Qt.black)
            painter.setFont(QFont("Arial", 8))
            # Центрируем подпись под линией.
            painter.drawText(x_pixel - 5, height - 5, str(x_val))
            painter.setPen(grid_pen)

        # Отрисовка горизонтальных линий и подписей по оси Y.
        grid_lines = 10
        y_step = (self.y_max - self.y_min) / grid_lines
        y_val = self.y_min
        while y_val <= self.y_max:
            y_pixel = to_pixel_y(y_val)
            painter.drawLine(0, y_pixel, width, y_pixel)
            painter.setPen(Qt.black)
            painter.setFont(QFont("Arial", 8))
            painter.drawText(width - 40, y_pixel + 5, f"{y_val:.1f}")
            painter.setPen(grid_pen)
            y_val += y_step

        # Отрисовка осей и выделение нулевой линии.
        axis_pen = QPen(Qt.black, 2)
        painter.setPen(axis_pen)
        if self.y_min <= 0 <= self.y_max:
            y_zero = to_pixel_y(0)
            zero_pen = QPen(Qt.darkGray, 3, Qt.SolidLine)
            painter.setPen(zero_pen)
            painter.drawLine(0, y_zero, width, y_zero)
        if grid_x_min <= 0 <= grid_x_max:
            x_zero = to_pixel_x(0)
            painter.setPen(axis_pen)
            painter.drawLine(x_zero, 0, x_zero, height)

        # Отрисовка графиков функций.
        for curve in self.data:
            pen = QPen(curve["color"], 2)
            painter.setPen(pen)
            points = []
            # Если разница по Y между точками превышает половину высоты виджета,
            # считаем, что произошёл разрыв и не соединяем точки линией.
            threshold = height / 2
            for x_val, y_val in zip(curve["x"], curve["y"]):
                if np.isnan(y_val):
                    if points:
                        painter.drawPolyline(QPolygon(points))
                        points = []
                else:
                    new_point = QPoint(to_pixel_x(x_val), to_pixel_y(y_val))
                    if points:
                        if abs(new_point.y() - points[-1].y()) > threshold:
                            painter.drawPolyline(QPolygon(points))
                            points = [new_point]
                        else:
                            points.append(new_point)
                    else:
                        points.append(new_point)
            if points:
                painter.drawPolyline(QPolygon(points))

        # Отрисовка легенды.
        self.draw_legend(painter)

    def draw_legend(self, painter):
        legend_x = 20
        legend_y = 20
        box_size = 15
        spacing = 8
        num_funcs = len(self.data)
        legend_height = num_funcs * (box_size + spacing) + spacing
        legend_width = 150

        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(legend_x - 5, legend_y - 5, legend_width, legend_height)

        current_y = legend_y
        painter.setFont(QFont("Arial", 10))
        for curve in self.data:
            painter.setBrush(QBrush(curve["color"]))
            painter.drawRect(legend_x, current_y, box_size, box_size)
            painter.setPen(Qt.black)
            painter.drawText(legend_x + box_size + 5, current_y + box_size - 3, curve["label"])
            current_y += box_size + spacing

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Построение диаграммы нескольких функций")
        self.resize(1000, 800)

        main_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        main_layout.addWidget(self.plot_widget)

        controls_layout = QHBoxLayout()
        # Интервал: Начало.
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1000, 1000)
        self.start_spin.setValue(-10)
        controls_layout.addWidget(QLabel("Начало:"))
        controls_layout.addWidget(self.start_spin)

        # Интервал: Конец.
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1000, 1000)
        self.end_spin.setValue(10)
        controls_layout.addWidget(QLabel("Конец:"))
        controls_layout.addWidget(self.end_spin)

        # Количество точек.
        self.points_spin = QSpinBox()
        self.points_spin.setRange(10, 10000)
        self.points_spin.setValue(200)
        controls_layout.addWidget(QLabel("Количество точек:"))
        controls_layout.addWidget(self.points_spin)

        # Выбор функций (номера функций 1-3).
        self.func1_checkbox = QCheckBox("Функция 1 (5*cos(x))")
        self.func1_checkbox.setChecked(True)
        controls_layout.addWidget(self.func1_checkbox)

        self.func2_checkbox = QCheckBox("Функция 2 (10*sin(x)+5*cos(2*x))")
        self.func2_checkbox.setChecked(True)
        controls_layout.addWidget(self.func2_checkbox)

        self.func3_checkbox = QCheckBox("Функция 3 (10/(x-1))")
        self.func3_checkbox.setChecked(True)
        controls_layout.addWidget(self.func3_checkbox)

        main_layout.addLayout(controls_layout)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Автоматическое обновление графика при изменении значений.
        self.start_spin.valueChanged.connect(self.update_plot)
        self.end_spin.valueChanged.connect(self.update_plot)
        self.points_spin.valueChanged.connect(self.update_plot)
        self.func1_checkbox.stateChanged.connect(self.update_plot)
        self.func2_checkbox.stateChanged.connect(self.update_plot)
        self.func3_checkbox.stateChanged.connect(self.update_plot)

        # Инициализация отображения графика.
        self.update_plot()

    def update_plot(self):
        start = self.start_spin.value()
        end = self.end_spin.value()
        num_points = self.points_spin.value()
        # Сохраняем введённые значения для оси X.
        self.plot_widget.user_x_start = start
        self.plot_widget.user_x_end = end
        data_list = []
        if self.func1_checkbox.isChecked():
            x, y, label = get_function_data(1, start, end, num_points)
            data_list.append({"x": x, "y": y, "label": label, "color": FUNCTION_COLORS[1]})
        if self.func2_checkbox.isChecked():
            x, y, label = get_function_data(2, start, end, num_points)
            data_list.append({"x": x, "y": y, "label": label, "color": FUNCTION_COLORS[2]})
        if self.func3_checkbox.isChecked():
            x, y, label = get_function_data(3, start, end, num_points)
            data_list.append({"x": x, "y": y, "label": label, "color": FUNCTION_COLORS[3]})
        self.plot_widget.setData(data_list)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())