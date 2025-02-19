import sys
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QLabel
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QFont, QPainterPath, QColor
)
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
        Принимает список словарей с данными для построения гистограммы,
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

        # Расширяем диапазон по оси X с небольшим запасом.
        grid_x_min = self.user_x_start - 1
        grid_x_max = self.user_x_end + 1
        x_range = grid_x_max - grid_x_min
        if x_range == 0:
            x_range = 1e-6

        # Функции преобразования координат.
        def to_pixel_x(x):
            return int((x - grid_x_min) / x_range * width)
        y_range = self.y_max - self.y_min
        if y_range == 0:
            y_range = 1e-6
        def to_pixel_y(y):
            return int(height - (y - self.y_min) / y_range * height)

        # Определяем положение оси X (нулевая линия).
        zero_y = to_pixel_y(0)

        # Заливка фона.
        painter.fillRect(self.rect(), QBrush(Qt.white))

        # Отрисовка сетки.
        grid_pen = QPen(Qt.lightGray, 1, Qt.DashLine)
        painter.setPen(grid_pen)
        start_label = int(np.floor(grid_x_min))
        end_label = int(np.ceil(grid_x_max))
        for x_val in range(start_label, end_label + 1):
            x_pixel = to_pixel_x(x_val)
            painter.drawLine(x_pixel, 0, x_pixel, height)
            painter.setPen(Qt.black)
            painter.setFont(QFont("Arial", 8))
            painter.drawText(x_pixel - 5, height - 5, str(x_val))
            painter.setPen(grid_pen)
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

        # Отрисовка осей.
        axis_pen = QPen(Qt.black, 2)
        painter.setPen(axis_pen)
        if self.y_min <= 0 <= self.y_max:
            painter.drawLine(0, zero_y, width, zero_y)
        if grid_x_min <= 0 <= grid_x_max:
            x_zero = to_pixel_x(0)
            painter.drawLine(x_zero, 0, x_zero, height)

        # Если данных нет, завершаем отрисовку.
        if not self.data:
            return

        # Все функции используют один и тот же массив x, предполагая, что они синхронизированы.
        num_points = len(self.data[0]["x"])
        num_funcs = len(self.data)
        # Общая ширина группы для одного x-сегмента.
        group_width = width / (num_points * 1.5)
        # Ширина подконуса для каждой функции.
        sub_width = group_width / num_funcs
        # Высота эллиптического основания для эффекта 3D.
        ellipse_height = sub_width * 0.6

        # Рисуем конусы для каждой точки и каждой функции
        for i in range(num_points):
            # Берем общее значение x из первого набора (предполагается, что все x одинаковые).
            x_val = self.data[0]["x"][i]
            x_base = to_pixel_x(x_val)
            for func_index, curve in enumerate(self.data):
                y_val = curve["y"][i]
                if np.isnan(y_val):
                    continue
                # Вычисляем смещение по горизонтали для каждого конуса в группе,
                # чтобы они не пересекались: распределяем равномерно внутри group_width.
                offset = (func_index - (num_funcs - 1) / 2) * sub_width
                x_pixel = x_base + int(offset)
                # Апекс конуса расположен по центру подконуса.
                apex_x = x_pixel + sub_width / 2
                apex_y = to_pixel_y(y_val)

                # Боковая поверхность конуса – треугольник: апекс, левая и правая точки базы.
                base_left = (x_pixel, zero_y)
                base_right = (x_pixel + sub_width, zero_y)
                lateral_path = QPainterPath()
                lateral_path.moveTo(apex_x, apex_y)
                lateral_path.lineTo(*base_left)
                lateral_path.lineTo(*base_right)
                lateral_path.lineTo(apex_x, apex_y)
                lateral_path.closeSubpath()

                # Рисуем боковую поверхность.
                painter.setBrush(QBrush(curve["color"]))
                painter.setPen(QPen(QColor(curve["color"]).darker(150), 2))
                painter.drawPath(lateral_path)

                # Отрисовка основания конуса.
                # Здесь основание представлено эллипсом, который лежит точно на оси X.
                ellipse_rect = (x_pixel, zero_y - ellipse_height/2, sub_width, ellipse_height)
                painter.setBrush(QBrush(QColor(curve["color"]).darker(115)))
                painter.setPen(QPen(QColor(curve["color"]).darker(150), 2))
                painter.drawEllipse(*ellipse_rect)

                # Дополнительное выделение: полутень на основании.
                highlight_color = QColor(curve["color"]).lighter(160)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(highlight_color))
                # Рисуем верхнюю половину эллипса для положительных значений
                # и нижнюю половину для отрицательных.
                if y_val >= 0:
                    start_angle = 180 * 16
                    span_angle = -180 * 16
                else:
                    start_angle = 0
                    span_angle = 180 * 16
                painter.drawArc(*ellipse_rect, start_angle, span_angle)

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
        self.setWindowTitle("Гистограмма с конусами (не пересекаются, основание на оси X)")
        self.resize(1000, 800)

        main_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        main_layout.addWidget(self.plot_widget)

        controls_layout = QHBoxLayout()
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1000, 1000)
        self.start_spin.setValue(-10)
        controls_layout.addWidget(QLabel("Начало:"))
        controls_layout.addWidget(self.start_spin)

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1000, 1000)
        self.end_spin.setValue(10)
        controls_layout.addWidget(QLabel("Конец:"))
        controls_layout.addWidget(self.end_spin)

        self.points_spin = QSpinBox()
        self.points_spin.setRange(10, 10000)
        self.points_spin.setValue(200)
        controls_layout.addWidget(QLabel("Количество точек:"))
        controls_layout.addWidget(self.points_spin)

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

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_plot)

        self.start_spin.valueChanged.connect(self.schedule_update)
        self.end_spin.valueChanged.connect(self.schedule_update)
        self.points_spin.valueChanged.connect(self.schedule_update)
        self.func1_checkbox.stateChanged.connect(self.schedule_update)
        self.func2_checkbox.stateChanged.connect(self.schedule_update)
        self.func3_checkbox.stateChanged.connect(self.schedule_update)

        self.update_plot()

    def schedule_update(self):
        self.update_timer.start(200)

    def update_plot(self):
        start = self.start_spin.value()
        end = self.end_spin.value()
        num_points = self.points_spin.value()
        self.plot_widget.user_x_start = start
        self.plot_widget.user_x_end = end
        data_list = []
        if self.func1_checkbox.isChecked():
            x, y, label = get_function_data(1, start, end, num_points)
            data_list.append({
                "x": x,
                "y": y,
                "label": label,
                "color": FUNCTION_COLORS[1]
            })
        if self.func2_checkbox.isChecked():
            x, y, label = get_function_data(2, start, end, num_points)
            data_list.append({
                "x": x,
                "y": y,
                "label": label,
                "color": FUNCTION_COLORS[2]
            })
        if self.func3_checkbox.isChecked():
            x, y, label = get_function_data(3, start, end, num_points)
            data_list.append({
                "x": x,
                "y": y,
                "label": label,
                "color": FUNCTION_COLORS[3]
            })
        self.plot_widget.setData(data_list)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
