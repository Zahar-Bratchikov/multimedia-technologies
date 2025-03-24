import sys
import math
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QSpinBox, QLabel, QComboBox, QListView
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QFont, QPainterPath, QColor,
    QStandardItemModel, QStandardItem, QFontMetrics
)
from lab1 import functions

# Определяем цвета для функций.
FUNCTION_COLORS = {
    1: QColor(Qt.blue),
    2: QColor(Qt.green),
    3: QColor(Qt.red),
    4: QColor(Qt.magenta)
}


def get_available_functions():
    """
    Собирает список доступных функций из модуля functions.
    Для каждой функции формируется подпись на основе документации.
    Возвращает список кортежей (подпись, идентификатор функции).
    """
    func_list = []
    for func_id in range(1, 4):
        func = getattr(functions, f"function_{func_id}", None)
        if callable(func):
            doc = func.__doc__.strip() if func.__doc__ else "Нет описания"
            func_list.append((f"Функция {func_id} ({doc})", func_id))
    return func_list


class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        # Данные для построения: список словарей с ключами "x", "y", "label", "color"
        self.data = []
        # Диапазон по оси Y (будет пересчитан на основе данных)
        self.y_min = -10
        self.y_max = 10
        # Интервал по оси X, задаваемый пользователем.
        self.user_x_start = -10
        self.user_x_end = 10
        # Количество точек (если данных нет, используется значение по умолчанию)
        self.default_points = 20

        # Конфигурационные параметры
        self.extra_margin = 1  # Дополнительный отступ для горизонтальной сетки (для оси Y)
        self.cone_scale = 0.7  # Масштаб по оси X для конусов
        self.shift_y_factor = 2.5  # Вертикальный сдвиг (используется с отрицательным знаком)
        self.font_family = "Arial"  # Название шрифта для подписей
        self.font_size = 8  # Размер шрифта для подписей осей
        self.legend_font_size = 10  # Размер шрифта для легенды
        self.small_value = 1e-6  # Малое значение для предотвращения деления на ноль
        # Фиксированный размер ячейки (если задан), иначе используется динамический расчёт
        self.fixed_base_width = None
        # Отношение ширины основания конуса к ширине ячейки (примерно 2/3)
        self.cone_base_ratio = 2 / 3

    def setData(self, data_list):
        """
        Устанавливает данные для построения и пересчитывает диапазон оси Y так,
        чтобы на графике присутствовал нулевой уровень.
        """
        self.data = data_list
        if self.data:
            # Объединяем все валидные значения Y.
            all_y = np.concatenate([d["y"][~np.isnan(d["y"])] for d in self.data])
            if len(all_y) > 0:
                margin_y = (all_y.max() - all_y.min()) * 0.1
                computed_y_min = all_y.min() - margin_y
                computed_y_max = all_y.max() + margin_y
                self.y_min = min(computed_y_min, 0)
                self.y_max = max(computed_y_max, 0)
        self.update()

    def paintEvent(self, event):
        """
        Отрисовывает диаграмму функций в виде конусов.
        1. Рисует фон и горизонтальную сетку с автоматически вычисляемым шагом по оси Y.
           Если выбрана одна точка, линии привязываются к вершинам конусов.
        2. Рисует вертикальную сетку, где все ячейки имеют одинаковую ширину.
           Для этого вычисляется домен по оси X с дополнительной ячейкой слева и справа.
        3. Отрисовывает конусы, где основание конуса имеет ширину ≈ 2/3 ширины ячейки.
        4. Рисует легенду.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Ось Y строится с дополнительным отступом
        grid_y_min = self.y_min - self.extra_margin
        grid_y_max = self.y_max + self.extra_margin
        grid_y_range = grid_y_max - grid_y_min if grid_y_max != grid_y_min else self.small_value

        def to_pixel_y(y):
            return int(h - (y - grid_y_min) / grid_y_range * h)

        # Функция для вычисления "приятного" шага для оси Y.
        def niceNum(x, round_val):
            if x == 0:
                return 0
            exp = math.floor(math.log10(x))
            f = x / (10 ** exp)
            if round_val:
                if f < 1.5:
                    nf = 1
                elif f < 3:
                    nf = 2
                elif f < 7:
                    nf = 5
                else:
                    nf = 10
            else:
                if f <= 1:
                    nf = 1
                elif f <= 2:
                    nf = 2
                elif f <= 5:
                    nf = 5
                else:
                    nf = 10
            return nf * (10 ** exp)

        # Рисуем фон.
        painter.fillRect(self.rect(), QBrush(Qt.white))

        # Отрисовка горизонтальной сетки.
        if self.data and len(self.data[0]["x"]) == 1:
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
            for func_index, curve in enumerate(self.data):
                base_y = to_pixel_y(0)
                shift_y = to_pixel_y(curve["y"][0]) - base_y
                y_pos = to_pixel_y(curve["y"][0]) + int(func_index * shift_y)
                painter.drawLine(0, y_pos, w, y_pos)
                painter.setPen(QPen(Qt.black, 1))
                painter.drawText(w - 40, y_pos + 5, f"{curve['y'][0]:.2f}")
                painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        else:
            y_range = grid_y_max - grid_y_min
            target_lines = 5
            step_y = niceNum(y_range / target_lines, True)
            tick_start = math.floor(grid_y_min / step_y) * step_y
            tick_end = math.ceil(grid_y_max / step_y) * step_y
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
            tick = tick_start
            while tick <= tick_end:
                y_pix = to_pixel_y(tick)
                painter.drawLine(0, y_pix, w, y_pix)
                painter.setPen(QPen(Qt.black, 1))
                painter.setFont(QFont(self.font_family, self.font_size))
                painter.drawText(w - 40, y_pix + 5, f"{tick:.2f}")
                painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
                tick += step_y

        # Рисуем горизонтальную ось y = 0.
        zero_y = to_pixel_y(0)
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(0, zero_y, w, zero_y)

        # Вертикальная сетка.
        if self.default_points > 1:
            step_x = (self.user_x_end - self.user_x_start) / (self.default_points - 1)
        else:
            step_x = (self.user_x_end - self.user_x_start) if (self.user_x_end != self.user_x_start) else 1

        # Определяем домен для вертикальной сетки с дополнительной ячейкой слева и справа.
        grid_x_min = self.user_x_start - step_x
        grid_x_max = self.user_x_end + step_x
        grid_x_range = grid_x_max - grid_x_min if grid_x_max != grid_x_min else self.small_value

        def to_pixel_x(x):
            return int((x - grid_x_min) / grid_x_range * w)

        # Отрисовка вертикальной сетки от grid_x_min до grid_x_max с шагом step_x.
        painter.setPen(QPen(Qt.darkGray, 1, Qt.DashDotLine))
        x_val = grid_x_min
        while x_val <= grid_x_max + step_x / 2:
            x_pix = to_pixel_x(x_val)
            painter.drawLine(x_pix, 0, x_pix, h)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(x_pix - 20, h - 5, f"{x_val:.2f}")
            painter.setPen(QPen(Qt.darkGray, 1, Qt.DashDotLine))
            x_val += step_x

        # Рисуем вертикальную ось x = 0 (если она входит в область)
        x_zero = to_pixel_x(0)
        x_zero = max(0, min(w, x_zero))
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(x_zero, 0, x_zero, h)
        painter.setPen(QPen(Qt.black, 1))
        painter.drawText(x_zero - 20, h - 5, "0.00")

        if not self.data:
            return

        # Отрисовка конусов.
        num_points = len(self.data[0]["x"])
        num_funcs = len(self.data)
        # Используем основное множество точек (домен: [user_x_start, user_x_end])
        # Вычисляем ширину области построения по оси X (без дополнительных ячеек)
        plot_x_range = self.user_x_end - self.user_x_start
        # Вычисляем group_width как долю области построения, масштабированную на конусную область
        left_bound_plot = to_pixel_x(self.user_x_start + step_x)
        right_bound_plot = to_pixel_x(self.user_x_end - step_x)
        group_width = (right_bound_plot - left_bound_plot) * self.cone_scale / (num_points if num_points else 1)

        # Если точек всего 3, устанавливаем фиксированное значение
        # чтобы конусы не сливались в линии.
        if num_points == 3:
            self.fixed_base_width = 40
        else:
            self.fixed_base_width = None

        if self.fixed_base_width is not None:
            cell_width = self.fixed_base_width
        else:
            cell_width = group_width / num_funcs if num_funcs else group_width

        cone_base_width = cell_width * self.cone_base_ratio
        ellipse_height = int(cone_base_width * 0.6)
        shift_y = -int(ellipse_height * self.shift_y_factor)

        for i in range(num_points):
            center = self.data[0]["x"][i]
            center_pix = to_pixel_x(center)
            base_centers = []
            for func_index, curve in enumerate(self.data):
                y_val = curve["y"][i]
                if np.isnan(y_val):
                    continue
                x_offset = int((func_index - (num_funcs - 1) / 2) * cell_width)
                apex_x = center_pix + x_offset
                base_x = apex_x - int(cone_base_width / 2)
                apex_y = to_pixel_y(curve["y"][i]) + func_index * shift_y
                base_y = to_pixel_y(0) + func_index * shift_y
                base_centers.append((base_x + int(cone_base_width / 2), base_y))
                self.draw_cone(painter, apex_x, apex_y, base_x, base_y, int(cone_base_width), ellipse_height,
                               curve["color"])
            if len(base_centers) > 1:
                painter.setPen(QPen(Qt.black, 1))
                for j in range(len(base_centers) - 1):
                    p1 = base_centers[j]
                    p2 = base_centers[j + 1]
                    painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(0, zero_y, w, zero_y)
        self.draw_legend(painter)

    def draw_cone(self, painter, apex_x, apex_y, base_x, base_y, width_val, height_val, color):
        """
        Отрисовывает конус с треугольной боковой гранью и эллиптическим основанием.
        """
        path = QPainterPath()
        path.moveTo(apex_x, apex_y)
        path.lineTo(base_x, base_y)
        path.lineTo(base_x + width_val, base_y)
        path.closeSubpath()
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawPath(path)
        base_width = width_val
        base_height = int(height_val * 0.5)
        ellipse_rect = (int(base_x), int(base_y - base_height / 2), int(base_width), int(base_height))
        painter.setBrush(QBrush(color.darker(115)))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawEllipse(*ellipse_rect)
        highlight_color = color.lighter(160)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(highlight_color))
        if apex_x < base_x:
            start_angle = 180 * 16
            span_angle = -180 * 16
        else:
            start_angle = 0
            span_angle = 180 * 16
        painter.drawArc(*ellipse_rect, start_angle, span_angle)

    def draw_legend(self, painter):
        """
        Рисует легенду с подписями функций. Ширина легенды рассчитывается на основе максимальной ширины подписи.
        """
        font = QFont(self.font_family, self.legend_font_size)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        padding = 10
        marker_size = 15
        spacing = 8
        max_label_width = 0
        for curve in self.data:
            text = curve["label"]
            width_text = metrics.horizontalAdvance(text)
            if width_text > max_label_width:
                max_label_width = width_text
        legend_width = padding * 2 + marker_size + spacing + max_label_width
        legend_height = padding * 2 + len(self.data) * (marker_size + spacing) - spacing
        legend_x = 20
        legend_y = 20
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(legend_x, legend_y, legend_width, legend_height)
        current_y = legend_y + padding
        for curve in self.data:
            painter.setBrush(QBrush(curve["color"]))
            painter.drawRect(legend_x + padding, current_y, marker_size, marker_size)
            painter.setPen(Qt.black)
            painter.drawText(legend_x + padding + marker_size + spacing,
                             current_y + marker_size - 3, curve["label"])
            current_y += marker_size + spacing


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №1: Диаграмма функций (PySide)")
        self.resize(1000, 800)
        main_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        main_layout.addWidget(self.plot_widget)
        controls_layout = QHBoxLayout()
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1000, 1000)
        self.start_spin.setValue(-10)
        controls_layout.addWidget(QLabel("Начало интервала:"))
        controls_layout.addWidget(self.start_spin)
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1000, 1000)
        self.end_spin.setValue(10)
        controls_layout.addWidget(QLabel("Конец интервала:"))
        controls_layout.addWidget(self.end_spin)
        self.points_spin = QSpinBox()
        self.points_spin.setRange(1, 10000)
        self.points_spin.setValue(20)
        controls_layout.addWidget(QLabel("Количество точек:"))
        controls_layout.addWidget(self.points_spin)
        self.func_combo = QComboBox()
        self.func_combo.setView(QListView())
        self.func_combo.setEditable(True)
        self.func_combo.lineEdit().setReadOnly(True)
        self.func_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self.func_combo.setInsertPolicy(QComboBox.NoInsert)
        self.func_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.func_combo.setMaxVisibleItems(10)
        self.func_combo.setMinimumContentsLength(20)
        self.func_model = QStandardItemModel(self.func_combo)
        for label, func_id in get_available_functions():
            item = QStandardItem(label)
            item.setData(func_id)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setData(Qt.Unchecked, Qt.CheckStateRole)
            self.func_model.appendRow(item)
        self.func_combo.setModel(self.func_model)
        controls_layout.addWidget(QLabel("Функции (выберите):"))
        controls_layout.addWidget(self.func_combo)
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
        self.func_model.itemChanged.connect(self.schedule_update)
        self.update_plot()

    def schedule_update(self):
        self.update_timer.start(200)

    def update_plot(self):
        start = self.start_spin.value()
        end = self.end_spin.value()
        num_points = self.points_spin.value()
        if num_points < 2:
            num_points = 1  # Если выбрана одна точка, размещаем её в центре
        self.plot_widget.user_x_start = start
        self.plot_widget.user_x_end = end
        self.plot_widget.default_points = num_points
        data_list = []
        for i in range(self.func_model.rowCount()):
            item = self.func_model.item(i)
            if item.checkState() == Qt.Checked:
                func_id = item.data()
                x, y, label = functions.get_function_data(func_id, start, end, num_points)
                data_list.append({
                    "x": x,
                    "y": y,
                    "label": label,
                    "color": FUNCTION_COLORS.get(func_id, QColor(Qt.black))
                })
        self.plot_widget.setData(data_list)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())