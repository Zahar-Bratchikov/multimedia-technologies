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
import functions

# Константы для настройки интерфейса
WINDOW_TITLE = "Лабораторная работа №1: Диаграмма функций (PySide)"
WINDOW_SIZE = (1000, 800)
PLOT_MIN_SIZE = (800, 600)
MARGIN_LEFT = 60
MARGIN_RIGHT = 40
MARGIN_BOTTOM = 40
MARGIN_TOP = 20
LEGEND_PADDING = 10
LEGEND_MARKER_SIZE = 15
LEGEND_SPACING = 8
LEGEND_POSITION = (20, 20)

# Константы для настройки графика
DEFAULT_X_RANGE = (-10, 10)
DEFAULT_POINTS = 20
DEFAULT_SPIN_RANGE = (-1000, 1000)
DEFAULT_POINTS_RANGE = (1, 10000)
SMALL_VALUE = 1e-6

# Константы для настройки конусов
CONE_SCALE = 0.7
CONE_BASE_RATIO = 2 / 3
SHIFT_Y_FACTOR = 2.5
FIXED_BASE_WIDTH = 40  # Для случая с 3 точками

# Определяем цвета для функций
FUNCTION_COLORS = {
    1: QColor(Qt.blue),
    2: QColor(Qt.green),
    3: QColor(Qt.red),
    4: QColor(Qt.magenta)
}

# Настройки шрифтов
FONT_FAMILY = "Arial"
FONT_SIZE = 8
LEGEND_FONT_SIZE = 10

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
        self.setMinimumSize(*PLOT_MIN_SIZE)
        
        # Инициализация данных
        self.data = []
        self.y_min = -10
        self.y_max = 10
        self.user_x_start = DEFAULT_X_RANGE[0]
        self.user_x_end = DEFAULT_X_RANGE[1]
        self.default_points = DEFAULT_POINTS
        self.extra_margin = 1
        self.fixed_base_width = None

    def setData(self, data_list):
        """
        Устанавливает данные для построения и пересчитывает диапазон оси Y так,
        чтобы на графике присутствовал нулевой уровень.
        """
        self.data = data_list
        if self.data:
            # Объединяем все валидные значения Y
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
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Вычисляем рабочую область для графика
        plot_width = w - MARGIN_LEFT - MARGIN_RIGHT
        plot_height = h - MARGIN_TOP - MARGIN_BOTTOM

        # Ось Y строится с дополнительным отступом
        grid_y_min = self.y_min - self.extra_margin
        grid_y_max = self.y_max + self.extra_margin
        grid_y_range = grid_y_max - grid_y_min if grid_y_max != grid_y_min else SMALL_VALUE

        def to_pixel_y(y):
            return int(h - MARGIN_BOTTOM - (y - grid_y_min) / grid_y_range * plot_height)

        def to_pixel_x(x):
            return int(MARGIN_LEFT + (x - grid_x_min) / grid_x_range * plot_width)

        # Рисуем фон
        painter.fillRect(self.rect(), QBrush(Qt.white))

        # Отрисовка горизонтальной сетки
        if self.data and len(self.data[0]["x"]) == 1:
            self._draw_single_point_grid(painter, w, h, to_pixel_y)
        else:
            self._draw_horizontal_grid(painter, w, h, to_pixel_y)

        # Рисуем горизонтальную ось y = 0
        zero_y = to_pixel_y(0)
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(MARGIN_LEFT, zero_y, w - MARGIN_RIGHT, zero_y)

        # Вертикальная сетка
        if self.default_points > 1:
            step_x = (self.user_x_end - self.user_x_start) / (self.default_points - 1)
        else:
            step_x = (self.user_x_end - self.user_x_start) if (self.user_x_end != self.user_x_start) else 1

        # Определяем домен для вертикальной сетки
        grid_x_min = self.user_x_start - step_x
        grid_x_max = self.user_x_end + step_x
        grid_x_range = grid_x_max - grid_x_min if grid_x_max != grid_x_min else SMALL_VALUE

        # Отрисовка вертикальной сетки
        self._draw_vertical_grid(painter, h, grid_x_min, grid_x_max, step_x, to_pixel_x)

        # Рисуем вертикальную ось x = 0
        x_zero = to_pixel_x(0)
        x_zero = max(MARGIN_LEFT, min(w - MARGIN_RIGHT, x_zero))
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(x_zero, MARGIN_TOP, x_zero, h - MARGIN_BOTTOM)
        painter.setPen(QPen(Qt.black, 1))
        painter.drawText(x_zero - 20, h - MARGIN_BOTTOM + 20, "0.00")

        if not self.data:
            return

        # Отрисовка конусов
        self._draw_cones(painter, w, h, to_pixel_x, to_pixel_y, grid_x_min, grid_x_range, step_x)

        # Рисуем горизонтальную ось y = 0
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(MARGIN_LEFT, zero_y, w - MARGIN_RIGHT, zero_y)
        
        # Рисуем легенду
        self._draw_legend(painter)

    def _draw_single_point_grid(self, painter, w, h, to_pixel_y):
        """Отрисовка сетки для случая с одной точкой"""
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        for func_index, curve in enumerate(self.data):
            base_y = to_pixel_y(0)
            shift_y = to_pixel_y(curve["y"][0]) - base_y
            y_pos = to_pixel_y(curve["y"][0]) + int(func_index * shift_y)
            painter.drawLine(MARGIN_LEFT, y_pos, w - MARGIN_RIGHT, y_pos)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(w - MARGIN_RIGHT + 5, y_pos + 5, f"{curve['y'][0]:.2f}")
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))

    def _draw_horizontal_grid(self, painter, w, h, to_pixel_y):
        """Отрисовка горизонтальной сетки"""
        y_range = self.y_max + self.extra_margin - (self.y_min - self.extra_margin)
        target_lines = 5
        step_y = self._nice_num(y_range / target_lines, True)
        tick_start = math.floor((self.y_min - self.extra_margin) / step_y) * step_y
        tick_end = math.ceil((self.y_max + self.extra_margin) / step_y) * step_y
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        tick = tick_start
        while tick <= tick_end:
            y_pix = to_pixel_y(tick)
            painter.drawLine(MARGIN_LEFT, y_pix, w - MARGIN_RIGHT, y_pix)
            painter.setPen(QPen(Qt.black, 1))
            painter.setFont(QFont(FONT_FAMILY, FONT_SIZE))
            painter.drawText(w - MARGIN_RIGHT + 5, y_pix + 5, f"{tick:.2f}")
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
            tick += step_y

    def _draw_vertical_grid(self, painter, h, grid_x_min, grid_x_max, step_x, to_pixel_x):
        """Отрисовка вертикальной сетки"""
        painter.setPen(QPen(Qt.darkGray, 1, Qt.DashDotLine))
        x_val = grid_x_min
        while x_val <= grid_x_max + step_x / 2:
            x_pix = to_pixel_x(x_val)
            painter.drawLine(x_pix, MARGIN_TOP, x_pix, h - MARGIN_BOTTOM)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(x_pix - 20, h - MARGIN_BOTTOM + 20, f"{x_val:.2f}")
            painter.setPen(QPen(Qt.darkGray, 1, Qt.DashDotLine))
            x_val += step_x

    def _draw_cones(self, painter, w, h, to_pixel_x, to_pixel_y, grid_x_min, grid_x_range, step_x):
        """Отрисовка конусов"""
        num_points = len(self.data[0]["x"])
        num_funcs = len(self.data)
        
        # Вычисляем размеры для конусов
        left_bound_plot = to_pixel_x(self.user_x_start + step_x)
        right_bound_plot = to_pixel_x(self.user_x_end - step_x)
        group_width = (right_bound_plot - left_bound_plot) * CONE_SCALE / (num_points if num_points else 1)

        # Определяем ширину ячейки
        if num_points == 3:
            cell_width = FIXED_BASE_WIDTH
        else:
            cell_width = group_width / num_funcs if num_funcs else group_width

        cone_base_width = cell_width * CONE_BASE_RATIO
        ellipse_height = int(cone_base_width * 0.6)
        shift_y = -int(ellipse_height * SHIFT_Y_FACTOR)

        # Рисуем сетку для конусов
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        for i in range(num_points):
            center = self.data[0]["x"][i]
            center_pix = to_pixel_x(center)
            painter.drawLine(center_pix, MARGIN_TOP, center_pix, h - MARGIN_BOTTOM)

        # Рисуем конусы
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
                
                # Для первого ряда данных (func_index == 0) основание всегда на оси X
                if func_index == 0:
                    base_y = to_pixel_y(0)
                    apex_y = to_pixel_y(curve["y"][i])
                else:
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

    def _draw_legend(self, painter):
        """Отрисовка легенды"""
        font = QFont(FONT_FAMILY, LEGEND_FONT_SIZE)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        
        # Вычисляем размеры легенды
        max_label_width = max(metrics.horizontalAdvance(curve["label"]) for curve in self.data)
        legend_width = LEGEND_PADDING * 2 + LEGEND_MARKER_SIZE + LEGEND_SPACING + max_label_width
        legend_height = LEGEND_PADDING * 2 + len(self.data) * (LEGEND_MARKER_SIZE + LEGEND_SPACING) - LEGEND_SPACING
        
        # Рисуем фон легенды
        painter.setBrush(QBrush(Qt.white))
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(*LEGEND_POSITION, legend_width, legend_height)
        
        # Рисуем маркеры и подписи
        current_y = LEGEND_POSITION[1] + LEGEND_PADDING
        for curve in self.data:
            # Рисуем цветной маркер
            painter.setBrush(QBrush(curve["color"]))
            painter.setPen(QPen(curve["color"].darker(150), 1))
            painter.drawRect(LEGEND_POSITION[0] + LEGEND_PADDING, current_y, 
                           LEGEND_MARKER_SIZE, LEGEND_MARKER_SIZE)
            
            # Рисуем текст
            painter.setPen(Qt.black)
            painter.drawText(LEGEND_POSITION[0] + LEGEND_PADDING + LEGEND_MARKER_SIZE + LEGEND_SPACING,
                           current_y + LEGEND_MARKER_SIZE - 3, curve["label"])
            current_y += LEGEND_MARKER_SIZE + LEGEND_SPACING

    @staticmethod
    def _nice_num(x, round_val):
        """Вычисляет "приятное" число для шага сетки"""
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

    def draw_cone(self, painter, apex_x, apex_y, base_x, base_y, width_val, height_val, color):
        """
        Отрисовывает конус с треугольной боковой гранью и эллиптическим основанием.
        Для конусов, направленных вверх, отрисовывается нижняя часть основания.
        """
        # Рисуем боковые грани конуса
        path = QPainterPath()
        path.moveTo(apex_x, apex_y)
        path.lineTo(base_x, base_y)
        path.lineTo(base_x + width_val, base_y)
        path.closeSubpath()
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawPath(path)

        # Рисуем основание конуса (эллипс)
        base_width = width_val
        base_height = int(height_val * 0.5)
        ellipse_rect = (int(base_x), int(base_y - base_height / 2), int(base_width), int(base_height))
        
        # Определяем направление конуса
        is_upward = apex_y < base_y
        
        if is_upward:
            # Для конусов, направленных вверх, рисуем нижнюю часть основания
            painter.setBrush(QBrush(color.darker(115)))
            painter.setPen(QPen(color.darker(150), 2))
            painter.drawEllipse(*ellipse_rect)
            
            # Рисуем видимую часть основания (нижняя половина эллипса)
            highlight_color = color.lighter(160)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(highlight_color))
            painter.drawArc(*ellipse_rect, 180 * 16, 180 * 16)  # Рисуем нижнюю половину
        else:
            # Для конусов, направленных вниз, рисуем полное основание
            # Рисуем невидимую часть основания (нижняя половина эллипса)
            painter.setBrush(QBrush(color.darker(115)))
            painter.setPen(QPen(color.darker(150), 2))
            painter.drawEllipse(*ellipse_rect)
            
            # Рисуем видимую часть основания (верхняя половина эллипса)
            highlight_color = color.lighter(160)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(highlight_color))
            painter.drawArc(*ellipse_rect, 180 * 16, -180 * 16)
        
        # Добавляем блик на боковой грани
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color.lighter(120)))
        highlight_path = QPainterPath()
        highlight_path.moveTo(apex_x, apex_y)
        highlight_path.lineTo(base_x + width_val * 0.3, base_y)
        highlight_path.lineTo(base_x + width_val * 0.7, base_y)
        highlight_path.closeSubpath()
        painter.drawPath(highlight_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №1: Диаграмма функций (PySide)")
        self.resize(1000, 800)
        main_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        main_layout.addWidget(self.plot_widget)
        controls_layout = QHBoxLayout()
        
        # Настройка спинбокса для начала интервала
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1000, 1000)
        self.start_spin.setValue(-10)
        self.start_spin.setDecimals(2)  # Устанавливаем 2 знака после запятой
        self.start_spin.setSingleStep(0.1)  # Шаг изменения 0.1
        controls_layout.addWidget(QLabel("Начало интервала:"))
        controls_layout.addWidget(self.start_spin)
        
        # Настройка спинбокса для конца интервала
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1000, 1000)
        self.end_spin.setValue(10)
        self.end_spin.setDecimals(2)  # Устанавливаем 2 знака после запятой
        self.end_spin.setSingleStep(0.1)  # Шаг изменения 0.1
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
        
        # Настройка таймера обновления
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_plot)
        
        # Подключаем обработчики событий
        self.start_spin.valueChanged.connect(self.on_start_changed)
        self.end_spin.valueChanged.connect(self.on_end_changed)
        self.points_spin.valueChanged.connect(self.schedule_update)
        self.func_model.itemChanged.connect(self.schedule_update)
        
        # Запускаем начальное обновление
        self.update_plot()

    def schedule_update(self):
        self.update_timer.start(200)

    def update_plot(self):
        start = self.start_spin.value()
        end = self.end_spin.value()
        num_points = self.points_spin.value()
        
        # Проверяем корректность интервала
        if start >= end:
            # Если начало больше или равно концу, очищаем данные и выходим
            self.plot_widget.setData([])
            return
            
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

    def on_start_changed(self, value):
        """
        Обработчик изменения значения начала интервала.
        Обновляет минимальное значение конца интервала.
        """
        self.end_spin.setMinimum(value + 0.1)
        if self.end_spin.value() <= value:
            self.end_spin.setValue(value + 0.1)
        self.schedule_update()

    def on_end_changed(self, value):
        """
        Обработчик изменения значения конца интервала.
        Обновляет максимальное значение начала интервала.
        """
        self.start_spin.setMaximum(value - 0.1)
        if self.start_spin.value() >= value:
            self.start_spin.setValue(value - 0.1)
        self.schedule_update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())