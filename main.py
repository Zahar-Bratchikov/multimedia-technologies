import sys
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QSpinBox, QLabel, QComboBox, QListView
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QFont, QPainterPath, QColor, QStandardItemModel, QStandardItem
)
import functions

# Определяем цвета для функций.
FUNCTION_COLORS = {
    1: QColor(Qt.blue),
    2: QColor(Qt.green),
    3: QColor(Qt.red),
    4: QColor(Qt.magenta),
    5: QColor(Qt.darkCyan),
    6: QColor(Qt.darkYellow),
    7: QColor(Qt.darkGray),
    8: QColor(Qt.cyan),
    9: QColor(Qt.darkRed)
}

def get_available_functions():
    """
    Собираем функции из модуля functions.
    Функции, имена которых имеют вид "function_<id>", где id от 1 до 9.
    """
    func_list = []
    for func_id in range(1, 10):  # от 1 до 9
        # Получаем функцию по имени "function_<id>"
        func = getattr(functions, f"function_{func_id}", None)
        if callable(func):
            doc = func.__doc__.strip() if func.__doc__ else "No description"
            # Отображаем номер и описание
            func_list.append((f"Функция {func_id} ({doc})", func_id))
    return func_list

class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        # Данные для построения графика: список словарей с ключами: x, y, label, color.
        self.data = []
        # Диапазон оси Y будет пересчитан на основе данных.
        self.y_min = -10
        self.y_max = 10
        # Интервал построения по оси X задается пользователем.
        self.user_x_start = -10
        self.user_x_end = 10

    def setData(self, data_list):
        """
        Устанавливает данные для построения и пересчитывает диапазон оси Y.
        """
        self.data = data_list
        if self.data:
            all_y = np.concatenate([d["y"][~np.isnan(d["y"])] for d in self.data])
            if len(all_y) > 0:
                margin_y = (all_y.max() - all_y.min()) * 0.1
                computed_y_min = all_y.min() - margin_y
                computed_y_max = all_y.max() + margin_y
                # Обеспечим, что нулевая линия входит в диапазон.
                self.y_min = min(computed_y_min, 0)
                self.y_max = max(computed_y_max, 0)
        self.update()

    def paintEvent(self, event):
        """
        Отрисовка диаграммы.
          1. Рисуется сетка с дополнительными "клетками" по краям.
          2. Подписи делений выводятся снизу (для оси X) и справа (для оси Y).
          3. Выделяется нулевая линия (Y=0) жирной линией.
          4. На основной области построения рисуются графики (конусы),
             которые не выходят за пределы этой области.
          5. Рисуется легенда с маркировкой функций.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # 1. Определяем параметры для сетки.
        extra_margin = 1  # дополнительный отступ в единицах координат
        grid_x_min = self.user_x_start - extra_margin
        grid_x_max = self.user_x_end + extra_margin
        grid_y_min = self.y_min - extra_margin
        grid_y_max = self.y_max + extra_margin
        grid_x_range = grid_x_max - grid_x_min if grid_x_max != grid_x_min else 1e-6
        grid_y_range = grid_y_max - grid_y_min if grid_y_max != grid_y_min else 1e-6

        def to_pixel_x_grid(x):
            return int((x - grid_x_min) / grid_x_range * w)

        def to_pixel_y_grid(y):
            return int(h - (y - grid_y_min) / grid_y_range * h)

        # 2. Определяем параметры для построения графиков (фигур) в основной области.
        plot_x_range = self.user_x_end - self.user_x_start if self.user_x_end != self.user_x_start else 1e-6
        # Определяем пиксельные границы основной области.
        left_bound = to_pixel_x_grid(self.user_x_start)
        right_bound = to_pixel_x_grid(self.user_x_end)
        top_bound = to_pixel_y_grid(self.y_max)
        bottom_bound = to_pixel_y_grid(self.y_min)

        def to_pixel_x_plot(x):
            return int(left_bound + (x - self.user_x_start) / plot_x_range * (right_bound - left_bound))

        def to_pixel_y_plot(y):
            return int(bottom_bound - (y - self.y_min) / (self.y_max - self.y_min) * (bottom_bound - top_bound))

        # 3. Рисуем фон и сетку.
        painter.fillRect(self.rect(), QBrush(Qt.white))
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        # Вертикальные линии: метки выводим снизу.
        for x_val in range(int(np.floor(grid_x_min)), int(np.ceil(grid_x_max)) + 1):
            x_pix = to_pixel_x_grid(x_val)
            painter.drawLine(x_pix, 0, x_pix, h)
            painter.setPen(QPen(Qt.black, 1))
            painter.setFont(QFont("Arial", 8))
            # Подписи снизу (только для внешних линий)
            if x_val >= grid_x_min and x_val <= grid_x_max:
                painter.drawText(x_pix - 10, h - 5, f"{x_val}")
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        # Горизонтальные линии: метки выводим справа.
        for y_val in range(int(np.floor(grid_y_min)), int(np.ceil(grid_y_max)) + 1):
            y_pix = to_pixel_y_grid(y_val)
            painter.drawLine(0, y_pix, w, y_pix)
            painter.setPen(QPen(Qt.black, 1))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(w - 30, y_pix + 5, f"{y_val}")
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))

        # 4. Рисуем оси (выделяем нулевую линию).
        painter.setPen(QPen(Qt.black, 2))
        zero_y_pix = to_pixel_y_grid(0)
        painter.drawLine(0, zero_y_pix, w, zero_y_pix)
        if grid_x_min <= 0 <= grid_x_max:
            x_zero_pix = to_pixel_x_grid(0)
            painter.drawLine(x_zero_pix, 0, x_zero_pix, h)

        # 5. Если данных для графиков нет, дальше не рисуем.
        if not self.data:
            return

        # Масштабирование размеров фигур (уменьшенные конусы).
        cone_scale = 0.7
        num_points = len(self.data[0]["x"])
        num_funcs = len(self.data)
        group_width = ((right_bound - left_bound) / (num_points * 1.5)) * cone_scale
        sub_width = group_width / num_funcs
        ellipse_height = sub_width * 0.6
        shift_x = sub_width * 0.05  # горизонтальный сдвиг
        shift_y = -ellipse_height * 2.5  # вертикальный сдвиг (отрицательное - вверх)

        # 6. Рисуем графики (конусы) только для точек внутри основного диапазона.
        for i in range(num_points):
            x_val = self.data[0]["x"][i]
            if not (self.user_x_start <= x_val <= self.user_x_end):
                continue
            x_base = to_pixel_x_plot(x_val)
            base_points = []
            for func_index, curve in enumerate(self.data):
                y_val = curve["y"][i]
                if np.isnan(y_val):
                    continue
                x_offset = int((func_index - (num_funcs - 1) / 2) * sub_width) + int(func_index * shift_x)
                x_pixel = x_base + x_offset
                apex_x = x_pixel + sub_width // 2
                apex_y = to_pixel_y_plot(y_val) + int(func_index * shift_y)
                base_y = to_pixel_y_plot(0) + int(func_index * shift_y)
                base_points.append((x_pixel, base_y))
                self.draw_cone(painter, apex_x, apex_y, x_pixel, base_y, sub_width, ellipse_height, curve["color"])
            if len(base_points) > 1:
                painter.setPen(QPen(Qt.black, 1))
                for j in range(len(base_points) - 1):
                    pt1 = base_points[j]
                    pt2 = base_points[j + 1]
                    painter.drawLine(pt1[0] + sub_width // 2, pt1[1], pt2[0] + sub_width // 2, pt2[1])

        # Перерисовываем ось X поверх графиков.
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(0, zero_y_pix, w, zero_y_pix)

        # 7. Рисуем легенду.
        self.draw_legend(painter)

    def draw_cone(self, painter, apex_x, apex_y, base_x, base_y, width, height, color):
        """
        Отрисовывает конус с боковой гранью и эллиптическим основанием.
        """
        path = QPainterPath()
        path.moveTo(apex_x, apex_y)
        path.lineTo(base_x, base_y)
        path.lineTo(base_x + width, base_y)
        path.lineTo(apex_x, apex_y)
        path.closeSubpath()

        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawPath(path)

        ellipse_rect = (base_x, base_y - height // 2, width, height)
        painter.setBrush(QBrush(color.darker(115)))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawEllipse(*ellipse_rect)

        highlight_color = color.lighter(160)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(highlight_color))
        if apex_y < base_y:
            start_angle = 180 * 16
            span_angle = -180 * 16
        else:
            start_angle = 0
            span_angle = 180 * 16
        painter.drawArc(*ellipse_rect, start_angle, span_angle)

    def draw_legend(self, painter):
        """
        Рисует легенду с прямоугольной рамкой, маркерами и подписями функций.
        Подписи черного цвета.
        """
        legend_x = 20
        legend_y = 20
        box_size = 15
        spacing = 8
        num_funcs = len(self.data)
        legend_height = num_funcs * (box_size + spacing) + spacing
        legend_width = 180

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
        self.setWindowTitle("Лабораторная работа №1: Диаграмма функций (PySide)")
        self.resize(1000, 800)

        main_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        main_layout.addWidget(self.plot_widget)

        # Панель управления для ввода данных.
        controls_layout = QHBoxLayout()

        # Интервал построения: начало.
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1000, 1000)
        self.start_spin.setValue(-10)
        controls_layout.addWidget(QLabel("Начало интервала:"))
        controls_layout.addWidget(self.start_spin)

        # Интервал построения: конец.
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1000, 1000)
        self.end_spin.setValue(10)
        controls_layout.addWidget(QLabel("Конец интервала:"))
        controls_layout.addWidget(self.end_spin)

        # Количество точек.
        self.points_spin = QSpinBox()
        self.points_spin.setRange(1, 10000)
        self.points_spin.setValue(200)
        controls_layout.addWidget(QLabel("Количество точек:"))
        controls_layout.addWidget(self.points_spin)

        # Выбор функций. Используем QComboBox с чекбоксами.
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

        # Таймер для обновления графика.
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
        self.plot_widget.user_x_start = start
        self.plot_widget.user_x_end = end
        data_list = []
        # Собираем выбранные функции.
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