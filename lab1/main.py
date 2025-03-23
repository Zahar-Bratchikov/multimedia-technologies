import sys
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

# Определяем цвета для каждой функции.
FUNCTION_COLORS = {
    1: QColor(Qt.blue),
    2: QColor(Qt.green),
    3: QColor(Qt.red),
    # 4: QColor(Qt.magenta)
}


def get_available_functions():
    """
    Собирает список доступных функций из модуля functions.
    Возвращает список кортежей (подпись, id функции).
    """
    func_list = []
    for func_id in range(1, 4):  # при добавлении функций увеличить конец рэнжа
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
        # Интервал по оси Y пересчитывается на основе данных функций.
        self.y_min = -10
        self.y_max = 10
        # Интервал по оси X, задаваемый пользователем.
        self.user_x_start = -10
        self.user_x_end = 10

    def setData(self, data_list):
        """
        Устанавливает данные для построения и пересчитывает диапазон y так, чтобы y=0 было видно.
        """
        self.data = data_list
        if self.data:
            # Собираем все валидные y-значения из всех наборов данных.
            all_y = np.concatenate([d["y"][~np.isnan(d["y"])] for d in self.data])
            if len(all_y) > 0:
                margin_y = (all_y.max() - all_y.min()) * 0.1
                computed_y_min = all_y.min() - margin_y
                computed_y_max = all_y.max() + margin_y
                # Обеспечиваем отображение y=0.
                self.y_min = min(computed_y_min, 0)
                self.y_max = max(computed_y_max, 0)
        self.update()

    def paintEvent(self, event):
        """
        Рисует диаграмму согласно следующим правилам:
          1. Отрисовывается сетка с дополнительным отступом и подписанными осями.
          2. Выделяется ось y=0.
          3. Графики функций отрисовываются в виде конусов. Шаг вычисляется по формуле:
             step = (end - start) / num_points.
             Первый конус строится в точке 'start', последующие – с шагом step.
             Если для одного x-значения построено несколько конусов, они смещаются по горизонтали,
             а их основания соединяются линиями.
          4. Легенда рисуется с маркерами и подписями. Поле легенды расширяется под самую длинную подпись.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()

        # 1. Определяем параметры сетки с дополнительным отступом.
        extra_margin = 1  # дополнительный отступ в единицах координат
        grid_x_min = self.user_x_start - extra_margin
        grid_x_max = self.user_x_end + extra_margin
        grid_y_min = self.y_min - extra_margin
        grid_y_max = self.y_max + extra_margin
        grid_x_range = grid_x_max - grid_x_min if grid_x_max != grid_x_min else 1e-6
        grid_y_range = grid_y_max - grid_y_min if grid_y_max != grid_y_min else 1e-6

        def to_pixel_x_grid(x):
            # Перевод координаты x в пиксели для сетки
            return int((x - grid_x_min) / grid_x_range * width)

        def to_pixel_y_grid(y):
            # Перевод координаты y в пиксели для сетки
            return int(height - (y - grid_y_min) / grid_y_range * height)

        # 2. Определяем область построения (без учета дополнительного отступа)
        plot_x_range = self.user_x_end - self.user_x_start if self.user_x_end != self.user_x_start else 1e-6
        left_bound = to_pixel_x_grid(self.user_x_start)
        right_bound = to_pixel_x_grid(self.user_x_end)
        top_bound = to_pixel_y_grid(self.y_max)
        bottom_bound = to_pixel_y_grid(self.y_min)

        def to_pixel_x_plot(x):
            # Преобразует координату x в пиксели для области графика
            return int(left_bound + (x - self.user_x_start) / plot_x_range * (right_bound - left_bound))

        def to_pixel_y_plot(y):
            # Преобразует координату y в пиксели для области графика
            return int(bottom_bound - (y - self.y_min) / (self.y_max - self.y_min) * (bottom_bound - top_bound))

        # 3. Рисуем фон и сетку.
        painter.fillRect(self.rect(), QBrush(Qt.white))
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        for x_val in range(int(np.floor(grid_x_min)), int(np.ceil(grid_x_max)) + 1):
            x_pix = to_pixel_x_grid(x_val)
            painter.drawLine(x_pix, 0, x_pix, height)
            painter.setPen(QPen(Qt.black, 1))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(x_pix - 10, height - 5, f"{x_val}")
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        for y_val in range(int(np.floor(grid_y_min)), int(np.ceil(grid_y_max)) + 1):
            y_pix = to_pixel_y_grid(y_val)
            painter.drawLine(0, y_pix, width, y_pix)
            painter.setPen(QPen(Qt.black, 1))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(width - 30, y_pix + 5, f"{y_val}")
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))

        # 4. Рисуем оси, выделяя линию y=0.
        painter.setPen(QPen(Qt.black, 2))
        zero_y_pix = to_pixel_y_grid(0)
        painter.drawLine(0, zero_y_pix, width, zero_y_pix)
        if grid_x_min <= 0 <= grid_x_max:
            x_zero_pix = to_pixel_x_grid(0)
            painter.drawLine(x_zero_pix, 0, x_zero_pix, height)

        # 5. Если данных нет, дальнейшая отрисовка не производится.
        if not self.data:
            return

        # 6. Построение и отрисовка конусов.
        # Предполагается, что массив x из первой функции сгенерирован как:
        # x = [start + i * step for i in range(num_points)]
        num_points = len(self.data[0]["x"])
        num_funcs = len(self.data)

        # Вычисляем ширину группы в пикселях (влияет на размер конусов)
        group_width = ((right_bound - left_bound) / (num_points * 1.5))
        cone_scale = 0.7
        group_width *= cone_scale
        # Ширина одного конуса в пикселях.
        sub_width = group_width / num_funcs
        ellipse_height = sub_width * 0.6  # высота эллиптического основания
        shift_x = sub_width * 0.05  # горизонтальное смещение для эффекта 3D
        shift_y = -ellipse_height * 2.5  # вертикальное смещение (отрицательное смещение поднимает вверх)

        # Для каждого x-значения.
        for i in range(num_points):
            # Находим центральный пиксель для текущего x.
            center_pixel = to_pixel_x_plot(self.data[0]["x"][i])
            base_centers = []  # список для хранения центров оснований конусов
            for func_index, curve in enumerate(self.data):
                y_val = curve["y"][i]
                if np.isnan(y_val):
                    continue
                # Если строится несколько конусов, смещаем их по горизонтали.
                x_offset = int((func_index - (num_funcs - 1) / 2) * sub_width) + int(func_index * shift_x)
                apex_x = center_pixel + x_offset  # вершина конуса расположена по центру с учетом смещения
                base_x = apex_x - sub_width // 2  # вычисляем левую координату основания, чтобы вершина была в середине
                apex_y = to_pixel_y_plot(y_val) + int(func_index * shift_y)
                base_y = to_pixel_y_plot(0) + int(func_index * shift_y)
                # Сохраняем координаты центра основания для последующего соединения линиями
                base_center = (base_x + sub_width // 2, base_y)
                base_centers.append(base_center)
                self.draw_cone(painter, apex_x, apex_y, base_x, base_y, sub_width, ellipse_height, curve["color"])
            # Если для одного x-значения построено несколько конусов, соединяем их основания линиями.
            if len(base_centers) > 1:
                painter.setPen(QPen(Qt.black, 1))
                for j in range(len(base_centers) - 1):
                    p1 = base_centers[j]
                    p2 = base_centers[j + 1]
                    painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        # 7. Рисуем ось X поверх всего.
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(0, zero_y_pix, width, zero_y_pix)

        # 8. Рисуем легенду.
        self.draw_legend(painter)

    def draw_cone(self, painter, apex_x, apex_y, base_x, base_y, width, height, color):
        """
        Отрисовывает конус.
        Боковая грань рисуется треугольником, основание – эллипсом.
        """
        # Рисуем боковую грань (треугольник).
        path = QPainterPath()
        path.moveTo(apex_x, apex_y)
        path.lineTo(base_x, base_y)
        path.lineTo(base_x + width, base_y)
        path.closeSubpath()
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawPath(path)

        # Рисуем основание в виде эллипса.
        base_width = width
        base_height = height * 0.5  # половина высоты для эллипса
        ellipse_rect = (int(base_x), int(base_y - base_height / 2), int(base_width), int(base_height))
        painter.setBrush(QBrush(color.darker(115)))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawEllipse(*ellipse_rect)

    def draw_legend(self, painter):
        """
        Рисует легенду с рамкой, цветными маркерами и подписями.
        Ширина поля легенды подстраивается под самую длинную подпись.
        """
        # Устанавливаем шрифт для легенды.
        font = QFont("Arial", 10)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        padding = 10
        marker_size = 15
        spacing = 8
        max_label_width = 0

        # Вычисляем максимальную ширину подписи.
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
            # Рисуем маркер цвета.
            painter.setBrush(QBrush(curve["color"]))
            painter.drawRect(legend_x + padding, current_y, marker_size, marker_size)
            painter.setPen(Qt.black)
            # Рисуем текст подписи.
            painter.drawText(legend_x + padding + marker_size + spacing, current_y + marker_size - 3, curve["label"])
            current_y += marker_size + spacing


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №1: Диаграмма функций (PySide)")
        self.resize(1000, 800)

        main_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        main_layout.addWidget(self.plot_widget)

        # Панель управления для ввода параметров.
        controls_layout = QHBoxLayout()

        # Начало интервала.
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1000, 1000)
        self.start_spin.setValue(-10)
        controls_layout.addWidget(QLabel("Начало интервала:"))
        controls_layout.addWidget(self.start_spin)

        # Конец интервала.
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1000, 1000)
        self.end_spin.setValue(10)
        controls_layout.addWidget(QLabel("Конец интервала:"))
        controls_layout.addWidget(self.end_spin)

        # Количество точек.
        self.points_spin = QSpinBox()
        self.points_spin.setRange(1, 10000)
        self.points_spin.setValue(20)
        controls_layout.addWidget(QLabel("Количество точек:"))
        controls_layout.addWidget(self.points_spin)

        # Выбор функций.
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
        # Получаем параметры из панели управления.
        start = self.start_spin.value()
        end = self.end_spin.value()
        num_points = self.points_spin.value()
        self.plot_widget.user_x_start = start
        self.plot_widget.user_x_end = end
        data_list = []
        # Собираем данные для выбранных функций.
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
