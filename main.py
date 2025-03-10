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

# Define colors for each function.
FUNCTION_COLORS = {
    1: QColor(Qt.blue),
    2: QColor(Qt.green),
    3: QColor(Qt.red)
}

def get_available_functions():
    """
    Collects the three functions available in the functions module.
    """
    func_list = []
    for func_id in range(1, 4):
        func = getattr(functions, f"function_{func_id}", None)
        if callable(func):
            doc = func.__doc__.strip() if func.__doc__ else "No description"
            func_list.append((f"Функция {func_id} ({doc})", func_id))
    return func_list

class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        # Data for plotting: list of dicts with keys "x", "y", "label", "color".
        self.data = []
        # Y-range will be recalculated based on function data.
        self.y_min = -10
        self.y_max = 10
        # User-defined x-range.
        self.user_x_start = -10
        self.user_x_end = 10

    def setData(self, data_list):
        """
        Set plotting data and recalculate the y-range ensuring y=0 is included.
        """
        self.data = data_list
        if self.data:
            # Concatenate valid y values from all datasets.
            all_y = np.concatenate([d["y"][~np.isnan(d["y"])] for d in self.data])
            if len(all_y) > 0:
                margin_y = (all_y.max() - all_y.min()) * 0.1
                computed_y_min = all_y.min() - margin_y
                computed_y_max = all_y.max() + margin_y
                # Ensure that y=0 is visible.
                self.y_min = min(computed_y_min, 0)
                self.y_max = max(computed_y_max, 0)
        self.update()

    def paintEvent(self, event):
        """
        Draws the diagram according to the following rules:
          1. A grid is drawn with extra margin and with labels along the bottom and right.
          2. The zero line (y=0) is highlighted.
          3. The graphs are drawn as cones ("КОУСЫ"). If the generated x-values are integer,
             they are snapped to integer positions so that the cone apex (center of the base)
             is positioned exactly at those values.
          4. A legend is drawn with markers and labels.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()

        # 1. Define grid parameters with extra margin.
        extra_margin = 1  # extra margin in coordinate units.
        grid_x_min = self.user_x_start - extra_margin
        grid_x_max = self.user_x_end + extra_margin
        grid_y_min = self.y_min - extra_margin
        grid_y_max = self.y_max + extra_margin
        grid_x_range = grid_x_max - grid_x_min if grid_x_max != grid_x_min else 1e-6
        grid_y_range = grid_y_max - grid_y_min if grid_y_max != grid_y_min else 1e-6

        def to_pixel_x_grid(x):
            return int((x - grid_x_min) / grid_x_range * width)

        def to_pixel_y_grid(y):
            return int(height - (y - grid_y_min) / grid_y_range * height)

        # 2. Define the plotting area (without extra margin).
        plot_x_range = self.user_x_end - self.user_x_start if self.user_x_end != self.user_x_start else 1e-6
        left_bound = to_pixel_x_grid(self.user_x_start)
        right_bound = to_pixel_x_grid(self.user_x_end)
        top_bound = to_pixel_y_grid(self.y_max)
        bottom_bound = to_pixel_y_grid(self.y_min)

        def to_pixel_x_plot(x):
            return int(left_bound + (x - self.user_x_start) / plot_x_range * (right_bound - left_bound))

        def to_pixel_y_plot(y):
            return int(bottom_bound - (y - self.y_min) / (self.y_max - self.y_min) * (bottom_bound - top_bound))

        # 3. Draw background and grid.
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

        # 4. Draw axes, highlighting the zero (y=0) line.
        painter.setPen(QPen(Qt.black, 2))
        zero_y_pix = to_pixel_y_grid(0)
        painter.drawLine(0, zero_y_pix, width, zero_y_pix)
        if grid_x_min <= 0 <= grid_x_max:
            x_zero_pix = to_pixel_x_grid(0)
            painter.drawLine(x_zero_pix, 0, x_zero_pix, height)

        # 5. If no function data, nothing else to draw.
        if not self.data:
            return

        # 6. Build and draw the cones.
        cone_scale = 0.7
        # Use the x array from the first function as the reference.
        num_points = len(self.data[0]["x"])
        num_funcs = len(self.data)

        # Group width in pixel space based on the plotting area.
        group_width = ((right_bound - left_bound) / (num_points * 1.5)) * cone_scale
        # Each cone within a group will have a width.
        sub_width = group_width / num_funcs
        ellipse_height = sub_width * 0.6  # base height of the ellipse
        shift_x = sub_width * 0.05  # horizontal shift for a 3D effect (can be zeroed if not needed)
        shift_y = -ellipse_height * 2.5  # vertical shift; negative shifts upward

        tol = 1e-6
        for i in range(num_points):
            x_val = self.data[0]["x"][i]
            # If x_val is near an integer, round it.
            if abs(x_val - round(x_val)) < tol:
                x_val = round(x_val)
            if not (self.user_x_start <= x_val <= self.user_x_end):
                continue
            x_base = to_pixel_x_plot(x_val)
            base_points = []
            for func_index, curve in enumerate(self.data):
                y_val = curve["y"][i]
                if np.isnan(y_val):
                    continue
                # Calculate horizontal offset so that cones from different functions are side by side.
                x_offset = int((func_index - (num_funcs - 1) / 2) * sub_width) + int(func_index * shift_x)
                x_pixel = x_base + x_offset
                apex_x = x_pixel + sub_width // 2  # apex (center of the cone) position
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

        # 7. Draw the x-axis again atop everything.
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(0, zero_y_pix, width, zero_y_pix)

        # 8. Draw a legend.
        self.draw_legend(painter)

    def draw_cone(self, painter, apex_x, apex_y, base_x, base_y, width, height, color):
        """
        Draws a cone (КОУС) with a lateral face and a base.
        The lateral face is drawn as a triangle and the base is an ellipse with the same width.
        """
        # Draw lateral face as a filled triangle.
        path = QPainterPath()
        path.moveTo(apex_x, apex_y)
        path.lineTo(base_x, base_y)
        path.lineTo(base_x + width, base_y)
        path.closeSubpath()
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawPath(path)

        # Draw the base as an ellipse.
        base_width = width
        base_height = height * 0.5  # half the ellipse height
        ellipse_rect = (int(base_x), int(base_y - base_height / 2), int(base_width), int(base_height))
        painter.setBrush(QBrush(color.darker(115)))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawEllipse(*ellipse_rect)

        # Draw an accent (highlight) over the base.
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
        Draws a legend with a border, color markers, and text labels.
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

        # Control panel for input parameters.
        controls_layout = QHBoxLayout()

        # X-range: start.
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1000, 1000)
        self.start_spin.setValue(-10)
        controls_layout.addWidget(QLabel("Начало интервала:"))
        controls_layout.addWidget(self.start_spin)

        # X-range: end.
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1000, 1000)
        self.end_spin.setValue(10)
        controls_layout.addWidget(QLabel("Конец интервала:"))
        controls_layout.addWidget(self.end_spin)

        # Number of points.
        self.points_spin = QSpinBox()
        self.points_spin.setRange(1, 10000)
        self.points_spin.setValue(21)
        controls_layout.addWidget(QLabel("Количество точек:"))
        controls_layout.addWidget(self.points_spin)

        # Function selection (QComboBox with checkboxes).
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

        # Timer for updating the plot.
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
        # Collect data for the selected functions.
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