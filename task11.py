import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtCore import Qt

# Виджет для отрисовки графиков
class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)  # Устанавливаем минимальный размер окна

    # Метод, вызываемый при перерисовке окна
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # Включаем сглаживание

        # Получаем размеры окна
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        grid_step = min(width, height) / 20  # Шаг сетки

        # Заливка фона белым цветом
        painter.fillRect(self.rect(), QBrush(Qt.white))

        # Отрисовка сетки
        pen = QPen(Qt.lightGray, 1, Qt.SolidLine)
        painter.setPen(pen)
        for i in range(-10, 11):
            x = center_x + i * grid_step
            y = center_y - i * grid_step
            painter.drawLine(x, 0, x, height)  # Вертикальные линии
            painter.drawLine(0, y, width, y)  # Горизонтальные линии

        # Отрисовка осей координат
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)
        painter.drawLine(0, center_y, width, center_y)  # Ось X
        painter.drawLine(center_x, 0, center_x, height)  # Ось Y

        # Подписи осей
        font = QFont()
        font.setPointSize(max(10, min(width, height) // 50))  # Размер шрифта зависит от размера окна
        painter.setFont(font)
        painter.drawText(width - 50, center_y - 5, "Ось X")
        painter.drawText(center_x + 10, 30, "Ось Y")

        # Отрисовка отметок на осях
        pen.setWidth(1)
        painter.setPen(pen)
        for i in range(-10, 11):
            x = center_x + i * grid_step
            y = center_y - i * grid_step
            painter.drawLine(x, center_y - 5, x, center_y + 5)  # Вертикальные отметки
            painter.drawLine(center_x - 5, y, center_x + 5, y)  # Горизонтальные отметки
            if i != 0:
                painter.drawText(x - 10, center_y + 20, str(i))  # Подписи по оси X
                painter.drawText(center_x + 10, y + 5, str(-i))  # Подписи по оси Y

        # Отрисовка графиков функций
        pen.setWidth(2)

        # График x^2
        pen.setColor(Qt.blue)
        painter.setPen(pen)
        self.draw_function(painter, lambda x: x ** 2, center_x, center_y, grid_step, -10, 10)

        # График sin(x) * exp(-0.1x^2)
        pen.setColor(Qt.green)
        painter.setPen(pen)
        self.draw_function(painter, lambda x: np.sin(x) * np.exp(-0.1 * x ** 2), center_x, center_y, grid_step, -10, 10)

        # График 1/x (разрывная функция)
        pen.setColor(Qt.red)
        painter.setPen(pen)
        self.draw_function(painter, lambda x: 1 / x if x != 0 else None, center_x, center_y, grid_step, -10, -0.1)
        self.draw_function(painter, lambda x: 1 / x if x != 0 else None, center_x, center_y, grid_step, 0.1, 10)

        # Отрисовка легенды
        self.draw_legend(painter, width - 180, 20)

    # Метод для отрисовки функции
    def draw_function(self, painter, func, cx, cy, scale, x_start, x_end, step=0.05):
        prev_point = None
        x = x_start
        while x <= x_end:
            y = func(x)
            if y is not None and abs(y) < 10:  # Ограничиваем область отображения по Y
                px = cx + x * scale
                py = cy - y * scale
                if prev_point:
                    painter.drawLine(prev_point[0], prev_point[1], px, py)  # Соединяем точки линиями
                prev_point = (px, py)
            else:
                prev_point = None  # Разрыв в функции (например, 1/x в x=0)
            x += step

    # Метод для отрисовки легенды
    def draw_legend(self, painter, x, y):
        pen = QPen()
        colors = [Qt.blue, Qt.green, Qt.red]
        labels = ["x^2 (простая)", "sin(x) * exp(-0.1x^2) (сложная)", "1/x (разрыв)"]
        for i, label in enumerate(labels):
            pen.setColor(colors[i])
            painter.setPen(pen)
            painter.drawLine(x, y + i * 20 + 5, x + 20, y + i * 20 + 5)  # Цветные линии
            pen.setColor(Qt.black)
            painter.setPen(pen)
            painter.drawText(x + 30, y + i * 20 + 10, label)  # Подписи к линиям

# Главное окно приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Графики функций")
        self.setGeometry(100, 100, 800, 600)  # Начальный размер окна

        layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.plot_widget = PlotWidget()  # Добавляем виджет с графиками
        layout.addWidget(self.plot_widget)

# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
