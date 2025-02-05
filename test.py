import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QPainter, QPen, QFont
from PySide6.QtCore import Qt


class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width, height = self.width(), self.height()
        center_x, center_y = width // 2, height // 2
        grid_step = min(width, height) / 20

        # Рисуем фон
        painter.fillRect(self.rect(), Qt.white)

        # Рисуем сетку
        pen = QPen(Qt.lightGray, 1, Qt.SolidLine)
        painter.setPen(pen)
        for i in range(-10, 11):
            x = center_x + i * grid_step
            y = center_y - i * grid_step
            painter.drawLine(x, 0, x, height)
            painter.drawLine(0, y, width, y)

        # Рисуем оси
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)
        painter.drawLine(0, center_y, width, center_y)  # Ось X
        painter.drawLine(center_x, 0, center_x, height)  # Ось Y

        # Подписываем оси
        font = QFont()
        font.setPointSize(max(10, min(width, height) // 50))
        painter.setFont(font)
        painter.drawText(width - 50, center_y - 5, "Ось X")
        painter.drawText(center_x + 10, 20, "Ось Y")

        # Подписываем метки на осях
        pen.setWidth(1)
        painter.setPen(pen)
        for i in range(-10, 11):
            if i != 0:
                x = center_x + i * grid_step
                y = center_y - i * grid_step
                painter.drawText(x - 10, center_y + 15, str(i))
                painter.drawText(center_x - 25, y + 5, str(-i))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Координатная плоскость")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
