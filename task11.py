import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Laba 1")


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
