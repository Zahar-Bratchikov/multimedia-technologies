import sys
import math
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QSlider, QPushButton, QSpinBox,
                              QGroupBox, QDoubleSpinBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor

class Vector3D:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

class Matrix4x4:
    @staticmethod
    def identity():
        return np.identity(4, dtype=np.float64)
    
    @staticmethod
    def rotation_x(angle_degrees):
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        result = Matrix4x4.identity()
        result[1, 1] = cos_a
        result[1, 2] = -sin_a
        result[2, 1] = sin_a
        result[2, 2] = cos_a
        
        return result
    
    @staticmethod
    def rotation_y(angle_degrees):
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Это теперь будет вращение по Z
        result = Matrix4x4.identity()
        result[0, 0] = cos_a
        result[0, 1] = -sin_a
        result[1, 0] = sin_a
        result[1, 1] = cos_a
        
        return result
    
    @staticmethod
    def rotation_z(angle_degrees):
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Это теперь будет вращение по Y
        result = Matrix4x4.identity()
        result[0, 0] = cos_a
        result[0, 2] = sin_a
        result[2, 0] = -sin_a
        result[2, 2] = cos_a
        
        return result
    
    @staticmethod
    def translate(x, y, z):
        result = Matrix4x4.identity()
        result[0, 3] = x
        result[1, 3] = y
        result[2, 3] = z
        
        return result
    
    @staticmethod
    def scale(sx, sy, sz):
        result = Matrix4x4.identity()
        result[0, 0] = sx
        result[1, 1] = sy
        result[2, 2] = sz
        
        return result
    
    @staticmethod
    def perspective(fov_degrees, aspect_ratio, near, far):
        fov = math.radians(fov_degrees)
        f = 1.0 / math.tan(fov / 2)
        
        result = np.zeros((4, 4), dtype=np.float64)
        result[0, 0] = f / aspect_ratio
        result[1, 1] = f
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = (2 * far * near) / (near - far)
        result[3, 2] = -1
        
        return result
    
    @staticmethod
    def reflection_x():
        result = Matrix4x4.identity()
        result[0, 0] = -1
        return result
    
    @staticmethod
    def reflection_y():
        result = Matrix4x4.identity()
        result[1, 1] = -1
        return result
    
    @staticmethod
    def reflection_z():
        result = Matrix4x4.identity()
        result[2, 2] = -1
        return result

class Cube:
    def __init__(self, width=100, height=100, depth=100):
        self.width = width
        self.height = height
        self.depth = depth
        self.position = Vector3D(0, 0, 0)
        self.rotation = Vector3D(0, 0, 0)
        self.scale_factors = Vector3D(1, 1, 1)  # Инициализация множителей масштабирования
        self.transform_matrix = Matrix4x4.identity()
        self.vertices = []
        self.edges = []
        # Флаги отражений
        self.x_flipped = False
        self.y_flipped = False
        self.z_flipped = False
        self.update_geometry()
    
    def update_geometry(self):
        # Создаем вершины куба
        w = self.width / 2
        h = self.height / 2
        d = self.depth / 2
        
        # Вершины куба (8 точек)
        self.vertices = [
            Vector3D(-w, -h, -d),  # 0: задняя нижняя левая
            Vector3D(w, -h, -d),   # 1: задняя нижняя правая
            Vector3D(w, h, -d),    # 2: задняя верхняя правая
            Vector3D(-w, h, -d),   # 3: задняя верхняя левая
            Vector3D(-w, -h, d),   # 4: передняя нижняя левая
            Vector3D(w, -h, d),    # 5: передняя нижняя правая
            Vector3D(w, h, d),     # 6: передняя верхняя правая
            Vector3D(-w, h, d)     # 7: передняя верхняя левая
        ]
        
        # Ребра куба (12 пар индексов вершин)
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # задняя грань
            (4, 5), (5, 6), (6, 7), (7, 4),  # передняя грань
            (0, 4), (1, 5), (2, 6), (3, 7)   # соединяющие ребра
        ]
    
    def resize(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.update_geometry()
    
    def update_transform(self):
        # Создаем матрицы трансформации
        scale_matrix = Matrix4x4.scale(self.scale_factors.x, self.scale_factors.y, self.scale_factors.z)
        
        # Применяем вращения в правильном порядке: сначала X, затем Y, затем Z
        rotation_x = Matrix4x4.rotation_x(self.rotation.x)
        rotation_y = Matrix4x4.rotation_y(self.rotation.y)
        rotation_z = Matrix4x4.rotation_z(self.rotation.z)
        rotation_matrix = rotation_z @ rotation_y @ rotation_x
        
        # Добавляем отражения при необходимости
        if self.x_flipped:
            reflection_x = Matrix4x4.reflection_x()
            rotation_matrix = rotation_matrix @ reflection_x
        
        if self.y_flipped:
            reflection_y = Matrix4x4.reflection_y()
            rotation_matrix = rotation_matrix @ reflection_y
        
        if self.z_flipped:
            reflection_z = Matrix4x4.reflection_z()
            rotation_matrix = rotation_matrix @ reflection_z
        
        translation_matrix = Matrix4x4.translate(self.position.x, self.position.y, self.position.z)
        
        # Объединяем трансформации: сначала масштабирование, затем вращение, затем перемещение
        self.transform_matrix = translation_matrix @ rotation_matrix @ scale_matrix
    
    def flip_x(self):
        # Переключаем флаг отражения по X
        self.x_flipped = not self.x_flipped
        self.update_transform()
    
    def flip_y(self):
        # Переключаем флаг отражения по Y
        self.y_flipped = not self.y_flipped
        self.update_transform()
    
    def flip_z(self):
        # Переключаем флаг отражения по Z
        self.z_flipped = not self.z_flipped
        self.update_transform()

class Camera:
    def __init__(self):
        self.position = Vector3D(0, 0, -500)
        self.target = Vector3D(0, 0, 0)
        self.up = Vector3D(0, 1, 0)
        self.rotation = Vector3D(0, 0, 0)
        
        self.fov = 60.0
        self.aspect_ratio = 1.0
        self.near = 0.1
        self.far = 1000.0
        
        self.view_matrix = Matrix4x4.identity()
        self.projection_matrix = Matrix4x4.identity()
    
    def update_view_matrix(self):
        # Создаем матрицу вращения для камеры
        rotation_x = Matrix4x4.rotation_x(self.rotation.x)
        rotation_y = Matrix4x4.rotation_y(self.rotation.y)
        rotation_z = Matrix4x4.rotation_z(self.rotation.z)
        rotation = rotation_z @ rotation_y @ rotation_x
        
        # Создаем матрицу перемещения
        translation = Matrix4x4.translate(-self.position.x, -self.position.y, -self.position.z)
        
        # Вид = Вращение * Перемещение
        self.view_matrix = rotation @ translation
    
    def update_projection_matrix(self):
        self.projection_matrix = Matrix4x4.perspective(self.fov, self.aspect_ratio, self.near, self.far)

class Renderer:
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.center_x = canvas_width // 2
        self.center_y = canvas_height // 2
    
    def project_vertex(self, vertex, model_matrix, view_matrix, projection_matrix):
        # Конвертируем вершину в однородные координаты
        v_homo = np.array([vertex.x, vertex.y, vertex.z, 1.0], dtype=np.float64)
        
        # Применяем модельную матрицу (преобразования объекта)
        v_model = model_matrix @ v_homo
        
        # Применяем матрицу вида (камеры)
        v_view = view_matrix @ v_model
        
        # Применяем матрицу проекции
        v_projected = projection_matrix @ v_view
        
        # Выполняем перспективное деление
        if abs(v_projected[3]) > 1e-6:  # Защита от деления на ноль
            v_ndc = v_projected / v_projected[3]
        else:
            v_ndc = v_projected
        
        # Преобразуем в экранные координаты
        screen_x = int(self.center_x + v_ndc[0] * self.center_x)
        screen_y = int(self.center_y - v_ndc[1] * self.center_y)  # Инвертируем Y
        
        return (screen_x, screen_y)

class Canvas3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        
        # Устанавливаем черный фон
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(0, 0, 0))
        self.setPalette(palette)
        
        # Инициализируем куб и камеру
        self.cube = Cube(100, 100, 100)
        self.camera = Camera()
        self.camera.aspect_ratio = self.width() / max(self.height(), 1)  # Избегаем деления на ноль
        
        # Обновляем матрицы
        self.camera.update_view_matrix()
        self.camera.update_projection_matrix()
        self.cube.update_transform()
        
        # Создаем рендерер
        self.renderer = Renderer(self.width(), self.height())
    
    def resizeEvent(self, event):
        # При изменении размера обновляем соотношение сторон и рендерер
        self.camera.aspect_ratio = self.width() / max(self.height(), 1)
        self.camera.update_projection_matrix()
        self.renderer = Renderer(self.width(), self.height())
        super().resizeEvent(event)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Настройка пера для рисования
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        
        # Получаем текущие матрицы
        model_matrix = self.cube.transform_matrix
        view_matrix = self.camera.view_matrix
        projection_matrix = self.camera.projection_matrix
        
        # Проецируем вершины куба
        screen_vertices = []
        for vertex in self.cube.vertices:
            screen_vertex = self.renderer.project_vertex(vertex, model_matrix, view_matrix, projection_matrix)
            screen_vertices.append(screen_vertex)
        
        # Рисуем ребра куба
        for edge in self.cube.edges:
            start = screen_vertices[edge[0]]
            end = screen_vertices[edge[1]]
            painter.drawLine(start[0], start[1], end[0], end[1])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("3D Куб - Лабораторная работа №2")
        self.resize(800, 600)
        
        # Отслеживание состояния отражений
        self.x_reflected = False
        self.y_reflected = False
        self.z_reflected = False
        
        # Создаем главный виджет и компоновку
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # Создаем 3D-холст
        self.canvas = Canvas3D()
        main_layout.addWidget(self.canvas, 2)
        
        # Панель управления
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, 1)
        
        # Группа параметров куба
        cube_group = QGroupBox("Параметры куба")
        cube_layout = QVBoxLayout(cube_group)
        
        # Размеры куба
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Ширина:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(10, 300)
        self.width_spin.setValue(100)
        self.width_spin.valueChanged.connect(self.update_cube_size)
        width_layout.addWidget(self.width_spin)
        cube_layout.addLayout(width_layout)
        
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Высота:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(10, 300)
        self.height_spin.setValue(100)
        self.height_spin.valueChanged.connect(self.update_cube_size)
        height_layout.addWidget(self.height_spin)
        cube_layout.addLayout(height_layout)
        
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Глубина:"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(10, 300)
        self.depth_spin.setValue(100)
        self.depth_spin.valueChanged.connect(self.update_cube_size)
        depth_layout.addWidget(self.depth_spin)
        cube_layout.addLayout(depth_layout)
        
        control_layout.addWidget(cube_group)
        
        # Группа вращения объекта
        rotation_group = QGroupBox("Вращение объекта")
        rotation_layout = QVBoxLayout(rotation_group)
        
        # Вращение по X
        rot_x_layout = QHBoxLayout()
        rot_x_layout.addWidget(QLabel("Вращение X:"))
        self.rot_x_slider = QSlider(Qt.Horizontal)
        self.rot_x_slider.setRange(0, 360)
        self.rot_x_slider.setValue(0)
        self.rot_x_slider.valueChanged.connect(self.update_cube_rotation)
        rot_x_layout.addWidget(self.rot_x_slider)
        rotation_layout.addLayout(rot_x_layout)
        
        # Вращение по Y
        rot_y_layout = QHBoxLayout()
        rot_y_layout.addWidget(QLabel("Вращение Y:"))
        self.rot_y_slider = QSlider(Qt.Horizontal)
        self.rot_y_slider.setRange(0, 360)
        self.rot_y_slider.setValue(0)
        self.rot_y_slider.valueChanged.connect(self.update_cube_rotation)
        rot_y_layout.addWidget(self.rot_y_slider)
        rotation_layout.addLayout(rot_y_layout)
        
        # Вращение по Z
        rot_z_layout = QHBoxLayout()
        rot_z_layout.addWidget(QLabel("Вращение Z:"))
        self.rot_z_slider = QSlider(Qt.Horizontal)
        self.rot_z_slider.setRange(0, 360)
        self.rot_z_slider.setValue(0)
        self.rot_z_slider.valueChanged.connect(self.update_cube_rotation)
        rot_z_layout.addWidget(self.rot_z_slider)
        rotation_layout.addLayout(rot_z_layout)
        
        control_layout.addWidget(rotation_group)
        
        # Группа перемещения объекта
        translation_group = QGroupBox("Перемещение объекта")
        translation_layout = QVBoxLayout(translation_group)
        
        # Перемещение по X
        trans_x_layout = QHBoxLayout()
        trans_x_layout.addWidget(QLabel("X:"))
        self.trans_x_spin = QSpinBox()
        self.trans_x_spin.setRange(-200, 200)
        self.trans_x_spin.setValue(0)
        self.trans_x_spin.valueChanged.connect(self.update_cube_position)
        trans_x_layout.addWidget(self.trans_x_spin)
        translation_layout.addLayout(trans_x_layout)
        
        # Перемещение по Y
        trans_y_layout = QHBoxLayout()
        trans_y_layout.addWidget(QLabel("Y:"))
        self.trans_y_spin = QSpinBox()
        self.trans_y_spin.setRange(-200, 200)
        self.trans_y_spin.setValue(0)
        self.trans_y_spin.valueChanged.connect(self.update_cube_position)
        trans_y_layout.addWidget(self.trans_y_spin)
        translation_layout.addLayout(trans_y_layout)
        
        # Перемещение по Z
        trans_z_layout = QHBoxLayout()
        trans_z_layout.addWidget(QLabel("Z:"))
        self.trans_z_spin = QSpinBox()
        self.trans_z_spin.setRange(-200, 200)
        self.trans_z_spin.setValue(0)
        self.trans_z_spin.valueChanged.connect(self.update_cube_position)
        trans_z_layout.addWidget(self.trans_z_spin)
        translation_layout.addLayout(trans_z_layout)
        
        control_layout.addWidget(translation_group)
        
        # Группа управления камерой
        camera_group = QGroupBox("Камера")
        camera_layout = QVBoxLayout(camera_group)
        
        # Вращение камеры по X
        cam_x_layout = QHBoxLayout()
        cam_x_layout.addWidget(QLabel("Вращение X:"))
        self.cam_x_slider = QSlider(Qt.Horizontal)
        self.cam_x_slider.setRange(-90, 90)
        self.cam_x_slider.setValue(0)
        self.cam_x_slider.valueChanged.connect(self.update_camera_rotation)
        cam_x_layout.addWidget(self.cam_x_slider)
        camera_layout.addLayout(cam_x_layout)
        
        # Вращение камеры по Y
        cam_y_layout = QHBoxLayout()
        cam_y_layout.addWidget(QLabel("Вращение Y:"))
        self.cam_y_slider = QSlider(Qt.Horizontal)
        self.cam_y_slider.setRange(-90, 90)
        self.cam_y_slider.setValue(0)
        self.cam_y_slider.valueChanged.connect(self.update_camera_rotation)
        cam_y_layout.addWidget(self.cam_y_slider)
        camera_layout.addLayout(cam_y_layout)
        
        # Расстояние камеры (Z)
        cam_z_layout = QHBoxLayout()
        cam_z_layout.addWidget(QLabel("Расстояние:"))
        self.cam_z_spin = QSpinBox()
        self.cam_z_spin.setRange(100, 1000)
        self.cam_z_spin.setValue(500)
        self.cam_z_spin.valueChanged.connect(self.update_camera_position)
        cam_z_layout.addWidget(self.cam_z_spin)
        camera_layout.addLayout(cam_z_layout)
        
        control_layout.addWidget(camera_group)
        
        # Дополнительные кнопки
        buttons_layout = QHBoxLayout()
        
        # Кнопки отражения
        self.reflect_x_btn = QPushButton("Отразить по X")
        self.reflect_x_btn.clicked.connect(self.reflect_x)
        buttons_layout.addWidget(self.reflect_x_btn)
        
        self.reflect_y_btn = QPushButton("Отразить по Y")
        self.reflect_y_btn.clicked.connect(self.reflect_y)
        buttons_layout.addWidget(self.reflect_y_btn)
        
        self.reflect_z_btn = QPushButton("Отразить по Z")
        self.reflect_z_btn.clicked.connect(self.reflect_z)
        buttons_layout.addWidget(self.reflect_z_btn)
        
        control_layout.addLayout(buttons_layout)
        
        # Кнопка автомасштабирования
        self.auto_scale_btn = QPushButton("Автомасштабирование")
        self.auto_scale_btn.clicked.connect(self.auto_scale)
        control_layout.addWidget(self.auto_scale_btn)
        
        # Установка центрального виджета
        self.setCentralWidget(central_widget)
    
    def update_cube_size(self):
        width = self.width_spin.value()
        height = self.height_spin.value()
        depth = self.depth_spin.value()
        self.canvas.cube.resize(width, height, depth)
        self.canvas.update()
    
    def update_cube_rotation(self):
        self.canvas.cube.rotation.x = self.rot_x_slider.value()
        self.canvas.cube.rotation.y = self.rot_y_slider.value()
        self.canvas.cube.rotation.z = self.rot_z_slider.value()
        self.canvas.cube.update_transform()
        self.canvas.update()
    
    def update_cube_position(self):
        self.canvas.cube.position.x = self.trans_x_spin.value()
        self.canvas.cube.position.y = self.trans_y_spin.value()
        self.canvas.cube.position.z = self.trans_z_spin.value()
        self.canvas.cube.update_transform()
        self.canvas.update()
    
    def update_camera_rotation(self):
        self.canvas.camera.rotation.x = self.cam_x_slider.value()
        self.canvas.camera.rotation.y = self.cam_y_slider.value()
        self.canvas.camera.update_view_matrix()
        self.canvas.update()
    
    def update_camera_position(self):
        self.canvas.camera.position.z = -self.cam_z_spin.value()
        self.canvas.camera.update_view_matrix()
        self.canvas.update()
    
    def reflect_x(self):
        # Применяем отражение по X
        self.canvas.cube.flip_x()
        self.canvas.update()
        
        # Отслеживаем текущее состояние отражения
        self.x_reflected = not self.x_reflected
        
        # Обновляем отображение текущего состояния отражения
        if self.x_reflected:
            self.reflect_x_btn.setText("Отменить отражение по X")
        else:
            self.reflect_x_btn.setText("Отразить по X")
    
    def reflect_y(self):
        # Применяем отражение по Y
        self.canvas.cube.flip_y()
        self.canvas.update()
        
        # Отслеживаем текущее состояние отражения
        self.y_reflected = not self.y_reflected
        
        # Обновляем отображение текущего состояния отражения
        if self.y_reflected:
            self.reflect_y_btn.setText("Отменить отражение по Y")
        else:
            self.reflect_y_btn.setText("Отразить по Y")
    
    def reflect_z(self):
        # Применяем отражение по Z
        self.canvas.cube.flip_z()
        self.canvas.update()
        
        # Отслеживаем текущее состояние отражения
        self.z_reflected = not self.z_reflected
        
        # Обновляем отображение текущего состояния отражения
        if self.z_reflected:
            self.reflect_z_btn.setText("Отменить отражение по Z")
        else:
            self.reflect_z_btn.setText("Отразить по Z")
    
    def auto_scale(self):
        # Автомасштабирование объекта в области рисования
        canvas_size = min(self.canvas.width(), self.canvas.height())
        max_dim = max(self.canvas.cube.width, self.canvas.cube.height, self.canvas.cube.depth)
        scale_factor = canvas_size / max_dim / 4
        
        # Обновляем размеры с сохранением пропорций
        new_width = int(self.canvas.cube.width * scale_factor)
        new_height = int(self.canvas.cube.height * scale_factor)
        new_depth = int(self.canvas.cube.depth * scale_factor)
        
        # Обновляем значения спинбоксов
        self.width_spin.setValue(new_width)
        self.height_spin.setValue(new_height)
        self.depth_spin.setValue(new_depth)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
