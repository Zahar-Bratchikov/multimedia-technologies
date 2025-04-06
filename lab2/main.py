import sys
import math
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QSlider, QPushButton, QSpinBox,
                              QGroupBox, QDoubleSpinBox, QComboBox)
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
    def rotation_around_x(angle_degrees):
        """Вращение вокруг оси X (стандартная матрица вращения)"""
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        result = Matrix4x4.identity()
        # X-ось неподвижна, изменяются Y и Z
        result[1, 1] = cos_a
        result[1, 2] = -sin_a
        result[2, 1] = sin_a
        result[2, 2] = cos_a
        
        return result
    
    @staticmethod
    def rotation_around_y(angle_degrees):
        """Вращение вокруг оси Y (стандартная матрица вращения)"""
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        result = Matrix4x4.identity()
        # Y-ось неподвижна, изменяются X и Z
        result[0, 0] = cos_a
        result[0, 2] = sin_a
        result[2, 0] = -sin_a
        result[2, 2] = cos_a
        
        return result
    
    @staticmethod
    def rotation_around_z(angle_degrees):
        """Вращение вокруг оси Z (стандартная матрица вращения)"""
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        result = Matrix4x4.identity()
        # Z-ось неподвижна, изменяются X и Y
        result[0, 0] = cos_a
        result[0, 1] = -sin_a
        result[1, 0] = sin_a
        result[1, 1] = cos_a
        
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

    # Старые методы оставляем для обратной совместимости
    @staticmethod
    def rotation_x(angle_degrees):
        """Совместимость с предыдущей версией"""
        return Matrix4x4.rotation_around_x(angle_degrees)
    
    @staticmethod
    def rotation_y(angle_degrees):
        """Совместимость с предыдущей версией"""
        return Matrix4x4.rotation_around_y(angle_degrees)
    
    @staticmethod
    def rotation_z(angle_degrees):
        """Совместимость с предыдущей версией"""
        return Matrix4x4.rotation_around_z(angle_degrees)

    @staticmethod
    def look_at(eye, target, up):
        """Создает матрицу вида, смотрящую из точки eye на точку target с направлением вверх up"""
        # Вычисляем направление взгляда
        z_axis = np.array([eye.x - target.x, eye.y - target.y, eye.z - target.z])
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Вычисляем вектор "вправо"
        up_vec = np.array([up.x, up.y, up.z])
        x_axis = np.cross(up_vec, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Вычисляем новый вектор "вверх"
        y_axis = np.cross(z_axis, x_axis)
        
        # Создаем матрицу вращения
        rotation = np.identity(4, dtype=np.float64)
        rotation[0, 0:3] = x_axis
        rotation[1, 0:3] = y_axis
        rotation[2, 0:3] = z_axis
        
        # Создаем матрицу перемещения
        translation = Matrix4x4.translate(-eye.x, -eye.y, -eye.z)
        
        # Возвращаем результат
        return rotation @ translation

class Letter3D:
    def __init__(self, letter_type, size=100):
        self.letter_type = letter_type  # 'Б' или 'З'
        self.size = size
        self.position = Vector3D(0, 0, 0)
        self.rotation = Vector3D(0, 0, 0)
        self.scale_factors = Vector3D(1, 1, 1)
        self.transform_matrix = Matrix4x4.identity()
        self.vertices = []
        self.edges = []
        self.x_flipped = False
        self.y_flipped = False
        self.z_flipped = False
        self.update_geometry()
    
    def update_geometry(self):
        # Базовые размеры для букв
        w = self.size * 0.8  # ширина
        h = self.size        # высота
        d = self.size * 0.2  # глубина (толщина буквы)
        t = self.size * 0.15  # толщина основных линий

        if self.letter_type == 'Б':
            # Создаем максимально простую букву Б - только необходимые линии
            self.vertices = []
            self.edges = []
            
            # 1. Вертикальная линия слева
            v_offset = len(self.vertices)
            self.vertices.extend([
                # Передняя грань
                Vector3D(-w/2, -d/2, -h/2),      # 0 - левый нижний
                Vector3D(-w/2+t, -d/2, -h/2),    # 1 - правый нижний
                Vector3D(-w/2+t, -d/2, h/2),     # 2 - правый верхний
                Vector3D(-w/2, -d/2, h/2),       # 3 - левый верхний
                # Задняя грань
                Vector3D(-w/2, d/2, -h/2),       # 4 - левый нижний
                Vector3D(-w/2+t, d/2, -h/2),     # 5 - правый нижний
                Vector3D(-w/2+t, d/2, h/2),      # 6 - правый верхний
                Vector3D(-w/2, d/2, h/2)         # 7 - левый верхний
            ])
            
            # Рёбра вертикальной линии слева
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения

            # 2. Верхняя горизонтальная линия
            v_offset = len(self.vertices)
            self.vertices.extend([
                # Передняя грань
                Vector3D(-w/2+t, -d/2, h/2-t),   # 0 - левый нижний
                Vector3D(w/2, -d/2, h/2-t),      # 1 - правый нижний
                Vector3D(w/2, -d/2, h/2),        # 2 - правый верхний
                Vector3D(-w/2+t, -d/2, h/2),     # 3 - левый верхний
                # Задняя грань
                Vector3D(-w/2+t, d/2, h/2-t),    # 4 - левый нижний
                Vector3D(w/2, d/2, h/2-t),       # 5 - правый нижний
                Vector3D(w/2, d/2, h/2),         # 6 - правый верхний
                Vector3D(-w/2+t, d/2, h/2)       # 7 - левый верхний
            ])
            
            # Рёбра верхней горизонтальной линии
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения

            # 3. Средняя горизонтальная линия
            v_offset = len(self.vertices)
            self.vertices.extend([
                # Передняя грань
                Vector3D(-w/2+t, -d/2, 0),       # 0 - левый нижний
                Vector3D(w/2, -d/2, 0),          # 1 - правый нижний
                Vector3D(w/2, -d/2, t),          # 2 - правый верхний
                Vector3D(-w/2+t, -d/2, t),       # 3 - левый верхний
                # Задняя грань
                Vector3D(-w/2+t, d/2, 0),        # 4 - левый нижний
                Vector3D(w/2, d/2, 0),           # 5 - правый нижний
                Vector3D(w/2, d/2, t),           # 6 - правый верхний
                Vector3D(-w/2+t, d/2, t)         # 7 - левый верхний
            ])
            
            # Рёбра средней горизонтальной линии
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения
                
            # 4. Вертикальная линия справа (нижняя часть)
            v_offset = len(self.vertices)
            self.vertices.extend([
                # Передняя грань
                Vector3D(w/2-t, -d/2, -h/2),      # 0 - левый нижний
                Vector3D(w/2, -d/2, -h/2),        # 1 - правый нижний
                Vector3D(w/2, -d/2, 0),           # 2 - правый верхний
                Vector3D(w/2-t, -d/2, 0),         # 3 - левый верхний
                # Задняя грань
                Vector3D(w/2-t, d/2, -h/2),       # 4 - левый нижний
                Vector3D(w/2, d/2, -h/2),         # 5 - правый нижний
                Vector3D(w/2, d/2, 0),            # 6 - правый верхний
                Vector3D(w/2-t, d/2, 0)           # 7 - левый верхний
            ])
            
            # Рёбра вертикальной линии справа (нижняя часть)
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения
                
            # 5. Соединяем дополнительными рёбрами для замыкания нижней части буквы
            # Находим вершины предыдущих блоков
            vert_left_bottom_front = 0  # левый вертикальный блок, нижняя правая точка спереди
            vert_left_bottom_back = 4   # левый вертикальный блок, нижняя правая точка сзади
            vert_right_bottom_front = v_offset    # правый вертикальный блок, нижняя левая точка спереди
            vert_right_bottom_back = v_offset + 4 # правый вертикальный блок, нижняя левая точка сзади
            
            # Соединяем нижнюю часть горизонтальной линией
            self.edges.append((vert_left_bottom_front, vert_right_bottom_front))
            self.edges.append((vert_left_bottom_back, vert_right_bottom_back))

        elif self.letter_type == 'З':
            # Создаем максимально простую букву З - только необходимые линии
            self.vertices = []
            self.edges = []
            
            # 1. Верхняя горизонтальная линия
            v_offset = len(self.vertices)
            self.vertices.extend([
                # Передняя грань
                Vector3D(-w/2, -d/2, h/2-t),     # 0 - левый нижний
                Vector3D(w/2, -d/2, h/2-t),      # 1 - правый нижний
                Vector3D(w/2, -d/2, h/2),        # 2 - правый верхний
                Vector3D(-w/2, -d/2, h/2),       # 3 - левый верхний
                # Задняя грань
                Vector3D(-w/2, d/2, h/2-t),      # 4 - левый нижний
                Vector3D(w/2, d/2, h/2-t),       # 5 - правый нижний
                Vector3D(w/2, d/2, h/2),         # 6 - правый верхний
                Vector3D(-w/2, d/2, h/2)         # 7 - левый верхний
            ])
            
            # Рёбра верхней горизонтальной линии
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения

            # 2. Средняя горизонтальная линия
            v_offset = len(self.vertices)
            self.vertices.extend([
                # Передняя грань
                Vector3D(-w/4, -d/2, -t/2),      # 0 - левый нижний
                Vector3D(w/2, -d/2, -t/2),       # 1 - правый нижний
                Vector3D(w/2, -d/2, t/2),        # 2 - правый верхний
                Vector3D(-w/4, -d/2, t/2),       # 3 - левый верхний
                # Задняя грань
                Vector3D(-w/4, d/2, -t/2),       # 4 - левый нижний
                Vector3D(w/2, d/2, -t/2),        # 5 - правый нижний
                Vector3D(w/2, d/2, t/2),         # 6 - правый верхний
                Vector3D(-w/4, d/2, t/2)         # 7 - левый верхний
            ])
            
            # Рёбра средней горизонтальной линии
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения

            # 3. Нижняя горизонтальная линия
            v_offset = len(self.vertices)
            self.vertices.extend([
                # Передняя грань
                Vector3D(-w/2, -d/2, -h/2),      # 0 - левый нижний
                Vector3D(w/2, -d/2, -h/2),       # 1 - правый нижний
                Vector3D(w/2, -d/2, -h/2+t),     # 2 - правый верхний
                Vector3D(-w/2, -d/2, -h/2+t),    # 3 - левый верхний
                # Задняя грань
                Vector3D(-w/2, d/2, -h/2),       # 4 - левый нижний
                Vector3D(w/2, d/2, -h/2),        # 5 - правый нижний
                Vector3D(w/2, d/2, -h/2+t),      # 6 - правый верхний
                Vector3D(-w/2, d/2, -h/2+t)      # 7 - левый верхний
            ])
            
            # Рёбра нижней горизонтальной линии
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения

            # 4. Правая вертикальная линия (верхняя часть)
            v_offset = len(self.vertices)
            top_vertical_offset = v_offset  # запоминаем для соединения
            self.vertices.extend([
                # Передняя грань
                Vector3D(w/2-t, -d/2, h/2-t),    # 0 - левый нижний
                Vector3D(w/2, -d/2, h/2-t),      # 1 - правый нижний
                Vector3D(w/2, -d/2, t/2),        # 2 - правый верхний
                Vector3D(w/2-t, -d/2, t/2),      # 3 - левый верхний
                # Задняя грань
                Vector3D(w/2-t, d/2, h/2-t),     # 4 - левый нижний
                Vector3D(w/2, d/2, h/2-t),       # 5 - правый нижний
                Vector3D(w/2, d/2, t/2),         # 6 - правый верхний
                Vector3D(w/2-t, d/2, t/2)        # 7 - левый верхний
            ])
            
            # Рёбра правой вертикальной линии (верхняя часть)
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения

            # 5. Правая вертикальная линия (нижняя часть)
            v_offset = len(self.vertices)
            bottom_vertical_offset = v_offset  # запоминаем для соединения
            self.vertices.extend([
                # Передняя грань
                Vector3D(w/2-t, -d/2, -t/2),     # 0 - левый нижний
                Vector3D(w/2, -d/2, -t/2),       # 1 - правый нижний
                Vector3D(w/2, -d/2, -h/2+t),     # 2 - правый верхний
                Vector3D(w/2-t, -d/2, -h/2+t),   # 3 - левый верхний
                # Задняя грань
                Vector3D(w/2-t, d/2, -t/2),      # 4 - левый нижний
                Vector3D(w/2, d/2, -t/2),        # 5 - правый нижний
                Vector3D(w/2, d/2, -h/2+t),      # 6 - правый верхний
                Vector3D(w/2-t, d/2, -h/2+t)     # 7 - левый верхний
            ])
            
            # Рёбра правой вертикальной линии (нижняя часть)
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения

    def resize(self, size):
        self.size = size
        self.update_geometry()
    
    def update_transform(self):
        # Создаем матрицы трансформации
        # В этой версии все трансформации применяются относительно мировых координат,
        # а не относительно осей объекта
        
        # Начинаем с единичной матрицы
        self.transform_matrix = Matrix4x4.identity()
        
        # 1. Сначала применяем масштабирование (относительно начала координат)
        scale_matrix = Matrix4x4.scale(self.scale_factors.x, self.scale_factors.y, self.scale_factors.z)
        self.transform_matrix = self.transform_matrix @ scale_matrix
        
        # 2. Применяем отражения относительно мировых осей
        if self.x_flipped:
            reflection_x = Matrix4x4.reflection_x()
            self.transform_matrix = self.transform_matrix @ reflection_x
        
        if self.y_flipped:
            reflection_y = Matrix4x4.reflection_y()
            self.transform_matrix = self.transform_matrix @ reflection_y
        
        if self.z_flipped:
            reflection_z = Matrix4x4.reflection_z()
            self.transform_matrix = self.transform_matrix @ reflection_z
        
        # 3. Применяем вращения относительно мировых осей
        rotation_x = Matrix4x4.rotation_around_x(self.rotation.x)
        rotation_y = Matrix4x4.rotation_around_y(self.rotation.y)
        rotation_z = Matrix4x4.rotation_around_z(self.rotation.z)
        
        # Порядок важен! Сначала Z, потом Y, потом X
        self.transform_matrix = self.transform_matrix @ rotation_z @ rotation_y @ rotation_x
        
        # 4. И наконец применяем перемещение
        translation_matrix = Matrix4x4.translate(self.position.x, self.position.y, self.position.z)
        self.transform_matrix = self.transform_matrix @ translation_matrix
    
    def flip_x(self):
        self.x_flipped = not self.x_flipped
        self.update_transform()
    
    def flip_y(self):
        self.y_flipped = not self.y_flipped
        self.update_transform()
    
    def flip_z(self):
        self.z_flipped = not self.z_flipped
        self.update_transform()

class Camera:
    def __init__(self):
        # Позиция камеры в начальном состоянии
        self.position = Vector3D(0, -500, 0)
        self.target = Vector3D(0, 0, 0)
        # Направление "вверх" для камеры
        self.up = Vector3D(0, 0, 1)
        self.rotation = Vector3D(0, 0, 0)
        
        self.fov = 60.0
        self.aspect_ratio = 1.0
        self.near = 0.1
        self.far = 1000.0
        
        self.view_matrix = Matrix4x4.identity()
        self.projection_matrix = Matrix4x4.identity()
    
    def update_view_matrix(self):
        # Создаем матрицу вращения для камеры относительно мировых координат
        # Порядок вращений: сначала вокруг оси Z, затем Y, затем X
        rot_z = Matrix4x4.rotation_around_z(self.rotation.z)
        rot_y = Matrix4x4.rotation_around_y(self.rotation.y)
        rot_x = Matrix4x4.rotation_around_x(self.rotation.x)
        
        # Применяем вращения в правильном порядке: Z -> Y -> X
        rot_matrix = rot_z @ rot_y @ rot_x
        
        # Вычисляем вектор направления взгляда (от позиции камеры к целевой точке)
        direction = Vector3D(-self.position.x, -self.position.y, -self.position.z)
        
        # Преобразуем позицию камеры с учетом вращений
        pos_vector = np.array([self.position.x, self.position.y, self.position.z, 1.0])
        rotated_pos = rot_matrix @ pos_vector
        rotated_position = Vector3D(rotated_pos[0], rotated_pos[1], rotated_pos[2])
        
        # Вычисляем новую целевую точку
        target_vector = np.array([self.target.x, self.target.y, self.target.z, 1.0])
        rotated_target = rot_matrix @ target_vector
        rotated_target_position = Vector3D(rotated_target[0], rotated_target[1], rotated_target[2])
        
        # Используем Look-At матрицу для создания матрицы вида
        # Используем фиксированный вектор "вверх" в мировых координатах (0, 0, 1)
        up_rotated = np.array([self.up.x, self.up.y, self.up.z, 0.0])
        rotated_up = rot_matrix @ up_rotated
        up_vector = Vector3D(rotated_up[0], rotated_up[1], rotated_up[2])
        
        self.view_matrix = Matrix4x4.look_at(rotated_position, rotated_target_position, up_vector)
    
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
        # Переворачиваем Y, так как в экранных координатах Y растет вниз
        screen_x = int(self.center_x + v_ndc[0] * self.center_x)
        screen_y = int(self.center_y - v_ndc[1] * self.center_y)
        
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
        
        # Инициализируем букву с размерами по умолчанию
        self.letter = Letter3D('Б', 100)
        
        # Инициализируем камеру
        self.camera = Camera()
        self.camera.position = Vector3D(0, -500, 0)  # Камера смотрит с оси Y
        self.camera.target = Vector3D(0, 0, 0)       # Камера смотрит на начало координат
        self.camera.up = Vector3D(0, 0, 1)           # Направление "вверх" по оси Z
        self.camera.aspect_ratio = self.width() / max(self.height(), 1)
        
        # Обновляем матрицы преобразования
        self.camera.update_view_matrix()
        self.camera.update_projection_matrix()
        self.letter.update_transform()
        
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
        model_matrix = self.letter.transform_matrix
        view_matrix = self.camera.view_matrix
        projection_matrix = self.camera.projection_matrix
        
        # Проецируем вершины буквы
        screen_vertices = []
        for vertex in self.letter.vertices:
            screen_vertex = self.renderer.project_vertex(vertex, model_matrix, view_matrix, projection_matrix)
            screen_vertices.append(screen_vertex)
        
        # Рисуем ребра буквы
        for edge in self.letter.edges:
            start = screen_vertices[edge[0]]
            end = screen_vertices[edge[1]]
            painter.drawLine(start[0], start[1], end[0], end[1])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("3D Буквы Б и З - Лабораторная работа №2")
        self.resize(800, 600)
        
        # Создаем центральный виджет и его компоновку
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Создаем холст для отрисовки
        self.canvas = Canvas3D()
        main_layout.addWidget(self.canvas, 2)
        
        # Создаем панель управления
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, 1)
        
        # Группа выбора буквы
        letter_group = QGroupBox("Параметры буквы")
        letter_layout = QVBoxLayout(letter_group)
        
        # Выбор буквы
        letter_type_layout = QHBoxLayout()
        letter_type_layout.addWidget(QLabel("Буква:"))
        self.letter_type_combo = QComboBox()
        self.letter_type_combo.addItems(['Б', 'З'])
        self.letter_type_combo.currentTextChanged.connect(self.update_letter_type)
        letter_type_layout.addWidget(self.letter_type_combo)
        letter_layout.addLayout(letter_type_layout)
        
        # Размер буквы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(10, 300)
        self.size_spin.setValue(100)
        self.size_spin.valueChanged.connect(self.update_letter_size)
        size_layout.addWidget(self.size_spin)
        letter_layout.addLayout(size_layout)
        
        control_layout.addWidget(letter_group)
        
        # Добавляем остальные группы управления
        control_layout.addWidget(self.create_rotation_group())
        control_layout.addWidget(self.create_translation_group())
        control_layout.addWidget(self.create_reflection_group())
        control_layout.addWidget(self.create_camera_group())
        
        # Добавляем кнопку автомасштабирования
        auto_scale_button = QPushButton("Автомасштабирование")
        auto_scale_button.clicked.connect(self.auto_scale)
        control_layout.addWidget(auto_scale_button)
        
        # Устанавливаем начальные значения
        self.update_letter_type()
    
    def create_rotation_group(self):
        # Группа вращения объекта относительно мировых осей
        rotation_group = QGroupBox("Вращение относительно мировых осей")
        rotation_layout = QVBoxLayout(rotation_group)
        
        # Вращение вокруг мировой оси X
        rot_x_layout = QHBoxLayout()
        rot_x_layout.addWidget(QLabel("Вращение вокруг X:"))
        self.rot_x_slider = QSlider(Qt.Horizontal)
        self.rot_x_slider.setRange(0, 360)
        self.rot_x_slider.setValue(0)
        self.rot_x_slider.valueChanged.connect(self.update_letter_rotation)
        rot_x_layout.addWidget(self.rot_x_slider)
        rotation_layout.addLayout(rot_x_layout)
        
        # Вращение вокруг мировой оси Y
        rot_y_layout = QHBoxLayout()
        rot_y_layout.addWidget(QLabel("Вращение вокруг Y:"))
        self.rot_y_slider = QSlider(Qt.Horizontal)
        self.rot_y_slider.setRange(0, 360)
        self.rot_y_slider.setValue(0)
        self.rot_y_slider.valueChanged.connect(self.update_letter_rotation)
        rot_y_layout.addWidget(self.rot_y_slider)
        rotation_layout.addLayout(rot_y_layout)
        
        # Вращение вокруг мировой оси Z
        rot_z_layout = QHBoxLayout()
        rot_z_layout.addWidget(QLabel("Вращение вокруг Z:"))
        self.rot_z_slider = QSlider(Qt.Horizontal)
        self.rot_z_slider.setRange(0, 360)
        self.rot_z_slider.setValue(0)
        self.rot_z_slider.valueChanged.connect(self.update_letter_rotation)
        rot_z_layout.addWidget(self.rot_z_slider)
        rotation_layout.addLayout(rot_z_layout)
        
        return rotation_group
    
    def create_translation_group(self):
        # Группа перемещения объекта в мировых координатах
        translation_group = QGroupBox("Перемещение в мировых координатах")
        translation_layout = QVBoxLayout(translation_group)
        
        # Перемещение по мировой оси X
        trans_x_layout = QHBoxLayout()
        trans_x_layout.addWidget(QLabel("X (мировая ось):"))
        self.trans_x_spin = QSpinBox()
        self.trans_x_spin.setRange(-200, 200)
        self.trans_x_spin.setValue(0)
        self.trans_x_spin.valueChanged.connect(self.update_letter_position)
        trans_x_layout.addWidget(self.trans_x_spin)
        translation_layout.addLayout(trans_x_layout)
        
        # Перемещение по мировой оси Y
        trans_y_layout = QHBoxLayout()
        trans_y_layout.addWidget(QLabel("Y (мировая ось):"))
        self.trans_y_spin = QSpinBox()
        self.trans_y_spin.setRange(-200, 200)
        self.trans_y_spin.setValue(0)
        self.trans_y_spin.valueChanged.connect(self.update_letter_position)
        trans_y_layout.addWidget(self.trans_y_spin)
        translation_layout.addLayout(trans_y_layout)
        
        # Перемещение по мировой оси Z
        trans_z_layout = QHBoxLayout()
        trans_z_layout.addWidget(QLabel("Z (мировая ось):"))
        self.trans_z_spin = QSpinBox()
        self.trans_z_spin.setRange(-200, 200)
        self.trans_z_spin.setValue(0)
        self.trans_z_spin.valueChanged.connect(self.update_letter_position)
        trans_z_layout.addWidget(self.trans_z_spin)
        translation_layout.addLayout(trans_z_layout)
        
        return translation_group
    
    def create_reflection_group(self):
        # Группа управления отражениями относительно мировых осей
        reflection_group = QGroupBox("Отражение относительно мировых осей")
        reflection_layout = QVBoxLayout(reflection_group)
        
        # Кнопки отражения
        self.reflect_x_btn = QPushButton("Отразить относительно оси X")
        self.reflect_x_btn.clicked.connect(self.reflect_x)
        reflection_layout.addWidget(self.reflect_x_btn)
        
        self.reflect_y_btn = QPushButton("Отразить относительно оси Y")
        self.reflect_y_btn.clicked.connect(self.reflect_y)
        reflection_layout.addWidget(self.reflect_y_btn)
        
        self.reflect_z_btn = QPushButton("Отразить относительно оси Z")
        self.reflect_z_btn.clicked.connect(self.reflect_z)
        reflection_layout.addWidget(self.reflect_z_btn)
        
        return reflection_group
    
    def create_camera_group(self):
        # Группа управления камерой относительно мировых осей
        camera_group = QGroupBox("Вращение камеры относительно мировых осей")
        camera_layout = QVBoxLayout(camera_group)
        
        # Вращение камеры вокруг оси X
        cam_x_layout = QHBoxLayout()
        cam_x_layout.addWidget(QLabel("Вращение вокруг X:"))
        self.cam_x_slider = QSlider(Qt.Horizontal)
        self.cam_x_slider.setRange(-90, 90)
        self.cam_x_slider.setValue(0)
        self.cam_x_slider.valueChanged.connect(self.update_camera_rotation)
        cam_x_layout.addWidget(self.cam_x_slider)
        camera_layout.addLayout(cam_x_layout)
        
        # Вращение камеры вокруг оси Y
        cam_y_layout = QHBoxLayout()
        cam_y_layout.addWidget(QLabel("Вращение вокруг Y:"))
        self.cam_y_slider = QSlider(Qt.Horizontal)
        self.cam_y_slider.setRange(-90, 90)
        self.cam_y_slider.setValue(0)
        self.cam_y_slider.valueChanged.connect(self.update_camera_rotation)
        cam_y_layout.addWidget(self.cam_y_slider)
        camera_layout.addLayout(cam_y_layout)
        
        # Вращение камеры вокруг оси Z
        cam_z_layout = QHBoxLayout()
        cam_z_layout.addWidget(QLabel("Вращение вокруг Z:"))
        self.cam_z_slider = QSlider(Qt.Horizontal)
        self.cam_z_slider.setRange(-90, 90)
        self.cam_z_slider.setValue(0)
        self.cam_z_slider.valueChanged.connect(self.update_camera_rotation)
        cam_z_layout.addWidget(self.cam_z_slider)
        camera_layout.addLayout(cam_z_layout)
        
        # Расстояние камеры по оси Y в мировых координатах
        cam_dist_layout = QHBoxLayout()
        cam_dist_layout.addWidget(QLabel("Расстояние по оси Y:"))
        self.cam_dist_spin = QSpinBox()
        self.cam_dist_spin.setRange(100, 1000)
        self.cam_dist_spin.setValue(500)
        self.cam_dist_spin.valueChanged.connect(self.update_camera_position)
        cam_dist_layout.addWidget(self.cam_dist_spin)
        camera_layout.addLayout(cam_dist_layout)
        
        return camera_group
    
    def update_letter_type(self):
        self.canvas.letter.letter_type = self.letter_type_combo.currentText()
        self.canvas.update()
    
    def update_letter_size(self):
        self.canvas.letter.resize(self.size_spin.value())
        self.canvas.update()
    
    def update_letter_rotation(self):
        # Обновляем углы вращения вокруг мировых осей
        self.canvas.letter.rotation.x = self.rot_x_slider.value()
        self.canvas.letter.rotation.y = self.rot_y_slider.value()
        self.canvas.letter.rotation.z = self.rot_z_slider.value()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def update_letter_position(self):
        # Обновляем позицию в мировых координатах
        self.canvas.letter.position.x = self.trans_x_spin.value()
        self.canvas.letter.position.y = self.trans_y_spin.value()
        self.canvas.letter.position.z = self.trans_z_spin.value()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def update_camera_rotation(self):
        # Обновляем вращение камеры вокруг мировых осей
        self.canvas.camera.rotation.x = self.cam_x_slider.value()
        self.canvas.camera.rotation.y = self.cam_y_slider.value()
        self.canvas.camera.rotation.z = self.cam_z_slider.value()
        self.canvas.camera.update_view_matrix()
        self.canvas.update()
    
    def update_camera_position(self):
        # Обновляем позицию камеры в мировых координатах
        self.canvas.camera.position.y = -self.cam_dist_spin.value()
        self.canvas.camera.update_view_matrix()
        self.canvas.update()
    
    def reflect_x(self):
        # Отражение относительно мировой оси X
        self.canvas.letter.flip_x()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def reflect_y(self):
        # Отражение относительно мировой оси Y
        self.canvas.letter.flip_y()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def reflect_z(self):
        # Отражение относительно мировой оси Z
        self.canvas.letter.flip_z()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def auto_scale(self):
        # Автомасштабирование объекта в области рисования
        canvas_size = min(self.canvas.width(), self.canvas.height())
        max_dim = max(self.canvas.letter.size, self.canvas.letter.size, self.canvas.letter.size)
        scale_factor = canvas_size / max_dim / 4
        
        # Обновляем размеры с сохранением пропорций
        new_size = int(self.canvas.letter.size * scale_factor)
        
        # Обновляем значения спинбоксов
        self.size_spin.setValue(new_size)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
