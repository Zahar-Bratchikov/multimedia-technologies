import sys
import math
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QSlider, QPushButton, QSpinBox,
                              QGroupBox, QDoubleSpinBox, QComboBox, QCheckBox)
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
    def __init__(self, letter_type, width=100, height=100, depth=20):
        self.letter_type = letter_type  # 'Б' или 'З'
        self.width = width
        self.height = height
        self.depth = depth
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
        w = self.width      # ширина
        h = self.height     # высота
        d = self.depth      # глубина (толщина буквы)
        t = min(w, h) * 0.15  # толщина основных линий пропорционально минимальному размеру

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
                
            # 5. Нижняя горизонтальная линия (параллелепипед)
            v_offset = len(self.vertices)
            self.vertices.extend([
                # Передняя грань
                Vector3D(-w/2+t, -d/2, -h/2),     # 0 - левый нижний
                Vector3D(w/2-t, -d/2, -h/2),      # 1 - правый нижний
                Vector3D(w/2-t, -d/2, -h/2+t),    # 2 - правый верхний
                Vector3D(-w/2+t, -d/2, -h/2+t),   # 3 - левый верхний
                # Задняя грань
                Vector3D(-w/2+t, d/2, -h/2),      # 4 - левый нижний
                Vector3D(w/2-t, d/2, -h/2),       # 5 - правый нижний
                Vector3D(w/2-t, d/2, -h/2+t),     # 6 - правый верхний
                Vector3D(-w/2+t, d/2, -h/2+t)     # 7 - левый верхний
            ])
            
            # Рёбра нижней горизонтальной линии
            for i in range(4):
                self.edges.append((v_offset+i, v_offset+((i+1)%4))) # передняя грань
                self.edges.append((v_offset+i+4, v_offset+((i+1)%4)+4)) # задняя грань
                self.edges.append((v_offset+i, v_offset+i+4)) # соединения

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

    def set_dimensions(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
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
        self.letter = Letter3D('Б', 100, 100, 20)
        
        # Инициализируем камеру
        self.camera = Camera()
        self.camera.position = Vector3D(0, -500, 0)  # Камера смотрит с оси Y
        self.camera.target = Vector3D(0, 0, 0)       # Камера смотрит на начало координат
        self.camera.up = Vector3D(0, 0, 1)           # Направление "вверх" по оси Z
        self.camera.aspect_ratio = self.width() / max(self.height(), 1)
        
        # Создаем рендерер
        self.renderer = Renderer(self.width(), self.height())
        
        # Параметры мировых осей
        self.axes_length = 100  # Длина осей по умолчанию
        
        # Инициализация осей координат
        self.initialize_axes()
        
        # Обновляем матрицы преобразования
        self.camera.update_view_matrix()
        self.camera.update_projection_matrix()
        self.letter.update_transform()
    
    def initialize_axes(self):
        """Инициализация осей координат"""
        # Создание вершин для осей координат
        self.axes_vertices = [
            Vector3D(0, 0, 0),                   # начало координат
            Vector3D(self.axes_length, 0, 0),    # конец оси X
            Vector3D(0, self.axes_length, 0),    # конец оси Y
            Vector3D(0, 0, self.axes_length)     # конец оси Z
        ]
        
        # Создание ребер для осей координат
        self.axes_edges = [
            (0, 1),  # ось X
            (0, 2),  # ось Y
            (0, 3)   # ось Z
        ]
        
        # Цвета для осей
        self.axes_colors = [
            QColor(255, 0, 0),    # красный для X
            QColor(0, 255, 0),    # зеленый для Y
            QColor(0, 0, 255)     # синий для Z
        ]
    
    def set_axes_length(self, length):
        """Установка длины осей координат"""
        self.axes_length = length
        self.initialize_axes()
        self.update()
    
    def resizeEvent(self, event):
        # При изменении размера обновляем соотношение сторон и рендерер
        self.camera.aspect_ratio = self.width() / max(self.height(), 1)
        self.camera.update_projection_matrix()
        self.renderer = Renderer(self.width(), self.height())
        super().resizeEvent(event)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Получаем текущие матрицы
        model_matrix = self.letter.transform_matrix
        view_matrix = self.camera.view_matrix
        projection_matrix = self.camera.projection_matrix
        
        # Отрисовка мировых осей координат (всегда)
        self.draw_world_axes(painter, view_matrix, projection_matrix)
        
        # Настройка пера для рисования буквы
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        
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
    
    def draw_world_axes(self, painter, view_matrix, projection_matrix):
        # Используем единичную матрицу для модельных преобразований
        model_matrix = Matrix4x4.identity()
        
        # Проецируем вершины осей
        screen_vertices = []
        for vertex in self.axes_vertices:
            screen_vertex = self.renderer.project_vertex(vertex, model_matrix, view_matrix, projection_matrix)
            screen_vertices.append(screen_vertex)
        
        # Рисуем оси с соответствующими цветами
        for i, edge in enumerate(self.axes_edges):
            start = screen_vertices[edge[0]]
            end = screen_vertices[edge[1]]
            
            # Устанавливаем цвет для данной оси
            painter.setPen(QPen(self.axes_colors[i], 2))
            painter.drawLine(start[0], start[1], end[0], end[1])
            
            # Подписываем оси буквами
            axis_labels = ["X", "Y", "Z"]
            painter.drawText(end[0] + 5, end[1] + 5, axis_labels[i])

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
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Устанавливаем начальные значения и обновляем отображение
        self.update_letter_type()
    
    def create_control_panel(self):
        # Создаем панель управления
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Группа для параметров буквы
        letter_group = QGroupBox("Параметры буквы")
        letter_layout = QVBoxLayout(letter_group)
        
        # Выбор типа буквы
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Буква:"))
        self.letter_combo = QComboBox()
        self.letter_combo.addItems(["Б", "З"])
        self.letter_combo.currentTextChanged.connect(self.update_letter_type)
        type_layout.addWidget(self.letter_combo)
        letter_layout.addLayout(type_layout)
        
        # Размер буквы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Ширина:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(10, 300)
        self.width_spin.setValue(100)
        self.width_spin.valueChanged.connect(self.update_letter_size)
        size_layout.addWidget(self.width_spin)
        letter_layout.addLayout(size_layout)
        
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Высота:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(10, 300)
        self.height_spin.setValue(100)
        self.height_spin.valueChanged.connect(self.update_letter_size)
        height_layout.addWidget(self.height_spin)
        letter_layout.addLayout(height_layout)
        
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Глубина:"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(5, 100)
        self.depth_spin.setValue(20)
        self.depth_spin.valueChanged.connect(self.update_letter_size)
        depth_layout.addWidget(self.depth_spin)
        letter_layout.addLayout(depth_layout)
        
        # Кнопка автомасштабирования
        self.auto_scale_btn = QPushButton("Авто-масштабирование")
        self.auto_scale_btn.clicked.connect(self.auto_scale)
        letter_layout.addWidget(self.auto_scale_btn)
        
        control_layout.addWidget(letter_group)
        
        # Группа для вращения
        rotation_group = QGroupBox("Вращение")
        rotation_layout = QVBoxLayout(rotation_group)
        
        # X-вращение
        self.x_rot_slider = QSlider(Qt.Horizontal)
        self.x_rot_slider.setRange(0, 360)
        self.x_rot_slider.setValue(0)
        self.x_rot_slider.valueChanged.connect(self.rotate_x)
        rotation_layout.addWidget(QLabel("Вокруг оси X:"))
        rotation_layout.addWidget(self.x_rot_slider)
        
        # Y-вращение
        self.y_rot_slider = QSlider(Qt.Horizontal)
        self.y_rot_slider.setRange(0, 360)
        self.y_rot_slider.setValue(0)
        self.y_rot_slider.valueChanged.connect(self.rotate_y)
        rotation_layout.addWidget(QLabel("Вокруг оси Y:"))
        rotation_layout.addWidget(self.y_rot_slider)
        
        # Z-вращение
        self.z_rot_slider = QSlider(Qt.Horizontal)
        self.z_rot_slider.setRange(0, 360)
        self.z_rot_slider.setValue(0)
        self.z_rot_slider.valueChanged.connect(self.rotate_z)
        rotation_layout.addWidget(QLabel("Вокруг оси Z:"))
        rotation_layout.addWidget(self.z_rot_slider)
        
        control_layout.addWidget(rotation_group)
        
        # Группа для перемещения
        translation_group = QGroupBox("Перемещение")
        translation_layout = QVBoxLayout(translation_group)
        
        # X-перемещение
        self.x_trans_slider = QSlider(Qt.Horizontal)
        self.x_trans_slider.setRange(-200, 200)
        self.x_trans_slider.setValue(0)
        self.x_trans_slider.valueChanged.connect(self.translate_x)
        translation_layout.addWidget(QLabel("По оси X:"))
        translation_layout.addWidget(self.x_trans_slider)
        
        # Y-перемещение
        self.y_trans_slider = QSlider(Qt.Horizontal)
        self.y_trans_slider.setRange(-200, 200)
        self.y_trans_slider.setValue(0)
        self.y_trans_slider.valueChanged.connect(self.translate_y)
        translation_layout.addWidget(QLabel("По оси Y:"))
        translation_layout.addWidget(self.y_trans_slider)
        
        # Z-перемещение
        self.z_trans_slider = QSlider(Qt.Horizontal)
        self.z_trans_slider.setRange(-200, 200)
        self.z_trans_slider.setValue(0)
        self.z_trans_slider.valueChanged.connect(self.translate_z)
        translation_layout.addWidget(QLabel("По оси Z:"))
        translation_layout.addWidget(self.z_trans_slider)
        
        control_layout.addWidget(translation_group)
        
        # Группа для отражения
        reflection_group = QGroupBox("Отражение")
        reflection_layout = QVBoxLayout(reflection_group)
        
        # Кнопки отражения
        self.flip_x_btn = QPushButton("Отразить по X")
        self.flip_x_btn.clicked.connect(self.flip_x)
        reflection_layout.addWidget(self.flip_x_btn)
        
        self.flip_y_btn = QPushButton("Отразить по Y")
        self.flip_y_btn.clicked.connect(self.flip_y)
        reflection_layout.addWidget(self.flip_y_btn)
        
        self.flip_z_btn = QPushButton("Отразить по Z")
        self.flip_z_btn.clicked.connect(self.flip_z)
        reflection_layout.addWidget(self.flip_z_btn)
        
        control_layout.addWidget(reflection_group)
        
        # Группа для камеры
        camera_group = QGroupBox("Камера")
        camera_layout = QVBoxLayout(camera_group)
        
        # Вращение камеры по X
        camera_layout.addWidget(QLabel("Вращение камеры по X:"))
        self.cam_x_slider = QSlider(Qt.Horizontal)
        self.cam_x_slider.setRange(-90, 90)
        self.cam_x_slider.setValue(0)
        self.cam_x_slider.valueChanged.connect(self.rotate_camera_x)
        camera_layout.addWidget(self.cam_x_slider)
        
        # Вращение камеры по Y
        camera_layout.addWidget(QLabel("Вращение камеры по Y:"))
        self.cam_y_slider = QSlider(Qt.Horizontal)
        self.cam_y_slider.setRange(-90, 90)
        self.cam_y_slider.setValue(0)
        self.cam_y_slider.valueChanged.connect(self.rotate_camera_y)
        camera_layout.addWidget(self.cam_y_slider)
        
        # Вращение камеры по Z
        camera_layout.addWidget(QLabel("Вращение камеры по Z:"))
        self.cam_z_slider = QSlider(Qt.Horizontal)
        self.cam_z_slider.setRange(-90, 90)
        self.cam_z_slider.setValue(0)
        self.cam_z_slider.valueChanged.connect(self.rotate_camera_z)
        camera_layout.addWidget(self.cam_z_slider)
        
        # Расстояние до камеры
        camera_layout.addWidget(QLabel("Расстояние до камеры:"))
        self.cam_dist_slider = QSlider(Qt.Horizontal)
        self.cam_dist_slider.setRange(100, 1000)
        self.cam_dist_slider.setValue(500)
        self.cam_dist_slider.valueChanged.connect(self.change_camera_distance)
        camera_layout.addWidget(self.cam_dist_slider)
        
        # Масштаб
        camera_layout.addWidget(QLabel("Масштаб:"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(50, 300)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.change_scale)
        camera_layout.addWidget(self.scale_slider)
        
        # Длина осей координат
        camera_layout.addWidget(QLabel("Длина осей:"))
        self.axes_length_slider = QSlider(Qt.Horizontal)
        self.axes_length_slider.setRange(50, 300)
        self.axes_length_slider.setValue(100)
        self.axes_length_slider.valueChanged.connect(self.change_axes_length)
        camera_layout.addWidget(self.axes_length_slider)
        
        control_layout.addWidget(camera_group)
        
        return control_panel
    
    def update_letter_type(self):
        # При смене типа буквы полностью обновляем геометрию
        letter_type = self.letter_combo.currentText()
        width = self.width_spin.value()
        height = self.height_spin.value()
        depth = self.depth_spin.value()
        
        # Создаем новый экземпляр буквы с текущими параметрами
        new_letter = Letter3D(letter_type, width, height, depth)
        
        # Копируем текущие трансформации
        new_letter.position = self.canvas.letter.position
        new_letter.rotation = self.canvas.letter.rotation
        new_letter.scale_factors = self.canvas.letter.scale_factors
        new_letter.x_flipped = self.canvas.letter.x_flipped
        new_letter.y_flipped = self.canvas.letter.y_flipped
        new_letter.z_flipped = self.canvas.letter.z_flipped
        
        # Обновляем трансформацию и заменяем текущую букву
        new_letter.update_transform()
        self.canvas.letter = new_letter
        
        # Обновляем холст для немедленного отображения изменений
        self.canvas.update()
    
    def update_letter_size(self):
        # При изменении размера буквы полностью обновляем геометрию
        letter_type = self.canvas.letter.letter_type
        width = self.width_spin.value()
        height = self.height_spin.value()
        depth = self.depth_spin.value()
        
        # Создаем новый экземпляр буквы с текущими параметрами
        new_letter = Letter3D(letter_type, width, height, depth)
        
        # Копируем текущие трансформации
        new_letter.position = self.canvas.letter.position
        new_letter.rotation = self.canvas.letter.rotation
        new_letter.scale_factors = self.canvas.letter.scale_factors
        new_letter.x_flipped = self.canvas.letter.x_flipped
        new_letter.y_flipped = self.canvas.letter.y_flipped
        new_letter.z_flipped = self.canvas.letter.z_flipped
        
        # Обновляем трансформацию и заменяем текущую букву
        new_letter.update_transform()
        self.canvas.letter = new_letter
        
        # Обновляем холст для немедленного отображения изменений
        self.canvas.update()
    
    def rotate_x(self):
        # Обновляем угол вращения вокруг оси X
        self.canvas.letter.rotation.x = self.x_rot_slider.value()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def rotate_y(self):
        # Обновляем угол вращения вокруг оси Y
        self.canvas.letter.rotation.y = self.y_rot_slider.value()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def rotate_z(self):
        # Обновляем угол вращения вокруг оси Z
        self.canvas.letter.rotation.z = self.z_rot_slider.value()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def translate_x(self):
        # Обновляем смещение по оси X
        self.canvas.letter.position.x = self.x_trans_slider.value()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def translate_y(self):
        # Обновляем смещение по оси Y
        self.canvas.letter.position.y = self.y_trans_slider.value()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def translate_z(self):
        # Обновляем смещение по оси Z
        self.canvas.letter.position.z = self.z_trans_slider.value()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def flip_x(self):
        # Отражение относительно оси X
        self.canvas.letter.flip_x()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def flip_y(self):
        # Отражение относительно оси Y
        self.canvas.letter.flip_y()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def flip_z(self):
        # Отражение относительно оси Z
        self.canvas.letter.flip_z()
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def change_scale(self):
        # Обновляем масштаб
        scale_factor = self.scale_slider.value() / 100
        self.canvas.letter.scale_factors.x = scale_factor
        self.canvas.letter.scale_factors.y = scale_factor
        self.canvas.letter.scale_factors.z = scale_factor
        self.canvas.letter.update_transform()
        self.canvas.update()
    
    def rotate_camera_x(self):
        # Обновляем вращение камеры вокруг оси X
        self.canvas.camera.rotation.x = self.cam_x_slider.value()
        self.canvas.camera.update_view_matrix()
        self.canvas.update()
    
    def rotate_camera_y(self):
        # Обновляем вращение камеры вокруг оси Y
        self.canvas.camera.rotation.y = self.cam_y_slider.value()
        self.canvas.camera.update_view_matrix()
        self.canvas.update()
    
    def rotate_camera_z(self):
        # Обновляем вращение камеры вокруг оси Z
        self.canvas.camera.rotation.z = self.cam_z_slider.value()
        self.canvas.camera.update_view_matrix()
        self.canvas.update()
    
    def change_camera_distance(self):
        # Изменяем расстояние до камеры
        self.canvas.camera.position.y = -self.cam_dist_slider.value()
        self.canvas.camera.update_view_matrix()
        self.canvas.update()
    
    def auto_scale(self):
        # Автомасштабирование объекта в области рисования
        canvas_size = min(self.canvas.width(), self.canvas.height())
        max_dim = max(self.canvas.letter.width, self.canvas.letter.height, self.canvas.letter.depth)
        scale_factor = canvas_size / max_dim / 4
        
        # Обновляем размеры с сохранением пропорций
        new_width = int(self.canvas.letter.width * scale_factor)
        new_height = int(self.canvas.letter.height * scale_factor)
        new_depth = int(self.canvas.letter.depth * scale_factor)
        
        # Обновляем значения спинбоксов
        self.width_spin.setValue(new_width)
        self.height_spin.setValue(new_height)
        self.depth_spin.setValue(new_depth)
    
    def change_axes_length(self):
        # Изменение длины осей координат
        length = self.axes_length_slider.value()
        self.canvas.set_axes_length(length)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
