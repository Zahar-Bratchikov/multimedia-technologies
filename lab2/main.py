import sys
import math
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QSlider, QPushButton, QSpinBox,
                              QGroupBox, QDoubleSpinBox, QComboBox, QCheckBox, QTabWidget,
                              QColorDialog)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QImage

class Vector3D:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

class Triangle:
    def __init__(self, v1, v2, v3, color=None):
        self.vertices = [v1, v2, v3]
        self.color = color
        self.normal = None  # Для хранения нормали к треугольнику
        self.vertex_normals = [None, None, None]  # Для хранения нормалей в вершинах (для метода Фонга)
        self.vertex_intensity = [None, None, None]  # Для хранения интенсивности освещения в вершинах (для метода Гуро)
        self.face_type = None  # Для идентификации типа грани (внутренняя, внешняя и т.д.)
    
    def transform(self, matrix):
        result = Triangle(Vector3D(), Vector3D(), Vector3D(), self.color)
        for i in range(3):
            v_homo = np.array([self.vertices[i].x, self.vertices[i].y, self.vertices[i].z, 1.0], dtype=np.float64)
            v_transformed = matrix @ v_homo
            if abs(v_transformed[3]) > 1e-6:
                v_transformed = v_transformed / v_transformed[3]
            result.vertices[i] = Vector3D(v_transformed[0], v_transformed[1], v_transformed[2])
        
        # Копируем нормаль из исходного треугольника, если она была задана
        if self.normal is not None:
            # Для правильной трансформации нормали используем транспонированную обратную матрицу
            # без учёта перемещения (только вращение и масштабирование)
            # Но для упрощения просто копируем нормаль, предполагая, что нормаль задана в мировом пространстве
            result.normal = Vector3D(self.normal.x, self.normal.y, self.normal.z)
            
        # Копируем другие свойства треугольника
        result.face_type = self.face_type
        
        return result
    
    def calculate_normal(self):
        """Вычисляет нормаль к треугольнику (единичный вектор, перпендикулярный поверхности)"""
        # Если нормаль уже установлена, не вычисляем её заново
        if self.normal is not None:
            return self.normal
            
        v1 = self.vertices[0]
        v2 = self.vertices[1]
        v3 = self.vertices[2]
        
        # Вычисляем векторы сторон треугольника
        vector1 = Vector3D(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z)
        vector2 = Vector3D(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z)
        
        # Вычисляем векторное произведение для нормали
        normal = Vector3D(
            vector1.y * vector2.z - vector1.z * vector2.y,
            vector1.z * vector2.x - vector1.x * vector2.z,
            vector1.x * vector2.y - vector1.y * vector2.x
        )
        
        # Нормализуем вектор для получения единичной нормали
        length = math.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
        if length > 1e-6:  # Избегаем деления на очень маленькие значения
            normal.x /= length
            normal.y /= length
            normal.z /= length
        else:
            # Если треугольник вырожденный (площадь близка к нулю), устанавливаем произвольную нормаль
            # чтобы предотвратить проблемы отображения
            normal.x = 0
            normal.y = 0
            normal.z = 1
            
        self.normal = normal
        return normal
        
    def is_visible(self, camera_position):
        """Определяет видимость треугольника относительно камеры с учетом ориентации и свойств"""
        if self.normal is None:
            self.calculate_normal()
            
        # Получаем вектор от вершины треугольника к камере
        v_to_camera = Vector3D(
            camera_position.x - self.vertices[0].x,
            camera_position.y - self.vertices[0].y,
            camera_position.z - self.vertices[0].z
        )
        
        # Нормализуем вектор для более точного скалярного произведения
        v_len = math.sqrt(v_to_camera.x**2 + v_to_camera.y**2 + v_to_camera.z**2)
        if v_len > 1e-6:
            v_to_camera.x /= v_len
            v_to_camera.y /= v_len
            v_to_camera.z /= v_len
        
        # Вычисляем скалярное произведение нормали и вектора к камере
        # Если скалярное произведение положительное, грань обращена к камере
        dot_product = self.normal.x * v_to_camera.x + self.normal.y * v_to_camera.y + self.normal.z * v_to_camera.z
        
        # Используем более строгое значение порога для стабильной отрисовки
        visibility_threshold = 1e-4
        
        # Треугольник видим, если его нормаль направлена в сторону камеры
        return dot_product > visibility_threshold
    
    def get_depth(self):
        """Возвращает среднюю глубину треугольника (расстояние от камеры)"""
        # Простой способ - взять среднее значение Z координат всех вершин
        return (self.vertices[0].z + self.vertices[1].z + self.vertices[2].z) / 3.0
    
    def calculate_shading(self, light_position, shading_method="flat", camera_position=None, light_color=None):
        """Рассчитывает цвет треугольника на основе освещения"""
        if self.normal is None:
            self.calculate_normal()
            
        # Направление света (от точки на треугольнике к источнику света)
        center = Vector3D(
            (self.vertices[0].x + self.vertices[1].x + self.vertices[2].x) / 3.0,
            (self.vertices[0].y + self.vertices[1].y + self.vertices[2].y) / 3.0,
            (self.vertices[0].z + self.vertices[1].z + self.vertices[2].z) / 3.0
        )
        
        light_dir = Vector3D(
            light_position.x - center.x,
            light_position.y - center.y,
            light_position.z - center.z
        )
        
        # Нормализуем вектор направления света
        light_len = math.sqrt(light_dir.x**2 + light_dir.y**2 + light_dir.z**2)
        if light_len > 1e-6:
            light_dir.x /= light_len
            light_dir.y /= light_len
            light_dir.z /= light_len
        
        # Вычисляем коэффициент диффузного отражения (косинус угла между нормалью и направлением света)
        diffuse = max(0, self.normal.x * light_dir.x + self.normal.y * light_dir.y + self.normal.z * light_dir.z)
        
        # Для метода Фонга дополнительно вычисляем отраженный свет и взгляд наблюдателя
        if shading_method == "phong" and camera_position is not None:
            # Вектор отражения света (отраженный луч)
            reflect_coeff = 2.0 * (self.normal.x * light_dir.x + self.normal.y * light_dir.y + self.normal.z * light_dir.z)
            reflect = Vector3D(
                self.normal.x * reflect_coeff - light_dir.x,
                self.normal.y * reflect_coeff - light_dir.y,
                self.normal.z * reflect_coeff - light_dir.z
            )
            
            # Вектор взгляда (от точки на треугольнике к камере)
            view = Vector3D(
                camera_position.x - center.x,
                camera_position.y - center.y,
                camera_position.z - center.z
            )
            
            # Нормализуем вектор взгляда
            view_len = math.sqrt(view.x**2 + view.y**2 + view.z**2)
            if view_len > 1e-6:
                view.x /= view_len
                view.y /= view_len
                view.z /= view_len
            
            # Вычисляем коэффициент зеркального отражения (косинус угла между отраженным лучом и взглядом)
            # Увеличиваем степень для более выраженного блика
            specular = max(0, reflect.x * view.x + reflect.y * view.y + reflect.z * view.z) ** 25
            
            # Итоговое освещение с улучшенными параметрами для объемного эффекта
            ambient = 0.2  # Фоновое освещение
            diffuse_factor = 0.7  # Увеличиваем для лучшего объемного эффекта
            specular_factor = 0.4  # Увеличиваем для более яркого блика
            
            light_intensity = min(1.0, ambient + diffuse * diffuse_factor + specular * specular_factor)
        else:
            # Для плоского затенения или затенения Гуро используем только диффузную компоненту
            # с улучшенными параметрами для объемного эффекта
            ambient = 0.2  # Увеличиваем фоновую составляющую
            diffuse_factor = 0.8  # Увеличиваем диффузную составляющую
            light_intensity = min(1.0, ambient + diffuse * diffuse_factor)
        
        # Возвращаем интенсивность освещения
        return light_intensity
    
    def calculate_vertex_normals(self):
        """Вычисляет нормали к вершинам для метода Фонга"""
        if self.normal is None:
            self.calculate_normal()
        
        # Простой случай - просто используем нормаль к грани для всех вершин
        # В реальном сценарии нормали вершин должны быть усреднены по соседним граням
        for i in range(3):
            self.vertex_normals[i] = Vector3D(self.normal.x, self.normal.y, self.normal.z)
    
    def calculate_vertex_intensity(self, light_position, camera_position=None, light_color=None):
        """Рассчитывает интенсивность освещения в вершинах для метода Гуро"""
        if self.normal is None:
            self.calculate_normal()
            
        # Если нормали вершин не вычислены, делаем это сейчас
        if self.vertex_normals[0] is None:
            self.calculate_vertex_normals()
            
        # Для каждой вершины вычисляем освещенность
        for i in range(3):
            vertex = self.vertices[i]
            normal = self.vertex_normals[i]
            
            # Направление от вершины к источнику света
            light_dir = Vector3D(
                light_position.x - vertex.x,
                light_position.y - vertex.y,
                light_position.z - vertex.z
            )
            
            # Нормализуем вектор
            light_len = math.sqrt(light_dir.x**2 + light_dir.y**2 + light_dir.z**2)
            if light_len > 1e-6:
                light_dir.x /= light_len
                light_dir.y /= light_len
                light_dir.z /= light_len
            
            # Косинус угла между нормалью и направлением света (диффузное отражение)
            diffuse = max(0, normal.x * light_dir.x + normal.y * light_dir.y + normal.z * light_dir.z)
            
            # Для метода Фонга добавляем расчет зеркального отражения
            if camera_position is not None:
                # Вектор отражения света
                reflect_coeff = 2.0 * (normal.x * light_dir.x + normal.y * light_dir.y + normal.z * light_dir.z)
                reflect = Vector3D(
                    normal.x * reflect_coeff - light_dir.x,
                    normal.y * reflect_coeff - light_dir.y,
                    normal.z * reflect_coeff - light_dir.z
                )
                
                # Вектор от вершины к камере
                view = Vector3D(
                    camera_position.x - vertex.x,
                    camera_position.y - vertex.y,
                    camera_position.z - vertex.z
                )
                
                # Нормализуем вектор взгляда
                view_len = math.sqrt(view.x**2 + view.y**2 + view.z**2)
                if view_len > 1e-6:
                    view.x /= view_len
                    view.y /= view_len
                    view.z /= view_len
                
                # Вычисляем зеркальное отражение
                specular = max(0, reflect.x * view.x + reflect.y * view.y + reflect.z * view.z) ** 20
                
                # Итоговая интенсивность с учетом зеркального отражения
                self.vertex_intensity[i] = min(1.0, 0.2 + diffuse * 0.5 + specular * 0.3)
            else:
                # Итоговая интенсивность только с диффузным отражением
                self.vertex_intensity[i] = min(1.0, 0.2 + diffuse * 0.8)
                
        return self.vertex_intensity
    
    def interpolate_intensity(self, barycentric_coords):
        """Интерполирует интенсивность освещения внутри треугольника по барицентрическим координатам"""
        if self.vertex_intensity[0] is None:
            return 1.0  # По умолчанию, если интенсивность не рассчитана
            
        # Интерполяция по барицентрическим координатам
        w0, w1, w2 = barycentric_coords
        return w0 * self.vertex_intensity[0] + w1 * self.vertex_intensity[1] + w2 * self.vertex_intensity[2]
    
    def get_interpolated_intensity(self, x, y, screen_coords):
        """Получает интерполированную интенсивность в точке (x,y) экрана по формулам из слайдов"""
        # Распаковываем экранные координаты вершин
        (x1, y1), (x2, y2), (x3, y3) = screen_coords
        
        # Если интенсивность в вершинах не вычислена, возвращаем значение по умолчанию
        if self.vertex_intensity[0] is None:
            return 1.0
            
        # Извлекаем интенсивности в вершинах
        i1, i2, i3 = self.vertex_intensity
        
        # Получаем барицентрические координаты для интерполяции
        w0, w1, w2 = self._barycentric_coords((x1, y1), (x2, y2), (x3, y3), (x, y))
        
        # Если точка не в треугольнике, возвращаем значение по умолчанию
        if w0 < 0 or w1 < 0 or w2 < 0:
            return 1.0
            
        # Интерполяция по барицентрическим координатам
        return w0 * i1 + w1 * i2 + w2 * i3
    
    def _barycentric_coords(self, v0, v1, v2, p):
        """Вычисление барицентрических координат точки p относительно треугольника (v0, v1, v2)"""
        # Конвертируем в float для точности вычислений
        v0 = (float(v0[0]), float(v0[1]))
        v1 = (float(v1[0]), float(v1[1]))
        v2 = (float(v2[0]), float(v2[1]))
        p = (float(p[0]), float(p[1]))
        
        # Вычисляем векторы
        v0v1 = (v1[0] - v0[0], v1[1] - v0[1])
        v0v2 = (v2[0] - v0[0], v2[1] - v0[1])
        v0p = (p[0] - v0[0], p[1] - v0[1])
        
        # Вычисляем определители для барицентрических координат
        d00 = v0v1[0] * v0v1[0] + v0v1[1] * v0v1[1]
        d01 = v0v1[0] * v0v2[0] + v0v1[1] * v0v2[1]
        d11 = v0v2[0] * v0v2[0] + v0v2[1] * v0v2[1]
        d20 = v0p[0] * v0v1[0] + v0p[1] * v0v1[1]
        d21 = v0p[0] * v0v2[0] + v0p[1] * v0v2[1]
        
        # Вычисляем барицентрические координаты
        denom = d00 * d11 - d01 * d01
        # Защита от деления на ноль
        if abs(denom) < 1e-6:
            return (-1, -1, -1)  # Вырожденный треугольник
            
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return (u, v, w)

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

class Light:
    def __init__(self, position=None):
        # Если позиция не указана, устанавливаем источник света над сценой по умолчанию
        if position is None:
            self.position = Vector3D(0, -200, 200)
        else:
            self.position = position
        self.color = QColor(255, 255, 255)  # Белый цвет по умолчанию
        self.intensity = 1.0  # Интенсивность по умолчанию
    
    def set_color(self, color):
        """Устанавливает цвет источника света"""
        self.color = color

class Letter3D:
    def __init__(self, letter_type, width=100, height=100, depth=20, color=None):
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
        self.triangles = []  # Добавляем список треугольников для закрашивания
        self.x_flipped = False
        self.y_flipped = False
        self.z_flipped = False
        self.color = color if color else QColor(200, 200, 200)  # Цвет буквы
        self.segments = 16  # Увеличиваем количество сегментов для лучших скруглений
        self.update_geometry()
    
    def set_color(self, color):
        """Установка цвета буквы"""
        self.color = color
        self.update_geometry()
    
    def update_geometry(self):
        """Обновляет геометрию буквы на основе текущих параметров"""
        # Базовые размеры для букв
        w = self.width      # ширина
        h = self.height     # высота
        d = self.depth      # глубина (толщина буквы)
        t = min(w, h) * 0.15  # толщина основных линий пропорционально минимальному размеру

        # Сбрасываем текущие вершины, рёбра и треугольники
        self.vertices = []
        self.edges = []
        self.triangles = []

        if self.letter_type == 'Б':
            self._create_letter_b(w, h, d, t)
        elif self.letter_type == 'З':
            self._create_letter_z(w, h, d, t)
    
    def _create_letter_b(self, w, h, d, t):
        """Создает геометрию буквы 'Б'"""
        # 1. Вертикальная линия слева (полная высота)
        self._add_box_with_triangles(
            Vector3D(-w/2, -d/2, -h/2-7),       # левый нижний передний
            Vector3D(-w/2+t, d/2, h/2),         # правый верхний задний
            self.color
        )
        
        # 2. Верхняя горизонтальная линия
        self._add_box_with_triangles(
            Vector3D(-w/2+t, -d/2, h/2-t),      # левый нижний передний
            Vector3D(35, d/2, h/2),             # правый верхний задний
            self.color
        )
        
        # 3. Средняя горизонтальная линия
        self._add_box_with_triangles(
            Vector3D(-w/2+t, -d/2, 0 - t/2),    # левый нижний передний
            Vector3D(0, d/2, 0 + t/2),          # правый верхний задний
            self.color
        )
        
        # 4. Полуокружность (с улучшенным закрашиванием)
        radius = h/4
        center_x = 1.5                          # Центр справа от средней линии
        center_z = -h/4                         # Центр на уровне средней линии
        
        # Создаем полуокружность с правильным закрашиванием граней
        self._add_smooth_arc_segment(
            Vector3D(center_x, -d/2, center_z),  # центр передней полуокружности
            Vector3D(center_x, d/2, center_z),   # центр задней полуокружности
            radius, t, -90, 90,                 # Углы полуокружности
            self.color
        )
        
        # 5. Нижняя горизонтальная линия
        self._add_box_with_triangles(
            Vector3D(-w/2+t, -d/2, -h/2-7),     # левый нижний передний
            Vector3D(0, d/2, -h/2-7+t),         # правый верхний задний
            self.color
        )
    
    def _create_letter_z(self, w, h, d, t):
        """Создает геометрию буквы 'З'"""
        t_half = t/2
        
        # 1. Верхняя полуокружность с улучшенным закрашиванием
        radius_top = h/4
        extension_factor_top = 1.3               # Коэффициент удлинения верхней части
        center_top_x = 0                         # Центр в середине
        center_top_z = h/4
        
        # Диапазон углов для верхней полуокружности
        start_angle_top = 270
        end_angle_top = 360 + (450 - 360) * extension_factor_top
        
        self._add_smooth_arc_segment(
            Vector3D(center_top_x, -d/2, center_top_z),
            Vector3D(center_top_x, d/2, center_top_z),
            radius_top, t, start_angle_top, end_angle_top,
            self.color
        )
        
        # 2. Нижняя полуокружность с улучшенным закрашиванием
        radius_bottom = h/4
        extension_factor_bottom = 0.55           # Коэффициент удлинения нижней части
        center_bottom_x = 0                      # Центр в середине
        center_bottom_z = -h/4
        
        # Диапазон углов для нижней полуокружности
        base_angle_bottom = 180
        mid_angle = 270
        start_angle_bottom = mid_angle - (mid_angle - base_angle_bottom) * extension_factor_bottom
        end_angle_bottom = 450
        
        self._add_smooth_arc_segment(
            Vector3D(center_bottom_x, -d/2, center_bottom_z),
            Vector3D(center_bottom_x, d/2, center_bottom_z),
            radius_bottom, t, start_angle_bottom, end_angle_bottom,
            self.color
        )
        
        # 3. Средняя горизонтальная линия - увеличиваем размер для лучшего соединения с дугами
        # Расчёт координат для лучшего соединения с верхней и нижней дугами
        left_x = -w/6 - t/4     # Немного увеличиваем ширину влево 
        right_x = w/6 + t/4     # Немного увеличиваем ширину вправо
        
        # Переопределяем положение прямоугольника для устранения щелей
        self._add_box_with_triangles(
            Vector3D(left_x, -d/2 - 0.1, -t_half),   # левый нижний передний, небольшой нахлест по Y
            Vector3D(right_x, d/2 + 0.1, t_half),    # правый верхний задний, небольшой нахлест по Y
            self.color
        )
        
        # 4. Добавляем дополнительные прямоугольники для заполнения щелей в стыках
        # Соединение с верхней дугой
        top_arc_x = center_top_x + radius_top * math.cos(math.radians(start_angle_top))
        top_arc_z = center_top_z + radius_top * math.sin(math.radians(start_angle_top))
        
        self._add_box_with_triangles(
            Vector3D(top_arc_x - t/2, -d/2 - 0.1, top_arc_z - t/2),
            Vector3D(top_arc_x + t/2, d/2 + 0.1, top_arc_z + t/2),
            self.color
        )
        
        # Соединение с нижней дугой
        bottom_arc_x = center_bottom_x + radius_bottom * math.cos(math.radians(end_angle_bottom))
        bottom_arc_z = center_bottom_z + radius_bottom * math.sin(math.radians(end_angle_bottom))
        
        self._add_box_with_triangles(
            Vector3D(bottom_arc_x - t/2, -d/2 - 0.1, bottom_arc_z - t/2),
            Vector3D(bottom_arc_x + t/2, d/2 + 0.1, bottom_arc_z + t/2),
            self.color
        )
    
    def _add_box_with_triangles(self, min_point, max_point, color):
        """Добавляет параллелепипед с правильным заполнением треугольниками"""
        # Сохраняем текущий индекс вершины для создания граней
        v_offset = len(self.vertices)
        
        # Передняя грань (y = min_y)
        self.vertices.append(Vector3D(min_point.x, min_point.y, min_point.z))  # 0: левый нижний
        self.vertices.append(Vector3D(max_point.x, min_point.y, min_point.z))  # 1: правый нижний
        self.vertices.append(Vector3D(max_point.x, min_point.y, max_point.z))  # 2: правый верхний
        self.vertices.append(Vector3D(min_point.x, min_point.y, max_point.z))  # 3: левый верхний
        
        # Задняя грань (y = max_y)
        self.vertices.append(Vector3D(min_point.x, max_point.y, min_point.z))  # 4: левый нижний
        self.vertices.append(Vector3D(max_point.x, max_point.y, min_point.z))  # 5: правый нижний
        self.vertices.append(Vector3D(max_point.x, max_point.y, max_point.z))  # 6: правый верхний
        self.vertices.append(Vector3D(min_point.x, max_point.y, max_point.z))  # 7: левый верхний
        
        # Добавляем рёбра для каркасного отображения
        # Передняя грань
        self.edges.append((v_offset+0, v_offset+1))
        self.edges.append((v_offset+1, v_offset+2))
        self.edges.append((v_offset+2, v_offset+3))
        self.edges.append((v_offset+3, v_offset+0))
        
        # Задняя грань
        self.edges.append((v_offset+4, v_offset+5))
        self.edges.append((v_offset+5, v_offset+6))
        self.edges.append((v_offset+6, v_offset+7))
        self.edges.append((v_offset+7, v_offset+4))
        
        # Соединения между гранями
        self.edges.append((v_offset+0, v_offset+4))
        self.edges.append((v_offset+1, v_offset+5))
        self.edges.append((v_offset+2, v_offset+6))
        self.edges.append((v_offset+3, v_offset+7))
        
        # Добавляем треугольники для закрашивания
        # Создаем треугольники с правильным порядком вершин для определения внешних граней
        
        # Передняя грань (2 треугольника) - внешняя
        triangle1 = Triangle(
            self.vertices[v_offset+0], 
            self.vertices[v_offset+1], 
            self.vertices[v_offset+2], 
            color
        )
        triangle1.face_type = "external"
        self.triangles.append(triangle1)
        
        triangle2 = Triangle(
            self.vertices[v_offset+0], 
            self.vertices[v_offset+2], 
            self.vertices[v_offset+3], 
            color
        )
        triangle2.face_type = "external"
        self.triangles.append(triangle2)
        
        # Задняя грань (2 треугольника) - внешняя
        triangle3 = Triangle(
            self.vertices[v_offset+4], 
            self.vertices[v_offset+6], 
            self.vertices[v_offset+5], 
            color
        )
        triangle3.face_type = "external"
        self.triangles.append(triangle3)
        
        triangle4 = Triangle(
            self.vertices[v_offset+4], 
            self.vertices[v_offset+7], 
            self.vertices[v_offset+6], 
            color
        )
        triangle4.face_type = "external"
        self.triangles.append(triangle4)
        
        # Верхняя грань (2 треугольника) - внешняя
        triangle5 = Triangle(
            self.vertices[v_offset+3], 
            self.vertices[v_offset+2], 
            self.vertices[v_offset+6], 
            color
        )
        triangle5.face_type = "external"
        self.triangles.append(triangle5)
        
        triangle6 = Triangle(
            self.vertices[v_offset+3], 
            self.vertices[v_offset+6], 
            self.vertices[v_offset+7], 
            color
        )
        triangle6.face_type = "external"
        self.triangles.append(triangle6)
        
        # Нижняя грань (2 треугольника) - внешняя
        triangle7 = Triangle(
            self.vertices[v_offset+0], 
            self.vertices[v_offset+5], 
            self.vertices[v_offset+1], 
            color
        )
        triangle7.face_type = "external"
        self.triangles.append(triangle7)
        
        triangle8 = Triangle(
            self.vertices[v_offset+0], 
            self.vertices[v_offset+4], 
            self.vertices[v_offset+5], 
            color
        )
        triangle8.face_type = "external"
        self.triangles.append(triangle8)
        
        # Левая грань (2 треугольника) - внешняя
        triangle9 = Triangle(
            self.vertices[v_offset+0], 
            self.vertices[v_offset+3], 
            self.vertices[v_offset+7], 
            color
        )
        triangle9.face_type = "external"
        self.triangles.append(triangle9)
        
        triangle10 = Triangle(
            self.vertices[v_offset+0], 
            self.vertices[v_offset+7], 
            self.vertices[v_offset+4], 
            color
        )
        triangle10.face_type = "external"
        self.triangles.append(triangle10)
        
        # Правая грань (2 треугольника) - внешняя
        triangle11 = Triangle(
            self.vertices[v_offset+1], 
            self.vertices[v_offset+5], 
            self.vertices[v_offset+6], 
            color
        )
        triangle11.face_type = "external"
        self.triangles.append(triangle11)
        
        triangle12 = Triangle(
            self.vertices[v_offset+1], 
            self.vertices[v_offset+6], 
            self.vertices[v_offset+2], 
            color
        )
        triangle12.face_type = "external"
        self.triangles.append(triangle12)
    
    def _add_smooth_arc_segment(self, front_center, back_center, radius, thickness, start_angle, end_angle, color):
        """Добавляет сегмент арки с улучшенным заполнением треугольников для плавного закрашивания"""
        # Создаем внутренний и внешний радиусы
        inner_radius = radius - thickness/2
        outer_radius = radius + thickness/2
        
        # Массивы для хранения индексов вершин
        vertices_front_inner = []
        vertices_front_outer = []
        vertices_back_inner = []
        vertices_back_outer = []
        
        # Вычисляем вершины дуги
        for i in range(self.segments + 1):
            # Интерполируем угол от начального до конечного
            angle = start_angle + (end_angle - start_angle) * (i / self.segments)
            angle_rad = math.radians(angle)
            
            # Вычисляем позицию на окружности
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            # Передняя внутренняя точка
            x_inner = front_center.x + inner_radius * cos_a
            y_inner = front_center.y
            z_inner = front_center.z + inner_radius * sin_a
            vertices_front_inner.append(len(self.vertices))
            self.vertices.append(Vector3D(x_inner, y_inner, z_inner))
            
            # Передняя внешняя точка
            x_outer = front_center.x + outer_radius * cos_a
            y_outer = front_center.y
            z_outer = front_center.z + outer_radius * sin_a
            vertices_front_outer.append(len(self.vertices))
            self.vertices.append(Vector3D(x_outer, y_outer, z_outer))
            
            # Задняя внутренняя точка
            vertices_back_inner.append(len(self.vertices))
            self.vertices.append(Vector3D(x_inner, back_center.y, z_inner))
            
            # Задняя внешняя точка
            vertices_back_outer.append(len(self.vertices))
            self.vertices.append(Vector3D(x_outer, back_center.y, z_outer))
        
        # Создаем треугольники и рёбра
        for i in range(self.segments):
            # Передняя грань (два треугольника) - внешняя
            front_triangle1 = Triangle(
                self.vertices[vertices_front_inner[i]],
                self.vertices[vertices_front_outer[i]],
                self.vertices[vertices_front_outer[i+1]],
                color
            )
            front_triangle1.face_type = "external"
            self.triangles.append(front_triangle1)
            
            front_triangle2 = Triangle(
                self.vertices[vertices_front_inner[i]],
                self.vertices[vertices_front_outer[i+1]],
                self.vertices[vertices_front_inner[i+1]],
                color
            )
            front_triangle2.face_type = "external"
            self.triangles.append(front_triangle2)
            
            # Задняя грань (два треугольника) - внешняя
            back_triangle1 = Triangle(
                self.vertices[vertices_back_inner[i]],
                self.vertices[vertices_back_outer[i+1]],
                self.vertices[vertices_back_outer[i]],
                color
            )
            back_triangle1.face_type = "external"
            self.triangles.append(back_triangle1)
            
            back_triangle2 = Triangle(
                self.vertices[vertices_back_inner[i]],
                self.vertices[vertices_back_inner[i+1]],
                self.vertices[vertices_back_outer[i+1]],
                color
            )
            back_triangle2.face_type = "external"
            self.triangles.append(back_triangle2)
            
            # Внешняя грань (два треугольника) - внешняя
            outer_triangle1 = Triangle(
                self.vertices[vertices_front_outer[i]],
                self.vertices[vertices_back_outer[i]],
                self.vertices[vertices_back_outer[i+1]],
                color
            )
            outer_triangle1.face_type = "external"
            self.triangles.append(outer_triangle1)
            
            outer_triangle2 = Triangle(
                self.vertices[vertices_front_outer[i]],
                self.vertices[vertices_back_outer[i+1]],
                self.vertices[vertices_front_outer[i+1]],
                color
            )
            outer_triangle2.face_type = "external"
            self.triangles.append(outer_triangle2)
            
            # Внутренняя грань (два треугольника) - внешняя
            inner_triangle1 = Triangle(
                self.vertices[vertices_front_inner[i]],
                self.vertices[vertices_back_inner[i+1]],
                self.vertices[vertices_back_inner[i]],
                color
            )
            inner_triangle1.face_type = "external"
            self.triangles.append(inner_triangle1)
            
            inner_triangle2 = Triangle(
                self.vertices[vertices_front_inner[i]],
                self.vertices[vertices_front_inner[i+1]],
                self.vertices[vertices_back_inner[i+1]],
                color
            )
            inner_triangle2.face_type = "external"
            self.triangles.append(inner_triangle2)
            
            # Добавляем рёбра для каркасного режима
            self.edges.append((vertices_front_inner[i], vertices_front_inner[i+1]))
            self.edges.append((vertices_front_outer[i], vertices_front_outer[i+1]))
            self.edges.append((vertices_back_inner[i], vertices_back_inner[i+1]))
            self.edges.append((vertices_back_outer[i], vertices_back_outer[i+1]))
            
            # Соединяющие рёбра между передней и задней гранями (боковые грани)
            # Добавляем их для всех сегментов, чтобы обеспечить полное заполнение
            self.edges.append((vertices_front_inner[i], vertices_back_inner[i]))
            self.edges.append((vertices_front_outer[i], vertices_back_outer[i]))
            
            # Радиальные рёбра для лучшей визуализации
            if i % 2 == 0:  # Добавляем больше радиальных рёбер для плавности
                self.edges.append((vertices_front_inner[i], vertices_front_outer[i]))
                self.edges.append((vertices_back_inner[i], vertices_back_outer[i]))
        
        # Добавляем последние соединяющие рёбра для замыкания формы
        last_idx = self.segments
        self.edges.append((vertices_front_inner[last_idx], vertices_back_inner[last_idx]))
        self.edges.append((vertices_front_outer[last_idx], vertices_back_outer[last_idx]))
        
        # Добавляем закрывающие треугольники на концах дуги (крышки)
        
        # Вычисляем нормали для крышек
        start_angle_rad = math.radians(start_angle)
        end_angle_rad = math.radians(end_angle)
        
        # Нормаль для начальной крышки - направлена в сторону, противоположную началу дуги
        start_normal_x = -math.cos(start_angle_rad)
        start_normal_z = -math.sin(start_angle_rad)
        
        # Нормаль для конечной крышки - направлена в сторону конца дуги
        end_normal_x = math.cos(end_angle_rad)
        end_normal_z = math.sin(end_angle_rad)
        
        # Начальная крышка (соединяет внутренний и внешний радиусы на начале дуги)
        start_front_triangle = Triangle(
            self.vertices[vertices_front_inner[0]],
            self.vertices[vertices_front_outer[0]],
            self.vertices[vertices_back_inner[0]],
            color
        )
        start_front_triangle.face_type = "external"
        start_front_triangle.normal = Vector3D(start_normal_x, 0, start_normal_z)  # Устанавливаем нормаль вручную
        self.triangles.append(start_front_triangle)
        
        start_back_triangle = Triangle(
            self.vertices[vertices_back_inner[0]],
            self.vertices[vertices_front_outer[0]],
            self.vertices[vertices_back_outer[0]],
            color
        )
        start_back_triangle.face_type = "external"
        start_back_triangle.normal = Vector3D(start_normal_x, 0, start_normal_z)  # Устанавливаем нормаль вручную
        self.triangles.append(start_back_triangle)
        
        # Добавляем рёбра для каркасного отображения начальной крышки
        self.edges.append((vertices_front_inner[0], vertices_front_outer[0]))
        self.edges.append((vertices_back_inner[0], vertices_back_outer[0]))
        
        # Конечная крышка (соединяет внутренний и внешний радиусы на конце дуги)
        end_idx = self.segments
        
        end_front_triangle = Triangle(
            self.vertices[vertices_front_inner[end_idx]],
            self.vertices[vertices_back_inner[end_idx]],
            self.vertices[vertices_front_outer[end_idx]],
            color
        )
        end_front_triangle.face_type = "external"
        end_front_triangle.normal = Vector3D(end_normal_x, 0, end_normal_z)  # Устанавливаем нормаль вручную
        self.triangles.append(end_front_triangle)
        
        end_back_triangle = Triangle(
            self.vertices[vertices_back_inner[end_idx]],
            self.vertices[vertices_back_outer[end_idx]],
            self.vertices[vertices_front_outer[end_idx]],
            color
        )
        end_back_triangle.face_type = "external"
        end_back_triangle.normal = Vector3D(end_normal_x, 0, end_normal_z)  # Устанавливаем нормаль вручную
        self.triangles.append(end_back_triangle)
        
        # Добавляем рёбра для каркасного отображения конечной крышки
        self.edges.append((vertices_front_inner[end_idx], vertices_front_outer[end_idx]))
        self.edges.append((vertices_back_inner[end_idx], vertices_back_outer[end_idx]))
    
    def set_dimensions(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.update_geometry()
    
    def update_transform(self):
        # Создаем матрицы трансформации
        self.transform_matrix = Matrix4x4.identity()
        
        # Порядок применения трансформаций для вращения относительно мировых координат:
        # 1. Применяем перемещение
        translation_matrix = Matrix4x4.translate(self.position.x, self.position.y, self.position.z)
        
        # 2. Применяем вращения относительно мировых осей
        rotation_x = Matrix4x4.rotation_around_x(self.rotation.x)
        rotation_y = Matrix4x4.rotation_around_y(self.rotation.y)
        rotation_z = Matrix4x4.rotation_around_z(self.rotation.z)
        
        # 3. Применяем отражения
        reflection_matrix = Matrix4x4.identity()
        if self.x_flipped:
            reflection_matrix = reflection_matrix @ Matrix4x4.reflection_x()
        if self.y_flipped:
            reflection_matrix = reflection_matrix @ Matrix4x4.reflection_y()
        if self.z_flipped:
            reflection_matrix = reflection_matrix @ Matrix4x4.reflection_z()
        
        # 4. Применяем масштабирование
        scale_matrix = Matrix4x4.scale(self.scale_factors.x, self.scale_factors.y, self.scale_factors.z)
        
        # Объединяем все трансформации в правильном порядке
        # Для вращения относительно мировых координат сначала применяем масштабирование и отражение к объекту,
        # затем вращаем относительно мировых осей, и в конце перемещаем
        self.transform_matrix = translation_matrix @ rotation_z @ rotation_y @ rotation_x @ reflection_matrix @ scale_matrix
    
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
        """Проецирует вершину на плоскость экрана"""
        # Преобразуем вершину в однородные координаты
        v_homo = np.array([vertex.x, vertex.y, vertex.z, 1.0], dtype=np.float64)
        
        # Применяем модельную матрицу для трансформации объекта в мировое пространство
        v_world = model_matrix @ v_homo
        
        # Применяем матрицу вида для перехода в пространство камеры
        v_view = view_matrix @ v_world
        
        # Применяем проекционную матрицу для перехода в нормализованное пространство устройства
        v_clip = projection_matrix @ v_view
        
        # Выполняем деление перспективы для получения нормализованных координат устройства (NDC)
        if abs(v_clip[3]) > 1e-6:
            v_ndc = v_clip / v_clip[3]
        else:
            v_ndc = v_clip
        
        # Преобразуем нормализованные координаты в экранные координаты
        screen_x = (v_ndc[0] + 1.0) * 0.5 * self.canvas_width
        screen_y = (1.0 - v_ndc[1]) * 0.5 * self.canvas_height  # Инвертируем Y-координату
        
        # Сохраняем преобразованную Z-координату для Z-буфера
        # Нормализуем ее к диапазону [0, 1], где 0 - ближняя плоскость, 1 - дальняя
        screen_z = v_ndc[2]
        
        return [screen_x, screen_y, screen_z]

class ZBuffer:
    def __init__(self, width, height):
        """Инициализация Z-буфера"""
        self.width = width
        self.height = height
        self.buffer = np.full((height, width), np.inf, dtype=np.float64)  # Буфер глубины (z-значения)
        self.color_buffer = np.zeros((height, width, 4), dtype=np.uint8)  # Цветовой буфер (RGBA)
    
    def clear_buffer(self):
        """Очистка Z-буфера и цветового буфера"""
        self.buffer.fill(np.inf)
        self.color_buffer.fill(0)
    
    def resize(self, width, height):
        """Изменение размера буферов"""
        self.width = width
        self.height = height
        self.buffer = np.full((height, width), np.inf, dtype=np.float64)
        self.color_buffer = np.zeros((height, width, 4), dtype=np.uint8)
    
    def set_pixel(self, x, y, z, color):
        """Устанавливает пиксель с учетом Z-буфера"""
        # Проверяем, находится ли пиксель в пределах буфера
        if 0 <= x < self.width and 0 <= y < self.height:
            # Если z-координата меньше, чем хранящаяся в буфере, значит пиксель ближе к камере
            if z < self.buffer[y, x]:
                # Обновляем Z-буфер
                self.buffer[y, x] = z
                
                # Обновляем цветовой буфер с полной непрозрачностью для объемного эффекта
                self.color_buffer[y, x] = [color.red(), color.green(), color.blue(), 255]
                return True
        return False
    
    def fill_triangle(self, triangle, screen_coords, colors):
        """Закрашивание треугольника с использованием Z-буфера и попиксельного сканирования"""
        # Получаем вершины треугольника и их z-координаты
        v0, v1, v2 = screen_coords
        z0, z1, z2 = v0[2], v1[2], v2[2]  # Используем z из проекции, а не из исходной вершины
        
        # Проверка вырожденного треугольника (с нулевой площадью)
        if (v0[0] == v1[0] and v0[1] == v1[1]) or (v0[0] == v2[0] and v0[1] == v2[1]) or (v1[0] == v2[0] and v1[1] == v2[1]):
            return  # Пропускаем вырожденные треугольники
        
        # Проверяем, полностью ли треугольник перед или за плоскостью отсечения
        if (z0 > 1.0 and z1 > 1.0 and z2 > 1.0) or (z0 < -1.0 and z1 < -1.0 and z2 < -1.0):
            return  # За пределами области видимости
        
        # Находим границы треугольника на экране
        min_x = max(0, int(min(v0[0], v1[0], v2[0])))
        max_x = min(self.width - 1, int(max(v0[0], v1[0], v2[0])))
        min_y = max(0, int(min(v0[1], v1[1], v2[1])))
        max_y = min(self.height - 1, int(max(v0[1], v1[1], v2[1])))
        
        # Если треугольник находится за пределами экрана, пропускаем его
        if max_x < min_x or max_y < min_y:
            return
        
        # Извлекаем координаты вершин для удобства
        v0_x, v0_y = v0[0], v0[1]
        v1_x, v1_y = v1[0], v1[1]
        v2_x, v2_y = v2[0], v2[1]
        
        # Вычисляем площадь треугольника для барицентрических координат
        area = (v1_x - v0_x) * (v2_y - v0_y) - (v2_x - v0_x) * (v1_y - v0_y)
        
        # Если площадь слишком мала, пропускаем треугольник
        if abs(area) < 1e-6:
            return
        
        inv_area = 1.0 / area
        
        # Проходим по всем пикселям в ограничивающем прямоугольнике
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Вычисляем барицентрические координаты пикселя
                w0 = ((v1_x - x) * (v2_y - y) - (v2_x - x) * (v1_y - y)) * inv_area
                w1 = ((v2_x - x) * (v0_y - y) - (v0_x - x) * (v2_y - y)) * inv_area
                w2 = 1.0 - w0 - w1
                
                # Если точка внутри треугольника
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Интерполируем z-координату с перспективной коррекцией
                    if abs(z0) > 1e-6 and abs(z1) > 1e-6 and abs(z2) > 1e-6:
                        # Вычисляем обратные значения z для перспективной коррекции
                        inv_z0 = 1.0 / z0
                        inv_z1 = 1.0 / z1
                        inv_z2 = 1.0 / z2
                        
                        # Перспективно-корректные веса
                        perspective_w0 = w0 * inv_z0
                        perspective_w1 = w1 * inv_z1
                        perspective_w2 = w2 * inv_z2
                        
                        # Нормализуем веса
                        weight_sum = perspective_w0 + perspective_w1 + perspective_w2
                        if abs(weight_sum) > 1e-6:
                            inv_w_sum = 1.0 / weight_sum
                            perspective_w0 *= inv_w_sum
                            perspective_w1 *= inv_w_sum
                            perspective_w2 *= inv_w_sum
                            
                            # Расчет итоговой глубины
                            z = 1.0 / (perspective_w0 * inv_z0 + perspective_w1 * inv_z1 + perspective_w2 * inv_z2)
                        else:
                            # Линейная интерполяция при проблемах с весами
                            z = w0 * z0 + w1 * z1 + w2 * z2
                    else:
                        # Линейная интерполяция для малых z-значений
                        z = w0 * z0 + w1 * z1 + w2 * z2
                    
                    # Преобразуем z в диапазон [0, Infinity], где 0 - ближняя плоскость, Infinity - дальняя
                    # Это преобразование гарантирует, что z-буфер корректно обрабатывает видимость
                    buffer_z = (z + 1.0) / 2.0
                    
                    # Если пиксель ближе текущего в z-буфере
                    if buffer_z < self.buffer[y, x]:
                        # Интерполируем цвет
                        color0, color1, color2 = colors
                        
                        if color0 == color1 and color1 == color2:
                            # Оптимизация для одноцветных треугольников
                            final_color = color0
                        else:
                            # Перспективно-корректная интерполяция цвета
                            if abs(z0) > 1e-6 and abs(z1) > 1e-6 and abs(z2) > 1e-6 and abs(weight_sum) > 1e-6:
                                r = int(perspective_w0 * color0.red() + perspective_w1 * color1.red() + perspective_w2 * color2.red())
                                g = int(perspective_w0 * color0.green() + perspective_w1 * color1.green() + perspective_w2 * color2.green())
                                b = int(perspective_w0 * color0.blue() + perspective_w1 * color1.blue() + perspective_w2 * color2.blue())
                            else:
                                # Линейная интерполяция цвета
                                r = int(w0 * color0.red() + w1 * color1.red() + w2 * color2.red())
                                g = int(w0 * color0.green() + w1 * color1.green() + w2 * color2.green())
                                b = int(w0 * color0.blue() + w1 * color1.blue() + w2 * color2.blue())
                            
                            # Ограничиваем значения цветов
                            r = max(0, min(255, r))
                            g = max(0, min(255, g))
                            b = max(0, min(255, b))
                            
                            final_color = QColor(r, g, b, 255)  # Полная непрозрачность
                        
                        # Устанавливаем пиксель в буфер
                        self.buffer[y, x] = buffer_z
                        self.color_buffer[y, x] = [final_color.red(), final_color.green(), final_color.blue(), 255]
    
    def fill_triangle_phong(self, triangle, screen_coords, base_color, light_position, camera_position, light_color=None):
        """Закрашивание треугольника с моделью освещения Фонга и попиксельным расчетом освещения"""
        # Получаем вершины треугольника и их z-координаты
        v0, v1, v2 = screen_coords
        
        # Важно: используем z-координаты из экранных координат, которые уже прошли проекцию
        z0, z1, z2 = v0[2], v1[2], v2[2]
        
        # Проверка вырожденного треугольника
        if (v0[0] == v1[0] and v0[1] == v1[1]) or (v0[0] == v2[0] and v0[1] == v2[1]) or (v1[0] == v2[0] and v1[1] == v2[1]):
            return
        
        # Проверяем, полностью ли треугольник перед или за плоскостью отсечения
        if (z0 > 1.0 and z1 > 1.0 and z2 > 1.0) or (z0 < -1.0 and z1 < -1.0 and z2 < -1.0):
            return  # За пределами области видимости
        
        # Находим границы треугольника на экране
        min_x = max(0, int(min(v0[0], v1[0], v2[0])))
        max_x = min(self.width - 1, int(max(v0[0], v1[0], v2[0])))
        min_y = max(0, int(min(v0[1], v1[1], v2[1])))
        max_y = min(self.height - 1, int(max(v0[1], v1[1], v2[1])))
        
        # Если треугольник находится за пределами экрана, пропускаем его
        if max_x < min_x or max_y < min_y:
            return
        
        # Вычисляем нормали вершин, если еще не вычислены
        if triangle.vertex_normals[0] is None:
            triangle.calculate_vertex_normals()
        
        # Извлекаем нормали вершин
        n0, n1, n2 = triangle.vertex_normals
        
        # Извлекаем координаты вершин для удобства
        v0_x, v0_y = v0[0], v0[1]
        v1_x, v1_y = v1[0], v1[1]
        v2_x, v2_y = v2[0], v2[1]
        
        # Вычисляем площадь треугольника
        area = (v1_x - v0_x) * (v2_y - v0_y) - (v2_x - v0_x) * (v1_y - v0_y)
        
        # Если площадь слишком мала, пропускаем треугольник
        if abs(area) < 1e-6:
            return
        
        inv_area = 1.0 / area
        
        # Заранее получаем мировые координаты вершин
        world_v0 = triangle.vertices[0]
        world_v1 = triangle.vertices[1]
        world_v2 = triangle.vertices[2]
        
        # Заранее подготовим вектор к камере от вершин для оптимизации
        view_dir0 = Vector3D(
            camera_position.x - world_v0.x,
            camera_position.y - world_v0.y,
            camera_position.z - world_v0.z
        )
        view_dir1 = Vector3D(
            camera_position.x - world_v1.x,
            camera_position.y - world_v1.y,
            camera_position.z - world_v1.z
        )
        view_dir2 = Vector3D(
            camera_position.x - world_v2.x,
            camera_position.y - world_v2.y,
            camera_position.z - world_v2.z
        )
        
        # Нормализуем векторы взгляда
        self._normalize_vector(view_dir0)
        self._normalize_vector(view_dir1)
        self._normalize_vector(view_dir2)
        
        # Заранее подготовим векторы света для вершин
        light_dir0 = Vector3D(
            light_position.x - world_v0.x,
            light_position.y - world_v0.y,
            light_position.z - world_v0.z
        )
        light_dir1 = Vector3D(
            light_position.x - world_v1.x,
            light_position.y - world_v1.y,
            light_position.z - world_v1.z
        )
        light_dir2 = Vector3D(
            light_position.x - world_v2.x,
            light_position.y - world_v2.y,
            light_position.z - world_v2.z
        )
        
        # Нормализуем векторы света
        self._normalize_vector(light_dir0)
        self._normalize_vector(light_dir1)
        self._normalize_vector(light_dir2)
        
        # Фоновое освещение (амбиентная составляющая)
        ambient = 0.2
        
        # Коэффициенты для разных компонентов освещения
        diffuse_factor = 0.7
        specular_factor = 0.4
        specular_power = 30  # Степень для зеркального блеска
        
        # Проходим по всем пикселям в ограничивающем прямоугольнике
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Вычисляем барицентрические координаты пикселя
                w0 = ((v1_x - x) * (v2_y - y) - (v2_x - x) * (v1_y - y)) * inv_area
                w1 = ((v2_x - x) * (v0_y - y) - (v0_x - x) * (v2_y - y)) * inv_area
                w2 = 1.0 - w0 - w1
                
                # Если точка внутри треугольника
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Интерполируем z-координату с перспективной коррекцией
                    # Используем значения z из проекции (NDC), в диапазоне [-1, 1]
                    buffer_z = w0 * z0 + w1 * z1 + w2 * z2
                    
                    # Преобразуем z из диапазона [-1, 1] в диапазон [0, 1] для Z-буфера
                    # где 0 - ближняя плоскость, 1 - дальняя
                    normalized_z = (buffer_z + 1.0) / 2.0
                    
                    # Если пиксель ближе текущего в z-буфере
                    if normalized_z < self.buffer[y, x]:
                        # Запоминаем глубину текущего пикселя
                        self.buffer[y, x] = normalized_z
                        
                        # Интерполируем нормаль с перспективной коррекцией
                        nx = w0 * n0.x + w1 * n1.x + w2 * n2.x
                        ny = w0 * n0.y + w1 * n1.y + w2 * n2.y
                        nz = w0 * n0.z + w1 * n1.z + w2 * n2.z
                        
                        # Нормализуем интерполированную нормаль
                        n_length = math.sqrt(nx*nx + ny*ny + nz*nz)
                        if n_length > 1e-6:
                            nx /= n_length
                            ny /= n_length
                            nz /= n_length
                        
                        # Интерполируем вектор взгляда
                        view_x = w0 * view_dir0.x + w1 * view_dir1.x + w2 * view_dir2.x
                        view_y = w0 * view_dir0.y + w1 * view_dir1.y + w2 * view_dir2.y
                        view_z = w0 * view_dir0.z + w1 * view_dir1.z + w2 * view_dir2.z
                        
                        # Нормализуем интерполированный вектор взгляда
                        view_length = math.sqrt(view_x*view_x + view_y*view_y + view_z*view_z)
                        if view_length > 1e-6:
                            view_x /= view_length
                            view_y /= view_length
                            view_z /= view_length
                        
                        # Интерполируем вектор света
                        light_x = w0 * light_dir0.x + w1 * light_dir1.x + w2 * light_dir2.x
                        light_y = w0 * light_dir0.y + w1 * light_dir1.y + w2 * light_dir2.y
                        light_z = w0 * light_dir0.z + w1 * light_dir1.z + w2 * light_dir2.z
                        
                        # Нормализуем интерполированный вектор света
                        light_length = math.sqrt(light_x*light_x + light_y*light_y + light_z*light_z)
                        if light_length > 1e-6:
                            light_x /= light_length
                            light_y /= light_length
                            light_z /= light_length
                        
                        # Вычисляем диффузную составляющую
                        diffuse = max(0, nx * light_x + ny * light_y + nz * light_z)
                        
                        # Вычисляем вектор отражения
                        reflect_coeff = 2.0 * (nx * light_x + ny * light_y + nz * light_z)
                        reflect_x = nx * reflect_coeff - light_x
                        reflect_y = ny * reflect_coeff - light_y
                        reflect_z = nz * reflect_coeff - light_z
                        
                        # Нормализуем вектор отражения
                        reflect_length = math.sqrt(reflect_x*reflect_x + reflect_y*reflect_y + reflect_z*reflect_z)
                        if reflect_length > 1e-6:
                            reflect_x /= reflect_length
                            reflect_y /= reflect_length
                            reflect_z /= reflect_length
                        
                        # Вычисляем зеркальную составляющую
                        specular = max(0, reflect_x * view_x + reflect_y * view_y + reflect_z * view_z) ** specular_power
                        
                        # Итоговая интенсивность освещения
                        light_intensity = min(1.0, ambient + diffuse * diffuse_factor + specular * specular_factor)
                        
                        # Применяем освещение к базовому цвету
                        if light_color:
                            r = min(255, int(base_color.red() * light_intensity * light_color.red() / 255))
                            g = min(255, int(base_color.green() * light_intensity * light_color.green() / 255))
                            b = min(255, int(base_color.blue() * light_intensity * light_color.blue() / 255))
                        else:
                            r = min(255, int(base_color.red() * light_intensity))
                            g = min(255, int(base_color.green() * light_intensity))
                            b = min(255, int(base_color.blue() * light_intensity))
                        
                        # Устанавливаем пиксель в буфер
                        self.color_buffer[y, x] = [r, g, b, 255]  # Полная непрозрачность
    
    def _normalize_vector(self, vector):
        """Вспомогательный метод для нормализации вектора"""
        length = math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)
        if length > 1e-6:
            vector.x /= length
            vector.y /= length
            vector.z /= length
        return vector
    
    def render_to_image(self):
        """Создает QImage из текущего состояния цветового буфера"""
        image = QImage(self.width, self.height, QImage.Format_RGBA8888)
        
        # Заполняем изображение данными из цветового буфера
        for y in range(self.height):
            for x in range(self.width):
                # Если пиксель не был установлен (имеет бесконечную глубину), он остается прозрачным
                if self.buffer[y, x] < float('inf'):
                    c = self.color_buffer[y, x]
                    image.setPixelColor(x, y, QColor(c[0], c[1], c[2], c[3]))
                else:
                    # Полностью прозрачный пиксель для фона
                    image.setPixelColor(x, y, QColor(0, 0, 0, 0))
        
        return image

class Canvas3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        
        # Устанавливаем черный фон
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(0, 0, 0))
        self.setPalette(palette)
        
        # Инициализируем две буквы с размерами по умолчанию
        self.letter1 = Letter3D('Б', 100, 100, 20, QColor(0, 255, 0))  # Зеленый цвет
        self.letter1.position.x = -60  # Смещаем первую букву влево
        
        self.letter2 = Letter3D('З', 100, 100, 20, QColor(255, 255, 0))  # Желтый цвет
        self.letter2.position.x = 60   # Смещаем вторую букву вправо
        
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
        self.letter1.update_transform()
        self.letter2.update_transform()
        
        # Инициализация Z-буфера
        self.z_buffer = ZBuffer(self.width(), self.height())
        
        # Инициализация источника света
        self.light = Light()
        
        # Режим отображения: 'wireframe', 'flat', 'gouraud', 'phong'
        self.render_mode = 'flat'
    
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
    
    def set_render_mode(self, mode):
        """Установка режима отображения"""
        self.render_mode = mode
        self.update()
    
    def set_light_position(self, x, y, z):
        """Установка позиции источника света"""
        self.light.position = Vector3D(x, y, z)
        self.update()
    
    def resizeEvent(self, event):
        # При изменении размера обновляем соотношение сторон и рендерер
        self.camera.aspect_ratio = self.width() / max(self.height(), 1)
        self.camera.update_projection_matrix()
        self.renderer = Renderer(self.width(), self.height())
        
        # Обновляем размер Z-буфера
        self.z_buffer.resize(self.width(), self.height())
        
        super().resizeEvent(event)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Получаем матрицы преобразований
        view_matrix = self.camera.view_matrix
        projection_matrix = self.camera.projection_matrix
        
        # Очищаем Z-буфер перед отрисовкой
        self.z_buffer.clear_buffer()
        
        # Если режим отображения - каркасный, рисуем как раньше
        if self.render_mode == 'wireframe':
            # Отрисовка мировых осей координат (всегда)
            self.draw_world_axes(painter, view_matrix, projection_matrix)
            
            # Рисуем первую букву
            self.draw_letter_wireframe(painter, self.letter1, view_matrix, projection_matrix, QColor(0, 255, 0))
            
            # Рисуем вторую букву
            self.draw_letter_wireframe(painter, self.letter2, view_matrix, projection_matrix, QColor(255, 255, 0))
            
            # Отображаем источник света
            self.draw_light(painter, self.light, view_matrix, projection_matrix)
        else:
            # Используем Z-буфер для отрисовки c затенением
            self.draw_with_z_buffer(painter, view_matrix, projection_matrix)
            
            # Всегда рисуем оси и источник света поверх остального
            self.draw_world_axes(painter, view_matrix, projection_matrix)
            self.draw_light(painter, self.light, view_matrix, projection_matrix)
        
        # Завершаем работу с QPainter
        painter.end()
    
    def draw_with_z_buffer(self, painter, view_matrix, projection_matrix):
        """Отрисовка сцены с использованием Z-буфера и попиксельного рендеринга"""
        # Очищаем Z-буфер перед отрисовкой
        self.z_buffer.clear_buffer()
        
        # Создаем списки для всех треугольников сцены
        triangles = []
        
        # Обрабатываем треугольники первой буквы
        for triangle in self.letter1.triangles:
            # Трансформируем треугольник в мировое пространство
            transformed_triangle = triangle.transform(self.letter1.transform_matrix)
            # Вычисляем нормаль для определения видимости
            transformed_triangle.calculate_normal()
            # Проверяем, виден ли треугольник с позиции камеры
            if transformed_triangle.is_visible(self.camera.position):
                triangles.append((transformed_triangle, self.letter1.color))
        
        # Обрабатываем треугольники второй буквы
        for triangle in self.letter2.triangles:
            # Трансформируем треугольник в мировое пространство
            transformed_triangle = triangle.transform(self.letter2.transform_matrix)
            # Вычисляем нормаль для определения видимости
            transformed_triangle.calculate_normal()
            # Проверяем, виден ли треугольник с позиции камеры
            if transformed_triangle.is_visible(self.camera.position):
                triangles.append((transformed_triangle, self.letter2.color))
        
        # Сортировка треугольников по глубине (от дальних к ближним)
        # Это оптимизирует работу Z-буфера, так как ближние треугольники, которые 
        # перекрывают дальние, рисуются позже
        triangles.sort(key=lambda x: x[0].get_depth(), reverse=True)
        
        # Обработка всех треугольников
        for triangle, base_color in triangles:
            # Проецируем вершины треугольника на экран
            screen_coords = []
            for vertex in triangle.vertices:
                screen_coord = self.renderer.project_vertex(vertex, np.identity(4), view_matrix, projection_matrix)
                screen_coords.append(screen_coord)
            
            # Проверяем, находится ли треугольник позади камеры или вне области видимости
            if any(z > 1.0 or z < -1.0 for _, _, z in screen_coords):
                continue
            
            # Используем различные модели затенения в зависимости от выбранного режима
            if self.render_mode == 'flat':
                # Плоское затенение - просто используем базовый цвет объекта без учета освещения
                color = QColor(base_color.red(), base_color.green(), base_color.blue(), 255)  # Полная непрозрачность
                
                # Один цвет для всех вершин треугольника
                colors = [color, color, color]
                
                # Закрашиваем треугольник
                self.z_buffer.fill_triangle(triangle, screen_coords, colors)
            
            elif self.render_mode == 'gouraud':
                # Метод Гуро - интерполяция интенсивности освещения между вершинами
                triangle.calculate_vertex_intensity(self.light.position, self.camera.position, self.light.color)
                
                # Вычисляем цвета для каждой вершины
                colors = []
                light_color = self.light.color
                for i in range(3):
                    intensity = triangle.vertex_intensity[i]
                    r = min(255, int(base_color.red() * intensity * light_color.red() / 255))
                    g = min(255, int(base_color.green() * intensity * light_color.green() / 255))
                    b = min(255, int(base_color.blue() * intensity * light_color.blue() / 255))
                    colors.append(QColor(r, g, b, 255))  # Полная непрозрачность
                
                # Закрашиваем треугольник с интерполяцией цветов вершин
                self.z_buffer.fill_triangle(triangle, screen_coords, colors)
            
            elif self.render_mode == 'phong':
                # Метод Фонга - интерполяция нормалей и расчет освещения в каждой точке
                triangle.calculate_vertex_normals()
                
                # Создаем непрозрачный базовый цвет
                opaque_base_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 255)
                
                # Закрашиваем треугольник с моделью освещения Фонга
                self.z_buffer.fill_triangle_phong(
                    triangle, 
                    screen_coords, 
                    opaque_base_color, 
                    self.light.position, 
                    self.camera.position,
                    self.light.color
                )
        
        # Создаем изображение из Z-буфера и отображаем его
        image = self.z_buffer.render_to_image()
        painter.drawImage(0, 0, image)
    
    def draw_letter_wireframe(self, painter, letter, view_matrix, projection_matrix, color):
        """Отрисовка буквы в каркасном режиме"""
        # Настройка пера для рисования буквы
        painter.setPen(QPen(color, 2))
        
        # Получаем модельную матрицу для буквы
        model_matrix = letter.transform_matrix
        
        # Проецируем вершины буквы
        screen_vertices = []
        for vertex in letter.vertices:
            screen_vertex = self.renderer.project_vertex(vertex, model_matrix, view_matrix, projection_matrix)
            screen_vertices.append(screen_vertex)
        
        # Рисуем ребра буквы
        for edge in letter.edges:
            start = screen_vertices[edge[0]]
            end = screen_vertices[edge[1]]
            painter.drawLine(start[0], start[1], end[0], end[1])
            
        # Отрисовка центра локальных координат (точки вращения)
        # Создаем точку в начале координат объекта
        origin_point = Vector3D(0, 0, 0)
        screen_origin = self.renderer.project_vertex(origin_point, model_matrix, view_matrix, projection_matrix)
        
        # Рисуем точку как маленький круг
        painter.setBrush(QColor(255, 255, 255))  # Белый цвет для точки
        painter.setPen(QPen(Qt.white, 1))
        painter.drawEllipse(screen_origin[0] - 3, screen_origin[1] - 3, 6, 6)
    
    def draw_world_axes(self, painter, view_matrix, projection_matrix):
        """Отрисовка мировых осей координат"""
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
    
    def draw_light(self, painter, light, view_matrix, projection_matrix):
        """Отрисовка источника света"""
        # Проецируем позицию источника света, используя только матрицы вида и проекции
        # без применения дополнительных трансформаций
        light_position = Vector3D(light.position.x, light.position.y, light.position.z)
        screen_position = self.renderer.project_vertex(light_position, np.identity(4), view_matrix, projection_matrix)
        
        # Рисуем источник света как яркий круг
        painter.setBrush(QColor(255, 255, 200))  # Светло-желтый цвет для источника света
        painter.setPen(QPen(QColor(255, 255, 0), 1))
        painter.drawEllipse(screen_position[0] - 5, screen_position[1] - 5, 10, 10)
        
        # Добавляем подпись
        painter.drawText(screen_position[0] + 10, screen_position[1], "Свет")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("3D Буквы Б и З - Лабораторная работа №2")
        self.resize(1200, 700)
        
        # Создаем центральный виджет и его компоновку
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Создаем холст для отрисовки
        self.canvas = Canvas3D()
        main_layout.addWidget(self.canvas, 2)
        
        # Создаем вкладки для управления обеими буквами
        tabs = QTabWidget()
        main_layout.addWidget(tabs, 1)
        
        # Создаем панели управления для каждой буквы
        letter1_panel = self.create_letter_panel(1)
        letter2_panel = self.create_letter_panel(2)
        
        # Создаем вкладку для настройки отображения
        render_panel = self.create_render_panel()
        
        # Добавляем панели управления на вкладки
        tabs.addTab(letter1_panel, "Буква Б")
        tabs.addTab(letter2_panel, "Буква З")
        tabs.addTab(render_panel, "Отображение")
        
        # Создаем панель управления камерой
        camera_panel = self.create_camera_panel()
        main_layout.addWidget(camera_panel, 1)
    
    def create_letter_panel(self, letter_index):
        """Создает панель управления для указанной буквы"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Определяем, с какой буквой работаем
        letter_attr = f"letter{letter_index}"
        letter_obj = getattr(self.canvas, letter_attr)
        letter_type = letter_obj.letter_type
        
        # Группа для параметров буквы
        letter_group = QGroupBox(f"Параметры буквы {letter_type}")
        letter_layout = QVBoxLayout(letter_group)
        
        # Добавляем выбор цвета буквы через ColorDialog
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Цвет:"))
        
        # Создаем кнопку для выбора цвета
        color_btn = QPushButton("Выбрать цвет")
        # Устанавливаем текущий цвет как фон кнопки
        color_btn.setStyleSheet(f"background-color: rgb({letter_obj.color.red()}, {letter_obj.color.green()}, {letter_obj.color.blue()}); color: white;")
        # Сохраняем ссылку на кнопку, чтобы обновлять ее цвет
        setattr(self, f"color_btn_{letter_index}", color_btn)
        # Подключаем обработчик нажатия
        color_btn.clicked.connect(lambda: self.show_color_picker(letter_index))
        color_layout.addWidget(color_btn)
        
        letter_layout.addLayout(color_layout)
        
        # Размеры буквы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Ширина:"))
        width_spin = QSpinBox()
        width_spin.setRange(10, 300)
        width_spin.setValue(letter_obj.width)
        width_spin.valueChanged.connect(lambda value: self.update_letter_size(letter_index, "width", value))
        size_layout.addWidget(width_spin)
        letter_layout.addLayout(size_layout)
        
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Высота:"))
        height_spin = QSpinBox()
        height_spin.setRange(10, 300)
        height_spin.setValue(letter_obj.height)
        height_spin.valueChanged.connect(lambda value: self.update_letter_size(letter_index, "height", value))
        height_layout.addWidget(height_spin)
        letter_layout.addLayout(height_layout)
        
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Глубина:"))
        depth_spin = QSpinBox()
        depth_spin.setRange(5, 100)
        depth_spin.setValue(letter_obj.depth)
        depth_spin.valueChanged.connect(lambda value: self.update_letter_size(letter_index, "depth", value))
        depth_layout.addWidget(depth_spin)
        letter_layout.addLayout(depth_layout)
        
        # Кнопка автомасштабирования
        auto_scale_btn = QPushButton("Авто-масштабирование")
        auto_scale_btn.clicked.connect(lambda: self.auto_scale(letter_index))
        letter_layout.addWidget(auto_scale_btn)
        
        layout.addWidget(letter_group)
        
        # Группа для вращения
        rotation_group = QGroupBox("Вращение")
        rotation_layout = QVBoxLayout(rotation_group)
        
        # X-вращение
        rotation_layout.addWidget(QLabel("Вокруг оси X:"))
        x_rot_slider = QSlider(Qt.Horizontal)
        x_rot_slider.setRange(0, 360)
        x_rot_slider.setValue(int(letter_obj.rotation.x))
        x_rot_slider.valueChanged.connect(lambda value: self.rotate_letter(letter_index, "x", value))
        setattr(self, f"x_rot_slider_{letter_index}", x_rot_slider)
        rotation_layout.addWidget(x_rot_slider)
        
        # Y-вращение
        rotation_layout.addWidget(QLabel("Вокруг оси Y:"))
        y_rot_slider = QSlider(Qt.Horizontal)
        y_rot_slider.setRange(0, 360)
        y_rot_slider.setValue(int(letter_obj.rotation.y))
        y_rot_slider.valueChanged.connect(lambda value: self.rotate_letter(letter_index, "y", value))
        setattr(self, f"y_rot_slider_{letter_index}", y_rot_slider)
        rotation_layout.addWidget(y_rot_slider)
        
        # Z-вращение
        rotation_layout.addWidget(QLabel("Вокруг оси Z:"))
        z_rot_slider = QSlider(Qt.Horizontal)
        z_rot_slider.setRange(0, 360)
        z_rot_slider.setValue(int(letter_obj.rotation.z))
        z_rot_slider.valueChanged.connect(lambda value: self.rotate_letter(letter_index, "z", value))
        setattr(self, f"z_rot_slider_{letter_index}", z_rot_slider)
        rotation_layout.addWidget(z_rot_slider)
        
        layout.addWidget(rotation_group)
        
        # Группа для перемещения
        translation_group = QGroupBox("Перемещение")
        translation_layout = QVBoxLayout(translation_group)
        
        # X-перемещение
        translation_layout.addWidget(QLabel("По оси X:"))
        x_trans_slider = QSlider(Qt.Horizontal)
        x_trans_slider.setRange(-200, 200)
        x_trans_slider.setValue(int(letter_obj.position.x))
        x_trans_slider.valueChanged.connect(lambda value: self.translate_letter(letter_index, "x", value))
        setattr(self, f"x_trans_slider_{letter_index}", x_trans_slider)
        translation_layout.addWidget(x_trans_slider)
        
        # Y-перемещение
        translation_layout.addWidget(QLabel("По оси Y:"))
        y_trans_slider = QSlider(Qt.Horizontal)
        y_trans_slider.setRange(-200, 200)
        y_trans_slider.setValue(int(letter_obj.position.y))
        y_trans_slider.valueChanged.connect(lambda value: self.translate_letter(letter_index, "y", value))
        setattr(self, f"y_trans_slider_{letter_index}", y_trans_slider)
        translation_layout.addWidget(y_trans_slider)
        
        # Z-перемещение
        translation_layout.addWidget(QLabel("По оси Z:"))
        z_trans_slider = QSlider(Qt.Horizontal)
        z_trans_slider.setRange(-200, 200)
        z_trans_slider.setValue(int(letter_obj.position.z))
        z_trans_slider.valueChanged.connect(lambda value: self.translate_letter(letter_index, "z", value))
        setattr(self, f"z_trans_slider_{letter_index}", z_trans_slider)
        translation_layout.addWidget(z_trans_slider)
        
        layout.addWidget(translation_group)
        
        # Группа для отражения
        reflection_group = QGroupBox("Отражение")
        reflection_layout = QVBoxLayout(reflection_group)
        
        # Кнопки отражения
        flip_x_btn = QPushButton("Отразить по X")
        flip_x_btn.clicked.connect(lambda: self.flip_letter(letter_index, "x"))
        reflection_layout.addWidget(flip_x_btn)
        
        flip_y_btn = QPushButton("Отразить по Y")
        flip_y_btn.clicked.connect(lambda: self.flip_letter(letter_index, "y"))
        reflection_layout.addWidget(flip_y_btn)
        
        flip_z_btn = QPushButton("Отразить по Z")
        flip_z_btn.clicked.connect(lambda: self.flip_letter(letter_index, "z"))
        reflection_layout.addWidget(flip_z_btn)
        
        layout.addWidget(reflection_group)
        
        # Сохраняем спинбоксы размеров для обновления во время авто-масштабирования
        setattr(self, f"width_spin_{letter_index}", width_spin)
        setattr(self, f"height_spin_{letter_index}", height_spin)
        setattr(self, f"depth_spin_{letter_index}", depth_spin)
        
        return panel
    
    def create_render_panel(self):
        """Создает панель управления настройками отображения"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Группа режима отображения
        render_mode_group = QGroupBox("Режим отображения")
        render_mode_layout = QVBoxLayout(render_mode_group)
        
        # Выбор режима отображения
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Каркас", "Плоское закрашивание", "Метод Гуро", "Метод Фонга"])
        self.render_mode_combo.setCurrentIndex(1)  # Плоское закрашивание по умолчанию
        self.render_mode_combo.currentIndexChanged.connect(self.change_render_mode)
        render_mode_layout.addWidget(self.render_mode_combo)
        
        layout.addWidget(render_mode_group)
        
        # Группа для источника света
        light_group = QGroupBox("Источник света")
        light_layout = QVBoxLayout(light_group)
        
        # X-позиция источника света
        light_layout.addWidget(QLabel("Позиция X:"))
        self.light_x_slider = QSlider(Qt.Horizontal)
        self.light_x_slider.setRange(-500, 500)
        self.light_x_slider.setValue(int(self.canvas.light.position.x))
        self.light_x_slider.valueChanged.connect(self.update_light_position)
        light_layout.addWidget(self.light_x_slider)
        
        # Y-позиция источника света
        light_layout.addWidget(QLabel("Позиция Y:"))
        self.light_y_slider = QSlider(Qt.Horizontal)
        self.light_y_slider.setRange(-500, 500)
        self.light_y_slider.setValue(int(self.canvas.light.position.y))
        self.light_y_slider.valueChanged.connect(self.update_light_position)
        light_layout.addWidget(self.light_y_slider)
        
        # Z-позиция источника света
        light_layout.addWidget(QLabel("Позиция Z:"))
        self.light_z_slider = QSlider(Qt.Horizontal)
        self.light_z_slider.setRange(-500, 500)
        self.light_z_slider.setValue(int(self.canvas.light.position.z))
        self.light_z_slider.valueChanged.connect(self.update_light_position)
        light_layout.addWidget(self.light_z_slider)
        
        layout.addWidget(light_group)
        
        return panel
    
    def create_camera_panel(self):
        """Создает панель управления камерой"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
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
        
        # Масштаб обеих букв
        camera_layout.addWidget(QLabel("Масштаб (обе буквы):"))
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
        
        layout.addWidget(camera_group)
        
        return panel
    
    def update_letter_size(self, letter_index, dimension, value):
        """Обновляет размер указанной буквы"""
        letter = getattr(self.canvas, f"letter{letter_index}")
        
        # Обновляем указанное измерение
        setattr(letter, dimension, value)
        
        # Создаем новую букву с обновленными размерами
        new_letter = Letter3D(letter.letter_type, letter.width, letter.height, letter.depth)
        
        # Копируем текущие трансформации
        new_letter.position = letter.position
        new_letter.rotation = letter.rotation
        new_letter.scale_factors = letter.scale_factors
        new_letter.x_flipped = letter.x_flipped
        new_letter.y_flipped = letter.y_flipped
        new_letter.z_flipped = letter.z_flipped
        
        # Обновляем трансформацию и заменяем текущую букву
        new_letter.update_transform()
        setattr(self.canvas, f"letter{letter_index}", new_letter)
        
        # Обновляем холст
        self.canvas.update()
    
    def rotate_letter(self, letter_index, axis, value):
        """Вращает указанную букву вокруг указанной оси"""
        letter = getattr(self.canvas, f"letter{letter_index}")
        setattr(letter.rotation, axis, value)
        letter.update_transform()
        self.canvas.update()
    
    def translate_letter(self, letter_index, axis, value):
        """Перемещает указанную букву по указанной оси"""
        letter = getattr(self.canvas, f"letter{letter_index}")
        setattr(letter.position, axis, value)
        letter.update_transform()
        self.canvas.update()
    
    def flip_letter(self, letter_index, axis):
        """Отражает указанную букву относительно указанной оси"""
        letter = getattr(self.canvas, f"letter{letter_index}")
        if axis == "x":
            letter.flip_x()
        elif axis == "y":
            letter.flip_y()
        elif axis == "z":
            letter.flip_z()
        letter.update_transform()
        self.canvas.update()
    
    def auto_scale(self, letter_index):
        """Автоматически масштабирует указанную букву"""
        letter = getattr(self.canvas, f"letter{letter_index}")
        canvas_size = min(self.canvas.width(), self.canvas.height())
        max_dim = max(letter.width, letter.height, letter.depth)
        scale_factor = canvas_size / max_dim / 8  # Делим на 8 для двух букв
        
        # Обновляем размеры с сохранением пропорций
        new_width = int(letter.width * scale_factor)
        new_height = int(letter.height * scale_factor)
        new_depth = int(letter.depth * scale_factor)
        
        # Обновляем значения спинбоксов
        getattr(self, f"width_spin_{letter_index}").setValue(new_width)
        getattr(self, f"height_spin_{letter_index}").setValue(new_height)
        getattr(self, f"depth_spin_{letter_index}").setValue(new_depth)
    
    def change_render_mode(self, index):
        """Изменение режима отображения"""
        modes = ['wireframe', 'flat', 'gouraud', 'phong']
        if index < len(modes):
            self.canvas.set_render_mode(modes[index])
            self.canvas.update()
    
    def update_light_position(self):
        """Обновление позиции источника света"""
        # Получаем значения слайдеров
        x = self.light_x_slider.value()
        y = self.light_y_slider.value()
        z = self.light_z_slider.value()
        
        # Устанавливаем новую позицию источника света напрямую в мировых координатах
        self.canvas.set_light_position(x, y, z)
    
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
    
    def change_scale(self):
        # Обновляем масштаб обеих букв
        scale_factor = self.scale_slider.value() / 100
        
        # Обновляем масштаб для первой буквы
        self.canvas.letter1.scale_factors.x = scale_factor
        self.canvas.letter1.scale_factors.y = scale_factor
        self.canvas.letter1.scale_factors.z = scale_factor
        self.canvas.letter1.update_transform()
        
        # Обновляем масштаб для второй буквы
        self.canvas.letter2.scale_factors.x = scale_factor
        self.canvas.letter2.scale_factors.y = scale_factor
        self.canvas.letter2.scale_factors.z = scale_factor
        self.canvas.letter2.update_transform()
        
        self.canvas.update()
    
    def change_axes_length(self):
        # Изменение длины осей координат
        length = self.axes_length_slider.value()
        self.canvas.set_axes_length(length)
    
    def change_letter_color(self, letter_index, color):
        """Изменяет цвет указанной буквы"""
        letter = getattr(self.canvas, f"letter{letter_index}")
        letter.set_color(color)
        self.canvas.update()
    
    def show_color_picker(self, letter_index):
        """Отображает диалог выбора цвета и применяет выбранный цвет к букве"""
        # Получаем текущую букву
        letter = getattr(self.canvas, f"letter{letter_index}")
        
        # Создаем диалог выбора цвета с текущим цветом буквы
        color_dialog = QColorDialog(letter.color, self)
        color_dialog.setWindowTitle(f"Выбор цвета для буквы {letter.letter_type}")
        
        # Если пользователь выбрал цвет и нажал ОК
        if color_dialog.exec():
            # Получаем выбранный цвет
            selected_color = color_dialog.selectedColor()
            
            # Применяем новый цвет к букве
            self.change_letter_color(letter_index, selected_color)
            
            # Обновляем фон кнопки выбора цвета
            color_btn = getattr(self, f"color_btn_{letter_index}")
            color_btn.setStyleSheet(f"background-color: rgb({selected_color.red()}, {selected_color.green()}, {selected_color.blue()}); color: white;")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
