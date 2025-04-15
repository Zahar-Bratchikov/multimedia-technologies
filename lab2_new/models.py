import math
import numpy as np

class Point3D:
    """Класс для представления точки в трехмерном пространстве"""
    
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        return f"Point3D({self.x}, {self.y}, {self.z})"
    
    def to_array(self):
        """Преобразует точку в массив numpy"""
        return np.array([self.x, self.y, self.z, 1.0])
    
    @staticmethod
    def from_array(array):
        """Создает точку из массива numpy"""
        if len(array) < 4:
            # Если передан массив без компонента w, предполагаем, что это обычные координаты
            return Point3D(array[0], array[1], array[2])
            
        if array[3] != 0:
            return Point3D(array[0]/array[3], array[1]/array[3], array[2]/array[3])
        else:
            return Point3D(array[0], array[1], array[2])
    
    def distance_to(self, other):
        """Вычисляет расстояние до другой точки"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


class Vector3D:
    """Класс для представления вектора в трехмерном пространстве"""
    
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"
    
    @staticmethod
    def from_points(p1, p2):
        """Создает вектор из двух точек"""
        return Vector3D(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
    
    def length(self):
        """Вычисляет длину вектора"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        """Нормализует вектор"""
        length = self.length()
        if length > 0:
            self.x /= length
            self.y /= length
            self.z /= length
        return self
    
    def dot(self, other):
        """Скалярное произведение с другим вектором"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        """Векторное произведение с другим вектором"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
        
    def reflect(self, normal):
        """Отражает вектор относительно нормали"""
        dot_product = self.dot(normal)
        return Vector3D(
            self.x - 2 * dot_product * normal.x,
            self.y - 2 * dot_product * normal.y,
            self.z - 2 * dot_product * normal.z
        )


class Face:
    """Класс для представления грани 3D объекта"""
    
    def __init__(self, vertices, color=(200, 200, 200)):
        self.vertices = vertices  # Индексы вершин
        self.color = color  # Цвет грани RGB
        self.normal = None  # Нормаль к грани
    
    def calculate_normal(self, vertices):
        """Вычисляет нормаль к грани"""
        if len(self.vertices) >= 3:
            p1 = vertices[self.vertices[0]]
            p2 = vertices[self.vertices[1]]
            p3 = vertices[self.vertices[2]]
            
            v1 = Vector3D.from_points(p1, p2)
            v2 = Vector3D.from_points(p1, p3)
            
            self.normal = v1.cross(v2)
            self.normal.normalize()
            
            return self.normal
        return Vector3D(0, 0, 1)
    
    def get_center(self, vertices):
        """Вычисляет центр грани"""
        x_sum = y_sum = z_sum = 0
        for idx in self.vertices:
            x_sum += vertices[idx].x
            y_sum += vertices[idx].y
            z_sum += vertices[idx].z
        
        n = len(self.vertices)
        if n == 0:
            return Point3D(0, 0, 0)
        
        return Point3D(x_sum / n, y_sum / n, z_sum / n)
    
    def get_depth(self, vertices):
        """Вычисляет глубину грани (для алгоритма художника)"""
        return self.get_center(vertices).z


class Camera:
    """Класс для представления камеры"""
    
    def __init__(self, position=None, target=None):
        self.position = position or Point3D(0, 0, 10)
        self.target = target or Point3D(0, 0, 0)
        self.up = Vector3D(0, 1, 0)
        self.rot_x = 0
        self.rot_y = 0
    
    def set_position(self, position):
        """Устанавливает позицию камеры"""
        self.position = position
    
    def set_rotation(self, rot_x, rot_y):
        """Устанавливает вращение камеры"""
        self.rot_x = rot_x
        self.rot_y = rot_y
        
        # Обновляем позицию камеры на основе углов вращения
        dist = self.position.distance_to(self.target)
        
        # Преобразование градусов в радианы
        rad_x = math.radians(rot_x)
        rad_y = math.radians(rot_y)
        
        # Вычисление новой позиции камеры
        x = dist * math.sin(rad_y) * math.cos(rad_x)
        y = dist * math.sin(rad_x)
        z = dist * math.cos(rad_y) * math.cos(rad_x)
        
        self.position = Point3D(x + self.target.x, y + self.target.y, z + self.target.z)
    
    def get_view_matrix(self):
        """Возвращает матрицу вида"""
        # Вычисление векторов для матрицы вида
        forward = Vector3D.from_points(self.position, self.target).normalize()
        right = forward.cross(self.up).normalize()
        up = right.cross(forward).normalize()
        
        # Создание матрицы вида
        view_matrix = np.array([
            [right.x, right.y, right.z, 0],
            [up.x, up.y, up.z, 0],
            [-forward.x, -forward.y, -forward.z, 0],
            [0, 0, 0, 1]
        ])
        
        # Создание матрицы перемещения
        translate_matrix = np.array([
            [1, 0, 0, -self.position.x],
            [0, 1, 0, -self.position.y],
            [0, 0, 1, -self.position.z],
            [0, 0, 0, 1]
        ])
        
        # Умножение матриц
        return np.matmul(view_matrix, translate_matrix)


class Light:
    """Класс для представления источника света"""
    
    def __init__(self, position=None, color=(255, 255, 255), intensity=1.0):
        self.position = position or Point3D(5, 5, 5)
        self.color = color
        self.intensity = intensity
    
    def set_position(self, position):
        """Устанавливает позицию источника света"""
        self.position = position
    
    def calculate_lighting(self, point, normal, camera_pos, material):
        """Вычисляет освещение для точки"""
        light_dir = Vector3D.from_points(point, self.position).normalize()
        view_dir = Vector3D.from_points(point, camera_pos).normalize()
        
        # Рассеянный свет (диффузное освещение)
        diffuse_intensity = max(0, normal.dot(light_dir)) * self.intensity
        
        # Зеркальный свет (по Фонгу)
        # Используем новый метод reflect для получения отраженного вектора
        # Инвертируем light_dir, так как для отражения нужен входящий вектор
        inverted_light_dir = Vector3D(-light_dir.x, -light_dir.y, -light_dir.z)
        reflect_dir = inverted_light_dir.reflect(normal).normalize()
        
        specular_intensity = pow(max(0, reflect_dir.dot(view_dir)), material.shininess) * self.intensity
        
        # Окончательный цвет
        r = min(255, int(material.ambient[0] + 
                         material.diffuse[0] * diffuse_intensity * self.color[0] / 255 +
                         material.specular[0] * specular_intensity * self.color[0] / 255))
        g = min(255, int(material.ambient[1] + 
                         material.diffuse[1] * diffuse_intensity * self.color[1] / 255 +
                         material.specular[1] * specular_intensity * self.color[1] / 255))
        b = min(255, int(material.ambient[2] + 
                         material.diffuse[2] * diffuse_intensity * self.color[2] / 255 +
                         material.specular[2] * specular_intensity * self.color[2] / 255))
        
        return (r, g, b)


class Material:
    """Класс для представления материала объекта"""
    
    def __init__(self, ambient=(30, 30, 30), diffuse=(200, 200, 200), 
                 specular=(255, 255, 255), shininess=32):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess 

class Object3D:
    """Базовый класс для представления 3D объекта"""
    
    def __init__(self, position=None):
        self.position = position or Point3D(0, 0, 0)
        self.vertices = []  # Точки
        self.faces = []  # Грани
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.material = Material()
        
        # Кэширование преобразованных вершин
        self.transformed_vertices = []
        self.vertex_normals = []
    
    def set_position(self, position):
        """Устанавливает позицию объекта"""
        self.position = position
    
    def set_rotation(self, x, y, z):
        """Устанавливает вращение объекта"""
        self.rotation_x = x
        self.rotation_y = y
        self.rotation_z = z
    
    def set_dimensions(self, height, width, depth):
        """Устанавливает размеры объекта"""
        # Должен быть переопределен в производных классах
        pass
    
    def get_translation_matrix(self):
        """Возвращает матрицу перемещения объекта"""
        return np.array([
            [1, 0, 0, self.position.x],
            [0, 1, 0, self.position.y],
            [0, 0, 1, self.position.z],
            [0, 0, 0, 1]
        ])
    
    def get_rotation_matrix(self):
        """Возвращает матрицу вращения объекта"""
        # Преобразование градусов в радианы
        rad_x = math.radians(self.rotation_x)
        rad_y = math.radians(self.rotation_y)
        rad_z = math.radians(self.rotation_z)
        
        # Матрицы вращения вокруг осей
        rot_x = np.array([
            [1, 0, 0, 0],
            [0, math.cos(rad_x), -math.sin(rad_x), 0],
            [0, math.sin(rad_x), math.cos(rad_x), 0],
            [0, 0, 0, 1]
        ])
        
        rot_y = np.array([
            [math.cos(rad_y), 0, math.sin(rad_y), 0],
            [0, 1, 0, 0],
            [-math.sin(rad_y), 0, math.cos(rad_y), 0],
            [0, 0, 0, 1]
        ])
        
        rot_z = np.array([
            [math.cos(rad_z), -math.sin(rad_z), 0, 0],
            [math.sin(rad_z), math.cos(rad_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Перемножаем матрицы вращения
        return np.matmul(np.matmul(rot_x, rot_y), rot_z)
    
    def get_model_matrix(self):
        """Возвращает модельную матрицу объекта"""
        rotation = self.get_rotation_matrix()
        translation = self.get_translation_matrix()
        return np.matmul(translation, rotation)
    
    def transform_vertices(self, view_matrix, projection_matrix, width, height):
        """Преобразует вершины объекта из мировых координат в экранные"""
        self.transformed_vertices = []
        model_matrix = self.get_model_matrix()
        
        for vertex in self.vertices:
            # Преобразование в однородные координаты
            v = vertex.to_array()
            
            # Применение модельной матрицы
            v = np.matmul(model_matrix, v)
            
            # Применение матрицы вида
            v = np.matmul(view_matrix, v)
            
            # Применение матрицы проекции
            v = np.matmul(projection_matrix, v)
            
            # Нормализация координат
            if v[3] != 0:
                v = v / v[3]
            
            # Преобразование в экранные координаты
            x = int((v[0] + 1) * width / 2)
            y = int((1 - v[1]) * height / 2)
            z = v[2]  # Z-буфер
            
            self.transformed_vertices.append((x, y, z))
        
        # Вычисление нормалей к вершинам (для освещения)
        self.calculate_vertex_normals()
    
    def calculate_vertex_normals(self):
        """Вычисляет нормали к вершинам для сглаженного освещения"""
        # Инициализация массива нормалей вершин
        self.vertex_normals = [Vector3D(0, 0, 0) for _ in range(len(self.vertices))]
        
        # Для каждой грани
        for face in self.faces:
            # Вычисляем нормаль к грани
            normal = face.calculate_normal(self.vertices)
            
            # Проверка наличия нормали и атрибута vertices у грани
            if normal is None or not hasattr(face, 'vertices'):
                continue
            
            # Добавляем нормаль к каждой вершине грани
            for vertex_idx in face.vertices:
                # Проверка корректности индекса вершины
                if 0 <= vertex_idx < len(self.vertex_normals):
                    self.vertex_normals[vertex_idx].x += normal.x
                    self.vertex_normals[vertex_idx].y += normal.y
                    self.vertex_normals[vertex_idx].z += normal.z
        
        # Нормализуем нормали вершин
        for i in range(len(self.vertex_normals)):
            self.vertex_normals[i].normalize()
    
    def apply_transform(self, transform_matrix):
        """Применяет матрицу трансформации ко всем вершинам объекта"""
        self.transformed_vertices = []
        
        for vertex in self.vertices:
            # Преобразование в однородные координаты
            v = vertex.to_array()
            
            # Применение матрицы трансформации
            v = np.matmul(transform_matrix, v)
            
            # Создание трансформированной точки
            transformed_point = Point3D.from_array(v)
            
            # Сохранение координат
            self.transformed_vertices.append((transformed_point.x, transformed_point.y, transformed_point.z))
            
        return self
    
    def scale(self, scale_x, scale_y, scale_z):
        """Масштабирует объект по трем осям"""
        for i in range(len(self.transformed_vertices)):
            x, y, z = self.transformed_vertices[i]
            self.transformed_vertices[i] = (x * scale_x, y * scale_y, z * scale_z)
            
        return self
    
    def move(self, dx, dy, dz):
        """Перемещает объект на указанное расстояние по каждой оси"""
        for i in range(len(self.transformed_vertices)):
            x, y, z = self.transformed_vertices[i]
            self.transformed_vertices[i] = (x + dx, y + dy, z + dz)
            
        return self
    
    def is_face_visible(self, face_idx, camera_position):
        """Проверяет, видима ли грань с заданной камеры (для алгоритма удаления невидимых поверхностей)"""
        if face_idx >= len(self.faces):
            return False
            
        face = self.faces[face_idx]
        face.calculate_normal(self.vertices)
        
        if face.normal is None:
            return True
        
        # Получение центра грани
        center = face.get_center(self.vertices)
        
        # Вектор от центра грани к камере
        view_vector = Vector3D.from_points(center, camera_position)
        
        # Проверка на нулевой вектор
        if view_vector.length() == 0:
            return True
            
        # Скалярное произведение нормали и вектора обзора
        dot_product = face.normal.dot(view_vector)
        
        # Если скалярное произведение > 0, грань видима
        return dot_product > 0


class LetterB(Object3D):
    """Класс для представления буквы Б"""
    
    def __init__(self, position=None, height=2.0, width=1.0, depth=0.5):
        super().__init__(position)
        self.height = height
        self.width = width
        self.depth = depth
        self.material = Material(
            ambient=(20, 20, 50),
            diffuse=(70, 70, 200),
            specular=(255, 255, 255),
            shininess=64
        )
        self.build()
    
    def set_dimensions(self, height, width, depth):
        """Устанавливает размеры буквы"""
        self.height = height
        self.width = width
        self.depth = depth
        self.build()
    
    def build(self):
        """Строит геометрию буквы Б"""
        # Очищаем массивы
        self.vertices = []
        self.faces = []
        
        # Параметры для построения
        h = self.height / 2  # Половина высоты
        w = self.width / 2   # Половина ширины
        d = self.depth / 2   # Половина глубины
        thickness = self.width * 0.2  # Толщина вертикальной линии
        
        # Вершины передней части (вертикальная линия)
        self.vertices.append(Point3D(-w, -h, d))  # 0
        self.vertices.append(Point3D(-w + thickness, -h, d))  # 1
        self.vertices.append(Point3D(-w + thickness, h, d))  # 2
        self.vertices.append(Point3D(-w, h, d))  # 3
        
        # Верхняя горизонтальная часть (передняя сторона)
        self.vertices.append(Point3D(-w + thickness, h, d))  # 4 (совпадает с 2)
        self.vertices.append(Point3D(w, h, d))  # 5
        self.vertices.append(Point3D(w, h - thickness, d))  # 6
        self.vertices.append(Point3D(-w + thickness, h - thickness, d))  # 7
        
        # Средняя горизонтальная часть (передняя сторона)
        top_mid = h * 0.3  # Положение верхней средней части
        bottom_mid = -h * 0.1  # Положение нижней средней части
        
        self.vertices.append(Point3D(-w + thickness, top_mid + thickness/2, d))  # 8
        self.vertices.append(Point3D(w, top_mid + thickness/2, d))  # 9
        self.vertices.append(Point3D(w, top_mid - thickness/2, d))  # 10
        self.vertices.append(Point3D(-w + thickness, top_mid - thickness/2, d))  # 11
        
        # Нижняя скругленная часть (передняя сторона)
        segments = 8  # Количество сегментов для дуги
        radius = (top_mid - bottom_mid) * 0.8  # Радиус закругления
        center_y = bottom_mid + radius  # Y-координата центра дуги
        
        # Сохраняем начальный индекс для дуги
        arc_start_idx = len(self.vertices)
        
        # Добавляем точки верхней части дуги
        for i in range(segments + 1):
            angle = (math.pi / 2) * (1 - i / segments)
            x = w - radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            if x > w:
                x = w
            self.vertices.append(Point3D(x, y, d))
        
        # Добавляем точки нижней части дуги
        for i in range(segments + 1):
            angle = (math.pi / 2) * (i / segments)
            x = w - radius * math.cos(angle)
            y = center_y - radius * math.sin(angle)
            if x > w:
                x = w
            self.vertices.append(Point3D(x, y, d))
        
        # Точка соединения дуги с вертикальной частью
        self.vertices.append(Point3D(-w + thickness, bottom_mid, d))  # Последняя точка дуги
        
        # Задняя сторона (зеркальные копии передних точек)
        num_front_vertices = len(self.vertices)
        
        for i in range(num_front_vertices):
            front_vertex = self.vertices[i]
            self.vertices.append(Point3D(front_vertex.x, front_vertex.y, -d))
            
        # Цвета для граней
        front_color = (70, 70, 200)  # Синий для передней стороны
        back_color = (50, 50, 150)   # Темно-синий для задней стороны
        side_color = (60, 60, 180)   # Средний синий для боковых сторон
            
        # Грани передней стороны
        
        # Вертикальная линия
        self.faces.append(Face([0, 1, 2, 3], color=front_color))
        
        # Верхняя горизонтальная линия
        self.faces.append(Face([4, 5, 6, 7], color=front_color))
        
        # Средняя горизонтальная линия
        self.faces.append(Face([8, 9, 10, 11], color=front_color))
        
        # Дуга (соединяем последовательные точки)
        arc_vertices = []
        
        # Добавляем точки дуги
        for i in range(arc_start_idx, len(self.vertices) - 1):
            arc_vertices.append(i)
        
        # Замыкаем контур
        arc_vertices.append(len(self.vertices) - 1)
        
        # Добавляем грань для дуги
        self.faces.append(Face(arc_vertices, color=front_color))
        
        # Грани задней стороны (с обратным порядком вершин)
        
        # Вертикальная линия
        self.faces.append(Face([num_front_vertices + 3, num_front_vertices + 2, num_front_vertices + 1, num_front_vertices + 0], color=back_color))
        
        # Верхняя горизонтальная линия
        self.faces.append(Face([num_front_vertices + 7, num_front_vertices + 6, num_front_vertices + 5, num_front_vertices + 4], color=back_color))
        
        # Средняя горизонтальная линия
        self.faces.append(Face([num_front_vertices + 11, num_front_vertices + 10, num_front_vertices + 9, num_front_vertices + 8], color=back_color))
        
        # Дуга (с обратным порядком вершин)
        back_arc_vertices = []
        
        # Добавляем точки задней дуги в обратном порядке
        for i in range(len(self.vertices) - 1, arc_start_idx + num_front_vertices - 1, -1):
            back_arc_vertices.append(i)
        
        # Добавляем грань для задней дуги
        self.faces.append(Face(back_arc_vertices, color=back_color))
        
        # Боковые грани
        
        # Соединяем переднюю и заднюю вертикальные линии
        self.faces.append(Face([0, 3, num_front_vertices + 3, num_front_vertices + 0], color=side_color))  # Левая сторона
        self.faces.append(Face([1, num_front_vertices + 1, num_front_vertices + 2, 2], color=side_color))  # Правая сторона
        self.faces.append(Face([0, num_front_vertices + 0, num_front_vertices + 1, 1], color=side_color))  # Нижняя сторона
        
        # Соединяем переднюю и заднюю верхние горизонтальные линии
        self.faces.append(Face([4, 7, num_front_vertices + 7, num_front_vertices + 4], color=side_color))  # Левая сторона
        self.faces.append(Face([5, num_front_vertices + 5, num_front_vertices + 6, 6], color=side_color))  # Правая сторона
        self.faces.append(Face([6, num_front_vertices + 6, num_front_vertices + 7, 7], color=side_color))  # Нижняя сторона
        self.faces.append(Face([4, num_front_vertices + 4, num_front_vertices + 5, 5], color=side_color))  # Верхняя сторона
        
        # Соединяем переднюю и заднюю средние горизонтальные линии
        self.faces.append(Face([8, 11, num_front_vertices + 11, num_front_vertices + 8], color=side_color))  # Левая сторона
        self.faces.append(Face([9, num_front_vertices + 9, num_front_vertices + 10, 10], color=side_color))  # Правая сторона
        self.faces.append(Face([10, num_front_vertices + 10, num_front_vertices + 11, 11], color=side_color))  # Нижняя сторона
        self.faces.append(Face([8, num_front_vertices + 8, num_front_vertices + 9, 9], color=side_color))  # Верхняя сторона
        
        # Соединяем точки дуги (создаем боковые грани)
        for i in range(arc_start_idx, len(self.vertices) - num_front_vertices - 1):
            self.faces.append(Face([i, i + 1, i + 1 + num_front_vertices, i + num_front_vertices], color=side_color))
        
        # Замыкающая грань дуги
        self.faces.append(Face([len(self.vertices) - 1, num_front_vertices - 1, 
                                num_front_vertices * 2 - 1, num_front_vertices + len(self.vertices) - 1], color=side_color))


class LetterZ(Object3D):
    """Класс для представления буквы З"""
    
    def __init__(self, position=None, height=2.0, width=1.0, depth=0.5):
        super().__init__(position)
        self.height = height
        self.width = width
        self.depth = depth
        self.material = Material(
            ambient=(50, 20, 20),
            diffuse=(200, 70, 70),
            specular=(255, 255, 255),
            shininess=64
        )
        self.build()
    
    def set_dimensions(self, height, width, depth):
        """Устанавливает размеры буквы"""
        self.height = height
        self.width = width
        self.depth = depth
        self.build()
    
    def build(self):
        """Строит геометрию буквы З"""
        # Очищаем массивы
        self.vertices = []
        self.faces = []
        
        # Параметры для построения
        h = self.height / 2  # Половина высоты
        w = self.width / 2   # Половина ширины
        d = self.depth / 2   # Половина глубины
        thickness = self.width * 0.2  # Толщина горизонтальных линий
        
        # Цвета для граней
        front_color = (200, 70, 70)  # Красный для передней стороны
        back_color = (150, 50, 50)   # Темно-красный для задней стороны
        side_color = (180, 60, 60)   # Средний красный для боковых сторон
        
        # Верхняя горизонтальная часть (передняя сторона)
        self.vertices.append(Point3D(-w, h, d))  # 0
        self.vertices.append(Point3D(w, h, d))  # 1
        self.vertices.append(Point3D(w, h - thickness, d))  # 2
        self.vertices.append(Point3D(-w, h - thickness, d))  # 3
        
        # Параметры для верхней и нижней дуг
        segments = 10  # Количество сегментов для каждой дуги
        radius = self.width * 0.4  # Радиус закругления
        
        # Верхняя дуга (передняя сторона)
        upper_arc_center_y = h/3
        upper_arc_center_x = w - radius
        upper_arc_start_idx = len(self.vertices)
        
        # Добавляем точки верхней дуги сверху вниз
        for i in range(segments + 1):
            angle = math.pi * (i / segments)
            x = upper_arc_center_x + radius * math.cos(angle)
            y = upper_arc_center_y + radius * math.sin(angle)
            self.vertices.append(Point3D(x, y, d))
        
        # Нижняя дуга (передняя сторона)
        lower_arc_center_y = -h/3
        lower_arc_center_x = w - radius
        lower_arc_start_idx = len(self.vertices)
        
        # Добавляем точки нижней дуги сверху вниз
        for i in range(segments + 1):
            angle = math.pi * (i / segments)
            x = lower_arc_center_x + radius * math.cos(angle)
            y = lower_arc_center_y + radius * math.sin(angle)
            self.vertices.append(Point3D(x, y, d))
        
        # Нижняя горизонтальная часть (передняя сторона)
        self.vertices.append(Point3D(-w, -h + thickness, d))  # Первая точка нижней части
        self.vertices.append(Point3D(w, -h + thickness, d))
        self.vertices.append(Point3D(w, -h, d))
        self.vertices.append(Point3D(-w, -h, d))  # Последняя точка нижней части
        
        # Запоминаем количество передних вершин
        num_front_vertices = len(self.vertices)
        
        # Создаем задние вершины (зеркальные копии передних)
        for i in range(num_front_vertices):
            front_vertex = self.vertices[i]
            self.vertices.append(Point3D(front_vertex.x, front_vertex.y, -d))
        
        # Грани передней стороны
        
        # Верхняя горизонтальная часть
        self.faces.append(Face([0, 1, 2, 3], color=front_color))
        
        # Верхняя дуга
        upper_arc_vertices = []
        for i in range(upper_arc_start_idx, lower_arc_start_idx):
            upper_arc_vertices.append(i)
        self.faces.append(Face(upper_arc_vertices, color=front_color))
        
        # Нижняя дуга
        lower_arc_vertices = []
        for i in range(lower_arc_start_idx, num_front_vertices - 4):
            lower_arc_vertices.append(i)
        self.faces.append(Face(lower_arc_vertices, color=front_color))
        
        # Нижняя горизонтальная часть
        self.faces.append(Face([num_front_vertices - 4, num_front_vertices - 3, num_front_vertices - 2, num_front_vertices - 1], color=front_color))
        
        # Грани задней стороны (с обратным порядком вершин)
        
        # Верхняя горизонтальная часть
        self.faces.append(Face([num_front_vertices + 3, num_front_vertices + 2, num_front_vertices + 1, num_front_vertices + 0], color=back_color))
        
        # Верхняя дуга
        back_upper_arc_vertices = []
        for i in range(num_front_vertices + upper_arc_start_idx, num_front_vertices + lower_arc_start_idx):
            back_upper_arc_vertices.append(i)
        self.faces.append(Face(back_upper_arc_vertices, color=back_color))
        
        # Нижняя дуга
        back_lower_arc_vertices = []
        for i in range(num_front_vertices + lower_arc_start_idx, num_front_vertices * 2 - 4):
            back_lower_arc_vertices.append(i)
        self.faces.append(Face(back_lower_arc_vertices, color=back_color))
        
        # Нижняя горизонтальная часть
        self.faces.append(Face([num_front_vertices * 2 - 4, num_front_vertices * 2 - 3, num_front_vertices * 2 - 2, num_front_vertices * 2 - 1], color=back_color))
        
        # Боковые грани
        
        # Верхняя горизонтальная часть
        self.faces.append(Face([0, 3, num_front_vertices + 3, num_front_vertices + 0], color=side_color))  # Левая сторона
        self.faces.append(Face([1, num_front_vertices + 1, num_front_vertices + 2, 2], color=side_color))  # Правая сторона
        self.faces.append(Face([2, num_front_vertices + 2, num_front_vertices + 3, 3], color=side_color))  # Нижняя сторона
        self.faces.append(Face([0, num_front_vertices + 0, num_front_vertices + 1, 1], color=side_color))  # Верхняя сторона
        
        # Боковые грани для верхней дуги
        for i in range(upper_arc_start_idx, lower_arc_start_idx - 1):
            self.faces.append(Face([i, i + 1, i + 1 + num_front_vertices, i + num_front_vertices], color=side_color))
        
        # Боковые грани для нижней дуги
        for i in range(lower_arc_start_idx, num_front_vertices - 5):
            self.faces.append(Face([i, i + 1, i + 1 + num_front_vertices, i + num_front_vertices], color=side_color))
        
        # Нижняя горизонтальная часть
        self.faces.append(Face([num_front_vertices - 4, num_front_vertices - 1, num_front_vertices * 2 - 1, num_front_vertices * 2 - 4], color=side_color))  # Левая сторона
        self.faces.append(Face([num_front_vertices - 3, num_front_vertices * 2 - 3, num_front_vertices * 2 - 2, num_front_vertices - 2], color=side_color))  # Правая сторона
        self.faces.append(Face([num_front_vertices - 2, num_front_vertices * 2 - 2, num_front_vertices * 2 - 1, num_front_vertices - 1], color=side_color))  # Нижняя сторона
        self.faces.append(Face([num_front_vertices - 4, num_front_vertices * 2 - 4, num_front_vertices * 2 - 3, num_front_vertices - 3], color=side_color))  # Верхняя сторона
        
        # Соединение верхней дуги с верхней горизонтальной частью
        self.faces.append(Face([3, upper_arc_start_idx, upper_arc_start_idx + num_front_vertices, 3 + num_front_vertices], color=side_color))
        
        # Соединение верхней дуги с нижней дугой
        self.faces.append(Face([lower_arc_start_idx - 1, lower_arc_start_idx, lower_arc_start_idx + num_front_vertices, lower_arc_start_idx - 1 + num_front_vertices], color=side_color))
        
        # Соединение нижней дуги с нижней горизонтальной частью
        self.faces.append(Face([num_front_vertices - 5, num_front_vertices - 4, num_front_vertices * 2 - 4, num_front_vertices - 5 + num_front_vertices], color=side_color)) 

class CoordinateAxes(Object3D):
    """Класс для представления осей координат в 3D пространстве"""
    
    def __init__(self, length=5.0):
        super().__init__()
        self.length = length
        self.build()
    
    def build(self):
        """Строит геометрию осей координат"""
        # Очищаем массивы
        self.vertices = []
        self.faces = []
        
        # Начало координат
        self.vertices.append(Point3D(0, 0, 0))
        
        # Конечные точки осей
        self.vertices.append(Point3D(self.length, 0, 0))  # X-ось
        self.vertices.append(Point3D(0, self.length, 0))  # Y-ось
        self.vertices.append(Point3D(0, 0, self.length))  # Z-ось
        
        # Грани (линии для каждой оси)
        self.faces.append(Face([0, 1], color=(255, 0, 0)))  # X-ось (красная)
        self.faces.append(Face([0, 2], color=(0, 255, 0)))  # Y-ось (зеленая)
        self.faces.append(Face([0, 3], color=(0, 0, 255)))  # Z-ось (синяя) 