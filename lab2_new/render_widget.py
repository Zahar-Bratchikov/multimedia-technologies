import math
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPoint, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QImage, QPixmap
import copy

from models import Point3D, Vector3D, Object3D, Face, CoordinateAxes

class RenderWidget(QWidget):
    """Виджет для отрисовки 3D объектов"""
    
    def __init__(self, letter_b, letter_z, camera, light, parent=None):
        super().__init__(parent)
        
        # Объекты для отрисовки
        self.letter_b = letter_b
        self.letter_z = letter_z
        
        # Камера и освещение
        self.camera = camera
        self.light = light
        
        # Параметры отображения
        self.render_mode = "wireframe"  # wireframe, hidden_surface, solid
        self.shading_mode = "flat"  # flat, gouraud, phong
        self.auto_scale = True
        
        # Параметры проекции
        self.fov = 60  # Угол обзора в градусах
        self.near = 0.1  # Ближняя плоскость отсечения
        self.far = 100.0  # Дальняя плоскость отсечения
        
        # Буферы для методов отображения
        self.z_buffer = None
        self.color_buffer = None
        
        # Минимальный размер виджета
        self.setMinimumSize(500, 500)
        
        # Настройка фокуса для обработки событий клавиатуры
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Обработчики для взаимодействия с мышью
        self.last_mouse_pos = QPoint()
        self.mouse_pressed = False
        
        # Добавляем оси координат
        self.coordinate_axes = self.create_coordinate_axes()
        
    def set_render_mode(self, mode):
        """Устанавливает режим отображения"""
        self.render_mode = mode
        
    def set_shading_mode(self, mode):
        """Устанавливает режим закрашивания"""
        self.shading_mode = mode
    
    def set_auto_scale(self, enabled):
        """Включает/выключает автомасштабирование"""
        self.auto_scale = enabled
    
    def paintEvent(self, event):
        """Обработчик события перерисовки"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Очистка фона
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Получение размеров виджета
        width = self.width()
        height = self.height()
        
        # Инициализация буферов при необходимости
        if self.z_buffer is None or self.z_buffer.shape[0] != height or self.z_buffer.shape[1] != width:
            self.z_buffer = np.ones((height, width)) * float('inf')
            self.color_buffer = QImage(width, height, QImage.Format_ARGB32)
            self.color_buffer.fill(Qt.black)
        
        # Очистка буферов
        self.z_buffer.fill(float('inf'))
        self.color_buffer.fill(Qt.black)
        
        # Получение матрицы проекции
        projection_matrix = self.get_projection_matrix()
        
        # Получение матрицы вида
        view_matrix = self.camera.get_view_matrix()
        
        # Преобразование координат объектов
        self.letter_b.transform_vertices(view_matrix, projection_matrix, width, height)
        self.letter_z.transform_vertices(view_matrix, projection_matrix, width, height)
        
        # Отрисовка объектов в зависимости от режима
        if self.render_mode == "wireframe":
            self.draw_wireframe(painter, self.letter_b)
            self.draw_wireframe(painter, self.letter_z)
        elif self.render_mode == "hidden_surface":
            self.draw_hidden_surface(painter, self.letter_b)
            self.draw_hidden_surface(painter, self.letter_z)
        elif self.render_mode == "solid":
            # Сначала очищаем буферы
            self.z_buffer.fill(float('inf'))
            self.color_buffer.fill(Qt.black)
            
            # Отрисовка с закрашиванием
            self.draw_solid(self.letter_b)
            self.draw_solid(self.letter_z)
            
            # Отображение результата
            painter.drawImage(0, 0, self.color_buffer)
        
        # Отрисовка источника света
        self.draw_light(painter, view_matrix, projection_matrix, width, height)
        
        # Рисуем координатные оси
        self.draw_coordinate_axes(painter)
        
        # Отображение информации о режиме
        self.draw_info(painter)
        
    def get_projection_matrix(self):
        """Возвращает матрицу проекции"""
        aspect = self.width() / self.height()
        f = 1.0 / math.tan(math.radians(self.fov / 2))
        
        # Перспективная проекционная матрица
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far + self.near) / (self.near - self.far), (2 * self.far * self.near) / (self.near - self.far)],
            [0, 0, -1, 0]
        ])
    
    def draw_wireframe(self, painter, obj):
        """Отрисовка каркасной модели"""
        # Настройка пера для отрисовки линий
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Для каждой грани объекта
        for face in obj.faces:
            # Для каждой пары соседних вершин грани
            for i in range(len(face.vertices)):
                # Индексы текущей и следующей вершины
                idx1 = face.vertices[i]
                idx2 = face.vertices[(i + 1) % len(face.vertices)]
                
                # Получение координат вершин
                p1 = obj.transformed_vertices[idx1]
                p2 = obj.transformed_vertices[idx2]
                
                # Отрисовка линии
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])
    
    def draw_hidden_surface(self, painter, obj):
        """Отрисовка модели с удалением невидимых поверхностей"""
        # Копируем грани и их индексы
        faces_to_draw = [(i, face) for i, face in enumerate(obj.faces)]
        
        # Сортировка граней по глубине (алгоритм художника)
        faces_to_draw.sort(key=lambda f: -self.get_face_depth(obj, f[0]))
        
        # Для каждой грани
        for face_idx, face in faces_to_draw:
            # Проверяем, видима ли грань
            if obj.is_face_visible(face_idx, self.camera.position):
                # Отрисовка видимой грани
                points = [QPoint(obj.transformed_vertices[idx][0], obj.transformed_vertices[idx][1]) 
                         for idx in face.vertices]
                
                # Устанавливаем цвет грани
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(*face.color)))
                
                # Рисуем многоугольник
                painter.drawPolygon(points)
                
                # Рисуем контур
                painter.setPen(QPen(QColor(0, 0, 0)))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(points)
    
    def get_face_depth(self, obj, face_idx):
        """Вычисляет среднюю глубину грани для сортировки"""
        face = obj.faces[face_idx]
        total_z = 0
        for vertex_idx in face.vertices:
            total_z += obj.transformed_vertices[vertex_idx][2]
        return total_z / len(face.vertices)
    
    def draw_solid(self, obj):
        """Отрисовка модели с закрашиванием поверхностей"""
        # Получаем размеры буфера
        width = self.width()
        height = self.height()
        
        # Копируем грани и их индексы
        faces_to_draw = [(i, face) for i, face in enumerate(obj.faces)]
        
        # Отрисовка каждой грани
        for face_idx, face in faces_to_draw:
            # Проверяем, видима ли грань
            if obj.is_face_visible(face_idx, self.camera.position):
                # Получаем вершины грани
                vertices = [obj.transformed_vertices[idx] for idx in face.vertices]
                
                # Вычисляем нормаль к грани
                face.calculate_normal(obj.vertices)
                
                if self.shading_mode == "flat":
                    # Монотонное закрашивание (вычисляем цвет один раз для всей грани)
                    face_center = face.get_center(obj.vertices)
                    face_color = self.light.calculate_lighting(
                        face_center, 
                        face.normal, 
                        self.camera.position, 
                        obj.material
                    )
                    
                    # Заливка грани одним цветом
                    self.fill_polygon(vertices, face_color)
                    
                elif self.shading_mode == "gouraud":
                    # Закрашивание по Гуро (интерполяция цветов вершин)
                    vertex_colors = []
                    
                    # Вычисляем цвет для каждой вершины
                    for i, idx in enumerate(face.vertices):
                        vertex_pos = obj.vertices[idx]
                        vertex_normal = obj.vertex_normals[idx]
                        
                        # Вычисляем освещение для данной вершины
                        color = self.light.calculate_lighting(
                            vertex_pos,
                            vertex_normal,
                            self.camera.position,
                            obj.material
                        )
                        vertex_colors.append(color)
                    
                    # Заливка полигона с интерполяцией цветов
                    self.fill_polygon_gouraud(vertices, vertex_colors)
                    
                elif self.shading_mode == "phong":
                    # Закрашивание по Фонгу (интерполяция нормалей)
                    vertex_normals = []
                    
                    # Сохраняем нормаль для каждой вершины
                    for idx in face.vertices:
                        vertex_normals.append(obj.vertex_normals[idx])
                    
                    # Заливка полигона с интерполяцией нормалей
                    self.fill_polygon_phong(vertices, vertex_normals, obj.material)
    
    def fill_polygon(self, vertices, color):
        """Заливка полигона одним цветом (для метода плоского закрашивания)"""
        # Преобразуем вершины в формат (x, y)
        points = [(int(v[0]), int(v[1])) for v in vertices]
        
        # Находим минимальные и максимальные координаты для создания ограничивающего прямоугольника
        min_x = max(0, min(p[0] for p in points))
        max_x = min(self.width() - 1, max(p[0] for p in points))
        min_y = max(0, min(p[1] for p in points))
        max_y = min(self.height() - 1, max(p[1] for p in points))
        
        # Создаем список ребер полигона
        edges = []
        for i in range(len(points)):
            edges.append((points[i], points[(i + 1) % len(points)]))
        
        # Для каждой строки в ограничивающем прямоугольнике
        for y in range(min_y, max_y + 1):
            # Находим пересечения с ребрами полигона
            intersections = []
            
            for edge in edges:
                # Извлекаем координаты
                x1, y1 = edge[0]
                x2, y2 = edge[1]
                
                # Пропускаем горизонтальные ребра
                if y1 == y2:
                    continue
                
                # Проверяем, пересекает ли текущая строка ребро
                if (y1 <= y <= y2) or (y2 <= y <= y1):
                    # Вычисляем x-координату пересечения (линейная интерполяция)
                    if y2 != y1:
                        x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                        intersections.append(int(x))
            
            # Сортируем пересечения по возрастанию x
            intersections.sort()
            
            # Заполняем пиксели между парами пересечений
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    start_x = max(min_x, intersections[i])
                    end_x = min(max_x, intersections[i + 1])
                    
                    # Заполняем пиксели между start_x и end_x
                    for x in range(start_x, end_x + 1):
                        # Проверяем z-буфер
                        z = self.interpolate_z(vertices, x, y)
                        if z < self.z_buffer[y, x]:
                            self.z_buffer[y, x] = z
                            self.color_buffer.setPixelColor(x, y, QColor(*color))
    
    def fill_polygon_gouraud(self, vertices, colors):
        """Заливка полигона с интерполяцией цветов (для метода Гуро)"""
        # Преобразуем вершины в формат (x, y)
        points = [(int(v[0]), int(v[1])) for v in vertices]
        
        # Находим минимальные и максимальные координаты для создания ограничивающего прямоугольника
        min_x = max(0, min(p[0] for p in points))
        max_x = min(self.width() - 1, max(p[0] for p in points))
        min_y = max(0, min(p[1] for p in points))
        max_y = min(self.height() - 1, max(p[1] for p in points))
        
        # Создаем список ребер полигона
        edges = []
        for i in range(len(points)):
            edges.append((points[i], points[(i + 1) % len(points)], colors[i], colors[(i + 1) % len(colors)]))
        
        # Для каждой строки в ограничивающем прямоугольнике
        for y in range(min_y, max_y + 1):
            # Находим пересечения с ребрами полигона и интерполируем цвета
            intersections = []
            
            for edge in edges:
                # Извлекаем координаты и цвета
                (x1, y1), (x2, y2), color1, color2 = edge
                
                # Пропускаем горизонтальные ребра
                if y1 == y2:
                    continue
                
                # Проверяем, пересекает ли текущая строка ребро
                if (y1 <= y <= y2) or (y2 <= y <= y1):
                    # Вычисляем x-координату пересечения
                    if y2 != y1:
                        t = (y - y1) / (y2 - y1)
                        x = x1 + t * (x2 - x1)
                        
                        # Интерполируем цвет
                        r = int(color1[0] + t * (color2[0] - color1[0]))
                        g = int(color1[1] + t * (color2[1] - color1[1]))
                        b = int(color1[2] + t * (color2[2] - color1[2]))
                        
                        intersections.append((int(x), (r, g, b)))
            
            # Сортируем пересечения по возрастанию x
            intersections.sort(key=lambda item: item[0])
            
            # Заполняем пиксели между парами пересечений
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    start_x, start_color = intersections[i]
                    end_x, end_color = intersections[i + 1]
                    
                    start_x = max(min_x, start_x)
                    end_x = min(max_x, end_x)
                    
                    # Заполняем пиксели между start_x и end_x с интерполяцией цвета
                    for x in range(start_x, end_x + 1):
                        # Линейная интерполяция цвета
                        if start_x != end_x:
                            t = (x - start_x) / (end_x - start_x)
                            r = int(start_color[0] + t * (end_color[0] - start_color[0]))
                            g = int(start_color[1] + t * (end_color[1] - start_color[1]))
                            b = int(start_color[2] + t * (end_color[2] - start_color[2]))
                        else:
                            r, g, b = start_color
                        
                        # Проверяем z-буфер
                        z = self.interpolate_z(vertices, x, y)
                        if z < self.z_buffer[y, x]:
                            self.z_buffer[y, x] = z
                            self.color_buffer.setPixelColor(x, y, QColor(r, g, b))
    
    def fill_polygon_phong(self, vertices, normals, material):
        """Заливка полигона с интерполяцией нормалей (для метода Фонга)"""
        # Преобразуем вершины в формат (x, y)
        points = [(int(v[0]), int(v[1])) for v in vertices]
        
        # Находим минимальные и максимальные координаты для создания ограничивающего прямоугольника
        min_x = max(0, min(p[0] for p in points))
        max_x = min(self.width() - 1, max(p[0] for p in points))
        min_y = max(0, min(p[1] for p in points))
        max_y = min(self.height() - 1, max(p[1] for p in points))
        
        # Создаем список ребер полигона
        edges = []
        for i in range(len(points)):
            edges.append((points[i], points[(i + 1) % len(points)], normals[i], normals[(i + 1) % len(normals)]))
        
        # Для каждой строки в ограничивающем прямоугольнике
        for y in range(min_y, max_y + 1):
            # Находим пересечения с ребрами полигона и интерполируем нормали
            intersections = []
            
            for edge in edges:
                # Извлекаем координаты и нормали
                (x1, y1), (x2, y2), normal1, normal2 = edge
                
                # Пропускаем горизонтальные ребра
                if y1 == y2:
                    continue
                
                # Проверяем, пересекает ли текущая строка ребро
                if (y1 <= y <= y2) or (y2 <= y <= y1):
                    # Вычисляем x-координату пересечения
                    if y2 != y1:
                        t = (y - y1) / (y2 - y1)
                        x = x1 + t * (x2 - x1)
                        
                        # Интерполируем нормаль
                        nx = normal1.x + t * (normal2.x - normal1.x)
                        ny = normal1.y + t * (normal2.y - normal1.y)
                        nz = normal1.z + t * (normal2.z - normal1.z)
                        
                        # Нормализуем интерполированную нормаль
                        length = math.sqrt(nx**2 + ny**2 + nz**2)
                        if length > 0:
                            nx /= length
                            ny /= length
                            nz /= length
                        
                        interpolated_normal = Vector3D(nx, ny, nz)
                        intersections.append((int(x), interpolated_normal))
            
            # Сортируем пересечения по возрастанию x
            intersections.sort(key=lambda item: item[0])
            
            # Заполняем пиксели между парами пересечений
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    start_x, start_normal = intersections[i]
                    end_x, end_normal = intersections[i + 1]
                    
                    start_x = max(min_x, start_x)
                    end_x = min(max_x, end_x)
                    
                    # Заполняем пиксели между start_x и end_x с интерполяцией нормалей
                    for x in range(start_x, end_x + 1):
                        # Линейная интерполяция нормали
                        if start_x != end_x:
                            t = (x - start_x) / (end_x - start_x)
                            nx = start_normal.x + t * (end_normal.x - start_normal.x)
                            ny = start_normal.y + t * (end_normal.y - start_normal.y)
                            nz = start_normal.z + t * (end_normal.z - start_normal.z)
                            
                            # Нормализуем интерполированную нормаль
                            length = math.sqrt(nx**2 + ny**2 + nz**2)
                            if length > 0:
                                nx /= length
                                ny /= length
                                nz /= length
                            
                            pixel_normal = Vector3D(nx, ny, nz)
                        else:
                            pixel_normal = start_normal
                        
                        # Проверяем z-буфер
                        z = self.interpolate_z(vertices, x, y)
                        if z < self.z_buffer[y, x]:
                            # Вычисляем освещение для каждого пикселя
                            point = Point3D(x, y, z)  # Приближенная позиция в 3D
                            
                            # Вычисляем цвет с помощью модели освещения
                            color = self.light.calculate_lighting(
                                point,
                                pixel_normal,
                                self.camera.position,
                                material
                            )
                            
                            self.z_buffer[y, x] = z
                            self.color_buffer.setPixelColor(x, y, QColor(*color))
    
    def interpolate_z(self, vertices, x, y):
        """Интерполирует значение z для заданных x, y (для z-буфера)"""
        # Для простоты используем среднее значение z всех вершин
        # В реальном z-буфере необходима барицентрическая интерполяция
        return sum(v[2] for v in vertices) / len(vertices)
    
    def draw_light(self, painter, view_matrix, projection_matrix, width, height):
        """Отрисовка источника света"""
        # Преобразование координат источника света
        light_pos = self.light.position.to_array()
        
        # Применение матрицы вида
        light_pos = np.matmul(view_matrix, light_pos)
        
        # Применение матрицы проекции
        light_pos = np.matmul(projection_matrix, light_pos)
        
        # Нормализация координат
        if light_pos[3] != 0:
            light_pos = light_pos / light_pos[3]
        
        # Преобразование в экранные координаты
        x = int((light_pos[0] + 1) * width / 2)
        y = int((1 - light_pos[1]) * height / 2)
        
        # Проверка, находится ли источник света в области видимости
        if 0 <= x < width and 0 <= y < height and light_pos[2] <= 1:
            # Отрисовка источника света как жёлтого кружка
            painter.setPen(QPen(QColor(255, 255, 0)))
            painter.setBrush(QBrush(QColor(255, 255, 0)))
            painter.drawEllipse(x - 5, y - 5, 10, 10)
    
    def draw_info(self, painter):
        """Отображение информации о режиме отображения"""
        painter.setPen(QPen(QColor(255, 255, 255)))
        
        # Отображение режима отображения
        mode_text = ""
        if self.render_mode == "wireframe":
            mode_text = "Режим: Каркасная модель"
        elif self.render_mode == "hidden_surface":
            mode_text = "Режим: Удаление невидимых поверхностей"
        elif self.render_mode == "solid":
            mode_text = "Режим: Закрашивание поверхностей"
            
            # Добавляем информацию о методе закрашивания
            if self.shading_mode == "flat":
                mode_text += " (монотонное)"
            elif self.shading_mode == "gouraud":
                mode_text += " (по Гуро)"
            elif self.shading_mode == "phong":
                mode_text += " (по Фонгу)"
        
        painter.drawText(10, 20, mode_text)
    
    # Обработчики событий мыши и клавиатуры
    def mousePressEvent(self, event):
        """Обработчик нажатия кнопки мыши"""
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
            self.mouse_pressed = True
    
    def mouseReleaseEvent(self, event):
        """Обработчик отпускания кнопки мыши"""
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = False
    
    def mouseMoveEvent(self, event):
        """Обработчик движения мыши"""
        if self.mouse_pressed:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            
            # Вращение камеры
            rot_x = self.camera.rot_x - dy * 0.5
            rot_y = self.camera.rot_y + dx * 0.5
            
            self.camera.set_rotation(rot_x, rot_y)
            
            self.last_mouse_pos = event.pos()
            self.update()
    
    def wheelEvent(self, event):
        """Обработчик прокрутки колеса мыши"""
        # Изменение расстояния до камеры (масштабирование)
        delta = event.angleDelta().y() / 120
        
        # Текущее расстояние
        dist = self.camera.position.distance_to(self.camera.target)
        
        # Новое расстояние (с ограничениями)
        new_dist = max(1, min(30, dist - delta))
        
        # Текущие углы вращения
        rot_x = self.camera.rot_x
        rot_y = self.camera.rot_y
        
        # Обновление позиции камеры
        rad_x = math.radians(rot_x)
        rad_y = math.radians(rot_y)
        
        x = new_dist * math.sin(rad_y) * math.cos(rad_x)
        y = new_dist * math.sin(rad_x)
        z = new_dist * math.cos(rad_y) * math.cos(rad_x)
        
        self.camera.position = Point3D(x + self.camera.target.x, y + self.camera.target.y, z + self.camera.target.z)
        
        self.update()
    
    def keyPressEvent(self, event):
        """Обработчик нажатия клавиш"""
        # Управление с помощью клавиш
        if event.key() == Qt.Key_W:
            # Поворот камеры вверх
            rot_x = self.camera.rot_x + 5
            self.camera.set_rotation(rot_x, self.camera.rot_y)
        elif event.key() == Qt.Key_S:
            # Поворот камеры вниз
            rot_x = self.camera.rot_x - 5
            self.camera.set_rotation(rot_x, self.camera.rot_y)
        elif event.key() == Qt.Key_A:
            # Поворот камеры влево
            rot_y = self.camera.rot_y - 5
            self.camera.set_rotation(self.camera.rot_x, rot_y)
        elif event.key() == Qt.Key_D:
            # Поворот камеры вправо
            rot_y = self.camera.rot_y + 5
            self.camera.set_rotation(self.camera.rot_x, rot_y)
        
        self.update()
    
    def create_coordinate_axes(self):
        """Создает модель координатных осей"""
        return CoordinateAxes(length=5.0)
    
    def draw_coordinate_axes(self, painter):
        """Рисует координатные оси на сцене"""
        if not self.coordinate_axes:
            return
            
        # Получение матрицы проекции и вида
        projection_matrix = self.get_projection_matrix()
        view_matrix = self.camera.get_view_matrix()
        
        # Копируем оси координат
        axes = copy.deepcopy(self.coordinate_axes)
        
        # Преобразование координат осей
        axes.transform_vertices(view_matrix, projection_matrix, self.width(), self.height())
        
        # Рисуем оси в виде линий
        for face in axes.faces:
            if len(face.vertices) == 2:  # Линия
                # Индексы текущей и следующей вершины
                idx1 = face.vertices[0]
                idx2 = face.vertices[1]
                
                # Получение координат вершин
                p1 = axes.transformed_vertices[idx1]
                p2 = axes.transformed_vertices[idx2]
                
                # Настройка пера для отрисовки
                pen = QPen(QColor(*face.color))
                pen.setWidth(2)
                painter.setPen(pen)
                
                # Отрисовка линии
                painter.drawLine(p1[0], p1[1], p2[0], p2[1]) 