"""
Основной модуль приложения для визуализации математических функций.

Этот модуль содержит графический интерфейс пользователя для отображения
математических функций в виде конусов. Приложение позволяет:
- Выбирать функции для отображения
- Настраивать диапазон и количество точек
- Визуализировать данные в виде конусов
- Отображать легенду и координатную сетку
"""

import sys
import math
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QSpinBox, QLabel, QComboBox, QListView
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QFont, QPainterPath, QColor,
    QStandardItemModel, QStandardItem, QFontMetrics
)
import functions


class Settings:
    """
    Класс, содержащий все настройки интерфейса и отображения.
    
    Этот класс определяет константы для:
    - Размеров окна и графика
    - Отступов и позиционирования элементов
    - Настроек легенды
    - Параметров графика
    - Настроек конусов
    - Цветов функций
    - Шрифтов
    """
    
    # Настройки интерфейса
    WINDOW_TITLE = "Лабораторная работа №1: Диаграмма функций (PySide)"
    WINDOW_SIZE = (1000, 800)
    PLOT_MIN_SIZE = (800, 600)
    
    # Отступы
    MARGIN_LEFT = 60
    MARGIN_RIGHT = 40
    MARGIN_BOTTOM = 40
    MARGIN_TOP = 20
    
    # Настройки легенды
    LEGEND_PADDING = 10
    LEGEND_MARKER_SIZE = 15
    LEGEND_SPACING = 8
    LEGEND_POSITION = (20, 20)
    
    # Настройки графика
    DEFAULT_X_RANGE = (-10, 10)
    DEFAULT_POINTS = 20
    DEFAULT_SPIN_RANGE = (-1000, 1000)
    DEFAULT_POINTS_RANGE = (1, 10000)
    SMALL_VALUE = 1e-6
    
    # Настройки конусов
    CONE_SCALE = 0.7
    CONE_BASE_RATIO = 2 / 3
    SHIFT_Y_FACTOR = 2.5
    FIXED_BASE_WIDTH = 40
    
    # Цвета функций Для добавления новых функций добавить цвета
    FUNCTION_COLORS = {
        1: QColor(Qt.blue),
        2: QColor(Qt.green),
        3: QColor(Qt.red),
        4: QColor(Qt.magenta)
    }
    
    # Шрифты
    FONT_FAMILY = "Arial"
    FONT_SIZE = 8
    LEGEND_FONT_SIZE = 10


class FunctionManager:
    """
    Класс для управления функциями и их данными.
    
    Этот класс предоставляет методы для:
    - Получения списка доступных функций
    - Получения данных функций для заданного интервала
    - Создания данных для построения графика
    """
    
    @staticmethod
    def get_available_functions():
        """
        Собирает список доступных функций из модуля functions.
        
        Returns:
            list: Список кортежей (подпись, идентификатор функции).
        """
        func_list = []
        for func_id in range(1, 4):  # Увеличить диапазон для добавления новых функций  
            func = getattr(functions, f"function_{func_id}", None)
            if callable(func):
                doc = func.__doc__.strip() if func.__doc__ else "Нет описания"
                func_list.append((f"Функция {func_id} ({doc})", func_id))
        return func_list
    
    @staticmethod
    def get_function_data(func_id, start, end, num_points):
        """
        Получает данные функции для заданного интервала.
        
        Args:
            func_id (int): Идентификатор функции
            start (float): Начало интервала
            end (float): Конец интервала
            num_points (int): Количество точек
            
        Returns:
            tuple: Кортеж (x, y, label) с массивами данных и меткой функции
        """
        return functions.get_function_data(func_id, start, end, num_points)
    
    @staticmethod
    def create_plot_data(selected_functions, start, end, num_points):
        """
        Создает данные для построения графика.
        
        Args:
            selected_functions (list): Список идентификаторов выбранных функций
            start (float): Начало интервала
            end (float): Конец интервала
            num_points (int): Количество точек
            
        Returns:
            list: Список словарей с данными функций
        """
        data_list = []
        for func_id in selected_functions:
            x, y, label = FunctionManager.get_function_data(func_id, start, end, num_points)
            data_list.append({
                "x": x,
                "y": y,
                "label": label,
                "color": Settings.FUNCTION_COLORS.get(func_id, QColor(Qt.black))
            })
        return data_list


class LegendWidget(QWidget):
    """
    Виджет для отображения легенды графика.
    
    Этот виджет отображает список функций с их цветовыми маркерами
    и подписями. Легенда автоматически центрируется и масштабируется
    под размер виджета.
    """
    
    def __init__(self):
        """
        Инициализирует виджет легенды.
        """
        super().__init__()
        self.data = []
        self.setMinimumHeight(50)  # Минимальная высота для легенды
        self.setMaximumHeight(100)  # Максимальная высота для легенды

    def setData(self, data_list):
        """
        Устанавливает данные для отображения в легенде.
        
        Args:
            data_list (list): Список словарей с данными функций
        """
        self.data = data_list
        self.update()

    def paintEvent(self, event):
        """
        Отрисовывает легенду.
        
        Args:
            event: Событие отрисовки
        """
        if not self.data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Настройка шрифта
        font = QFont(Settings.FONT_FAMILY, Settings.LEGEND_FONT_SIZE)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        
        # Рисуем белый фон
        painter.fillRect(self.rect(), QBrush(Qt.white))
        
        # Вычисляем размеры элементов легенды
        item_spacing = 20  # Расстояние между элементами
        marker_size = 15
        text_spacing = 5
        
        # Вычисляем общую ширину всех элементов
        total_width = 0
        for curve in self.data:
            text_width = metrics.horizontalAdvance(curve["label"])
            item_width = marker_size + text_spacing + text_width
            total_width += item_width + item_spacing

        # Начальная позиция (центрируем легенду)
        start_x = (self.width() - total_width + item_spacing) / 2
        center_y = self.height() / 2
        
        # Рисуем элементы легенды
        current_x = start_x
        for curve in self.data:
            # Рисуем цветной маркер
            painter.setBrush(QBrush(curve["color"]))
            painter.setPen(QPen(curve["color"].darker(150), 1))
            painter.drawRect(int(current_x), int(center_y - marker_size/2), 
                           marker_size, marker_size)
            
            # Рисуем текст
            painter.setPen(Qt.black)
            text_x = current_x + marker_size + text_spacing
            text_y = center_y + metrics.height()/2 - 2
            painter.drawText(int(text_x), int(text_y), curve["label"])
            
            # Обновляем позицию для следующего элемента
            text_width = metrics.horizontalAdvance(curve["label"])
            current_x += marker_size + text_spacing + text_width + item_spacing


class PlotData:
    """
    Класс для хранения и управления данными графика.
    
    Этот класс отвечает за:
    - Хранение данных функций
    - Расчет диапазонов значений для осей
    - Подготовку данных для отображения
    - Настройку параметров сетки
    
    Attributes:
        data (list): Список словарей с данными функций
        y_min (float): Минимальное значение по оси Y
        y_max (float): Максимальное значение по оси Y
        user_x_start (float): Начало интервала по оси X, заданное пользователем
        user_x_end (float): Конец интервала по оси X, заданное пользователем
        default_points (int): Количество точек для отображения
        extra_margin (float): Дополнительный отступ для графика
    """
    
    def __init__(self):
        """
        Инициализирует объект данных графика со значениями по умолчанию.
        """
        self.data = []                               # Данные функций
        self.y_min = -10                             # Минимум по оси Y
        self.y_max = 10                              # Максимум по оси Y
        self.user_x_start = Settings.DEFAULT_X_RANGE[0]  # Начало интервала
        self.user_x_end = Settings.DEFAULT_X_RANGE[1]    # Конец интервала
        self.default_points = Settings.DEFAULT_POINTS    # Число точек
        self.extra_margin = 1                        # Дополнительный отступ

    def update_data(self, data_list):
        """
        Обновляет данные и пересчитывает диапазон оси Y.
        
        Этот метод анализирует значения функций, определяет минимум и максимум,
        а также добавляет небольшой отступ для лучшего отображения.
        
        Args:
            data_list (list): Список словарей с данными функций
                Каждый словарь содержит ключи: "x", "y", "label", "color"
        """
        self.data = data_list
        
        # Если есть данные, пересчитываем диапазон по оси Y
        if self.data:
            # Объединяем все значения Y всех функций, исключая NaN
            all_y = np.concatenate([d["y"][~np.isnan(d["y"])] for d in self.data])
            
            if len(all_y) > 0:
                # Добавляем 10% отступа для лучшего отображения
                margin_y = (all_y.max() - all_y.min()) * 0.1
                computed_y_min = all_y.min() - margin_y
                computed_y_max = all_y.max() + margin_y
                
                # Убеждаемся, что 0 входит в диапазон (для корректного отображения оси X)
                self.y_min = min(computed_y_min, 0)
                self.y_max = max(computed_y_max, 0)

    def get_grid_ranges(self, step_x):
        """
        Возвращает расширенные диапазоны для сетки.
        
        Расширяет диапазоны по осям X и Y с учетом дополнительных клеток сетки для
        лучшего визуального отображения графика.
        
        Args:
            step_x (float): Шаг по оси X
            
        Returns:
            tuple: Кортеж с диапазонами для сетки:
                (grid_x_min, grid_x_max, grid_x_range, grid_y_min, grid_y_max, grid_y_range)
        """
        # Расширяем диапазон по оси Y с учетом дополнительного отступа
        grid_y_min = self.y_min - self.extra_margin
        grid_y_max = self.y_max + self.extra_margin
        grid_y_range = grid_y_max - grid_y_min if grid_y_max != grid_y_min else Settings.SMALL_VALUE

        # Добавляем по одной дополнительной клетке с каждой стороны по оси X
        grid_x_min = self.user_x_start - step_x
        grid_x_max = self.user_x_end + step_x
        grid_x_range = grid_x_max - grid_x_min if grid_x_max != grid_x_min else Settings.SMALL_VALUE

        return grid_x_min, grid_x_max, grid_x_range, grid_y_min, grid_y_max, grid_y_range


class ConeRenderer:
    """
    Класс для отрисовки конусов.
    
    Этот класс предоставляет методы для отрисовки конусов с
    треугольной боковой гранью и эллиптическим основанием.
    """
    
    @staticmethod
    def draw_cone(painter, apex_x, apex_y, base_x, base_y, width_val, height_val, color):
        """
        Отрисовывает конус с треугольной боковой гранью и эллиптическим основанием.
        
        Args:
            painter (QPainter): Объект для отрисовки
            apex_x (int): X-координата вершины конуса
            apex_y (int): Y-координата вершины конуса
            base_x (int): X-координата основания конуса
            base_y (int): Y-координата основания конуса
            width_val (int): Ширина основания конуса
            height_val (int): Высота конуса
            color (QColor): Цвет конуса
        """
        # Рисуем основание конуса (эллипс)
        base_width = width_val
        base_height = int(height_val * 0.5)
        ellipse_rect = (int(base_x), int(base_y - base_height / 2), int(base_width), int(base_height))
        
        # Определяем направление конуса
        is_upward = apex_y < base_y
        
        # Сначала рисуем основание
        if is_upward:
            # Для конусов, направленных вверх
            painter.setBrush(QBrush(color.darker(115)))
            painter.setPen(QPen(color.darker(150), 2))
            painter.drawEllipse(*ellipse_rect)
        
        # Затем рисуем боковую грань
        path = QPainterPath()
        path.moveTo(apex_x, apex_y)
        path.lineTo(base_x, base_y)
        path.lineTo(base_x + width_val, base_y)
        path.closeSubpath()
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawPath(path)
        
        # Рисуем видимую часть основания поверх боковой грани
        if is_upward:
            # Для конусов, направленных вверх, рисуем нижнюю часть
            highlight_color = color.lighter(160)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(highlight_color))
            painter.drawArc(*ellipse_rect, 180 * 16, 180 * 16)
        else:
            # Для конусов, направленных вниз
            painter.setBrush(QBrush(color.darker(115)))
            painter.setPen(QPen(color.darker(150), 2))
            painter.drawEllipse(*ellipse_rect)
            highlight_color = color.lighter(160)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(highlight_color))
            painter.drawArc(*ellipse_rect, 180 * 16, -180 * 16)
        

class PlotWidget(QWidget):
    """
    Виджет для отображения графика с конусами.
    
    Этот виджет отвечает за:
    - Отрисовку координатной сетки
    - Отображение осей
    - Отрисовку конусов
    - Масштабирование и трансформацию координат
    """
    
    def __init__(self):
        """
        Инициализирует виджет графика.
        """
        super().__init__()
        self.setMinimumSize(*Settings.PLOT_MIN_SIZE)
        self.plot_data = PlotData()
        self.cone_renderer = ConeRenderer()
        self.coordinate_transformer = None  # Будет инициализирован в paintEvent

    def setData(self, data_list):
        """
        Обновляет данные для отображения и перерисовывает график.
        
        Args:
            data_list (list): Список словарей с данными функций
        """
        self.plot_data.update_data(data_list)
        self.update()

    def paintEvent(self, event):
        """
        Основной метод отрисовки виджета.
        
        Args:
            event: Событие отрисовки
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Инициализация трансформатора координат
        self.initialize_coordinate_transformer(w, h)
        
        # Рисуем фон и границы
        self.draw_background(painter, w, h)
        
        # Если нет данных для отображения, выходим
        if not self.plot_data.data:
            return
            
        # Рисуем сетку и оси
        self.draw_grid(painter, w, h)
        self.draw_axes(painter, w, h)
        
        # Рисуем конусы
        self.draw_cones(painter, w, h)

    def initialize_coordinate_transformer(self, w, h):
        """Инициализирует трансформатор координат."""
        # Вычисляем рабочую область для графика
        plot_width = w - Settings.MARGIN_LEFT - Settings.MARGIN_RIGHT
        plot_height = h - Settings.MARGIN_TOP - Settings.MARGIN_BOTTOM

        # Получаем шаг по оси X
        step_x = self._get_x_step()

        # Получаем диапазоны осей
        grid_x_min, grid_x_max, grid_x_range, grid_y_min, grid_y_max, grid_y_range = self.plot_data.get_grid_ranges(step_x)
        
        # Создаем объект трансформатора координат
        self.coordinate_transformer = {
            'w': w, 
            'h': h,
            'plot_width': plot_width,
            'plot_height': plot_height,
            'grid_x_min': grid_x_min,
            'grid_x_max': grid_x_max,
            'grid_x_range': grid_x_range,
            'grid_y_min': grid_y_min,
            'grid_y_max': grid_y_max,
            'grid_y_range': grid_y_range,
            'step_x': step_x
        }

    def _get_x_step(self):
        """
        Вычисляет шаг по оси X с учетом количества точек.
        
        Returns:
            float: Шаг между точками по оси X
        """
        if self.plot_data.default_points > 1:
            # Для множества точек вычисляем равномерный шаг
            return (self.plot_data.user_x_end - self.plot_data.user_x_start) / (self.plot_data.default_points - 1)
        else:
            # Для одной точки используем либо весь диапазон, либо значение по умолчанию
            return (self.plot_data.user_x_end - self.plot_data.user_x_start) if (self.plot_data.user_x_end != self.plot_data.user_x_start) else 1

    def _get_extended_x_range(self):
        """
        Возвращает расширенный диапазон по оси X с дополнительными клетками.
        
        Returns:
            tuple: Кортеж (grid_x_min, grid_x_max) с расширенным диапазоном
        """
        step_x = self._get_x_step()
        # Добавляем по одной дополнительной клетке с каждой стороны
        grid_x_min = self.plot_data.user_x_start - step_x
        grid_x_max = self.plot_data.user_x_end + step_x
        return grid_x_min, grid_x_max

    def to_pixel_x(self, x):
        """
        Преобразует X-координату из системы координат данных в пиксели.
        
        Args:
            x (float): Координата X в системе данных
            
        Returns:
            int: Координата X в пикселях
        """
        plot_width = self.width() - Settings.MARGIN_LEFT - Settings.MARGIN_RIGHT
        
        # Получаем расширенный диапазон X
        grid_x_min, grid_x_max = self._get_extended_x_range()
        x_range = grid_x_max - grid_x_min
        
        # Преобразуем координаты
        return int(Settings.MARGIN_LEFT + (x - grid_x_min) * plot_width / x_range)

    def to_pixel_y(self, y):
        """
        Преобразует Y-координату из системы координат данных в пиксели.
        
        Args:
            y (float): Координата Y в системе данных
            
        Returns:
            int: Координата Y в пикселях
        """
        if not self.plot_data.data:
            return self.height() - Settings.MARGIN_BOTTOM
            
        plot_height = self.height() - Settings.MARGIN_TOP - Settings.MARGIN_BOTTOM
        
        # Находим минимальное и максимальное значения функций
        min_y = float('inf')
        max_y = float('-inf')
        for curve in self.plot_data.data:
            valid_y = curve["y"][~np.isnan(curve["y"])]
            if len(valid_y) > 0:
                min_y = min(min_y, np.min(valid_y))
                max_y = max(max_y, np.max(valid_y))
        
        if min_y == float('inf') or max_y == float('-inf'):
            return self.height() - Settings.MARGIN_BOTTOM
            
        # Убеждаемся, что 0 входит в диапазон
        min_y = min(min_y, 0)
        max_y = max(max_y, 0)
        
        # Рассчитываем параметры отображения в зависимости от количества точек
        if self.plot_data.default_points == 1:
            # Создаем искусственный диапазон вокруг единственного значения
            cell_size = 1.0  # Фиксированный размер клетки для одной точки
            plot_min_y = min_y - cell_size
            plot_max_y = max_y + cell_size
        else:
            # Стандартная обработка для множества точек
            y_range = max_y - min_y
            cell_size = y_range / (self.plot_data.default_points - 1)
            plot_min_y = min_y - cell_size
            plot_max_y = max_y + cell_size
            
        plot_range = plot_max_y - plot_min_y
        
        # Инвертируем Y-координату (в Qt ось Y направлена вниз)
        return int(self.height() - Settings.MARGIN_BOTTOM - (y - plot_min_y) * plot_height / plot_range)

    def draw_background(self, painter, w, h):
        """Рисует фон и границы виджета."""
        painter.fillRect(self.rect(), QBrush(Qt.white))
        painter.setPen(QPen(Qt.black, 2))
        painter.drawRect(0, 0, w-1, h-1)

    def draw_grid(self, painter, w, h):
        """Рисует координатную сетку."""
        if len(self.plot_data.data[0]["x"]) == 1:
            self._draw_single_point_grid(painter, w, h)
        else:
            self._draw_horizontal_grid(painter, w, h)
            
        self._draw_vertical_grid(painter, h)

    def draw_axes(self, painter, w, h):
        """
        Рисует оси координат.
        
        Args:
            painter (QPainter): Объект для отрисовки
            w (int): Ширина области отрисовки
            h (int): Высота области отрисовки
        """
        # Рисуем горизонтальную ось y = 0
        zero_y = self.to_pixel_y(0)
        painter.setPen(QPen(Qt.black, 2))  # Делаем оси толще чем линии сетки
        painter.drawLine(Settings.MARGIN_LEFT, zero_y, w - Settings.MARGIN_RIGHT, zero_y)

        # Проверяем, попадает ли ось X = 0 в область отображения с учетом расширенного диапазона
        grid_x_min, grid_x_max = self._get_extended_x_range()
        if grid_x_min <= 0 <= grid_x_max:
            # Определяем координаты оси X = 0 в пикселях
            x_zero = self.to_pixel_x(0)
            # Ограничиваем координаты рабочей областью
            x_zero = max(Settings.MARGIN_LEFT, min(w - Settings.MARGIN_RIGHT, x_zero))
            
            # Рисуем вертикальную ось
            painter.drawLine(x_zero, Settings.MARGIN_TOP, x_zero, h - Settings.MARGIN_BOTTOM)
            
            # Добавляем подпись к началу координат
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(x_zero - 20, h - Settings.MARGIN_BOTTOM + 20, "0.00")

    def draw_cones(self, painter, w, h):
        """
        Рисует конусы для отображения данных функций.
        
        Конусы представляют значения функций в каждой точке. 
        Высота конуса соответствует значению функции, а цвет - выбранной функции.
        
        Args:
            painter (QPainter): Объект для отрисовки
            w (int): Ширина области отрисовки
            h (int): Высота области отрисовки
        """
        # Проверяем наличие точек и функций для отображения
        if not self.plot_data.data or len(self.plot_data.data[0]["x"]) == 0:
            return
            
        num_points = len(self.plot_data.data[0]["x"])  # Количество точек
        num_funcs = len(self.plot_data.data)           # Количество функций
            
        # Определяем границы для размещения конусов
        # Используем точки данных непосредственно, без учета дополнительных клеток
        left_bound_plot = self.to_pixel_x(self.plot_data.data[0]["x"][0])
        right_bound_plot = self.to_pixel_x(self.plot_data.data[0]["x"][-1])
        
        # Вычисляем ширину группы конусов для одной точки данных
        group_width = (right_bound_plot - left_bound_plot) * Settings.CONE_SCALE / (num_points if num_points > 1 else 1)

        # Настраиваем параметры отображения конусов
        # Для малого числа точек используем фиксированную ширину
        cell_width = Settings.FIXED_BASE_WIDTH if num_points <= 3 else group_width / (num_funcs + 1)
        cone_base_width = cell_width * Settings.CONE_BASE_RATIO  # Ширина основания конуса
        ellipse_height = int(cone_base_width * 0.6)              # Высота эллипса основания
        
        # Настраиваем смещения для размещения конусов разных функций
        horizontal_shift = cell_width * 1.2    # Горизонтальное смещение между конусами
        vertical_shift = ellipse_height * 0.8  # Вертикальное смещение для наглядности

        # Рисуем вертикальные линии сетки в точках данных
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        for i in range(num_points):
            center = self.plot_data.data[0]["x"][i]
            center_pix = self.to_pixel_x(center)
            painter.drawLine(center_pix, Settings.MARGIN_TOP, center_pix, h - Settings.MARGIN_BOTTOM)

        # Отрисовка конусов для каждой точки и каждой функции
        for i in range(num_points):
            center = self.plot_data.data[0]["x"][i]       # X-координата точки
            center_pix = self.to_pixel_x(center)          # X-координата в пикселях
            base_centers = []  # Запоминаем центры оснований для соединения линиями
            
            # Рисуем конусы для каждой функции в текущей точке
            for func_index, curve in enumerate(self.plot_data.data):
                y_val = curve["y"][i]
                # Пропускаем NaN значения (вне области определения функции)
                if np.isnan(y_val):
                    continue
                
                # Вычисляем смещения конусов для разных функций
                x_offset = int(func_index * horizontal_shift)
                y_offset = int(func_index * vertical_shift)
                
                # Позиция вершины и основания конуса
                apex_x = center_pix + x_offset
                base_x = apex_x - int(cone_base_width / 2)
                
                # Вычисляем y-координаты вершины и основания
                base_y = self.to_pixel_y(0) - y_offset     # Основание на оси X
                apex_y = self.to_pixel_y(y_val) - y_offset # Вершина на высоте значения функции
                
                # Запоминаем центр основания для соединительных линий
                base_centers.append((base_x + int(cone_base_width / 2), base_y))
                
                # Отрисовываем конус с помощью специализированного класса
                self.cone_renderer.draw_cone(
                    painter, apex_x, apex_y, base_x, base_y, 
                    int(cone_base_width), ellipse_height, curve["color"]
                )
            
            # Рисуем соединительные линии между основаниями конусов для наглядности
            if len(base_centers) > 1:
                painter.setPen(QPen(Qt.black, 1))
                for j in range(len(base_centers) - 1):
                    p1 = base_centers[j]
                    p2 = base_centers[j + 1]
                    painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

    def _draw_single_point_grid(self, painter, w, h):
        """
        Отрисовка сетки для случая с одной точкой.
        
        Специальная обработка для случая, когда у нас только одна точка данных.
        Рисует горизонтальные линии на каждой функции и на нуле.
        
        Args:
            painter (QPainter): Объект для отрисовки
            w (int): Ширина области отрисовки
            h (int): Высота области отрисовки
        """
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        
        # Проходим по каждой функции
        for func_index, curve in enumerate(self.plot_data.data):
            # Определяем позицию базовой линии (на оси X)
            base_y = self.to_pixel_y(0)
            
            # Вычисляем смещение точки от базовой линии
            shift_y = self.to_pixel_y(curve["y"][0]) - base_y
            
            # Y-координата для точки с учетом смещения функций
            y_pos = self.to_pixel_y(curve["y"][0]) + int(func_index * shift_y)
            
            # Рисуем горизонтальную линию через точку
            painter.drawLine(Settings.MARGIN_LEFT, y_pos, w - Settings.MARGIN_RIGHT, y_pos)
            
            # Добавляем подпись значения
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(w - Settings.MARGIN_RIGHT + 5, y_pos + 5, f"{curve['y'][0]:.3f}")
            
            # Возвращаем стиль линии для сетки
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))

    def _draw_horizontal_grid(self, painter, w, h):
        """
        Отрисовка горизонтальной сетки.
        
        Args:
            painter (QPainter): Объект для отрисовки
            w (int): Ширина области отрисовки
            h (int): Высота области отрисовки
        """
        # Проверяем наличие данных для построения
        if not self.plot_data.data:
            return
            
        # Находим минимальное и максимальное значения функций
        min_y = float('inf')
        max_y = float('-inf')
        for curve in self.plot_data.data:
            # Фильтруем NaN значения
            valid_y = curve["y"][~np.isnan(curve["y"])]
            if len(valid_y) > 0:
                min_y = min(min_y, np.min(valid_y))
                max_y = max(max_y, np.max(valid_y))
        
        # Если нет данных, выходим
        if min_y == float('inf') or max_y == float('-inf'):
            return
            
        # Убеждаемся, что 0 входит в диапазон (для корректного отображения оси X)
        min_y = min(min_y, 0)
        max_y = max(max_y, 0)
        
        # Особая обработка для случая с одной точкой
        if self.plot_data.default_points == 1:
            # Для одной точки создаем искусственный диапазон с фиксированным размером клетки
            cell_size = 1.0
            plot_min_y = min_y - cell_size  # Одна клетка снизу
            plot_max_y = max_y + cell_size  # Одна клетка сверху
            num_lines = 3  # Три линии: снизу, в точке и сверху
        else:
            # Для множества точек вычисляем размер клетки на основе диапазона значений
            y_range = max_y - min_y
            cell_size = y_range / (self.plot_data.default_points - 1)
            plot_min_y = min_y - cell_size  # Одна клетка снизу
            plot_max_y = max_y + cell_size  # Одна клетка сверху
            num_lines = self.plot_data.default_points + 2  # +2 для дополнительных линий
        
        # Рисуем горизонтальные линии сетки с равным шагом
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        step = (plot_max_y - plot_min_y) / (num_lines - 1) if num_lines > 1 else 0
        
        # Рисуем линии сетки и подписи к ним
        for i in range(num_lines):
            y_val = plot_min_y + i * step
            y_pix = self.to_pixel_y(y_val)
            
            # Используем более жирную линию для оси X (если близко к 0)
            if abs(y_val) < step / 1000:  # Проверка на близость к 0
                painter.setPen(QPen(Qt.black, 2))
            else:
                painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
                
            # Рисуем горизонтальную линию сетки
            painter.drawLine(Settings.MARGIN_LEFT, y_pix, w - Settings.MARGIN_RIGHT, y_pix)
            
            # Рисуем подпись значения
            painter.setPen(QPen(Qt.black, 1))
            painter.setFont(QFont(Settings.FONT_FAMILY, Settings.FONT_SIZE))
            # Используем 3 десятичных знака для повышения точности отображения
            painter.drawText(w - Settings.MARGIN_RIGHT + 5, y_pix + 5, f"{y_val:.3f}")
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))

    def _draw_vertical_grid(self, painter, h):
        """
        Отрисовка вертикальной сетки.
        
        Args:
            painter (QPainter): Объект для отрисовки
            h (int): Высота области отрисовки
        """
        painter.setPen(QPen(Qt.darkGray, 1, Qt.DashDotLine))
        
        # Получаем шаг и диапазон для отрисовки сетки
        step_x = self._get_x_step()
        grid_x_min, grid_x_max = self._get_extended_x_range()
        
        # Рисуем вертикальные линии сетки с шагом step_x
        x_val = grid_x_min
        while x_val <= grid_x_max + step_x/2:  # step_x/2 для компенсации погрешности округления
            x_pix = self.to_pixel_x(x_val)
            
            # Рисуем линию сетки
            painter.drawLine(x_pix, Settings.MARGIN_TOP, x_pix, h - Settings.MARGIN_BOTTOM)
            
            # Добавляем подпись значения
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(x_pix - 20, h - Settings.MARGIN_BOTTOM + 20, f"{x_val:.2f}")
            
            # Возвращаем стиль линии для сетки
            painter.setPen(QPen(Qt.darkGray, 1, Qt.DashDotLine))
            
            # Переходим к следующей линии
            x_val += step_x

    @staticmethod
    def _nice_num(x, round_val):
        """
        Вычисляет 'приятное' число для шага сетки.
        
        Функция выбирает округленные значения, которые лучше воспринимаются
        человеком (например, 1, 2, 5, 10 или их степени).
        
        Args:
            x (float): Исходное число
            round_val (bool): Нужно ли округлять вверх (True) или вниз (False)
            
        Returns:
            float: 'Приятное' число для шага сетки
        """
        # Особый случай - нулевое значение
        if x == 0:
            return 0
            
        # Находим порядок числа
        exp = math.floor(math.log10(x))
        f = x / (10 ** exp)  # Мантисса (десятичная часть)
        
        # Выбираем 'приятное' значение
        if round_val:
            # Округление с приоритетом округления вверх
            if f < 1.5:
                nf = 1      # Для значений < 1.5, используем 1
            elif f < 3:
                nf = 2      # Для значений 1.5-3, используем 2
            elif f < 7:
                nf = 5      # Для значений 3-7, используем 5
            else:
                nf = 10     # Для значений > 7, используем 10
        else:
            # Округление с приоритетом округления вниз
            if f <= 1:
                nf = 1      # Для значений <= 1, используем 1
            elif f <= 2:
                nf = 2      # Для значений 1-2, используем 2
            elif f <= 5:
                nf = 5      # Для значений 2-5, используем 5
            else:
                nf = 10     # Для значений > 5, используем 10
                
        # Возвращаем число, умноженное на степень 10
        return nf * (10 ** exp)


class CustomComboBox(QComboBox):
    """
    Пользовательский комбобокс с улучшенным поведением.
    
    Этот класс расширяет стандартный QComboBox для улучшения
    пользовательского интерфейса при выборе функций.
    """
    
    def mouseReleaseEvent(self, event):
        """
        Обрабатывает событие отпускания кнопки мыши.
        
        Args:
            event: Событие мыши
        """
        if self.rect().contains(event.position().toPoint()):
            self.showPopup()
        super().mouseReleaseEvent(event)


class ControlPanel(QWidget):
    """
    Панель управления с элементами ввода для настройки графика.
    
    Этот виджет содержит:
    - Спинбоксы для настройки диапазона и количества точек
    - Комбобокс для выбора функций
    - Обработчики изменения параметров
    """
    
    def __init__(self, parent=None):
        """
        Инициализирует панель управления.
        
        Args:
            parent: Родительский виджет
        """
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """
        Создание и настройка элементов интерфейса.
        """
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Настройка спинбокса для начала интервала
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(*Settings.DEFAULT_SPIN_RANGE)
        self.start_spin.setValue(Settings.DEFAULT_X_RANGE[0])
        self.start_spin.setDecimals(2)
        self.start_spin.setSingleStep(0.1)
        layout.addWidget(QLabel("Начало интервала:"))
        layout.addWidget(self.start_spin)
        
        # Настройка спинбокса для конца интервала
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(*Settings.DEFAULT_SPIN_RANGE)
        self.end_spin.setValue(Settings.DEFAULT_X_RANGE[1])
        self.end_spin.setDecimals(2)
        self.end_spin.setSingleStep(0.1)
        layout.addWidget(QLabel("Конец интервала:"))
        layout.addWidget(self.end_spin)
        
        # Настройка спинбокса для количества точек
        self.points_spin = QSpinBox()
        self.points_spin.setRange(*Settings.DEFAULT_POINTS_RANGE)
        self.points_spin.setValue(Settings.DEFAULT_POINTS)
        layout.addWidget(QLabel("Количество точек:"))
        layout.addWidget(self.points_spin)
        
        # Настройка комбобокса для выбора функций
        self.func_combo = CustomComboBox()
        self.func_combo.setView(QListView())
        self.func_combo.setEditable(True)
        self.func_combo.lineEdit().setReadOnly(True)
        self.func_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self.func_combo.setInsertPolicy(QComboBox.NoInsert)
        self.func_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.func_combo.setMaxVisibleItems(10)
        self.func_combo.setMinimumContentsLength(20)
        
        self.func_model = QStandardItemModel(self.func_combo)
        for label, func_id in FunctionManager.get_available_functions():
            item = QStandardItem(label)
            item.setData(func_id)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setData(Qt.Unchecked, Qt.CheckStateRole)
            self.func_model.appendRow(item)
        self.func_combo.setModel(self.func_model)
        
        layout.addWidget(QLabel("Функции (выберите):"))
        layout.addWidget(self.func_combo)
        
    def setup_connections(self):
        """Настройка соединений сигнал-слот."""
        self.start_spin.valueChanged.connect(self.on_start_changed)
        self.end_spin.valueChanged.connect(self.on_end_changed)
    
    def on_start_changed(self, value):
        """Обработчик изменения значения начала интервала."""
        self.end_spin.setMinimum(value + 0.01)
        if self.end_spin.value() <= value:
            self.end_spin.setValue(value + 0.01)
    
    def on_end_changed(self, value):
        """Обработчик изменения значения конца интервала."""
        self.start_spin.setMaximum(value - 0.01)
        if self.start_spin.value() >= value:
            self.start_spin.setValue(value - 0.01)
    
    def get_plot_parameters(self):
        """Возвращает параметры для построения графика."""
        return {
            'start': self.start_spin.value(),
            'end': self.end_spin.value(),
            'num_points': self.points_spin.value()
        }
    
    def get_selected_functions(self):
        """Возвращает список выбранных функций."""
        selected = []
        for i in range(self.func_model.rowCount()):
            item = self.func_model.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data())
        return selected


class MainWindow(QMainWindow):
    """
    Главное окно приложения.
    
    Этот класс управляет:
    - Созданием и компоновкой виджетов
    - Обновлением графика
    - Обработкой пользовательского ввода
    """
    
    def __init__(self):
        """
        Инициализирует главное окно приложения.
        """
        super().__init__()
        self.setWindowTitle(Settings.WINDOW_TITLE)
        self.resize(*Settings.WINDOW_SIZE)
        
        # Создаем главный виджет и layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Добавляем график
        self.plot_widget = PlotWidget()
        main_layout.addWidget(self.plot_widget)
        
        # Добавляем легенду под графиком
        self.legend_widget = LegendWidget()
        main_layout.addWidget(self.legend_widget)
        
        # Добавляем панель управления
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel)
        
        self.setCentralWidget(main_widget)
        
        # Настройка таймера обновления
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_plot)
        
        # Подключаем обработчики событий
        self.control_panel.points_spin.valueChanged.connect(self.schedule_update)
        self.control_panel.start_spin.valueChanged.connect(self.schedule_update)
        self.control_panel.end_spin.valueChanged.connect(self.schedule_update)
        self.control_panel.func_model.itemChanged.connect(self.schedule_update)
        
        # Запускаем начальное обновление
        self.update_plot()

    def schedule_update(self):
        """Запланировать обновление графика с небольшой задержкой."""
        self.update_timer.start(200)

    def update_plot(self):
        """Обновить график с текущими параметрами."""
        params = self.control_panel.get_plot_parameters()
        start = params['start']
        end = params['end']
        num_points = params['num_points']
        
        # Проверяем корректность интервала
        if start >= end:
            # Если начало больше или равно концу, очищаем данные и выходим
            self.plot_widget.setData([])
            self.legend_widget.setData([])
            return
            
        if num_points < 2:
            num_points = 1  # Если выбрана одна точка, размещаем её в центре
            
        # Обновляем параметры графика
        self.plot_widget.plot_data.user_x_start = start
        self.plot_widget.plot_data.user_x_end = end
        self.plot_widget.plot_data.default_points = num_points
        
        # Получаем выбранные функции и создаем данные
        selected_functions = self.control_panel.get_selected_functions()
        data_list = FunctionManager.create_plot_data(selected_functions, start, end, num_points)
            
        # Обновляем данные графика и легенды
        self.plot_widget.setData(data_list)
        self.legend_widget.setData(data_list)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())