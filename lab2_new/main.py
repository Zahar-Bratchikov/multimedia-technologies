import sys
import math
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QSlider, QPushButton, 
                              QGroupBox, QRadioButton, QComboBox, QSpinBox, 
                              QDoubleSpinBox, QTabWidget, QGridLayout, QCheckBox)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QImage

from render_widget import RenderWidget
from models import LetterB, LetterZ, Camera, Light, Point3D

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Визуализация букв Б и З")
        self.setMinimumSize(1200, 800)
        
        # Создание моделей букв
        self.letter_b = LetterB(Point3D(-1.5, 0, 0), 2.0, 1.0, 0.5)
        self.letter_z = LetterZ(Point3D(1.5, 0, 0), 2.0, 1.0, 0.5)
        
        # Создание камеры и источника света
        self.camera = Camera(Point3D(0, 0, 10), Point3D(0, 0, 0))
        self.light = Light(Point3D(5, 5, 5))
        
        # Инициализация UI
        self.setup_ui()
    
    def setup_ui(self):
        # Основной виджет
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # Создание виджета для отрисовки 3D объектов
        self.render_widget = RenderWidget(self.letter_b, self.letter_z, 
                                         self.camera, self.light)
        
        # Панель управления
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Создание вкладок для управления
        tab_widget = QTabWidget()
        
        # Вкладка параметров букв
        letters_tab = QWidget()
        letters_layout = QVBoxLayout(letters_tab)
        
        # Группа параметров для буквы Б
        b_group = QGroupBox("Параметры буквы Б")
        b_layout = QGridLayout(b_group)
        
        b_layout.addWidget(QLabel("Высота:"), 0, 0)
        self.b_height_spin = QDoubleSpinBox()
        self.b_height_spin.setRange(0.5, 10.0)
        self.b_height_spin.setValue(2.0)
        self.b_height_spin.setSingleStep(0.1)
        self.b_height_spin.valueChanged.connect(self.update_letter_b_params)
        b_layout.addWidget(self.b_height_spin, 0, 1)
        
        b_layout.addWidget(QLabel("Ширина:"), 1, 0)
        self.b_width_spin = QDoubleSpinBox()
        self.b_width_spin.setRange(0.5, 10.0)
        self.b_width_spin.setValue(1.0)
        self.b_width_spin.setSingleStep(0.1)
        self.b_width_spin.valueChanged.connect(self.update_letter_b_params)
        b_layout.addWidget(self.b_width_spin, 1, 1)
        
        b_layout.addWidget(QLabel("Глубина:"), 2, 0)
        self.b_depth_spin = QDoubleSpinBox()
        self.b_depth_spin.setRange(0.1, 5.0)
        self.b_depth_spin.setValue(0.5)
        self.b_depth_spin.setSingleStep(0.1)
        self.b_depth_spin.valueChanged.connect(self.update_letter_b_params)
        b_layout.addWidget(self.b_depth_spin, 2, 1)
        
        # Группа параметров для буквы З
        z_group = QGroupBox("Параметры буквы З")
        z_layout = QGridLayout(z_group)
        
        z_layout.addWidget(QLabel("Высота:"), 0, 0)
        self.z_height_spin = QDoubleSpinBox()
        self.z_height_spin.setRange(0.5, 10.0)
        self.z_height_spin.setValue(2.0)
        self.z_height_spin.setSingleStep(0.1)
        self.z_height_spin.valueChanged.connect(self.update_letter_z_params)
        z_layout.addWidget(self.z_height_spin, 0, 1)
        
        z_layout.addWidget(QLabel("Ширина:"), 1, 0)
        self.z_width_spin = QDoubleSpinBox()
        self.z_width_spin.setRange(0.5, 10.0)
        self.z_width_spin.setValue(1.0)
        self.z_width_spin.setSingleStep(0.1)
        self.z_width_spin.valueChanged.connect(self.update_letter_z_params)
        z_layout.addWidget(self.z_width_spin, 1, 1)
        
        z_layout.addWidget(QLabel("Глубина:"), 2, 0)
        self.z_depth_spin = QDoubleSpinBox()
        self.z_depth_spin.setRange(0.1, 5.0)
        self.z_depth_spin.setValue(0.5)
        self.z_depth_spin.setSingleStep(0.1)
        self.z_depth_spin.valueChanged.connect(self.update_letter_z_params)
        z_layout.addWidget(self.z_depth_spin, 2, 1)
        
        letters_layout.addWidget(b_group)
        letters_layout.addWidget(z_group)
        
        # Вкладка трансформаций букв
        transform_tab = QWidget()
        transform_layout = QVBoxLayout(transform_tab)
        
        # Группа для трансформации буквы Б
        b_transform_group = QGroupBox("Трансформация буквы Б")
        b_transform_layout = QGridLayout(b_transform_group)
        
        # Вращение
        b_transform_layout.addWidget(QLabel("Вращение X:"), 0, 0)
        self.b_rot_x_slider = QSlider(Qt.Horizontal)
        self.b_rot_x_slider.setRange(0, 360)
        self.b_rot_x_slider.setValue(0)
        self.b_rot_x_slider.valueChanged.connect(self.update_letter_b_rotation)
        b_transform_layout.addWidget(self.b_rot_x_slider, 0, 1)
        
        b_transform_layout.addWidget(QLabel("Вращение Y:"), 1, 0)
        self.b_rot_y_slider = QSlider(Qt.Horizontal)
        self.b_rot_y_slider.setRange(0, 360)
        self.b_rot_y_slider.setValue(0)
        self.b_rot_y_slider.valueChanged.connect(self.update_letter_b_rotation)
        b_transform_layout.addWidget(self.b_rot_y_slider, 1, 1)
        
        b_transform_layout.addWidget(QLabel("Вращение Z:"), 2, 0)
        self.b_rot_z_slider = QSlider(Qt.Horizontal)
        self.b_rot_z_slider.setRange(0, 360)
        self.b_rot_z_slider.setValue(0)
        self.b_rot_z_slider.valueChanged.connect(self.update_letter_b_rotation)
        b_transform_layout.addWidget(self.b_rot_z_slider, 2, 1)
        
        # Перемещение
        b_transform_layout.addWidget(QLabel("Позиция X:"), 3, 0)
        self.b_pos_x_spin = QDoubleSpinBox()
        self.b_pos_x_spin.setRange(-10, 10)
        self.b_pos_x_spin.setValue(-1.5)
        self.b_pos_x_spin.setSingleStep(0.1)
        self.b_pos_x_spin.valueChanged.connect(self.update_letter_b_position)
        b_transform_layout.addWidget(self.b_pos_x_spin, 3, 1)
        
        b_transform_layout.addWidget(QLabel("Позиция Y:"), 4, 0)
        self.b_pos_y_spin = QDoubleSpinBox()
        self.b_pos_y_spin.setRange(-10, 10)
        self.b_pos_y_spin.setValue(0)
        self.b_pos_y_spin.setSingleStep(0.1)
        self.b_pos_y_spin.valueChanged.connect(self.update_letter_b_position)
        b_transform_layout.addWidget(self.b_pos_y_spin, 4, 1)
        
        b_transform_layout.addWidget(QLabel("Позиция Z:"), 5, 0)
        self.b_pos_z_spin = QDoubleSpinBox()
        self.b_pos_z_spin.setRange(-10, 10)
        self.b_pos_z_spin.setValue(0)
        self.b_pos_z_spin.setSingleStep(0.1)
        self.b_pos_z_spin.valueChanged.connect(self.update_letter_b_position)
        b_transform_layout.addWidget(self.b_pos_z_spin, 5, 1)
        
        # Группа для трансформации буквы З
        z_transform_group = QGroupBox("Трансформация буквы З")
        z_transform_layout = QGridLayout(z_transform_group)
        
        # Вращение
        z_transform_layout.addWidget(QLabel("Вращение X:"), 0, 0)
        self.z_rot_x_slider = QSlider(Qt.Horizontal)
        self.z_rot_x_slider.setRange(0, 360)
        self.z_rot_x_slider.setValue(0)
        self.z_rot_x_slider.valueChanged.connect(self.update_letter_z_rotation)
        z_transform_layout.addWidget(self.z_rot_x_slider, 0, 1)
        
        z_transform_layout.addWidget(QLabel("Вращение Y:"), 1, 0)
        self.z_rot_y_slider = QSlider(Qt.Horizontal)
        self.z_rot_y_slider.setRange(0, 360)
        self.z_rot_y_slider.setValue(0)
        self.z_rot_y_slider.valueChanged.connect(self.update_letter_z_rotation)
        z_transform_layout.addWidget(self.z_rot_y_slider, 1, 1)
        
        z_transform_layout.addWidget(QLabel("Вращение Z:"), 2, 0)
        self.z_rot_z_slider = QSlider(Qt.Horizontal)
        self.z_rot_z_slider.setRange(0, 360)
        self.z_rot_z_slider.setValue(0)
        self.z_rot_z_slider.valueChanged.connect(self.update_letter_z_rotation)
        z_transform_layout.addWidget(self.z_rot_z_slider, 2, 1)
        
        # Перемещение
        z_transform_layout.addWidget(QLabel("Позиция X:"), 3, 0)
        self.z_pos_x_spin = QDoubleSpinBox()
        self.z_pos_x_spin.setRange(-10, 10)
        self.z_pos_x_spin.setValue(1.5)
        self.z_pos_x_spin.setSingleStep(0.1)
        self.z_pos_x_spin.valueChanged.connect(self.update_letter_z_position)
        z_transform_layout.addWidget(self.z_pos_x_spin, 3, 1)
        
        z_transform_layout.addWidget(QLabel("Позиция Y:"), 4, 0)
        self.z_pos_y_spin = QDoubleSpinBox()
        self.z_pos_y_spin.setRange(-10, 10)
        self.z_pos_y_spin.setValue(0)
        self.z_pos_y_spin.setSingleStep(0.1)
        self.z_pos_y_spin.valueChanged.connect(self.update_letter_z_position)
        z_transform_layout.addWidget(self.z_pos_y_spin, 4, 1)
        
        z_transform_layout.addWidget(QLabel("Позиция Z:"), 5, 0)
        self.z_pos_z_spin = QDoubleSpinBox()
        self.z_pos_z_spin.setRange(-10, 10)
        self.z_pos_z_spin.setValue(0)
        self.z_pos_z_spin.setSingleStep(0.1)
        self.z_pos_z_spin.valueChanged.connect(self.update_letter_z_position)
        z_transform_layout.addWidget(self.z_pos_z_spin, 5, 1)
        
        transform_layout.addWidget(b_transform_group)
        transform_layout.addWidget(z_transform_group)
        
        # Вкладка управления камерой
        camera_tab = QWidget()
        camera_layout = QGridLayout(camera_tab)
        
        camera_layout.addWidget(QLabel("Позиция камеры X:"), 0, 0)
        self.camera_x_spin = QDoubleSpinBox()
        self.camera_x_spin.setRange(-20, 20)
        self.camera_x_spin.setValue(0)
        self.camera_x_spin.setSingleStep(0.5)
        self.camera_x_spin.valueChanged.connect(self.update_camera_position)
        camera_layout.addWidget(self.camera_x_spin, 0, 1)
        
        camera_layout.addWidget(QLabel("Позиция камеры Y:"), 1, 0)
        self.camera_y_spin = QDoubleSpinBox()
        self.camera_y_spin.setRange(-20, 20)
        self.camera_y_spin.setValue(0)
        self.camera_y_spin.setSingleStep(0.5)
        self.camera_y_spin.valueChanged.connect(self.update_camera_position)
        camera_layout.addWidget(self.camera_y_spin, 1, 1)
        
        camera_layout.addWidget(QLabel("Позиция камеры Z:"), 2, 0)
        self.camera_z_spin = QDoubleSpinBox()
        self.camera_z_spin.setRange(1, 30)
        self.camera_z_spin.setValue(10)
        self.camera_z_spin.setSingleStep(0.5)
        self.camera_z_spin.valueChanged.connect(self.update_camera_position)
        camera_layout.addWidget(self.camera_z_spin, 2, 1)
        
        camera_layout.addWidget(QLabel("Вращение камеры X:"), 3, 0)
        self.camera_rot_x_slider = QSlider(Qt.Horizontal)
        self.camera_rot_x_slider.setRange(-180, 180)
        self.camera_rot_x_slider.setValue(0)
        self.camera_rot_x_slider.valueChanged.connect(self.update_camera_rotation)
        camera_layout.addWidget(self.camera_rot_x_slider, 3, 1)
        
        camera_layout.addWidget(QLabel("Вращение камеры Y:"), 4, 0)
        self.camera_rot_y_slider = QSlider(Qt.Horizontal)
        self.camera_rot_y_slider.setRange(-180, 180)
        self.camera_rot_y_slider.setValue(0)
        self.camera_rot_y_slider.valueChanged.connect(self.update_camera_rotation)
        camera_layout.addWidget(self.camera_rot_y_slider, 4, 1)
        
        # Вкладка настроек отображения
        render_tab = QWidget()
        render_layout = QVBoxLayout(render_tab)
        
        # Группа режим отображения
        render_mode_group = QGroupBox("Режим отображения")
        render_mode_layout = QVBoxLayout(render_mode_group)
        
        self.wireframe_radio = QRadioButton("Каркасная модель")
        self.wireframe_radio.setChecked(True)
        self.wireframe_radio.toggled.connect(self.update_render_mode)
        render_mode_layout.addWidget(self.wireframe_radio)
        
        self.hidden_surface_radio = QRadioButton("Удаление невидимых поверхностей")
        self.hidden_surface_radio.toggled.connect(self.update_render_mode)
        render_mode_layout.addWidget(self.hidden_surface_radio)
        
        self.solid_radio = QRadioButton("Закрашивание поверхностей")
        self.solid_radio.toggled.connect(self.update_render_mode)
        render_mode_layout.addWidget(self.solid_radio)
        
        render_layout.addWidget(render_mode_group)
        
        # Группа методы закрашивания
        shading_group = QGroupBox("Метод закрашивания")
        shading_layout = QVBoxLayout(shading_group)
        
        self.flat_radio = QRadioButton("Монотонное")
        self.flat_radio.setChecked(True)
        self.flat_radio.toggled.connect(self.update_shading_mode)
        shading_layout.addWidget(self.flat_radio)
        
        self.gouraud_radio = QRadioButton("По Гуро")
        self.gouraud_radio.toggled.connect(self.update_shading_mode)
        shading_layout.addWidget(self.gouraud_radio)
        
        self.phong_radio = QRadioButton("По Фонгу")
        self.phong_radio.toggled.connect(self.update_shading_mode)
        shading_layout.addWidget(self.phong_radio)
        
        render_layout.addWidget(shading_group)
        
        # Настройка источника света
        light_group = QGroupBox("Источник света")
        light_layout = QGridLayout(light_group)
        
        light_layout.addWidget(QLabel("Позиция света X:"), 0, 0)
        self.light_x_spin = QDoubleSpinBox()
        self.light_x_spin.setRange(-20, 20)
        self.light_x_spin.setValue(5)
        self.light_x_spin.setSingleStep(0.5)
        self.light_x_spin.valueChanged.connect(self.update_light_position)
        light_layout.addWidget(self.light_x_spin, 0, 1)
        
        light_layout.addWidget(QLabel("Позиция света Y:"), 1, 0)
        self.light_y_spin = QDoubleSpinBox()
        self.light_y_spin.setRange(-20, 20)
        self.light_y_spin.setValue(5)
        self.light_y_spin.setSingleStep(0.5)
        self.light_y_spin.valueChanged.connect(self.update_light_position)
        light_layout.addWidget(self.light_y_spin, 1, 1)
        
        light_layout.addWidget(QLabel("Позиция света Z:"), 2, 0)
        self.light_z_spin = QDoubleSpinBox()
        self.light_z_spin.setRange(-20, 20)
        self.light_z_spin.setValue(5)
        self.light_z_spin.setSingleStep(0.5)
        self.light_z_spin.valueChanged.connect(self.update_light_position)
        light_layout.addWidget(self.light_z_spin, 2, 1)
        
        render_layout.addWidget(light_group)
        
        # Автомасштабирование и анимация
        misc_group = QGroupBox("Дополнительно")
        misc_layout = QVBoxLayout(misc_group)
        
        self.auto_scale_check = QCheckBox("Автомасштабирование")
        self.auto_scale_check.setChecked(True)
        self.auto_scale_check.toggled.connect(self.update_auto_scale)
        misc_layout.addWidget(self.auto_scale_check)
        
        render_layout.addWidget(misc_group)
        
        # Добавление вкладок
        tab_widget.addTab(letters_tab, "Параметры букв")
        tab_widget.addTab(transform_tab, "Трансформации")
        tab_widget.addTab(camera_tab, "Камера")
        tab_widget.addTab(render_tab, "Отображение")
        
        control_layout.addWidget(tab_widget)
        
        # Добавление виджетов в основной лейаут
        main_layout.addWidget(self.render_widget, 2)
        main_layout.addWidget(control_panel, 1)
        
        self.setCentralWidget(central_widget)
    
    def update_letter_b_params(self):
        height = self.b_height_spin.value()
        width = self.b_width_spin.value()
        depth = self.b_depth_spin.value()
        self.letter_b.set_dimensions(height, width, depth)
        self.render_widget.update()
    
    def update_letter_z_params(self):
        height = self.z_height_spin.value()
        width = self.z_width_spin.value()
        depth = self.z_depth_spin.value()
        self.letter_z.set_dimensions(height, width, depth)
        self.render_widget.update()
    
    def update_letter_b_rotation(self):
        x_rot = self.b_rot_x_slider.value()
        y_rot = self.b_rot_y_slider.value()
        z_rot = self.b_rot_z_slider.value()
        self.letter_b.set_rotation(x_rot, y_rot, z_rot)
        self.render_widget.update()
    
    def update_letter_z_rotation(self):
        x_rot = self.z_rot_x_slider.value()
        y_rot = self.z_rot_y_slider.value()
        z_rot = self.z_rot_z_slider.value()
        self.letter_z.set_rotation(x_rot, y_rot, z_rot)
        self.render_widget.update()
    
    def update_letter_b_position(self):
        x = self.b_pos_x_spin.value()
        y = self.b_pos_y_spin.value()
        z = self.b_pos_z_spin.value()
        self.letter_b.set_position(Point3D(x, y, z))
        self.render_widget.update()
    
    def update_letter_z_position(self):
        x = self.z_pos_x_spin.value()
        y = self.z_pos_y_spin.value()
        z = self.z_pos_z_spin.value()
        self.letter_z.set_position(Point3D(x, y, z))
        self.render_widget.update()
    
    def update_camera_position(self):
        x = self.camera_x_spin.value()
        y = self.camera_y_spin.value()
        z = self.camera_z_spin.value()
        self.camera.set_position(Point3D(x, y, z))
        self.render_widget.update()
    
    def update_camera_rotation(self):
        x_rot = self.camera_rot_x_slider.value()
        y_rot = self.camera_rot_y_slider.value()
        self.camera.set_rotation(x_rot, y_rot)
        self.render_widget.update()
    
    def update_render_mode(self):
        if self.wireframe_radio.isChecked():
            self.render_widget.set_render_mode("wireframe")
        elif self.hidden_surface_radio.isChecked():
            self.render_widget.set_render_mode("hidden_surface")
        elif self.solid_radio.isChecked():
            self.render_widget.set_render_mode("solid")
        self.render_widget.update()
    
    def update_shading_mode(self):
        if self.flat_radio.isChecked():
            self.render_widget.set_shading_mode("flat")
        elif self.gouraud_radio.isChecked():
            self.render_widget.set_shading_mode("gouraud")
        elif self.phong_radio.isChecked():
            self.render_widget.set_shading_mode("phong")
        self.render_widget.update()
    
    def update_light_position(self):
        x = self.light_x_spin.value()
        y = self.light_y_spin.value()
        z = self.light_z_spin.value()
        self.light.set_position(Point3D(x, y, z))
        self.render_widget.update()
    
    def update_auto_scale(self):
        self.render_widget.set_auto_scale(self.auto_scale_check.isChecked())
        self.render_widget.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 