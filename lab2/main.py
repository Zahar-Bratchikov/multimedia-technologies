import sys
import math
from PySide6 import QtCore, QtWidgets, QtGui, QtOpenGL
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import Qt
from OpenGL import GL

class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Параметры вращения куба относительно сетки
        self.cubeRotX = 0.0
        self.cubeRotY = 0.0
        self.cubeRotZ = 0.0
        # Параметры вращения сетки (всей сцены)
        self.gridRotX = 0.0
        self.gridRotY = 0.0
        self.gridRotZ = 0.0
        # Масштаб куба
        self.cubeScale = 1.0

        # Для отслеживания перемещения мыши
        self.lastPos = QtCore.QPoint()
        self._mouseButton = Qt.MouseButton.NoButton

        # Списки отображения для сетки и куба (будут созданы в initializeGL)
        self.gridList = None
        self.cubeList = None
        self.gl = None  # Initialize self.gl to None

    def initializeGL(self):
        context = self.context()
        if context is None:
            print("Error: OpenGL context is not valid.")
            return

        self.gl = context.functions()
        if self.gl is None:
             print("Error: OpenGL functions could not be loaded.")
             return
        self.gl.initializeOpenGLFunctions()
        self.gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        self.gl.glEnable(GL.GL_DEPTH_TEST)  # Use GL.GL_DEPTH_TEST
        self.gl.glEnable(GL.GL_CULL_FACE)   # Use GL.GL_CULL_FACE
        self.gl.glCullFace(GL.GL_BACK)     # Use GL.GL_BACK

        # Попытка создать display lists. Если они не поддерживаются, будем работать в режиме непосредственной отрисовки.
        try:
            self.gridList = GL.glGenLists(1)
            if self.gridList == 0:
                raise Exception("Display lists not supported.")
            GL.glNewList(self.gridList, GL.GL_COMPILE)  # Use GL.GL_COMPILE
            self.drawGridImmediate()
            GL.glEndList()

            self.cubeList = GL.glGenLists(1)
            if self.cubeList == 0:
                raise Exception("Display lists not supported.")
            GL.glNewList(self.cubeList, GL.GL_COMPILE)  # Use GL.GL_COMPILE
            self.drawCubeImmediate()
            GL.glEndList()
        except Exception as e:
            print("Error creating display lists:", e)
            self.gridList = None
            self.cubeList = None

    def resizeGL(self, width, height):
        if self.gl is not None:
            GL.glViewport(0, 0, width, height)

    def paintGL(self):
        if self.gl is not None:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT) # Use GL.GL_COLOR_BUFFER_BIT and GL.GL_DEPTH_BUFFER_BIT

            # Установка матрицы проекции через glFrustum (перспективная проекция)
            GL.glMatrixMode(GL.GL_PROJECTION) # Use GL.GL_PROJECTION
            GL.glLoadIdentity()
            aspect = self.width() / self.height() if self.height() != 0 else 1.0
            zNear = 1.0
            zFar = 100.0
            fov = 45.0  # угол обзора
            top = zNear * math.tan(math.radians(fov) / 2)
            bottom = -top
            right = top * aspect
            left = -right
            GL.glFrustum(left, right, bottom, top, zNear, zFar)

            # Настройка модели
            GL.glMatrixMode(GL.GL_MODELVIEW) # Use GL.GL_MODELVIEW
            GL.glLoadIdentity()
            GL.glTranslatef(0.0, 0.0, -15.0)

            # Применяем вращение для всей сетки (координатного пространства)
            GL.glRotatef(self.gridRotX, 1.0, 0.0, 0.0)
            GL.glRotatef(self.gridRotY, 0.0, 1.0, 0.0)
            GL.glRotatef(self.gridRotZ, 0.0, 0.0, 1.0)

            # Рисуем координатную сетку. Если display list создан, вызываем его, иначе – отрисовываем непосредственно.
            if self.gridList:
                GL.glCallList(self.gridList)
            else:
                self.drawGridImmediate()

            # Отрисовка куба
            GL.glPushMatrix()
            GL.glRotatef(self.cubeRotX, 1.0, 0.0, 0.0)
            GL.glRotatef(self.cubeRotY, 0.0, 1.0, 0.0)
            GL.glRotatef(self.cubeRotZ, 0.0, 0.0, 1.0)
            GL.glScalef(self.cubeScale, self.cubeScale, self.cubeScale)
            if self.cubeList:
                GL.glCallList(self.cubeList)
            else:
                self.drawCubeImmediate()
            GL.glPopMatrix()

    def drawGridImmediate(self):
        if self.gl is not None:
            GL.glLineWidth(1.0)
            GL.glBegin(GL.GL_LINES)  # Use GL.GL_LINES
            gridSize = 10
            step = 1
            # Линии, параллельные оси Z (изменение по оси X)
            for i in range(-gridSize, gridSize + 1):
                if i == 0:
                    GL.glColor3f(1, 0, 0)  # ось X красным
                else:
                    GL.glColor3f(0.5, 0.5, 0.5)
                GL.glVertex3f(i * step, 0, -gridSize * step)
                GL.glVertex3f(i * step, 0, gridSize * step)
            # Линии, параллельные оси X (изменение по оси Z)
            for i in range(-gridSize, gridSize + 1):
                if i == 0:
                    GL.glColor3f(0, 0, 1)  # ось Z синий
                else:
                    GL.glColor3f(0.5, 0.5, 0.5)
                GL.glVertex3f(-gridSize * step, 0, i * step)
                GL.glVertex3f(gridSize * step, 0, i * step)
            # Ось Y (вертикальная линия)
            GL.glColor3f(0, 1, 0)  # ось Y зеленый
            GL.glVertex3f(0, -gridSize * step, 0)
            GL.glVertex3f(0, gridSize * step, 0)
            GL.glEnd()

    def drawCubeImmediate(self):
        if self.gl is not None:
            GL.glBegin(GL.GL_QUADS) # Use GL.GL_QUADS
            # Лицевая грань (z = 1)
            GL.glColor3f(1, 0, 0)
            GL.glVertex3f(-1, -1, 1)
            GL.glVertex3f(1, -1, 1)
            GL.glVertex3f(1, 1, 1)
            GL.glVertex3f(-1, 1, 1)
            # Задняя грань (z = -1)
            GL.glColor3f(0, 1, 0)
            GL.glVertex3f(-1, -1, -1)
            GL.glVertex3f(-1, 1, -1)
            GL.glVertex3f(1, 1, -1)
            GL.glVertex3f(1, -1, -1)
            # Верхняя грань (y = 1)
            GL.glColor3f(0, 0, 1)
            GL.glVertex3f(-1, 1, -1)
            GL.glVertex3f(-1, 1, 1)
            GL.glVertex3f(1, 1, 1)
            GL.glVertex3f(1, 1, -1)
            # Нижняя грань (y = -1)
            GL.glColor3f(1, 1, 0)
            GL.glVertex3f(-1, -1, -1)
            GL.glVertex3f(1, -1, -1)
            GL.glVertex3f(1, -1, 1)
            GL.glVertex3f(-1, -1, 1)
            # Правая грань (x = 1)
            GL.glColor3f(1, 0, 1)
            GL.glVertex3f(1, -1, -1)
            GL.glVertex3f(1, 1, -1)
            GL.glVertex3f(1, 1, 1)
            GL.glVertex3f(1, -1, 1)
            # Левая грань (x = -1)
            GL.glColor3f(0, 1, 1)
            GL.glVertex3f(-1, -1, -1)
            GL.glVertex3f(-1, -1, 1)
            GL.glVertex3f(-1, 1, 1)
            GL.glVertex3f(-1, 1, -1)
            GL.glEnd()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.lastPos = event.position().toPoint()
        self._mouseButton = event.button()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        delta = event.position().toPoint() - self.lastPos
        self.lastPos = event.position().toPoint()
        if self._mouseButton == Qt.MouseButton.LeftButton:
            # Вращаем куб (относительно текущей сетки)
            self.cubeRotX += delta.y()
            self.cubeRotY += delta.x()
        elif self._mouseButton == Qt.MouseButton.RightButton:
            # Вращаем всю сетку
            self.gridRotX += delta.y()
            self.gridRotY += delta.x()
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y() / 120  # одно деление ~120
        self.cubeScale *= (1 + delta * 0.1)
        self.cubeScale = max(0.1, min(self.cubeScale, 10))
        self.update()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Cube and Grid Optimized")
        self.glWidget = GLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.resize(800, 600)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())