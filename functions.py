import numpy as np

def func1(start, end, num_points):
    """
    Генерирует данные для функции: 5*cos(x)
    Входные данные:
      start - начало интервала
      end - конец интервала
      num_points - количество точек
    Выходные данные:
      x, y, label – координаты точек и строковое представление функции.
    """
    x = np.linspace(start, end, num_points)
    y = 5 * np.cos(x)
    label = "5*cos(x)"
    return x, y, label

def func2(start, end, num_points):
    """
    Генерирует данные для функции: 10*sin(x) + 5*cos(2*x)
    Входные данные:
      start - начало интервала
      end - конец интервала
      num_points - количество точек
    Выходные данные:
      x, y, label – координаты точек и строковое представление функции.
    """
    x = np.linspace(start, end, num_points)
    y = 10 * np.sin(x) + 5 * np.cos(2 * x)
    label = "10*sin(x) + 5*cos(2*x)"
    return x, y, label

def func3(start, end, num_points):
    """
    Генерирует данные для функции: 10/(x-1)
    Функция имеет точку разрыва при x = 1. При приближении к x=1 значение заменяется на NaN.
    Входные данные:
      start - начало интервала
      end - конец интервала
      num_points - количество точек
    Выходные данные:
      x, y, label – координаты точек и строковое представление функции.
    """
    x = np.linspace(start, end, num_points)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.where(np.abs(x - 1) < 1e-6, np.nan, 10 / (x - 1))
    label = "10/(x-1)"
    return x, y, label

def get_function_data(func_num, start, end, num_points):
    """
    Возвращает данные для построения диаграммы по заданному номеру функции.
    Поддерживаемые номера: 1, 2, 3.
    """
    if func_num == 1:
        return func1(start, end, num_points)
    elif func_num == 2:
        return func2(start, end, num_points)
    elif func_num == 3:
        return func3(start, end, num_points)
    else:
        raise ValueError("Неверный номер функции. Допустимые значения: 1, 2, 3.")