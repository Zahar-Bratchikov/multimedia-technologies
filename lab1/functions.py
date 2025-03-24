import numpy as np


def function_1(x):
    """
    Вычисляет 5 * cos(x)
    """
    return 5 * np.cos(x)


def function_2(x):
    """
    Вычисляет 10 * sin(x) + 5 * cos(2 * x)
    """
    return 10 * np.sin(x) + 5 * np.cos(2 * x)


def function_3(x):
    """
    Вычисляет 10 / (x - 1) с обработкой деления на 0.
    Если модуль (x-1) меньше 1e-6, возвращается NaN.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 10 / (x - 1)
        y[np.abs(x - 1) < 1e-6] = np.nan
        return y


def get_function_data(func_id, start, end, num_points):
    """
    Генерирует массивы значений x и y для выбранной функции на интервале [start, end].
    Если num_points == 1, создаётся массив с одной точкой, размещаемой в центре диапазона.
    Для num_points > 1 шаг вычисляется как:
        step = (end - start) / (num_points - 1)
    Это гарантирует, что первый элемент равен start, а последний — end.

    Аргументы:
      - func_id: Идентификатор функции.
      - start: Начальное значение интервала.
      - end: Конечное значение интервала.
      - num_points: Количество точек для вычисления значений.

    Возвращает:
      Кортеж (x, y, label), где x и y – массивы значений, а label – подпись функции.
    """
    if num_points < 1:
        raise ValueError("num_points должен быть не менее 1 для построения графика.")
    if num_points == 1:
        x = np.array([(start + end) / 2])
    else:
        step = (end - start) / (num_points - 1)
        x = np.array([start + i * step for i in range(num_points)])
    if func_id == 1:
        y = function_1(x)
        label = "Функция 1 (5*cos(x))"
    elif func_id == 2:
        y = function_2(x)
        label = "Функция 2 (10*sin(x)+5*cos(2*x))"
    elif func_id == 3:
        y = function_3(x)
        label = "Функция 3 (10/(x-1))"
    else:
        raise ValueError("Неизвестный идентификатор функции")
    return x, y, label