import numpy as np


def function_1(x):
    """
    5*cos(x) - вычисляет 5*cos(x)
    """
    return 5 * np.cos(x)


def function_2(x):
    """
    10*sin(x) + 5*cos(2*x) - вычисляет 10*sin(x) + 5*cos(2*x)
    """
    return 10 * np.sin(x) + 5 * np.cos(2 * x)


def function_3(x):
    """
    10/(x-1) - вычисляет 10/(x-1) с обработкой деления на ноль
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 10 / (x - 1)
        # Если abs(x-1) меньше 1e-6, присваиваем значение NaN.
        y[np.abs(x - 1) < 1e-6] = np.nan
        return y


# def function_4(x):
#     """
#     x**2
#     """
#     return x ** 2


def get_function_data(func_id, start, end, num_points):
    """
    Генерирует данные x и y для выбранной функции на интервале [start, end].
    Шаг (step) вычисляется по формуле (end - start) / num_points.
    Конусы строятся для точек x, вычисленных как:
      x = start + i * step,  где i от 0 до num_points-1.
    Это гарантирует, что первый конус строится в точке 'start' и все конусы находятся в пределах [start, end].
    """
    step = (end - start) / num_points
    # Генерируем ровно num_points значений x, начиная с 'start'
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
    # elif func_id == 4:
    #     y = function_4(x)
    #     label = "Функция 4 (x ** 2)"
    else:
        raise ValueError("Неизвестный идентификатор функции")
    return x, y, label
