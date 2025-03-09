import numpy as np

def function_1(x):
    """
    5*cos(x)
    """
    return 5 * np.cos(x)

def function_2(x):
    """
    10*sin(x) + 5*cos(2*x)
    """
    return 10 * np.sin(x) + 5 * np.cos(2 * x)

def function_3(x):
    """
    10/(x-1)
    """
    return 10 / (x - 1)

def get_function_data(func_id, start, end, num_points):
    x = np.linspace(start, end, num_points)
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
        raise ValueError("Unknown function ID")
    return x, y, label