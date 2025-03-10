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
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 10 / (x - 1)
        y[np.abs(x - 1) < 1e-6] = np.nan
        return y

def get_function_data(func_id, start, end, num_points):
    """
    Generates x and y data for the selected function over the interval [start, end].
    The step is calculated as (end - start) / num_points.
    Cones are plotted for x values:
      x = start + i * step    for i in range(num_points)
    This ensures that the first cone is drawn at 'start' and all cones are within [start, end].
    """
    step = (end - start) / num_points
    # Generate exactly num_points x-values starting from 'start'
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
        raise ValueError("Unknown function ID")
    return x, y, label