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
    # Use errstate to ignore divide warnings and set values near the singularity to np.nan.
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 10 / (x - 1)
        # When x is close to 1, set the result to NaN.
        y[np.abs(x - 1) < 1e-6] = np.nan
        return y

def get_function_data(func_id, start, end, num_points):
    # If the number of points exactly equals end - start + 1, use integer values.
    if num_points == int(end - start) + 1:
        x = np.arange(start, end + 1)
    else:
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