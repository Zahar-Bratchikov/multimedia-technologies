"""
Модуль для работы с математическими функциями.

Этот модуль содержит функции для вычисления различных математических выражений
и генерации данных для их визуализации. Основные функции:

- function_1: 5 * cos(x)
- function_2: 10 * sin(x) + 5 * cos(2 * x)
- function_3: 10 / (x - 1)

Модуль также предоставляет функцию get_function_data для генерации данных
для построения графиков.
"""

import numpy as np


def function_1(x):
    """
    5 * cos(x).
    """
    return 5 * np.cos(x)


def function_2(x):
    """
    10 * sin(x) + 5 * cos(2 * x).
    """
    return 10 * np.sin(x) + 5 * np.cos(2 * x)


def function_3(x):
    """
    10 / (x - 1) с обработкой деления на 0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 10 / (x - 1)
        y[np.abs(x - 1) < 1e-6] = np.nan
        return y


def get_function_data(func_id, start, end, num_points):
    """
    Генерирует массивы значений x и y для выбранной функции на интервале [start, end].

    Функция создает равномерно распределенные точки на заданном интервале
    и вычисляет значения выбранной функции в этих точках.

    Args:
        func_id (int): Идентификатор функции (1, 2 или 3).
        start (float): Начальное значение интервала.
        end (float): Конечное значение интервала.
        num_points (int): Количество точек для вычисления значений.

    Returns:
        tuple: Кортеж (x, y, label), где:
            - x (numpy.ndarray): Массив значений аргумента
            - y (numpy.ndarray): Массив значений функции
            - label (str): Подпись функции для отображения на графике

    Raises:
        ValueError: Если num_points < 1 или указан неверный func_id.
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