import numpy as np
from typing import Tuple, List, Any


def detect_flat_surfaces(
    surface_matrix: np.ndarray,
    height_matrix: np.ndarray,
    flatness_threshold: float = 3,
) -> Tuple[list]:
    """
    Определяет плоские и неровные поверхности по разбросу высот внутри каждой области.

    :param surface_matrix: 2D numpy массив, где указаны номера областей
    :param height_matrix: 2D numpy массив, где указаны значения высот
    :param flatness_threshold: Порог отклонения, ниже которого поверхность считается плоской
    :return: Список номеров плоских и неровных поверхностей
    """
    unique_surfaces = np.unique(surface_matrix)
    flat_surfaces: list = []
    rough_surfaces: list = []
    for surface_id in unique_surfaces:
        # Создание маски для текущей поверхности
        mask = surface_matrix == surface_id
        heights = height_matrix[mask]

        if heights.size == 0:
            continue

        # Вычисление статистик высоты
        min_height = np.min(heights)
        max_height = np.max(heights)
        height_range = max_height - min_height  # Размах высот
        std_dev = np.std(heights)  # Стандартное отклонение
        median_height = np.median(heights)

        # Использование межквартильного размаха (IQR) для поиска выбросов
        q1, q3 = np.percentile(heights, [25, 75])
        iqr = q3 - q1

        # Определение "плоскости" по стандартному отклонению и IQR
        if height_range < flatness_threshold * median_height and std_dev < flatness_threshold and iqr < flatness_threshold:
            flat_surfaces.append(int(surface_id.item()))
        else:
            rough_surfaces.append(int(surface_id.item()))

    return flat_surfaces, rough_surfaces


def masked_mean_by_surfaces(
    surfaces_matrix: np.ndarray,
    value_matrix: np.ndarray,
    flat_surfaces: List[int],
    min_value: float = 0.0):
    """
    Вычисляет среднее значение в каждой области surface_matrix, исключая элементы value_matrix < min_value.

    :param surfaces_matrix: 2D numpy массив с целыми числами (метки областей)
    :param value_matrix: 2D numpy массив с плавающими значениями
    :param min_value: Порог для включения значений в среднее
    :return: Словарь {метка области: среднее значение}
    """
    surfaces_means: dict = {}

    for surface in flat_surfaces:
        mask = (surfaces_matrix == surface) & (value_matrix >= min_value)  # Учитываем только >= min_value
        values = value_matrix[mask]

        if len(values) > 0:
            surfaces_means[surface] = int(np.mean(values))
        else:
            surfaces_means[surface] = np.nan  # Если нет подходящих значений, ставим NaN

    return surfaces_means


def compute_robust_linear_gradient(
    surface_matrix: np.ndarray,
    height_matrix: np.ndarray,
    rough_surfaces: List[int],
) -> np.ndarray:
    """
    Создает линейный градиент внутри каждой области, используя метод наименьших квадратов 
    для нахождения направления наибольшего изменения высот.

    :param surface_matrix: 2D numpy массив, где указаны номера областей
    :param height_matrix: 2D numpy массив, где указаны значения высот
    :return: 2D numpy массив с линейным градиентом по областям
    """
    gradient_matrix = np.zeros_like(height_matrix, dtype=np.float32)

    for surface_id in rough_surfaces:
        if surface_id == 0:  # Пропускаем фон, если он есть
            continue
        
        # Маска текущей области
        mask = (surface_matrix == surface_id)
        indices = np.argwhere(mask)

        # Получаем высоты внутри области
        heights = height_matrix[mask]

        # Центрируем координаты относительно центра области
        centroid = np.mean(indices, axis=0)
        X = indices - centroid  # (N, 2)

        # Решаем уравнение Ax = b с помощью линейной регрессии
        # A = [x y], b = heights
        A = np.c_[X[:, 0], X[:, 1], np.ones(X.shape[0])]
        coeffs, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)  # [a, b, c] -> плоскость z = ax + by + c

        # Вычисляем значения градиента в направлении [grad_x, grad_y]
        for (x, y) in indices:
            gradient_matrix[x, y] = coeffs[0] * (x - centroid[0]) + coeffs[1] * (y - centroid[1]) + np.mean(heights)

    return gradient_matrix


def apply_flat_surface_heights(surface_matrix, gradient_matrix, flat_surface_heights):
    """
    Заменяет высоты в градиентной матрице фиксированными значениями для плоских поверхностей.

    :param surface_matrix: 2D numpy массив, где указаны номера областей
    :param gradient_matrix: 2D numpy массив с градиентом высот
    :param flat_surface_heights: словарь {номер поверхности: фиксированная высота}
    :return: Обновленная градиентная матрица
    """
    result = gradient_matrix.copy()

    for surface_id, height in flat_surface_heights.items():
        mask = (surface_matrix == surface_id)
        result[mask] = height  # Заменяем все пиксели данной области на высоту из словаря

    return result