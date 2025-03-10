import numpy as np
from typing import Tuple, List
from shapely.geometry import Polygon


def detect_flat_surfaces(
    surface_matrix: np.ndarray,
    height_matrix: np.ndarray,
    flatness_thd: float = 3.0,
) -> Tuple[list]:
    """
    Определяет плоские и неровные поверхности по разбросу высот внутри каждой области.

    :param surface_matrix: 2D numpy массив, где указаны номера областей
    :param height_matrix: 2D numpy массив, где указаны значения высот
    :param flatness_thd: Порог отклонения, ниже которого поверхность считается плоской
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

        height_range = np.max(heights) - np.min(heights)  # Размах высот
        std_dev = np.std(heights)  # Стандартное отклонение
        median_height = np.median(heights)

        # Использование межквартильного размаха (IQR) для поиска выбросов
        q1, q3 = np.percentile(heights, [25, 75])
        iqr = q3 - q1

        # Определение "плоскости" по стандартному отклонению и IQR
        if height_range < flatness_thd * median_height and std_dev < flatness_thd and iqr < flatness_thd:
            flat_surfaces.append(int(surface_id.item()))
        else:
            rough_surfaces.append(int(surface_id.item()))

    return flat_surfaces, rough_surfaces


def masked_mean_by_surfaces(
    surfaces_matrix: np.ndarray,
    value_matrix: np.ndarray,
    flat_surfaces: List[int],
    min_value: float = 0,
) -> dict:
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
        masked_values = value_matrix[mask]

        if len(masked_values) > 0:
            surfaces_means[surface] = int(np.mean(masked_values))
        else:
            surfaces_means[surface] = np.nan  # Если нет подходящих значений, ставим NaN

    return surfaces_means


def detect_right_vector(polygon: Polygon, non_rob_vector: np.ndarray) -> set:
    """ Определение направление вектора градиента вдоль какой либо оси плоскости

    Args:
        polygon (Polygon): Полигон плоскости
        non_rob_vector (np.ndarray): Исходный вектор направления градиента

    Returns:
        set: Обновленный вектор градиента
    """
    # Вычисляем векторы сторон и их коэффициенты направления
    vertices = list(polygon.exterior.coords)
    direction_vectors = set()
    for i in range(len(vertices) - 1):
        # Текущая и следующая вершины
        start = vertices[i]
        end = vertices[i + 1]

        # Вектор стороны
        vector = np.array(end) - np.array(start)

        # Нормализация вектора (коэффициенты направления)
        norm = np.linalg.norm(vector)
        if norm == 0:  # Избегаем деления на ноль
            direction_vector = vector  # Если длина вектора 0, оставляем как есть
        else:
            direction_vector = vector / norm
        direction_vector = tuple(direction_vector.astype(np.float16))
        if direction_vector not in direction_vectors:
            direction_vectors.add(direction_vector)

    angles = []
    for direction_vector in direction_vectors:
        # Скалярное произведение
        dot_product = np.dot(non_rob_vector, direction_vector)
        # Угол в радианах
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Ограничиваем значение для арккосинуса
        angles.append(angle)

    # Находим индекс ближайшего вектора
    closest_index = np.argmin(angles)
    direction_vectors = list(direction_vectors)
    closest_vector = direction_vectors[closest_index]

    return closest_vector


def compute_robust_linear_gradient(
    surface_matrix: np.ndarray,
    height_matrix: np.ndarray,
    rough_surfaces: List[int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Создает линейный градиент внутри каждой области, используя метод наименьших квадратов
    для нахождения направления наибольшего изменения высот.

    :param surface_matrix: 2D numpy массив, где указаны номера областей
    :param height_matrix: 2D numpy массив, где указаны значения высот
    :param rough_surfaces: Список идентификаторов областей, для которых нужно вычислить градиент
    :return: 2D numpy массив с линейным градиентом по областям и список матриц с градиентами для каждой области
    """
    gradient_matrix = np.zeros_like(height_matrix, dtype=np.float32)
    gradient_matrices = []

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
        A = np.c_[
            X[:, 0],
            X[:, 1],
            np.ones(X.shape[0]),
        ]
        coeffs, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)  # [a, b, c] -> плоскость z = ax + by + c

        # Создаем матрицу градиента для текущей области
        gradient_matrix_surface = np.zeros_like(height_matrix, dtype=np.float32)

        # Вычисляем значения градиента для всей области
        for x in range(height_matrix.shape[0]):
            for y in range(height_matrix.shape[1]):
                gradient_matrix_surface[x, y] = coeffs[0] * \
                    (x - centroid[0]) + coeffs[1] * \
                    (y - centroid[1]) + np.mean(heights)

        # Добавляем матрицу градиента для текущей области в список
        gradient_matrices.append(gradient_matrix_surface)

        # Обновляем общую матрицу градиентов только в пределах маски
        gradient_matrix[mask] = gradient_matrix_surface[mask]

    return gradient_matrix, gradient_matrices


def apply_flat_surface_heights(
    surface_matrix: np.ndarray,
    gradient_matrix: np.ndarray,
    flat_surface_heights: dict,
) -> np.ndarray:
    """
    Заменяет высоты в градиентной матрице фиксированными значениями для плоских поверхностей.

    :param surface_matrix: 2D numpy массив, где указаны номера областей
    :param gradient_matrix: 2D numpy массив с градиентом высот
    :param flat_surface_heights: словарь {номер поверхности: фиксированная высота}
    :return: Обновленная градиентная матрица
    """
    result_gm = gradient_matrix.copy()

    for surface_id, height in flat_surface_heights.items():
        mask = (surface_matrix == surface_id)
        result_gm[mask] = height

    return result_gm
