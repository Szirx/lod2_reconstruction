import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple


def create_tin(height_map: np.ndarray) -> Tuple[np.ndarray, Delaunay]:
    """ Создание нерегулярной триангулярной сети (TIN)

    Args:
        height_map (np.ndarray): Полученная карта высот | кластеризованная карта высот

    Returns:
        Tuple[np.ndarray, Delaunay]: Координаты точек (x, y) для построения визуализации и объект триангуляции
    """
    rows, cols = height_map.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.column_stack((x.ravel(), y.ravel()))

    return points, Delaunay(points)