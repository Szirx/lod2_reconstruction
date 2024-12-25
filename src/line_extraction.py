from shapely.geometry import LineString
from skimage.measure import find_contours
from scipy.spatial import Delaunay
import numpy as np
from typing import List


def extract_intersection_lines(
    triangulation: Delaunay,
    height_map: np.ndarray,
) -> List[LineString]:
    """ Извлечение границ кластеров высот

    Args:
        triangulation (Delaunay): нерегулярная триангуляционная сеть карты кластеризации высот
        height_map (np.ndarray): карта кластеризации высот

    Returns:
        List[LineString]: Список с линиями границ кластеров высот
    """
    edges = set()
    for simplex in triangulation.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            edges.add(edge)

    intersection_lines: list = []
    for edge in edges:
        p1, p2 = triangulation.points[edge[0]], triangulation.points[edge[1]]
        if height_map[int(p1[1]), int(p1[0])] != height_map[int(p2[1]), int(p2[0])]:
            intersection_lines.append(LineString([p1, p2]))
    return intersection_lines


def extract_step_lines(height_map: np.ndarray, tolerance: int = 1) -> List[LineString]:
    step_lines = []
    contours = find_contours(height_map, level=10)
    for contour in contours:
        line = LineString(contour)
        simplified = line.simplify(tolerance)
        step_lines.append(simplified)
    return step_lines