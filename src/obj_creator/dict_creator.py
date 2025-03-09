import numpy as np
import warnings

from shapely.geometry import Polygon
from shapely.vectorized import contains


def create_image_dict(height_map: np.ndarray, pixel_multipolygons: list, resolution: int | float) -> dict:
    """
    Функция принимает массив высот в виде numpy ndarray размером HxW,
    список пиксельных полигонов в виде:
        [
            [[x1, y1], [x2, y2], ...],
            [[x1, y1], [x2, y2], ...],
            ...
        ]
    и разрешение спутникового снимка в формате метры/пиксель.
    Возвращает словарь с контурами и соответствующими высотами.

    Args:
        height_map (np.ndarray): Массив высот размером HxW в формате numpy ndarray.
        pixel_multipolygons (list): Список пиксельных полигонов, где каждый полигон представлен списком координат,
                                     например, [[[x1, y1], [x2, y2], ...], ...].
        resolution (int | float): Разрешение спутникового снимка в метрах на пиксель.

    Returns:
        dict: Словарь, содержащий контура и соответствующие высоты, где ключами являются индексы полигонов,
              а значениями - словари с полями 'contours', 'height', 'height_meters' и 'category'.
    """

    contours_with_height = {}  
    h, w = height_map.shape  
    
    for i, (contour, category) in enumerate(pixel_multipolygons):
        if len(contour) < 3:
            continue
        
        polygon = Polygon(contour)
        
        # Получаем границы полигона
        minx, miny, maxx, maxy = polygon.bounds
        minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
        
        # Обрезаем карту высот по границам полигона
        cropped_height_map = height_map[miny:maxy + 1, minx:maxx + 1]
        
        # Корректируем координаты контура
        adjusted_contour = [(x - minx, y - miny) for x, y in contour]
        adjusted_polygon = Polygon(adjusted_contour)
        
        # Создаем сетку координат для обрезанной карты высот
        y, x = np.indices(cropped_height_map.shape)
        coords = np.c_[x.ravel(), y.ravel()]
        
        # Создаем маску, которая показывает, какие пиксели находятся внутри полигона
        mask = contains(adjusted_polygon, coords[:, 0], coords[:, 1]).reshape(cropped_height_map.shape)
        
        if not np.any(mask):
            warnings.warn(f'Warning: Polygon {i} does not contain any pixels in the height map.')
            building_height_mean_meters = 10 
        else:
            building_height_mean_meters = np.mean(cropped_height_map[mask])
        
        building_height_mean_obj = building_height_mean_meters / resolution
        
        new_contour = [[point[0], h - point[1]] for point in contour]
        contours_with_height[i] = {
            'contours': new_contour,
            'height': int(building_height_mean_obj),
            'height_meters': building_height_mean_meters,
            'category': category,
        }
    
    return contours_with_height  
