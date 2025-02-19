from shapely.geometry import Polygon, Point
from skimage.measure import find_contours
from typing import List, Dict, Tuple
from src.approximate import douglas_peucker
from skimage.morphology import remove_small_objects
import numpy as np
from scipy.ndimage import binary_closing

def check_close_contour(contour: List[List[float]]) -> bool:
    return True if contour[0][0] == contour[-1][0] and contour[0][1] == contour[-1][1] else False


def fix_and_simplification_contour(
    contour: List[np.ndarray],
    tolerance: int = 3,
) -> List[np.ndarray]:
    """ Фикс полигона для применения Дугласа-Пеккера

    Args:
        contour (List[np.ndarray]): Исходный контур
        tolerance (int, optional): Степень аппроксимации. Defaults to 3.

    Returns:
        List[np.ndarray]: Аппроксимированный контур объекта
    """
    if check_close_contour(contour):
        contour = contour[:-1]
    
    return douglas_peucker(contour, tolerance=tolerance)


def create_polygons(
    instance_masks: Dict[np.uint8, Tuple[np.ndarray, int]],
    tolerance: int = 3,
) -> Dict[np.uint8, List[Polygon]]:
    """ Создание shapely полигонов поверхностей

    Args:
        instance_masks (Dict[np.uint8, Tuple[np.ndarray, int]]): Словарь с масками поверхностей
        tolerance (int): степень аппроксимации (default: 3)
    Returns:
        Dict[np.uint8, List[Polygon]]: Словарь высота : набор полигонов поверхностей
    """
    
    contours_dict: dict = {}

    for height, (mask, quantity) in instance_masks.items():
        contours: list = []
        
        for instance in range(1, quantity + 1):
            contours_sample: List[np.ndarray | List[np.ndarray]] = find_contours(
                (mask == instance).astype(np.int8),
            )
            if len(contours_sample) == 1:
                cntr = fix_and_simplification_contour(contours_sample[0], tolerance=tolerance)
                contours.append(cntr)
            else:
                for cntr in contours_sample:
                    cntr = fix_and_simplification_contour(cntr, tolerance=tolerance)
                    contours.append(cntr)

        contours_dict[height] = contours
    return contours_dict


def smoothed_mask(
    instance_mask: np.ndarray,
    min_size_objects: int = 20,
    interations: int = 3,
) -> np.ndarray:
    """ Сглаживание формы объектов на карте кластеризации 

    Args:
        instance_mask (np.ndarray): 
        min_size_objects (int, optional): Минимальный размер возможных объектов. Defaults to 20.
        interations (int, optional): Количество итераций расширения и сжатия областей. Defaults to 3.

    Returns:
        np.ndarray: Сглаженная инстанс маска 
    """
    matrix_cleaned: np.ndarray = remove_small_objects(
        instance_mask.astype(np.uint8
),
        min_size=min_size_objects,
    ).astype(int)

    # Морфологические операции: расширение + сужение для сглаживания
    smoothed = np.zeros_like(matrix_cleaned)
    unique_labels = np.unique(matrix_cleaned)

    for label_value in unique_labels:
        if label_value == 0:
            continue
        # Обрабатываем каждую область по отдельности
        mask = (matrix_cleaned == label_value).astype(bool)  # Приводим к булевому типу
        smoothed_mask = binary_closing(mask, iterations=interations)
        smoothed[smoothed_mask] = label_value

    
    return smoothed


def concat_masks(
    instance_masks: Dict[np.uint8, Tuple[np.ndarray, int]],
) -> np.ndarray:
    """ Склейка всех инстанс масок кластеризованных поверхностей

    Args:
        instance_masks (Dict[np.uint8, Tuple[np.ndarray, int]]): 
        Словарь с высотами и инстанс масками их поверхностей + количество поверхностей

    Returns:
        np.ndarray: Склейка всех инстанс масок поверхностей
    """
    
    shape: Tuple[int] = next(iter(instance_masks.values()))[0].shape

    all_masks = np.zeros(shape)
    offset: int = 0

    for _, (mask, quantity) in instance_masks.items():
        all_masks += mask + (mask > 0) * offset
        offset += quantity
    
    return all_masks


def create_height_map(
    polygons_by_height: Dict[np.uint8, List[Polygon]],
    resolution: int = 1,
) -> np.ndarray:
    """
    Создаёт numpy карту высот из словаря полигонов.

    :param polygons_by_height: Словарь {высота: [список полигонов (shapely.geometry.Polygon)]}
    :param resolution: Разрешение сетки (шаг между точками по X и Y).
    :return: numpy массив, представляющий карту высот.
    """
    # Определение границ всех полигонов
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for height, polygons in polygons_by_height.items():
        for polygon in polygons:
            bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

    # Создание пустой карты высот
    width = int((max_x - min_x) / resolution) + 1
    height = int((max_y - min_y) / resolution) + 1
    height_map = np.zeros((height, width))

    # Заполнение карты высот
    for h, polygons in polygons_by_height.items():
        for polygon in polygons:
            if polygon.is_empty:
                continue
            area_threshold = 100
            if polygon.area < area_threshold:
                continue  # Пропускаем слишком маленькие полигоны
            # Получение индексов точек внутри полигона
            min_row = int((polygon.bounds[1] - min_y) / resolution)
            max_row = int((polygon.bounds[3] - min_y) / resolution)
            min_col = int((polygon.bounds[0] - min_x) / resolution)
            max_col = int((polygon.bounds[2] - min_x) / resolution)
            
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    x = min_x + col * resolution
                    y = min_y + row * resolution
                    point = Point([(x, y)])
                    if polygon.contains(point):
                        height_map[row, col] = h

    return height_map


def create_surface_map(
    polygons_by_idx: Dict[np.uint8, List[Polygon]],
    resolution: int = 1,
    area_threshold: float = 100,
) -> np.ndarray:
    """
    Создаёт numpy карту высот из словаря полигонов.

    :param polygons_by_idx: Словарь {номер поверхности: [список полигонов (shapely.geometry.Polygon)]}
    :param resolution: Разрешение сетки (шаг между точками по X и Y).
    :param area_threshold: Порог для вычеркивания слишком маленьких полигонов.
    :return: numpy массив, представляющий карту высот.
    """
    # Определение границ всех полигонов
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for _, polygon in polygons_by_idx.items():
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])

    # Создание пустой карты высот
    width = int((max_x - min_x) / resolution) + 1
    height = int((max_y - min_y) / resolution) + 1
    height_map = np.zeros((height, width))

    # Заполнение карты высот
    for idx, polygon in polygons_by_idx.items():
        if polygon.is_empty:
            continue
        if polygon.area < area_threshold:
            continue  # Пропускаем слишком маленькие полигоны
        # Получение индексов точек внутри полигона
        min_row = int((polygon.bounds[1] - min_y) / resolution)
        max_row = int((polygon.bounds[3] - min_y) / resolution)
        min_col = int((polygon.bounds[0] - min_x) / resolution)
        max_col = int((polygon.bounds[2] - min_x) / resolution)
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                x = min_x + col * resolution
                y = min_y + row * resolution
                point = Point([(x, y)])
                if polygon.contains(point):
                    height_map[row, col] = idx
    
    coords = (
        slice(round(min_x), round(min_x + width)),
        slice(round(min_y), round(min_y + height)),
    )

    return height_map, coords