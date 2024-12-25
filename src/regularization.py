from typing import List, Dict
import numpy as np
from shapely import unary_union
from shapely.geometry import Polygon, box
from sklearn.decomposition import PCA

def oriented_bounding_box(polygon: Polygon) -> np.ndarray:
    """ Вычисляет повернутый bounding box для заданного полигона.

    Args:
        polygon (Polygon): shapely полигон части здания
    Returns:
        np.ndarray: координаты повернутого bounding box в виде массива numpy
    """
    # Получение координат точек полигона
    coords = np.array(polygon.exterior.coords)

    pca = PCA(n_components=2).fit(coords)
    rotation_matrix = pca.components_

    # Поворот точек полигона в новое пространство (ориентированное по PCA)
    rotated_coords = coords @ rotation_matrix.T

    # Найти минимальный прямоугольник в повернутом пространстве
    min_x, min_y = np.min(rotated_coords, axis=0)
    max_x, max_y = np.max(rotated_coords, axis=0)

    # Координаты bounding box в повернутом пространстве
    box_rotated = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y],  # Замкнуть прямоугольник
    ])

    # Обратный поворот в исходное пространство
    box_original = box_rotated @ rotation_matrix

    return box_original, rotation_matrix, (min_x, min_y), (max_x, max_y)


def snap_to_grid(polygon: Polygon, resolution: float):
    """
    Приводит полигон к ближайшему прямоугольному виду по сетке с разрешением resolution.
    
    :param polygon: Shapely Polygon
    :param resolution: Разрешение сетки
    :return: Новый полигон, выровненный по сетке
    """
    # Вычисление минимального охватывающего прямоугольника
    min_rect = polygon.minimum_rotated_rectangle

    # Преобразование его в массив точек
    rect_coords = np.array(min_rect.exterior.coords)

    # Подгонка точек прямоугольника к сетке
    snapped_coords = np.round(rect_coords / resolution) * resolution

    # Создание нового полигона с выровненными координатами
    snapped_polygon = Polygon(snapped_coords)
    
    return snapped_polygon, np.array(snapped_coords[:-1])


def create_grid_lines(min_point, max_point, rotation_matrix, resolution):
    """
    Создает линии сетки внутри повернутого bounding box.
    
    :param min_point: Минимальные координаты (min_x, min_y) повернутого bounding box
    :param max_point: Максимальные координаты (max_x, max_y) повернутого bounding box
    :param rotation_matrix: Матрица поворота bounding box
    :param resolution: Шаг сетки
    :return: Горизонтальные и вертикальные линии в исходной системе координат
    """
    min_x, min_y = min_point
    max_x, max_y = max_point

    # Горизонтальные линии
    y_coords = np.arange(min_y, max_y + resolution, resolution)
    horizontal_lines = [
        [[min_x, y], [max_x, y]] for y in y_coords
    ]

    # Вертикальные линии
    x_coords = np.arange(min_x, max_x + resolution, resolution)
    vertical_lines = [
        [[x, min_y], [x, max_y]] for x in x_coords
    ]

    # Объединяем линии
    all_lines_rotated = np.array(horizontal_lines + vertical_lines)

    # Обратный поворот линий в исходное пространство
    all_lines_original = []
    for line in all_lines_rotated:
        transformed_line = line @ rotation_matrix
        all_lines_original.append(transformed_line)

    return np.array(all_lines_original)


def create_grid_cells(min_x: float, min_y: float, max_x: float, max_y: float, resolution: float) -> List[box]:
    """
    Создает сетку в виде прямоугольных ячеек внутри указанного диапазона.
    
    :param min_x: Минимальное значение по оси X
    :param min_y: Минимальное значение по оси Y
    :param max_x: Максимальное значение по оси X
    :param max_y: Максимальное значение по оси Y
    :param resolution: Размер ячеек сетки
    :return: Список прямоугольных ячеек (rectangles)
    """
    cells = []
    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    
    for x in x_coords:
        for y in y_coords:
            cell = box(x, y, x + resolution, y + resolution)
            cells.append(cell)
    
    return cells

def fill_grid_cells(
    polygon: Polygon,
    grid_cells: List[box],
    resolution: float,
) -> List[Polygon]:
    """
    Заполняет ячейки сетки, если площадь пересечения с полигоном больше половины площади ячейки.
    
    :param polygon: Shapely Polygon — исходный полигон
    :param grid_cells: Список прямоугольных ячеек (shapely.geometry.box)
    :param resolution: Размер ячеек сетки
    :return: Список заполненных ячеек
    """
    filled_cells: list = []

    for cell in grid_cells:
        intersection = polygon.intersection(cell)
        intersection_area = intersection.area
        cell_area = resolution * resolution
        
        if intersection_area > cell_area * 0.001:
            filled_cells.append(cell)
    
    return filled_cells


def convert_to_rectangles(
    contours_dict: Dict[np.uint8, List[Polygon]],
    resolution: float = 5.0,
) -> Dict[np.uint8, List[Polygon]]:
    """ Приведение полигонов к составной фигуре из прямоугольников

    Args:
        contours_dict (Dict[np.uint8, List[Polygon]]): Исходный словарь с высотами и полигонами
        resolution (float): Размер ячеек для преобразования объекта в комбинацию из прямоугольных форм

    Returns:
        Dict[np.uint8, List[Polygon]]: Преобразованный словарь с высотами и полигонами
    """
    contours_dict_new: dict = {}
    for key, values in contours_dict.items():
        contours: list = []
        for value in values:
            building_polygon = Polygon(value)

            # Минимальные и максимальные координаты bounding box
            min_x, min_y, max_x, max_y = building_polygon.bounds

            # Создание сетки
            grid_cells = create_grid_cells(min_x, min_y, max_x, max_y, resolution)

            # Заполнение ячеек сетки
            filled_cells = fill_grid_cells(building_polygon, grid_cells, resolution)
            contours.append(unary_union(filled_cells))
        contours_dict_new[key] = contours
    
    return contours_dict_new