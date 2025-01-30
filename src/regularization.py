from typing import List, Dict, Tuple
import numpy as np
from shapely import unary_union
from shapely.geometry import Polygon, box
from shapely.geometry import LineString
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay


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
    
    return snapped_polygon, np.array(snapped_coords[:-1]), min_rect,


def create_grid_lines(
    min_point: Tuple[float, float],
    max_point: Tuple[float, float],
    rotation_matrix: np.ndarray,
    resolution: float
) -> Tuple[np.ndarray, np.ndarray]:
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
    horizontal_lines = np.array([
        [[min_x, y], [max_x, y]] for y in y_coords
    ])

    # Вертикальные линии
    x_coords = np.arange(min_x, max_x + resolution, resolution)
    vertical_lines = np.array([
        [[x, min_y], [x, max_y]] for x in x_coords
    ])

    h_lines_original: list = []
    v_lines_original: list = []

    # Обратный поворот линий в исходное пространство
    for h_line in horizontal_lines:
        h_transformed_line = h_line @ rotation_matrix
        h_lines_original.append(h_transformed_line)

    for v_line in vertical_lines:
        v_transformed_line = v_line @ rotation_matrix
        v_lines_original.append(v_transformed_line)

    return np.array(h_lines_original), np.array(v_lines_original)


def scale_line(
    line: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """
    Увеличивает длину отрезка на заданный коэффициент.
    
    :param line: numpy массив с координатами концов линии [[x1, y1], [x2, y2]]
    :param scale_factor: Коэффициент увеличения длины
    :return: Масштабированная линия
    """
    # Найти центр линии
    center = np.mean(line, axis=0)

    # Вычислить направление линии (вектор)
    direction = line[1] - line[0]
    unit_direction = direction / np.linalg.norm(direction)  # Единичный вектор

    # Удлинение линии на scale_factor
    new_half_length = np.linalg.norm(direction) * scale_factor / 2
    new_start = center - new_half_length * unit_direction
    new_end = center + new_half_length * unit_direction

    return np.array([new_start, new_end])


# Функция для нахождения пересечений отрезков
def find_intersections(h_lines, v_lines):
    intersections = []  # Список для хранения пересечений

    for h_line in h_lines:
        h_line_string = LineString(h_line)  # Горизонтальная линия

        for v_line in v_lines:
            v_line_string = LineString(v_line)  # Вертикальная линия

            # Проверяем пересечение
            intersection = h_line_string.intersection(v_line_string)

            # Если пересечение существует, добавляем в список
            if not intersection.is_empty:
                # В зависимости от типа пересечения (точка или отрезок) мы извлекаем только точку
                if intersection.geom_type == 'Point':
                    intersections.append([intersection.x, intersection.y])
    
    return np.array(intersections)


def tin_grid_cells(points: np.ndarray) -> List[Polygon]:
    # Выполняем Delaunay триангуляцию
    tri = Delaunay(points)

    # Создаем список миниполигонов
    tin_polygons: list = []
    for simplex in tri.simplices:
        poly_points = points[simplex]  # Получаем точки треугольника
        polygon = Polygon(poly_points)  # Создаем полигон из точек
        tin_polygons.append(polygon)
    return tin_polygons


def create_grid_cells(
    polygon: Polygon,
    resolution: float,
) -> List[box]:
    """
    Создает сетку в виде прямоугольных ячеек внутри указанного диапазона.
    
    :param polygon: Полигон отдельной части объекта
    :param resolution: Размер ячеек сетки
    :return: Список прямоугольных ячеек (rectangles)
    """
    min_x, min_y, max_x, max_y = polygon.bounds

    cells: list = []
    
    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    
    for x in x_coords:
        for y in y_coords:
            cell = box(x, y, x + resolution, y + resolution)
            cells.append(cell)
    
    return cells

def fill_grid_cells(
    polygon: Polygon,
    grid_cells: List[Polygon],
) -> List[Polygon]:
    """
    Заполняет ячейки сетки, если площадь пересечения с полигоном больше половины площади ячейки.
    
    :param polygon: Shapely Polygon — исходный полигон
    :param grid_cells: Список прямоугольных ячеек (shapely.geometry.Polygon)
    :return: Список заполненных ячеек
    """
    filled_cells: list = []

    for cell in grid_cells:
        intersection = polygon.intersection(cell)
        intersection_area = intersection.area
        cell_area = grid_cells[0].area
        
        if intersection_area > cell_area * 0.0001:
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


def minimum_rectangle(polygon: Polygon) -> Tuple[Polygon, list]:
    """
    Приводит полигон к ближайшему прямоугольному виду по сетке с разрешением resolution.
    
    :param polygon: Shapely Polygon
    :return: Новый полигон, выровненный по сетке
    """
    min_rect = polygon.minimum_rotated_rectangle
    
    return min_rect, np.array(min_rect.exterior.coords)

def create_grid_lines_from_rect(rect_coords: np.ndarray, resolution: float):
    """
    Создает линии сетки внутри повернутого bounding box без матрицы поворота.
    
    :param min_rect: Shapely Polygon — минимальный охватывающий прямоугольник
    :param rect_coords: Координаты углов прямоугольника (4x2 массив)
    :param resolution: Разрешение сетки
    :return: Горизонтальные и вертикальные линии в исходной системе координат
    """
    # Вектора сторон прямоугольника
    vec1 = rect_coords[1] - rect_coords[0]  # Первый вектор (длина)
    vec2 = rect_coords[3] - rect_coords[0]  # Второй вектор (ширина)

    # Нормализуем вектора
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)

    # Определяем размеры bounding box
    length = np.linalg.norm(rect_coords[1] - rect_coords[0])
    width = np.linalg.norm(rect_coords[3] - rect_coords[0])

    # Генерируем координаты сетки вдоль первой и второй оси
    num_steps_1 = int(np.ceil(length / resolution)) + 1
    num_steps_2 = int(np.ceil(width / resolution)) + 1

    # Генерация точек вдоль первой оси
    line1_points = [rect_coords[0] + vec1 * i * resolution for i in range(num_steps_1)]
    line2_points = [rect_coords[0] + vec2 * i * resolution for i in range(num_steps_2)]

    # Генерация горизонтальных линий
    h_lines = []
    for p in line2_points:
        line = [p + vec1 * 0, p + vec1 * length]  # Линия вдоль vec1
        h_lines.append(line)

    # Генерация вертикальных линий
    v_lines = []
    for p in line1_points:
        line = [p + vec2 * 0, p + vec2 * width]  # Линия вдоль vec2
        v_lines.append(line)

    return np.array(h_lines), np.array(v_lines)