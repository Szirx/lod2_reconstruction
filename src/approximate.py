from typing import List, Tuple
from shapely.geometry import Polygon
Point = Tuple[float, float]
LineSegment = Tuple[Point, Point]


def dist_point_line_segment(
    line_segment: LineSegment,
    point: Point,
) -> float:
    x, y = point
    x1, y1 = line_segment[0]
    x2, y2 = line_segment[1]

    degree: float = 0.5
    double_area = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    line_segment_length = (
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    ) ** degree
    if line_segment_length == 0:
        return 0
    else:
        return double_area / line_segment_length


def simplify_points(
    src_points: List[Point],
    tolerance: float,
    begin: int,
    end: int,
) -> List[Point]:

    if begin + 1 == end:
        return []

    max_distance = -1.0
    max_index = 0
    for i in range(begin + 1, end):
        cur_point = src_points[i]
        start_point = src_points[begin]
        end_point = src_points[end]
        distance = dist_point_line_segment((start_point, end_point), cur_point)
        if distance > max_distance:
            max_distance = distance
            max_index = i

    dest_points = []
    if max_distance > tolerance:
        dest_points += simplify_points(src_points, tolerance, begin, max_index)
        dest_points.append(src_points[max_index])
        dest_points += simplify_points(src_points, tolerance, max_index, end)
    return dest_points


def douglas_peucker(
    src_points: List[Point],
    tolerance: float = 3,
) -> List[Point]:
    if tolerance <= 0:
        return src_points

    dest_points = [src_points[0]]
    dest_points += simplify_points(src_points, tolerance, 0, len(src_points) - 1)
    dest_points.append(src_points[-1])
    dest_points.append(src_points[0])
    return dest_points


def poly_adjustment(instances_dict, epsilon_smooth=None):
    complete_list = []
    for class_id, prediction_data in instances_dict.items():
        if not Polygon(prediction_data[0]).exterior.is_ccw:
            prediction_data = [
                contour[::-1] for contour in prediction_data
            ]
        if epsilon_smooth:
            contour_corrected_direction = [
                douglas_peucker(
                    contour[:-1],
                    tolerance=epsilon_smooth,
                )
                for contour in prediction_data
            ]
        for contour in contour_corrected_direction:
            complete_list.append([contour, class_id])
    return complete_list
