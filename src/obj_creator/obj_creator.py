from typing import List, Tuple
from shapely import Polygon


def is_ear(p1, p2, p3, polygon):
    """
    Проверка, является ли треугольник (p1, p2, p3) "ухом" полигона.
    :param p1, p2, p3: Точки треугольника.
    :param polygon: Полигон.
    :return: True, если треугольник является "ухом".
    """
    for point in polygon:
        if point not in (p1, p2, p3):
            if point_in_triangle(point, p1, p2, p3):
                return False
    return True


def cross_product(a: float, b: float) -> float:
    return a[0] * b[1] - a[1] * b[0]


def area(a: float, b: float, c: float) -> float:
    return abs(
        cross_product(
            (
                b[0] - a[0],
                b[1] - a[1],
            ),
            (
                c[0] - a[0],
                c[1] - a[1],
            ),
        ) / 2
    )


def point_in_triangle(p, p1, p2, p3):
    """
    Проверка, находится ли точка p внутри треугольника (p1, p2, p3).
    :param p: Точка.
    :param p1, p2, p3: Точки треугольника.
    :return: True, если точка внутри треугольника.
    """
    epsilon: float = 1e-6

    area_total = area(p1, p2, p3)
    area1 = area(p, p1, p2)
    area2 = area(p, p2, p3)
    area3 = area(p, p3, p1)

    return abs((area1 + area2 + area3) - area_total) < epsilon


def triangulate_polygon(polygon: List[Tuple[float]]) -> list:
    """Improved polygon triangulation to prevent infinite loops."""
    len_polygon: int = len(polygon)
    if len_polygon < 3:
        return []

    triangles = []
    polygon = list(polygon)
    attempts = 0

    while len_polygon > 3:
        ear_found = False
        for i in range(len_polygon):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len_polygon]
            p3 = polygon[(i + 2) % len_polygon]

            if is_ear(p1, p2, p3, polygon):
                triangles.append((p1, p2, p3))
                polygon.pop((i + 1) % len_polygon)
                len_polygon = len(polygon)
                ear_found = True
                break

        if not ear_found:
            attempts += 1
            if attempts > len_polygon * 2:  # Fail-safe to prevent infinite loops
                return []

    triangles.append(tuple(polygon))  # Append the remaining triangle
    return triangles


def triangulate_polygon_with_heights(polygon: List[Tuple[float]]) -> list:
    """
    Триангуляция полигона с учетом высот для создания скатной крыши.
    :param polygon: Список точек полигона в формате [(x1, y1, z1), (x2, y2, z2), ...].
    :return: Список треугольников в формате [(p1, p2, p3), ...].
    """
    len_polygon: int = len(polygon)
    if len_polygon < 3:
        return []

    triangles = []
    polygon = list(polygon)
    attempts = 0

    while len_polygon > 3:
        ear_found = False
        for i in range(len_polygon):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len_polygon]
            p3 = polygon[(i + 2) % len_polygon]

            if is_ear(p1, p2, p3, polygon):
                triangles.append((p1, p2, p3))
                polygon.pop((i + 1) % len_polygon)
                len_polygon = len(polygon)
                ear_found = True
                break

        if not ear_found:
            attempts += 1
            if attempts > len_polygon * 2:  # Защита от бесконечного цикла
                return []

    triangles.append(tuple(polygon))  # Добавляем последний треугольник
    return triangles


def create_building_obj(
    i: int,
    obj_dict: dict,
) -> str:
    """Creates a 3D .obj representation of multiple buildings given a dictionary of building building_data."""
    def add_building_to_obj(
        outline: List[Tuple[int]],
        heights: list | int,
        obj_str_list: list,
        start_vertex_index: int = 1,
    ) -> None:
        """Helper function to add a single building's building_data to a list of .obj strings."""
        # Create the 3D vertices
        vertices = []
        if isinstance(heights, int):
            for (x, y) in outline:
                # Bottom vertices
                vertices.append((x, y, 0))
                # Top vertices
                vertices.append((x, y, heights))
            # Triangulate base (bottom) and top faces using ear clipping
            triangles_bottom = triangulate_polygon(outline)
            for triangle in triangles_bottom:
                indices = [start_vertex_index + outline.index(point) * 2 for point in triangle]
                obj_str_list.append(f"f {indices[0]}//1 {indices[1]}//1 {indices[2]}//1")
            triangles_top = triangulate_polygon(outline)
            for triangle in triangles_top:
                indices = [start_vertex_index + outline.index(point) * 2 + 1 for point in triangle]
                obj_str_list.append(f"f {indices[0]}//2 {indices[1]}//2 {indices[2]}//2")
        else:
            for (x, y), height in zip(outline, heights):
                vertices.append((x, y, 0))
                # Top vertices
                vertices.append((x, y, height))
            triangles_bottom = triangulate_polygon(outline)
            for triangle in triangles_bottom:
                indices = [start_vertex_index + outline.index(point) * 2 for point in triangle]
                obj_str_list.append(f"f {indices[0]}//1 {indices[1]}//1 {indices[2]}//1")
            outline_with_heights = [(x, y, height) for (x, y), height in zip(outline, heights)]
            triangles_top = triangulate_polygon_with_heights(outline_with_heights)
            for triangle in triangles_top:
                indices = [start_vertex_index + outline_with_heights.index(point) * 2 + 1 for point in triangle]
                obj_str_list.append(f"f {indices[0]}//2 {indices[1]}//2 {indices[2]}//2")

        for v in vertices:
            obj_str_list.append(f"v {v[0]} {v[1]} {v[2]}")

        obj_str_list.append("vn 0 0 -1")
        obj_str_list.append("vn 0 0 1")

        # Side faces with appropriate normals
        for i in range(len(outline)):
            p1 = outline[i]
            p2 = outline[(i + 1) % len(outline)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Compute the normal vector (perpendicular to the edge)
            nx, ny = dy, -dx  # perpendicular vector
            magnitude = (nx ** 2 + ny ** 2) ** 0.5
            if magnitude > 0:
                nx, ny = nx / magnitude, ny / magnitude
            else:
                continue
            cross_product = nx  # Simplified cross product with (0, 0, 1)
            if cross_product < 0:
                nx, ny = -nx, -ny  # Flip the normal if necessary

            obj_str_list.append(f"vn {nx} {ny} 0")  # Normal for this side face

            bottom_left = start_vertex_index + i * 2
            bottom_right = start_vertex_index + ((i + 1) % len(outline)) * 2
            top_left = bottom_left + 1
            top_right = bottom_right + 1
            obj_str_list.append(
                f"f {bottom_left}//{2 + i + 1} "
                f"{bottom_right}//{2 + i + 1} "
                f"{top_right}//{2 + i + 1} "
                f"{top_left}//{2 + i + 1}"
            )

    obj_str_list: list = []
    vertex_count: int = 0
    for building_id, building_data in obj_dict.items():
        obj_str_list.append(f"o 0_{building_id}_{i}")
        outline = building_data['polygon']
        if building_data['heights'] is None:
            height = building_data['mean_height']
        else:
            height = building_data['heights']
        if not Polygon(outline).is_valid:
            continue
        add_building_to_obj(outline, height, obj_str_list, start_vertex_index=vertex_count + 1)
        vertex_count += len(outline) * 2  # Each outline results in twice the number of 3D vertices

    merged_obj_str_list: str = '\n'.join(obj_str_list)
    return f'{merged_obj_str_list}\n'


def merge_objs(info_objects: dict, output_file_path: str) -> None:

    vertices: list = []
    normals: list = []
    texcoords: list = []
    faces: list = []
    vertex_offset: int = 0

    for i, building in enumerate(info_objects['buildings']):
        if building is None:
            continue
        obj_list = create_building_obj(i, building).split('\n')
        for line in obj_list:
            if line.startswith('v '):  # вершины
                vertices.append(f'{line}\n')
            elif line.startswith('vn '):  # нормали
                normals.append(f'{line}\n')
            elif line.startswith('vt '):  # текстурные координаты
                texcoords.append(f'{line}\n')
            elif line.startswith('f '):  # грани (faces)
                # Пересчитываем индексы вершин, нормалей и текстур
                face_building_data = line.split()[1:]
                new_face = ['f']
                for face in face_building_data:
                    parts = face.split('//')
                    if len(parts) == 2:  # формат v//vn
                        v, vn = parts
                        new_v = int(v) + vertex_offset
                        new_vn = int(vn) + vertex_offset
                        new_face.append(f"{new_v}//{new_vn}")
                    elif len(parts) == 3:  # формат v//vt//vn
                        v, vt, vn = parts
                        new_v = int(v) + vertex_offset
                        new_vt = int(vt) + vertex_offset
                        new_vn = int(vn) + vertex_offset
                        new_face.append(f"{new_v}//{new_vt}//{new_vn}")
                    else:
                        raise ValueError(f"Неизвестный формат грани: {face}")
                faces.append(' '.join(new_face) + '\n')
            elif line.startswith('o '):  # объект
                # Добавляем объект с новым именем (если нужно)
                faces.append(f'{line}\n')

        # Обновляем смещение для следующего файла
        vertex_offset = len(vertices)

    # Записываем объединенный файл
    with open(output_file_path, 'w') as f:
        f.write("# Merged OBJ file\n")
        f.writelines(vertices)
        f.writelines(normals)
        f.writelines(texcoords)
        f.writelines(faces)
