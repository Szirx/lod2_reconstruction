from typing import Tuple, Dict
import argparse
import numpy as np
from PIL import Image
from ultralytics import YOLO
from scipy.ndimage import find_objects

from src.create_polygons import create_surface_map
from src.regularization import convert_to_rectangles
from src.obj_creator import (
    merge_objs,
    create_dict_single_obj,
)
from src.modified_contours import (
    apply_canny_to_mask,
    thin_contours_skeletonization,
    close_contours,
)
from src.surfaces import (
    masked_mean_by_surfaces,
    detect_flat_surfaces,
    apply_flat_surface_heights,
    compute_robust_linear_gradient,
)
from src.coordinates import (
    replace_crop,
    crop_patches,
)


def create_single_object(
    img_array: np.ndarray,
    heatmap_array: np.ndarray,
    instance_mask: np.ndarray,
    crop_slice: Tuple[slice],
    resolution: float = 7.0,
    flatness_threshold: int = 20,
    cell_area_threshold: float = 0,
    polygon_area_threshold: float = 0,
) -> Tuple[Tuple[slice], np.ndarray, dict]:
    """ Обработка составляющих одного объекта на изображении

    Args:
        img_array (np.ndarray): Матрица изображения
        heatmap_array (np.ndarray): Матрица карты высот
        instance_mask (np.ndarray): Набор instance-масок полученных моделью YOLO
        crop_slice (Tuple[slice]): Координаты выреза объекта
        resolution (float, optional): Размер полигонов сетки апроксимации полигонов. Defaults to 7.0.
        flatness_threshold (float, optional): Порог определения горизонтальных крыш. Defaults to 20.
        cell_area_threshold (float, optional): Порог взятия клетки апроксимационной сетки. Defaults to 0.
        polygon_area_threshold (float, optional): Порог рассмотрения полигона по его площади. Defaults to 0.

    Returns:
        Tuple[Tuple[slice], np.ndarray, dict]: Координаты выреза объекта,
        матрица с плоскостями объекта, словарь с информацией о всех частях объекта
    """
    cropped_image, cropped_heatmap, cropped_mask = crop_patches(
        img_array, heatmap_array, instance_mask, crop_slice,
    )

    # TODO: проблемное место извлечения контуров здания для построения плоскостей
    edges = apply_canny_to_mask(cropped_image, cropped_mask)
    closed_edges = close_contours(edges)
    thin_edges = thin_contours_skeletonization(closed_edges)

    rectangle_contours = convert_to_rectangles(
        thin_edges=thin_edges,
        cell_area_threshold=cell_area_threshold,
        resolution=resolution,
    )

    if rectangle_contours:
        mapp, coords = create_surface_map(rectangle_contours, area_threshold=polygon_area_threshold)
        replaced_crop = replace_crop(crop_slice, coords)

        surfaces_map, height_map = mapp.T, cropped_heatmap[coords]

        flat_surfaces, rough_surfaces = detect_flat_surfaces(surfaces_map, height_map, flatness_threshold)
        surfaces_mean = masked_mean_by_surfaces(surfaces_map, height_map, flat_surfaces)
        gradient_height, gradient_matrices = compute_robust_linear_gradient(surfaces_map, height_map, rough_surfaces)
        gradient_matrix = apply_flat_surface_heights(surfaces_map, gradient_height, surfaces_mean)
        info_dict_single_object = create_dict_single_obj(
            rectangle_contours,
            gradient_matrices,
            rough_surfaces,
            flat_surfaces,
            surfaces_mean,
            coords,
            replaced_crop,
        )

        return replaced_crop, gradient_matrix.T, info_dict_single_object
    else:
        return None, None, None


def main(
    weights_path: str,
    heatmap_path: str,
    rgb_path: str,
    resolution: float = 7.0,
    flatness_threshold: float = 20,
    cell_area_threshold: float = 0,
    polygon_area_threshold: float = 0,
    output_name: str = 'lod2',
) -> None:
    """ Главная функция для сбора всех данных по спутниковому изображению
    и создания общего .obj файла

    Args:
        weights_path (str): Путь до весов модели YOLO
        heatmap_path (str): Путь до .tif файла карты высот изображения
        rgb_path (str): Путь до .tif файла RGB изображения
        resolution (float, optional): Размер полигонов сетки апроксимации полигонов. Defaults to 7.0.
        flatness_threshold (float, optional): Порог определения горизонтальных крыш. Defaults to 20.
        cell_area_threshold (float, optional): Порог взятия клетки апроксимационной сетки. Defaults to 0.
        polygon_area_threshold (float, optional): Порог рассмотрения полигона по его площади. Defaults to 0.
        output_name (str, optional): Имя выходного файла (без расширения). Defaults to 'lod2'.
    """
    heatmap = np.array(Image.open(heatmap_path).resize((576, 1024)))
    image = Image.open(rgb_path).resize((576, 1024))
    model = YOLO(weights_path)
    results_seg = model(image)
    instance_masks = results_seg[0].masks.data.cpu()
    image = np.array(image)

    expansion: int = 10
    replaced_crops: list = []
    created_matrices: list = []
    info_objects: Dict[str, list] = {'buildings': []}

    for instance_mask in instance_masks:
        instance_mask = np.asarray(instance_mask, dtype=np.uint8)

        crop_slice: Tuple[slice] = find_objects(instance_mask)[0]

        crop_slice = (
            slice(
                crop_slice[0].start - expansion,
                crop_slice[0].stop + expansion,
                crop_slice[0].step,
            ),
            slice(
                crop_slice[1].start - expansion,
                crop_slice[1].stop + expansion,
                crop_slice[1].step,
            ),
        )

        replaced_crop, created_matrix, info_single_object = create_single_object(
            image,
            heatmap,
            instance_mask,
            crop_slice,
            resolution=resolution,
            flatness_threshold=flatness_threshold,
            cell_area_threshold=cell_area_threshold,
            polygon_area_threshold=polygon_area_threshold,
        )
        replaced_crops.append(replaced_crop)
        created_matrices.append(created_matrix)
        info_objects['buildings'].append(info_single_object)

    merge_objs(info_objects, f'{output_name}.obj')


if __name__ == '__main__':

    RESOLUTION: float = 7.0
    FLATNESS_THRESHOLD: int = 20
    parser = argparse.ArgumentParser(description="LoD2 for satellite image.")
    parser.add_argument(
        '--weights_path', type=str, required=True, help='Path to the weights file',
    )
    parser.add_argument(
        '--heatmap_path', type=str, required=True, help='Path to the heatmap file',
    )
    parser.add_argument(
        '--rgb_path', type=str, required=True, help='Path to the RGB image file',
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=RESOLUTION,
        help='Resolution of grid cells for rectangle polygons (default: 7.0)',
    )
    parser.add_argument(
        '--flatness_thd', type=int, default=FLATNESS_THRESHOLD, help='Flatness threshold for surfaces (default: 20)',
    )
    parser.add_argument(
        '--cell_area_thd',
        type=float,
        default=0,
        help='Cell area threshold for adding rectangle minipolygons in final polygon of surface (default: 0.0)',
    )
    parser.add_argument(
        '--poly_area_thd', type=float, default=0, help='Area of polygon for removing from processing (default: 0.0)',
    )
    parser.add_argument(
        '--output_name', type=str, default='lod2', help='Filename of output file with .obj format (default: "lod2")',
    )

    args = parser.parse_args()

    main(
        args.weights_path,
        args.heatmap_path,
        args.rgb_path,
        args.resolution,
        args.flatness_thd,
        args.cell_area_thd,
        args.poly_area_thd,
        args.output_name,
    )
