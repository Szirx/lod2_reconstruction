from typing import Tuple, List
import numpy as np
from src.create_polygons import create_surface_map
from shapely.geometry import Polygon
from src.regularization import convert_to_rectangles
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
    flatness_threshold: float = 0.0,
    cell_area_threshold: float = 0.0,
    polygon_area_threshold: float = 0.0,
    ) -> Tuple[Tuple[slice], np.ndarray, dict]:

    cropped_image, cropped_heatmap, cropped_mask = crop_patches(
        img_array, heatmap_array, instance_mask, crop_slice,
    )    

    # TODO: проблемное место извлечения контуров здания для построения плоскостей
    edges = apply_canny_to_mask(cropped_image, cropped_mask)
    closed_edges = close_contours(edges)
    thin_edges = thin_contours_skeletonization(closed_edges)

    r_contours = convert_to_rectangles(
        thin_edges=thin_edges,
        cell_area_threshold=cell_area_threshold,
        resolution=resolution,
    )

    if r_contours:
        mapp, coords = create_surface_map(r_contours, area_threshold=polygon_area_threshold)
        replaced_crop = replace_crop(crop_slice, coords)

        surfaces_map, height_map = mapp.T, cropped_heatmap[coords]

        flat_surfaces, rough_surfaces = detect_flat_surfaces(surfaces_map, height_map, flatness_threshold)
        surfaces_mean = masked_mean_by_surfaces(surfaces_map, height_map, flat_surfaces)
        gradient_height = compute_robust_linear_gradient(surfaces_map, height_map, rough_surfaces, r_contours)
        gradient_matrix = apply_flat_surface_heights(surfaces_map, gradient_height, surfaces_mean)

        right_contours: dict = {}

        for k, v in r_contours.items():
            right_contours[k] = Polygon([(x - coords[0].start,y - coords[1].start) for (x, y) in list(v.exterior.coords)])

        info_dict_single_object: dict = {}

        threshold: int = 7
        for index, polygon in right_contours.items():
            replaced_polygon: List[Tuple[float, float]] = [
                (x + replaced_crop[0].start, y + replaced_crop[1].start)
                for (x, y) in list(polygon.exterior.coords)
            ]
            if index in rough_surfaces:
                
                heights: list = []
                
                for (x, y) in list(polygon.exterior.coords):
                    x_start = max(int(x) - threshold, 0)
                    x_end = min(int(x) + threshold, gradient_matrix.shape[0])
                    y_start = max(int(y) - threshold, 0)
                    y_end = min(int(y) + threshold, gradient_matrix.shape[1])
                    slice_heights = gradient_matrix[
                                x_start: x_end,
                                y_start: y_end,
                            ].T
                    heights.append(np.max(slice_heights))
            
            if Polygon(replaced_polygon).exterior.is_ccw:
                replaced_polygon = replaced_polygon[::-1]
                heights = heights[::-1]

            info_dict_single_object[index] = {
                'polygon': replaced_polygon,
                'is_flat': True if index in flat_surfaces else False,
                'mean_height': surfaces_mean[index] if index in flat_surfaces else None,
                'heights': heights if index in rough_surfaces else None,
            }

        return replaced_crop, gradient_matrix.T, info_dict_single_object
    else:
        return None, None, None

if __name__ == '__main__':
    weights_path: str = '../../../shared_data/users/avlasov/vaihingen.pt'
    vai: str = '../../shared_data/datasets/Vaihingen/train/NDSM/area34.tif'
    vai_rgb: str = '../../shared_data/datasets/Vaihingen/train/RGB/area34.tif'