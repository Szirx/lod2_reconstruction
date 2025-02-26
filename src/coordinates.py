from typing import Tuple
import numpy as np

def crop_sizes(crop_slice: Tuple[slice]) -> Tuple[int]:
    """_summary_

    Args:
        crop_slice (Tuple[slice]): _description_

    Returns:
        Tuple[int]: _description_
    """
    crop_height = crop_slice[0].stop - crop_slice[0].start
    crop_width = crop_slice[1].stop - crop_slice[1].start
    return crop_height, crop_width

def create_start_slices(
    crop_slice: Tuple[slice],
    img_array: np.ndarray,
) -> Tuple[slice]:
    """_summary_

    Args:
        crop_slice (Tuple[slice]): _description_
        img_array (np.ndarray): _description_

    Returns:
        Tuple[slice]: _description_
    """
    y_start = max(crop_slice[0].start, 0)
    y_end = min(crop_slice[0].stop, img_array.shape[0])
    x_start = max(crop_slice[1].start, 0)
    x_end = min(crop_slice[1].stop, img_array.shape[1])
    # Вычисляем смещения в пустой матрице
    y_offset = y_start - crop_slice[0].start
    x_offset = x_start - crop_slice[1].start

    primary_slice = (
        slice(y_start, y_end),
        slice(x_start, x_end),
    )
    template_slice = (
        slice(
            y_offset,
            y_offset + (y_end - y_start),
        ),
        slice(
            x_offset,
            x_offset + (x_end - x_start),
        )
    )
    return primary_slice, template_slice

def crop_patches(
    img_array: np.ndarray,
    heatmap_array: np.ndarray,
    instance_mask: np.ndarray,
    crop_slice: Tuple[slice]
) -> Tuple[np.ndarray]:
    """_summary_

    Args:
        img_array (np.ndarray): _description_
        heatmap_array (np.ndarray): _description_
        instance_mask (np.ndarray): _description_
        crop_slice (Tuple[slice]): _description_

    Returns:
        Tuple[np.ndarray]: _description_
    """
    crop_height, crop_width = crop_sizes(crop_slice=crop_slice)
    # Создаем пустую матрицу для кропа (заполненную нулями)
    cropped_image = np.zeros((crop_height, crop_width, img_array.shape[2]), dtype=np.uint8)
    cropped_heatmap = np.zeros((crop_height, crop_width), dtype=np.uint8)
    cropped_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)

    primary_slice, template_slice = create_start_slices(crop_slice, img_array)
    # Копируем данные из изображения в пустую матрицу
    cropped_image[template_slice] = img_array[primary_slice]
    cropped_heatmap[template_slice] = heatmap_array[primary_slice]
    cropped_mask[template_slice] = instance_mask[primary_slice]

    return cropped_image, cropped_heatmap, cropped_mask


def replace_crop(
    crop_slice: Tuple[slice],
    coords: Tuple[slice],
) -> Tuple[slice]:
    """_summary_

    Args:
        crop_slice (Tuple[slice]): _description_
        coords (Tuple[slice]): _description_

    Returns:
        Tuple[slice]: _description_
    """
    replaced_crop = (
        slice(
            crop_slice[0].start + coords[0].start,
            crop_slice[0].stop - (crop_slice[0].stop - (crop_slice[0].start + coords[0].stop))
        ),
        slice(
            crop_slice[1].start + coords[1].start,
            crop_slice[1].stop - (crop_slice[1].stop - (crop_slice[1].start + coords[1].stop))
        )
    )

    return replaced_crop