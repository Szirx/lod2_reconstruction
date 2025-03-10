from typing import Tuple
import numpy as np


def crop_sizes(crop: Tuple[slice]) -> Tuple[int]:
    """ Расчет высоты и ширины вырезки

    Args:
        crop (Tuple[slice]): Кортеж координат вырезки

    Returns:
        Tuple[int]: Кортеж высоты и ширины вырезки
    """
    crop_height = crop[0].stop - crop[0].start
    crop_width = crop[1].stop - crop[1].start
    return crop_height, crop_width


def create_start_slices(
    crop_slice: Tuple[slice],
    img_array: np.ndarray,
) -> Tuple[slice]:
    """ Создание начальных вырезок для проведения операций LoD2 алгоритма

    Args:
        crop_slice (Tuple[slice]): Кортеж координат вырезки
        img_array (np.ndarray): Матрица изображения

    Returns:
        Tuple[slice]: Вырезка исходной матрицы и вырезка несущей матрицы
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
    """ Создание патчей с помощью координат вырезки

    Args:
        img_array (np.ndarray): Матрица исходного изображения
        heatmap_array (np.ndarray): Матрица карты высот
        instance_mask (np.ndarray): Instance-маски полученные моделью сегментации
        crop_slice (Tuple[slice]): Кортеж координат вырезки

    Returns:
        Tuple[np.ndarray]: Кропы матриц изображения, карты высот и маски сегментации
    """
    crop_height, crop_width = crop_sizes(crop=crop_slice)

    cropped_image = np.zeros(
        (crop_height, crop_width, img_array.shape[2]),
        dtype=np.uint8,
    )
    cropped_heatmap = np.zeros((crop_height, crop_width), dtype=np.uint8)
    cropped_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)

    primary_slice, template_slice = create_start_slices(crop_slice, img_array)

    cropped_image[template_slice] = img_array[primary_slice]
    cropped_heatmap[template_slice] = heatmap_array[primary_slice]
    cropped_mask[template_slice] = instance_mask[primary_slice]

    return cropped_image, cropped_heatmap, cropped_mask


def replace_crop(
    crop_slice: Tuple[slice],
    coords: Tuple[slice],
) -> Tuple[slice]:
    """ Перенос кортежа координат вырезки

    Args:
        crop_slice (Tuple[slice]): Исходный кортеж координат вырезки
        coords (Tuple[slice]): Координаты смещения

    Returns:
        Tuple[slice]: Смещенный кортеж координат вырезки
    """
    replaced_crop = (
        slice(
            crop_slice[0].start + coords[0].start,
            crop_slice[0].stop - (
                crop_slice[0].stop - (
                    crop_slice[0].start + coords[0].stop
                )
            )
        ),
        slice(
            crop_slice[1].start + coords[1].start,
            crop_slice[1].stop - (
                crop_slice[1].stop - (
                    crop_slice[1].start + coords[1].stop
                )
            )
        )
    )

    return replaced_crop
