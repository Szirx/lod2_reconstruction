from typing import Tuple
from skimage.morphology import skeletonize
import numpy as np
import cv2

MAX_PIXEL_VALUE: int = 255
CANNY_PARAMETERS: Tuple[int] = (100, 300)


def thin_contours_skeletonization(binary_image: np.ndarray) -> np.ndarray:
    """ Скелетонизация границ

    Args:
        binary_image (np.ndarray): Бинарная матрица где 1 - граница, 0 - область

    Returns:
        np.ndarray: Матрица со скелетонизированными границами
    """
    skeleton = skeletonize(binary_image.astype(np.bool))
    return (skeleton * MAX_PIXEL_VALUE).astype(np.uint8)


def apply_canny_to_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Применение фильтра Канни к чб изображению

    Args:
        image (np.ndarray): ЧБ изображение
        mask (np.ndarray): Маска instance-сегментации

    Returns:
        np.ndarray: Отфильтрованная матрица
    """
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    # gray_image = cv2.equalizeHist(gray_image)
    # blurred = cv2.GaussianBlur(gray_image, (5, 5), 1)
    # Применяем медианную фильтрацию для уменьшения шума
    blurred = cv2.medianBlur(gray_image, 5)
    # blurred = cv2.bilateralFilter(image_eq, 5, 75, 75)
    # blurred = gray_image
    edges = cv2.Canny(
        blurred.astype(np.uint8),
        CANNY_PARAMETERS[0],
        CANNY_PARAMETERS[1],
    )
    return edges


def close_contours(edges: np.ndarray) -> np.ndarray:
    """ Алгоритм закрытия контуров для выделения полигонов

    Args:
        edges (np.ndarray): Матрица отфильтрованная алгоритмом Канни

    Returns:
        np.ndarray: Матрица с закрытыми контурами областей
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    closed_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_CLOSE, kernel)

    return closed_edges
