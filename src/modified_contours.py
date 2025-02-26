from skimage.morphology import skeletonize
import numpy as np
import cv2


# Функция для тонкой окантовки
def thin_contours_skeletonization(binary_image: np.ndarray) -> np.ndarray:
    # Применение тонкой окантовки: превращаем изображение в двоичное (0 и 1)
    # skimage.skeletonize ожидает 0 и 1, поэтому преобразуем в тип bool
    skeleton = skeletonize(binary_image.astype(np.bool))
    # Преобразуем результат обратно в тип uint8 (0 или 255)
    return (skeleton * 255).astype(np.uint8)

def apply_canny(image: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.Canny(blurred, otsu_thresh * 0.5, otsu_thresh)

# Функция применения Canny к области, указанной маской
def apply_canny_to_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    # gray_image = cv2.equalizeHist(gray_image)
    # blurred = cv2.GaussianBlur(gray_image, (5, 5), 1)
    # Применяем медианную фильтрацию для уменьшения шума
    blurred = cv2.medianBlur(gray_image, 5)
    # blurred = cv2.bilateralFilter(image_eq, 5, 75, 75)
    # blurred = gray_image
    edges = cv2.Canny(blurred.astype(np.uint8), 100, 300)
    return edges

def close_contours(edges: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    closed_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_CLOSE, kernel)

    return closed_edges
