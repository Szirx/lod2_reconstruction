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
    # Применяем маску на изображение, чтобы извлечь только интересующую нас часть
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Преобразуем изображение в оттенки серого (grayscale)
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    
    # Применяем размытие (GaussianBlur) перед Canny для уменьшения шума
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Применяем фильтр Canny для выявления краев
    edges = cv2.Canny(blurred, 100, 300)  # Параметры могут быть настроены
    
    return edges