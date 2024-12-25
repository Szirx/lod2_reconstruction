from typing import Literal, Dict, Tuple
import os
import numpy as np
from PIL import Image
from joblib import dump
from scipy.ndimage import label
from sklearn.cluster import KMeans


def read_tiff(file_path: str) -> np.ndarray:
    return np.asarray(Image.open(file_path))


def prepare_dataset(tif_folder: str) -> np.ndarray:

    tif_files = [
        os.path.join(tif_folder, f) for f in os.listdir(tif_folder) if f.endswith('.tif')
    ]

    dataset: list = []

    for file in tif_files:
        image_tif = read_tiff(file)
        dataset.append(image_tif.reshape(-1, 1))

    return np.vstack(dataset)


def train_clusterisation(
    X: list | np.ndarray,
    num_clusters: int = 5,
    random_state: int = 42,
) -> None:
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=random_state,
    )
    kmeans.fit(X)
    dump(kmeans, 'kmeans_model.joblib')


def height_clustering(
    building_map: np.ndarray,
    building: np.ndarray,
    n_clusters: int = 5,
    mode: Literal['min', 'mean', 'max'] = 'mean',
) -> np.ndarray:
    """ Кластеризация здания по высоте

    Args:
        building_map (np.ndarray): Карта высот здания
        building (np.ndarray): Маска сегментации здания
        n_clusters (int, optional): Количество различных высот здания. Defaults to 5.
        mode (Literal['min', 'mean', 'max'], optional):
        Тип взятия высоты по кластеру. Defaults to 'mean'.

    Returns:
        np.ndarray: Кластеризованная карта высот здания
    """

    masked_height_map: np.ndarray = building_map * building
    height_values: np.ndarray = masked_height_map[building == 1]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(height_values.reshape(-1, 1))

    labels: np.ndarray = kmeans.labels_

    clustered_height_map = np.zeros_like(building_map)
    clustered_height_map[building == 1] = labels + 1
    
    clustered_building: np.ndarray = building_map.copy()
    
    labels = np.insert(np.unique(labels) + 1, 0, 0)

    for label in np.unique(labels):
        if mode == 'min':
            clustered_building[clustered_height_map == label] = building_map[clustered_height_map == label].min()
        elif mode == 'mean':
            clustered_building[clustered_height_map == label] = building_map[clustered_height_map == label].mean()
        elif mode == 'max':
            clustered_building[clustered_height_map == label] = building_map[clustered_height_map == label].max()

    return clustered_building


def surface_clustering(
        clustered_heights_building: np.ndarray,
) -> Dict[np.uint8, Tuple[np.ndarray, int]]:
    """ Кластеризация здания по поверхностям

    Args:
        clustered_heights_building (np.ndarray): кластеризованная маска высот здания

    Returns:
        Dict[np.uint8, Tuple[np.ndarray, int]]: Высота : (Набор инстанс масок, количество объектов на маске)
    """

    unique_heights: np.ndarray = np.unique(clustered_heights_building)
    instance_masks: dict = {}

    for height in unique_heights[1:]:
        binary_mask = (clustered_heights_building == height).astype(np.uint8)
        instance_masks[height] = label(binary_mask)

    return instance_masks
