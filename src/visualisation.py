from typing import List, Dict, Tuple
from scipy.spatial import Delaunay
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as Poly
from shapely.geometry import Polygon, LineString
from src.clustering import height_clustering, surface_clustering
from src.regularization import convert_to_rectangles
from src.create_polygons import create_height_map, create_polygons, concat_masks, smoothed_mask


def visualisation(
    building_height_map: np.ndarray,
    clustered_heights_building: np.ndarray,
    clustered_surfaces_building: np.ndarray,
) -> None:
    """ Визуализация исходной карты высот, кластеризации по высотам
    и кластеризации по поверхностям

    Args:
        building_height_map (np.ndarray): Исходная карта высот
        clustered_heights_building (np.ndarray): Кластеризация по высотам
        clustered_surfaces_building (np.ndarray): Кластеризация по поверхностям
    """
    num_colors: int = (clustered_heights_building.max() + 1).astype(int)
    colors: np.ndarray = np.random.rand(num_colors, 3)
    colors[0] = [0, 0, 0]
    cmap = ListedColormap(colors)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    im1 = axs[0].imshow(building_height_map)
    axs[0].set(title='Исходная карта высот')
    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar1.set_label('Значения высот')

    im2 = axs[1].imshow(clustered_heights_building)
    axs[1].set_title("Кластеризация высот")
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar2.set_label("Значения высот")

    im3 = axs[2].imshow(clustered_surfaces_building, cmap=cmap, interpolation='nearest')
    axs[2].set_title("Кластеризация поверхностей")
    cbar3 = fig.colorbar(im3, ax=axs[2])
    cbar3.set_label("i-ая поверхность")

    plt.tight_layout()
    plt.show()


def visualisation_lines(
    points: np.ndarray,
    triangulation: Delaunay,
    height_map: np.ndarray,
    intersection_lines: List[LineString],
) -> None:
    """ Визуализация линий границ кластеров на карте кластеризации высот

    Args:
        points (np.ndarray): 
        triangulation (Delaunay): _description_
        height_map (np.ndarray): _description_
        intersection_lines (List[LineString]): _description_
    """
    _, ax = plt.subplots(figsize=(15,15))
    ax.triplot(points[:, 0], points[:, 1], triangulation.simplices, color='gray')
    for line in intersection_lines:
        x, y = line.xy
        ax.plot(x, y, 'r--', linewidth=0.5)
    ax.imshow(height_map, cmap='gray', interpolation='none')
    plt.show()


def vis_heatmap_and_rgb(
    heatmap_path: str,
    rgb_path: str,
) -> None:
    """ Визуализация карты высот и RGB-изображения датасета

    Args:
        heatmap_path (str): Путь до карты высот изображения
        rgb_path (str): Путь до RGB-изображения
    """
    heatmap = Image.open(heatmap_path)
    image = Image.open(rgb_path)
    _, axes = plt.subplots(1, 2, figsize=(10, 6))
    i = axes[0].imshow(heatmap)
    plt.colorbar(i)
    axes[0].set(title='Карта высот')
    axes[0].axis('off')
    axes[1].imshow(image)
    axes[1].set(title='rgb изображение')
    axes[1].axis('off')


def vis_heatmap_and_mask(
    heatmap: np.ndarray,
    mask: np.ndarray,
) -> None:
    """_summary_

    Args:
        heatmap (np.ndarray): _description_
        mask (np.ndarray): _description_
    """
    _, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(heatmap)
    axes[0].set(title='Карта высот')
    axes[0].axis('off')
    axes[1].imshow(mask)
    axes[1].set(title='маска')
    axes[1].axis('off')


def create_approximate_vis(
    contours_dict: Dict[int, List[Tuple[float]]],
) -> None:
    """_summary_

    Args:
        contours_dict (Dict[int, List[Tuple[float]]]): _description_
    """
    # Создание фигуры и осей
    _, ax = plt.subplots(figsize=(10, 8))

    # Цветовая палитра
    _ = plt.cm.tab10.colors  # Выбираем палитру
    area_threshold: int = 20
    # Итерация по словарю
    patches = []
    for _, polygons in contours_dict.items():
        for polygon in polygons:
            # Проверяем площадь полигона
            if Polygon(polygon).area < area_threshold:
                continue  # Пропускаем слишком маленькие полигоны
            poly_patch = Poly(polygon, closed=True)
            patches.append(poly_patch)
    # Добавляем коллекцию полигонов с цветами
    collection = PatchCollection(patches, cmap='tab10', edgecolor='black', alpha=0.6)
    collection.set_array(np.arange(len(patches)))  # Установка индекса цветов

    ax.add_collection(collection)

    # Настройка осей
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')


def vis_3d_house(
    height_map: np.ndarray,
    shape: Tuple[int] = (150, 150, 200),
) -> None:

    rows, cols = height_map.shape

    # Создаем координаты (x, y) для каждой точки
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ls = LightSource(azdeg=100, altdeg=45)
    cmap = plt.cm.viridis
    rgb = ls.shade(height_map, cmap=cmap, blend_mode='soft')
    # Построение поверхности
    surf = ax.plot_surface(x, y, height_map, facecolors=rgb, rstride=1, cstride=1, antialiased=True, alpha=0.7)

    # Настройка осей
    ax.grid(False)
    ax.set_xlim3d(0, shape[0])
    ax.set_ylim3d(0, shape[1])
    ax.set_zlim3d(0, shape[2])
    ax.set_axis_off()
    plt.show()


def all_visualisation(
    crop_image_list: List[Image.Image],
    house_map_list: List[np.ndarray],
    instance_mask_list: List[np.ndarray],
) -> None:
    fig, axs = plt.subplots(5, 3, figsize=(20, 20))
    for i, (crop_image, house_map, house) in enumerate(zip(crop_image_list, house_map_list, instance_mask_list)):
        clustered_heights_building = height_clustering(house_map, house)
        instance_masks = surface_clustering(clustered_heights_building)
        all_masks = concat_masks(instance_masks)
        clustered_surfaces_building = smoothed_mask(all_masks, min_size_objects=40, interations=2)
        contours_dict = create_polygons(instance_masks, tolerance=1)
        contours_dict_new = convert_to_rectangles(contours_dict, resolution=8.0)
        height_map = create_height_map(contours_dict_new, resolution=1)
        rows, cols = height_map.shape

        axs[0][i].imshow(crop_image)
        axs[0][i].set(title=f'Изображение {i+1}')

        num_colors: int = (clustered_heights_building.max() + 1).astype(int)
        colors: np.ndarray = np.random.rand(num_colors, 3)
        colors[0] = [0, 0, 0]
        cmap = ListedColormap(colors)
        im1 = axs[1][i].imshow(house_map)
        axs[1][i].set(title=f'Исходная карта высот {i+1}')
        cbar1 = fig.colorbar(im1, ax=axs[1][i])
        cbar1.set_label("Значения высот")

        im2 = axs[2][i].imshow(clustered_heights_building)
        axs[2][i].set_title(f"Кластеризация высот {i+1}")
        cbar2 = fig.colorbar(im2, ax=axs[2][i])
        cbar2.set_label("Значения высот")

        im3 = axs[3][i].imshow(clustered_surfaces_building, cmap=cmap, interpolation='nearest')
        axs[3][i].set_title(f"Кластеризация поверхностей {i+1}")
        cbar3 = fig.colorbar(im3, ax=axs[3][i])
        cbar3.set_label("i-ая поверхность")

         # 3D-визуализация
        ax = fig.add_subplot(4, 3, 3 * 3 + i + 1, projection='3d')  # Добавляем 3D-график
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        ls = LightSource(azdeg=100, altdeg=45)
        rgb = ls.shade(height_map, cmap=plt.cm.viridis, blend_mode='soft')
        surf = ax.plot_surface(x, y, height_map, facecolors=rgb, rstride=1, cstride=1, antialiased=True)
        # Добавление шкалы значений
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Height')
        ax.set_xlim3d(0, 200)
        ax.set_ylim3d(0, 200)
        ax.set_zlim3d(0, 200)
        ax.grid(False)
        ax.axis('off')
    plt.show()