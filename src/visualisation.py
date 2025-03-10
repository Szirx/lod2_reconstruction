from typing import Tuple
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource

ALPHA: float = 0.7
ALTDEG: int = 45


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


def vis_3d_house(
    height_map: np.ndarray,
    shape: Tuple[int] = (150, 150, 200),
) -> None:

    rows, cols = height_map.shape

    x, y = np.meshgrid(
        np.arange(cols),
        np.arange(rows),
    )
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ls = LightSource(azdeg=100, altdeg=ALTDEG)
    cmap = plt.cm.viridis
    rgb = ls.shade(height_map, cmap=cmap, blend_mode='soft')
    # Построение поверхности
    ax.plot_surface(
        x,
        y,
        height_map,
        facecolors=rgb,
        rstride=1,
        cstride=1,
        antialiased=True,
        alpha=ALPHA,
    )

    ax.grid(False)
    ax.set_xlim3d(0, shape[0])
    ax.set_ylim3d(0, shape[1])
    ax.set_zlim3d(0, shape[2])
    ax.set_axis_off()
    plt.show()
