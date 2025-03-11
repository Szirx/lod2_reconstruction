# LoD 2 Reconstruction

Исследование и реализация методов восстановления LoD-2 моделей зданий на спутниковых снимках.

#### Ссылки на оригинальные статьи:

* *[Implicit Regularization for Reconstructing 3D Building Rooftop Models Using Airborne LiDAR Data](https://www.mdpi.com/1424-8220/17/3/621)*

* *[Simplification and Regularization Algorithm for Right-Angled Polygon Building Outlines with Jagged Edges](https://www.mdpi.com/2220-9964/12/12/469)*

![Visualisation of process LoD-2 reconstruction](/images/photo_2024-12-17_13-49-31.jpg)


#### Текущая версия пайплайна:

![Алгоритм](/images/reconstruction.drawio.png)

### Usage

Run the script from the command line with the following arguments:

```
python src/main.py --weights_path <path_to_weights> --heatmap_path <path_to_heatmap> --rgb_path <path_to_rgb> [--resolution <float>] [--flatness_thd <int>] [--cell_area_thd <int>] [--poly_area_thd <int>] [--output_name <str>]
```

### Arguments

| Argument         | Type  | Required | Default Value | Description                                                                 |
|------------------|-------|----------|---------------|-----------------------------------------------------------------------------|
| `--weights_path` | str   | Yes      | -             | Path to the YOLO model weights file (e.g., `vaihingen.pt`).                |
| `--heatmap_path` | str   | Yes      | -             | Path to the heatmap file (e.g., `area34.tif`).                             |
| `--rgb_path`     | str   | Yes      | -             | Path to the RGB image file (e.g., `area34.tif`).                           |
| `--resolution`   | float | No       | `7.0`         | Resolution of grid cells for rectangle polygons.                            |
| `--flatness_thd` | int   | No       | `20`          | Flatness threshold for surfaces.                                           |
| `--cell_area_thd`| int   | No       | `0`           | Cell area threshold for adding rectangle minipolygons to the final polygon. |
| `--poly_area_thd`| int   | No       | `0`           | Area threshold for removing small polygons from processing.                 |
| `--output_name`| str   | No       | `lod2`           |  Filename of output file with .obj format                |


### Example Commands

1. With all arguments specified:

```
python src/main.py --weights_path weights/vaihingen.pt --heatmap_path images/NDSM/area34.tif --rgb_path images/RGB/area34.tif --resolution 7.0 --flatness_thd 20 --cell_area_thd 0 --poly_area_thd 0 --output_name lod2
```

2. Using default values for optional arguments:

```
python src/main.py --weights_path weights/vaihingen.pt --heatmap_path images/NDSM/area34.tif --rgb_path images/RGB/area34.tif
```

In this case, the script will use the default values:
* resolution = 7.0
* flatness_thd = 20
* cell_area_thd = 0
* poly_area_thd = 0
* output_name = 'lod2'

### Output

The script generates the following output:

* 3D Object File: A .obj file containing the 3D representation of the processed surfaces, saved as filename.obj.


### Customization

You can adjust the following parameters to fine-tune the surface processing:

* Resolution (--resolution): Controls the granularity of the grid cells for generating rectangle polygons.
* Flatness Threshold (--flatness_thd): Determines the flatness required for a surface to be considered valid.
* Cell Area Threshold (--cell_area_thd): Filters out small cells from the final polygon.
* Polygon Area Threshold (--poly_area_thd): Removes small polygons from further processing.
* Output name (--output_name): Filename of .obj output file

### Notes

* Ensure that the paths provided for weights_path, heatmap_path, and rgb_path are correct and accessible.
* The script assumes that the input images and heatmaps are in a compatible format (e.g., .tif).
* The output .obj file can be visualized using 3D rendering tools like Blender or MeshLab.