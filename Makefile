run:
	python src/main.py \
	--weights_path weights/vaihingen.pt \
	--heatmap_path images/NDSM/area34.tif \
	--rgb_path images/RGB/area34.tif \
	--resolution 7.0 --flatness_thd 20 --cell_area_thd 0.0 --poly_area_thd 0.0 \
	--output_name lod2
