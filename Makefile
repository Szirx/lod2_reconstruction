run:
	python src/main.py \
	--weights_path ../../../shared_data/users/avlasov/vaihingen.pt \
	--heatmap_path ../../shared_data/datasets/Vaihingen/train/NDSM/area34.tif \
	--rgb_path ../../shared_data/datasets/Vaihingen/train/RGB/area34.tif \
	--resolution 7.0 --flatness_thd 20 --cell_area_thd 0.0 --poly_area_thd 0.0 \
	--output_name lod2
