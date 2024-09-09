
------------------------------------
### gs_flat3d_image

python scripts/train.py
--eval -s image_dir -m model_output_path --gs_type gs_flat3d_image --scene_image image --distance 1

python scripts/render.py 
-m model_output_path --scene_image image --distance 1

or mirror:

python scripts/train.py
--eval -s image_dir -m model_output_path --gs_type gs_flat3d_image --scene_image mirror --distance 1

python scripts/render.py 
-m model_output_path --scene_image mirror --distance 1 

python metrics.py
-m model_output_path

------------------------------------
### gs_flat2d_image
python scripts/train.py
--eval -s image_dir -m model_output_path --gs_type gs_flat2d_image --scene_image mirror --distance 1

python scripts/render.py 
-m model_output_path --scene_image mirror --distance 1  --gs_type gs_flat2D 

python metrics.py 
-m model_output_path --gs_type gs_flat2D 
------------------------------------
### gs_flat_slices
python scripts/train.py
--eval -s image_dir -m model_output_path --gs_type gs_flat_slices --scene_image mirror --distance 1

python scripts/render.py 
-m model_output_path --scene_image mirror --distance 1 

python metrics.py
-m model_output_path 

------------------------------------


python scripts/save_pseudomesh.py
--model_path model_output_path

python scripts/render_from_object.py 
-m model_output_path --object_path object_path.obj

python scripts/render_simulation.py 
-m model_output_path --simulation_path simulation_path --save_trajectory