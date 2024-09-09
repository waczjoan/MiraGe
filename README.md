python scripts/train.py
--eval -s dataset_path -m model_output_path --gs_type gs_flat2d 

python scripts/render.py 
-m model_output_path

python scripts/save_pseudomesh.py
--model_path model_output_path

python scripts/render_from_object.py 
-m model_output_path --object_path object_path.obj

python scripts/render_simulation.py 
-m model_output_path --simulation_path simulation_path --save_trajectory