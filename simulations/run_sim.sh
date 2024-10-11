#!/bin/bash
# run from main directory
export PYTHONPATH=.

DATA=apple
IMAGE_DIR=data/$DATA
MODEL_DIR=output/$DATA
type=2d
ITER=30000

python train.py -s $IMAGE_DIR -m $MODEL_DIR --gs_type $type --camera mirror --iterations $ITER -w
python scripts/render.py -m $MODEL_DIR --camera mirror --distance 1 --gs_type $type
python metrics.py -m $MODEL_DIR --gs_type $type

SIM="$MODEL_DIR/simulation"
python scripts/save_pseudomesh.py --model_path $MODEL_DIR
python simulations/apple.py -i "$MODEL_DIR/pseudomesh_info/ours_$ITER" -o $SIM
python scripts/render_simulation.py --model_path $MODEL_DIR --simulation_path $SIM --save_trajectory --scale 1