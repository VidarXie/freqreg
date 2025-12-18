#!/bin/bash

# List your datasets here
datasets=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for dataset in "${datasets[@]}"; do
    echo "Running sampling experiment for $dataset"
    
    # Generate a random seed
    seed=$RANDOM
    # Replace the following line with your actual BA experiment command
    python main.py --scene $dataset --data_root data/nerf_synthetic --exp_name Sampling_${dataset}_w --task nerf --seed $seed --max_steps 20000
    done
done