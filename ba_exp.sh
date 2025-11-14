#!/bin/bash

# List your datasets here
datasets=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

# Number of iterations
iterations=10

for dataset in "${datasets[@]}"; do
    echo "Running BA experiment for $dataset"
    for ((i=1; i<=iterations; i++)); do
        echo "Iteration $i for $dataset"
        # Generate a random seed
        seed=$RANDOM
        # Replace the following line with your actual BA experiment command
        python main.py --scene $dataset --data_root data/nerf_synthetic --exp_name ${dataset}_ba_exp_$i --task ba --seed $seed
    done
done