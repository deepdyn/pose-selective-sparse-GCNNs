#!/bin/bash

# Defining the datasets and seeds to run
DATASETS=(
    "mnist"
    "rot_mnist"
    "fashion_mnist"
    "rot_fashion_mnist"
    "cifar10"
    "cifar10_plus"
    "gtsrb"
    "gtsrb_plus"
)

SEEDS=(0 1 2 3 4)

# Loop through all combinations and submit a job
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CONFIG_PATH="configs/${dataset}.yaml"
        
        # Check if the config file exists before submitting
        if [ -f "$CONFIG_PATH" ]; then
            echo "Submitting job for Dataset: ${dataset}, Seed: ${seed}"
            sbatch scripts/run_experiment.sh ${CONFIG_PATH} ${seed}
        else
            echo "WARNING: Config file not found at ${CONFIG_PATH}. Skipping."
        fi
    done
done

echo "All jobs submitted. Use 'squeue -u \$USER' to check status."