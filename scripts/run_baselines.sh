#!/bin/bash
#SBATCH --job-name=run_baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/baselines_output.out
#SBATCH --error=logs/baselines_error.out

python src/calculate_baseline_flops.py --config configs/cifar10.yaml
