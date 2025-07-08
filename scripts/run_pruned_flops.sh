#!/bin/bash
#SBATCH --job-name=run_pruned_flops
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pruned_flops.out
#SBATCH --error=logs/pruned_flops.err

python src/calculate_pruned_flops.py --config configs/cifar10.yaml --checkpoint results/CIFAR10/PoseSelectiveSparse_ResNet44/seed_0/model_best.pth

