#!/bin/bash

#SBATCH --job-name=gcnn_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# --- Environment Setup ---
# Load the necessary modules provided by IITR HPC
# module load anaconda/3
# Activate conda environment
source /scratch/pradeep.cs.iitr/venv-kanishk/bin/activate

# --- Set Python Path ---
export PYTHONPATH=$PWD:$PYTHONPATH

# --- Argument Parsing ---
CONFIG_FILE=$1
SEED=$2

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================================"
echo "Starting job on host: $(hostname)"
echo "Timestamp: $(date)"
echo "Running experiment with config: ${CONFIG_FILE} and seed: ${SEED}"
echo "========================================================"

# --- Execute the Python Training Script ---
python src/train.py --config ${CONFIG_FILE} --seed ${SEED}

echo "========================================================"
echo "Job finished with exit code $?"
echo "Timestamp: $(date)"
echo "========================================================"
