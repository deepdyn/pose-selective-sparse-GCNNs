#!/bin/bash
#SBATCH --job-name=gcnn_exp      # Job name
#SBATCH --output=logs/%x_%j.out  # Standard output and error log (%x=job name, %j=job id)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1             # Request 1 GPU on Paramganga
#SBATCH --time=08:00:00          # Time limit hrs:min:sec
#SBATCH --mem=32G

# --- Environment Setup ---
# Load the necessary modules provided by IITR HPC
# module load anaconda/3
# Activate conda environment
# source activate your_pytorch_env 

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