#!/bin/bash

#SBATCH --job-name=run_all
#SBATCH --partition=gpu
#SBATCH --output=logs/run_all.out

# --- Configuration ---
# Maximum number of jobs to have in the queue at one time.
# From the user manual (p. 53), the GPU partition allows 3 running jobs.
MAX_JOBS=4
# Log file to keep track of jobs that have already been submitted.
SUBMITTED_JOBS_LOG="submitted_jobs.log"

# --- Job Definitions ---
DATASETS=(
    "mnist"
    "rot_mnist"
    "fashionmnist"
    "rot_fashion_mnist"
    "cifar10"
    "cifar10_plus"
    "gtsrb"
    "gtsrb_plus"
)
SEEDS=(0 1 2 3 4)

# Create the log file if it doesn't exist
touch ${SUBMITTED_JOBS_LOG}

# --- Main Loop ---
# Loop through all job combinations
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Create a unique identifier for this specific job
        JOB_ID="gcnn_${dataset}_seed_${seed}"
        CONFIG_PATH="configs/${dataset}.yaml"

        # Check if this job has already been submitted by looking in our log file
        if grep -q "${JOB_ID}" "${SUBMITTED_JOBS_LOG}"; then
            echo "Skipping already submitted job: ${JOB_ID}"
            continue
        fi

        # If the job is new, check if there is space in the queue
        while true; do
            current_jobs=$(squeue -u $USER -h | wc -l)
            if [ $current_jobs -lt $MAX_JOBS ]; then
                # There is space in the queue, so we can submit
                echo "Queue has space (${current_jobs}/${MAX_JOBS}). Submitting job: ${JOB_ID}"
                
                if [ -f "$CONFIG_PATH" ]; then
                    # Submit the job. Note that we now pass the unique Job ID to the script.
                    # Your run_experiment.sh should have a --job-name directive for this to be effective.
                    sbatch --job-name=${JOB_ID} --output=logs/${JOB_ID}_%j.out scripts/run_experiment.sh ${CONFIG_PATH} ${seed}
                    
                    # Log that this job has been successfully submitted
                    echo "${JOB_ID}" >> "${SUBMITTED_JOBS_LOG}"
                else
                    echo "WARNING: Config file not found at ${CONFIG_PATH}. Skipping ${JOB_ID}."
                fi
                
                # Break the inner 'while' loop to move to the next job in the 'for' loop
                break 
            else
                # If the queue is full, wait for a minute before checking again
                echo "Job limit (${MAX_JOBS}) reached. Waiting for 60 seconds..."
                sleep 60
            fi
        done
    done
done

echo "All jobs have been submitted."

