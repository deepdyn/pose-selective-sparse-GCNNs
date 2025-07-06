#!/bin/bash

#SBATCH --job-name=run_all
#SBATCH --partition=gpu
#SBATCH --output=logs/run_all.out

# --- Configuration ---
# Maximum number of jobs to have in the queue at one time.
# From the user manual (p. 53), the GPU partition allows 3 running jobs.
MAX_JOBS=3
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

        # If the job is new, wait until there is space in the queue
        while true; do
            # --- STAGE 1: Robustly check the queue status ---
            squeue_output=$(squeue -u $USER -h 2>&1)
            squeue_exit_code=$?

            if [ $squeue_exit_code -ne 0 ]; then
                echo "ERROR: squeue command failed with exit code ${squeue_exit_code}. Message: ${squeue_output}"
                echo "Waiting for 60 seconds before retrying..."
                sleep 60
                continue # Retry the squeue command
            fi
            
            current_jobs=$(echo "$squeue_output" | wc -l)

            if [ $current_jobs -lt $MAX_JOBS ]; then
                # There is space in the queue, so we can try to submit
                echo "Queue has space (${current_jobs}/${MAX_JOBS}). Attempting to submit job: ${JOB_ID}"
                
                if [ ! -f "$CONFIG_PATH" ]; then
                    echo "WARNING: Config file not found at ${CONFIG_PATH}. Skipping ${JOB_ID}."
                    # Log as "skipped" so we don't try again
                    echo "${JOB_ID} # SKIPPED - no config" >> "${SUBMITTED_JOBS_LOG}"
                    break
                fi
                
                # --- STAGE 2: Robustly submit the job and verify ---
                submission_output=$(sbatch --job-name=${JOB_ID} --output=logs/${JOB_ID}_%j.out scripts/run_experiment.sh ${CONFIG_PATH} ${seed} 2>&1)
                sbatch_exit_code=$?

                if [ $sbatch_exit_code -eq 0 ] && [[ "$submission_output" == *"Submitted batch job"* ]]; then
                    # SUCCESS: The job was accepted by Slurm.
                    echo "Successfully submitted job: ${JOB_ID}. Slurm response: ${submission_output}"
                    
                    # Log that this job has been successfully submitted
                    echo "${JOB_ID}" >> "${SUBMITTED_JOBS_LOG}"
                    
                    # Break the inner 'while' loop to move to the next job in the 'for' loop
                    break
                else
                    # FAILURE: The sbatch command failed or timed out.
                    echo "ERROR: Failed to submit job ${JOB_ID}. Slurm response: '${submission_output}'. Retrying..."
                    # The script will loop and re-check the queue before trying to submit again.
                    sleep 30 
                fi

            else
                # If the queue is full, wait for a minute before checking again
                echo "Job limit (${MAX_JOBS}) reached. Waiting for 60 seconds..."
                sleep 60
            fi
        done
    done
done

echo "All jobs have been submitted."

