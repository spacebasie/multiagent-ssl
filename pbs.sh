#!/bin/bash

# --- PBS Directives ---
# These lines are instructions for the job scheduler. They are not executed as commands.

# Set the name of the job. This will appear in queue listings.
#PBS -N vicreg_multiagent

# Request computing resources. This is a crucial step.
# We are requesting 1 node, 8 CPU cores, 32GB of memory, and 1 GPU.
# The GPU is essential for deep learning. 8 CPUs will help with data loading.
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1

# Set the maximum walltime for the job (e.g., 4 hours).
# The job will be terminated if it runs longer than this.
# Adjust this based on how long you expect your experiment to run.
#PBS -l walltime=04:00:00

# --- Job Execution ---

# It's a critical best practice to change to the directory where you submitted the job.
# This ensures that your script can find all your project files (main.py, config.py, etc.).
cd $PBS_O_WORKDIR

echo "Job started on $(hostname) at $(date)"
echo "Working directory is $(pwd)"

# Load the necessary software modules provided by the cluster.
echo "Loading modules..."
module load Python/3.10.4-GCCcore-11.3.0

# Activate your specific Python virtual environment.
echo "Activating virtual environment..."
source ~/venv/venv25/bin/activate

# These arguments will override the defaults in config.py.
echo "Starting Python script..."
# For office_hierarchical
# python main.py \
#    --mode 'decentralized' \
#    --dataset 'office_home' \
#    --heterogeneity_type 'office_hierarchical' \
#    --topology 'random' \
#    --num_neighborhoods 2 \
#    --alpha 100 \
#    --agents_per_neighborhood 4 \
#    --eval_every 2 \
#    --num_classes 10 \
#    --comm_rounds 10 \
#    --local_epochs 2

#REMEMBER TO CHECK THE LEARNING RATE!

#For cifar10
python main.py \
    --mode 'decentralized' \
    --dataset 'cifar10' \
    --heterogeneity_type 'label_skew_personalized' \
    --topology 'fully_connected' \
    --alpha 0.5 \
    --eval_every 5 \
    --num_classes 10 \
    --comm_rounds 200 \
    --num_agents 5 \
    --local_epochs 1

echo "Job finished at $(date)"