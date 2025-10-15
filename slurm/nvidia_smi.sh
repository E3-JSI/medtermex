#!/bin/bash
#SBATCH --job-name=nvidia-smi            # Name of the job
#SBATCH --account=[account_name]         # Account name
#SBATCH --output=logs/nvidia-smi-%j.out  # Standard output file (%j = job ID)
#SBATCH --error=logs/nvidia-smi-%j.out   # Standard error file (same as output file) (change to .err if you want to separate the output and error logs)
#SBATCH --time=00:01:00                  # Time limit (format: HH:MM:SS)
#SBATCH --partition=gpu                  # Partition (queue) to use
#SBATCH --gres=gpu:4                     # Number of GPUs per node
#SBATCH --nodes=1                        # Number of nodes to allocate
#SBATCH --ntasks=1                       # Total number of tasks (processes)
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=4G                         # Memory requirement per node

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=50202

# Print some information about the job
echo "Running on host: $(hostname)"
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}
echo "NODELIST="${SLURM_NODELIST}
echo "Current working directory: $(pwd)"
echo "Start time: $(date)"

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Print end time
echo "End time: $(date)"
