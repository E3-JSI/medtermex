# SLURM (Simple Linux Utility for Resource Management)

SLURM is a free and open-source job scheduling system designed for Linux clusters. It provides three key functions:

1. Allocating compute resources to users for specific durations
2. Providing a framework for starting, executing, and monitoring work on allocated resources
3. Arbitrating resource contention

## Essential Commands

### Job Management

#### sbatch (submit match jobs)

**URL:** https://slurm.schedmd.com/sbatch.html

The `sbatch` command is used to submit batch jobs to the SLURM scheduler. It is used for non-interactive jobs, resource-intensive processes, scheduled jobs, job dependencies, etc.

The key difference from `srun` is that `sbatch` submits a script to the job queue for later execution, while `srun` attempts to execute a command immediately. Think of `sbatch` as "submit and forget" - your job will run when resources are available, and you can check on it later using commands like `squeue` or `sacct`.

##### A typical SLURM batch script

```bash
#!/bin/bash
#SBATCH --job-name=job-name         # Name of the job
#SBATCH --output=output_%j.log      # Standard output file (%j = job ID)
#SBATCH --error=error_%j.log        # Standard error file
#SBATCH --time=12:00:00             # Time limit (format: HH:MM:SS)
#SBATCH --partition=compute         # Partition (queue) to use
#SBATCH --nodes=2                   # Number of nodes to allocate
#SBATCH --ntasks=16                 # Total number of tasks (processes)
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mail-type=END             # Email notification when job ends
#SBATCH --mail-user=user@email.com  # Email address for notifications

# Load necessary modules
module load python

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the application
srun ./my_application
```

#### srun (run interactive executions)

**URL:** https://slurm.schedmd.com/srun.html

Used for running interactive executions. The key benefit of `srun` is that it handles resource allocation through SLURM, ensuring your job gets the requested resources and runs with the proper environment setup.

```bash
# Basic syntax
srun [options] command [args]

# Example: Run a command on GPU partition with 2 GPUs
srun -p gpu --gres=gpu:2 -n 1 -c 1 nvidia-smi

# Common options:
#   -p, --partition     : Specify partition (queue)
#   --gres              : Request generic resources (e.g., GPUs)
#   -n, --ntasks        : Number of tasks (processes)
#   -c, --cpus-per-task : CPU cores per task
#   --mem               : Memory requirement
#   --time              : Time limit
```

#### squeue (view job information)

**URL:** https://slurm.schedmd.com/squeue.html

View information about jobs in the queue.

```bash
# View all jobs
squeue

# View your jobs
squeue -u $USER

# Monitor specific job
squeue -j <job_id>
```

#### scancel (cancel jobs)

**URL:** https://slurm.schedmd.com/scancel.html

Cancel a job or job step.

```bash
# Cancel a specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel all jobs in a partition
scancel -p <partition_name>
```

### Resource Information

#### sinfo (view node and partition information)

**URL:** https://slurm.schedmd.com/sinfo.html

View information about SLURM nodes and partitions.

```bash
# Basic partition and node information
sinfo

# Detailed format showing node counts and states
sinfo -o "%20P %5D %14F %8z %10m %10G %6w %8f %N"

# GPU partition information
sinfo -p gpu -o "%20P %5D %6t %8G %N"

# Common options:
#   -N, --nodes       : Show node information
#   -p, --partition   : Show specific partition
#   -o, --format      : Custom output format
```

#### scontrol (view and modify configuration)

**URL:** https://slurm.schedmd.com/scontrol.html

View and modify SLURM configuration and state.

```bash
# Show all nodes
scontrol show nodes

# Show specific node
scontrol show node=nodename

# Detailed node information
scontrol -d show node=nodename

# Show partition information
scontrol show partition=<partition_name>

# Show job information
scontrol show job=<job_id>
```

#### sacct (view accounting information)

**URL:** https://slurm.schedmd.com/sacct.html

View accounting information for jobs and job steps.

```bash
# View your recent jobs
sacct

# View specific job
sacct -j <job_id>

# Custom format with start/end times
sacct --format=JobID,JobName,Start,End,State,MaxRSS,MaxVMSize

# View jobs from specific user
sacct -u <username>
```

## Partition States

Understanding partition states is crucial for job scheduling:

- **idle**: Partition is available and ready to accept jobs
- **alloc**: All nodes in the partition are allocated to running jobs
- **mix**: Some nodes are allocated while others are idle
- **drain$**: Partition is being drained but will accept new jobs
- **drain\***: Partition is being drained and won't accept new jobs
- **down\***: Partition is completely unavailable
- **comp**: Compute partition (custom name, typically for general-purpose computing)

### Draining Process

When a partition is being "drained":

1. No new jobs will be scheduled on the partition
2. Running jobs continue until completion
3. Nodes become unavailable after all jobs finish
4. This is a graceful way to take nodes offline for maintenance

## Best Practices

1. **Resource Requests**

   - Always specify appropriate time limits
   - Request sufficient memory for your job
   - Use `--cpus-per-task` for multi-threaded applications

2. **Job Management**

   - Monitor your jobs using `squeue`
   - Use job arrays for similar tasks
   - Set up email notifications for long-running jobs

3. **Error Handling**

   - Check output and error logs
   - Use `sacct` to investigate failed jobs
   - Set appropriate resource limits to prevent job failures

4. **Performance**
   - Use appropriate partitions for your workload
   - Consider using job arrays for parallel tasks
   - Monitor resource usage with `seff <job_id>`
