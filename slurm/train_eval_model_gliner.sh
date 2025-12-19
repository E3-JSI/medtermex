#!/bin/bash
#SBATCH --job-name=TNERGLINER                       # Name of the job
#SBATCH --account=[account_name]                    # Account name
#SBATCH --output=logs/train-model-gliner-%j.out     # Standard output file (%j = job ID)
#SBATCH --error=logs/train-model-gliner-%j.out      # Standard error file (same as output file)
#SBATCH --time=04:00:00                             # Time limit (format: HH:MM:SS)
#SBATCH --partition=gpu                             # Partition (queue) to use
#SBATCH --gres=gpu:1                                # Number of GPUs per node
#SBATCH --nodes=1                                   # Number of nodes to allocate
#SBATCH --ntasks=1                                  # Total number of tasks (processes)
#SBATCH --ntasks-per-node=1                         # Number of tasks per node
#SBATCH --cpus-per-task=8                           # Number of CPU cores per task

set -e # exit on error

# ===============================================
# Python runner helper (uv or standard python)
# ===============================================
# Automatically use uv if available, otherwise fall back to python with venv
if command -v uv &> /dev/null; then
    RUN_PYTHON="uv run python"
else
    # Activate the appropriate virtual environment for GLiNER
    # Priority: .venv-gliner > .venv
    if [ -d ".venv-gliner" ]; then
        source .venv-gliner/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        echo "Warning: No virtual environment found. Please create one with the appropriate dependencies."
        exit 1
    fi
    RUN_PYTHON="python"
fi

echo "# ==============================================="
echo "# Job information"
echo "# ==============================================="
echo ""

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=50202

# Print some information about the job
echo "Running on host: $(hostname)"
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "NODELIST="${SLURM_NODELIST}
echo "Current working directory: $(pwd)"
echo "Start time: $(date)"
echo ""


echo "# ==============================================="
echo "# Make sure GPU is available"
echo "# ==============================================="
echo ""

# Print GPU info for debugging
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

echo ""

# ===============================================
# Load the dataset directory parameters
# ===============================================

# Storage directories
BASE_STORAGE_DIR=[base_storage_dir]
BASE_PROJECT_DIR=[base_project_dir]

# Subdirectories
DATASET_DIR=[dataset_dir]
MODELS_DIR=[models_dir]
RESULTS_DIR=[results_dir]

# Dataset file names
TRAIN_DATASET_FILE=[train_dataset_file]
TEST_DATASET_FILE=[test_dataset_file]

# ===============================================
# Export the environment variables
# ==============================================

# Hugging Face cache (set them to a place where you have enough space)
export HF_HOME=$BASE_STORAGE_DIR/huggingface
export HUGGINGFACE_HUB_CACHE=$BASE_STORAGE_DIR/huggingface/hub
export HF_HUB_CACHE=$BASE_STORAGE_DIR/huggingface/hub

# Create the directories if they don't exist
mkdir -p $HF_HOME
mkdir -p $HUGGINGFACE_HUB_CACHE
mkdir -p $HF_HUB_CACHE

# ===============================================
# Load the experiment parameters
# ===============================================
# Model parameters

# GLiNER model options:
# urchade/gliner_small-v2.1
# urchade/gliner_medium-v2.1
# urchade/gliner_large-v2.1
# urchade/gliner_multi-v2.1
# urchade/gliner_multi_pii-v1
# knowledgator/gliner-multitask-large-v0.5
MODEL_NAME=urchade/gliner_multi-v2.1

# Training parameters
TRAIN_NUM_EPOCHS=3
TRAIN_BATCH_SIZE=8
TRAIN_LEARNING_RATE=5e-6
TRAIN_WEIGHT_DECAY=0.01

# Evaluation parameters
EVAL_THRESHOLD=0.5
EVAL_METRICS="exact,relaxed,overlap"
EVAL_USE_CPU=false

# ===============================================
# Prepare the training and output directories
# ===============================================

# Load the training dataset file
TRAIN_DATASET_FILE_PATH=${BASE_STORAGE_DIR}/${DATASET_DIR}/${TRAIN_DATASET_FILE}
EVAL_DATASET_FILE_PATH=${BASE_STORAGE_DIR}/${DATASET_DIR}/${TEST_DATASET_FILE}

MODEL_STORE_NAME="${MODEL_NAME}-${SLURM_JOB_ID}"

# Load the train and test output directories
TRAIN_OUTPUT_DIR=${BASE_STORAGE_DIR}/${MODELS_DIR}/${MODEL_STORE_NAME}
TEST_OUTPUT_DIR=${BASE_PROJECT_DIR}/${RESULTS_DIR}/${MODEL_STORE_NAME}


echo "# ==============================================="
echo "# Parameters"
echo "# ==============================================="
echo ""
echo "MODEL_NAME=${MODEL_NAME}"
echo "TRAIN_NUM_EPOCHS=${TRAIN_NUM_EPOCHS}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "TRAIN_LEARNING_RATE=${TRAIN_LEARNING_RATE}"
echo "TRAIN_WEIGHT_DECAY=${TRAIN_WEIGHT_DECAY}"
echo "EVAL_THRESHOLD=${EVAL_THRESHOLD}"
echo "EVAL_METRICS=${EVAL_METRICS}"
echo "EVAL_USE_CPU=${EVAL_USE_CPU}"
echo ""
echo "TRAIN_DATASET_FILE_PATH=${TRAIN_DATASET_FILE_PATH}"
echo "EVAL_DATASET_FILE_PATH=${EVAL_DATASET_FILE_PATH}"
echo "TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}"
echo "TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo ""


echo "================================================"
echo "Training the model..."
echo "================================================"
echo ""

$RUN_PYTHON -m src.training.train_gliner \
    --train-dataset-file ${TRAIN_DATASET_FILE_PATH} \
    --model-name-or-path ${MODEL_NAME} \
    --model-output-dir ${TRAIN_OUTPUT_DIR} \
    --num-train-epochs ${TRAIN_NUM_EPOCHS} \
    --train-batch-size ${TRAIN_BATCH_SIZE} \
    --train-learning-rate ${TRAIN_LEARNING_RATE} \
    --train-weight-decay ${TRAIN_WEIGHT_DECAY} \
    --use-cpu ${EVAL_USE_CPU}

echo "================================================"
echo "Testing the model..."
echo "================================================"
echo ""

$RUN_PYTHON -m src.training.evaluate_gliner \
    --eval-dataset-file ${EVAL_DATASET_FILE_PATH} \
    --results-dir ${TEST_OUTPUT_DIR} \
    --model-dir ${TRAIN_OUTPUT_DIR} \
    --eval-threshold ${EVAL_THRESHOLD} \
    --eval-metrics ${EVAL_METRICS} \
    --use-cpu ${EVAL_USE_CPU}

echo "Removing the model directory..."
rm -rf ${TRAIN_OUTPUT_DIR}

# Print end time
echo ""
echo "End time: $(date)"

