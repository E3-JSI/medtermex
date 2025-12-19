#!/bin/bash
#SBATCH --job-name=TNERUNEMC                      # Name of the job
#SBATCH --account=[account_name]                  # Account name
#SBATCH --output=logs/train-model-unsloth-%j.out  # Standard output file (%j = job ID)
#SBATCH --error=logs/train-model-unsloth-%j.out   # Standard error file (same as output file) (change to .err if you want to separate the output and error logs)
#SBATCH --time=06:00:00                           # Time limit (format: HH:MM:SS)
#SBATCH --partition=gpu                           # Partition (queue) to use
#SBATCH --gres=gpu:1                              # Number of GPUs per node
#SBATCH --nodes=1                                 # Number of nodes to allocate
#SBATCH --ntasks=1                                # Total number of tasks (processes)
#SBATCH --ntasks-per-node=1                       # Number of tasks per node
#SBATCH --cpus-per-task=8                         # Number of CPU cores per task

set -e # exit on error

# ===============================================
# Python runner helper (uv or standard python)
# ===============================================
# Automatically use uv if available, otherwise fall back to python with venv
if command -v uv &> /dev/null; then
    RUN_PYTHON="uv run python"
else
    # Activate the appropriate virtual environment for Unsloth
    # Priority: .venv-unsloth > .venv
    if [ -d ".venv-unsloth" ]; then
        source .venv-unsloth/bin/activate
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

# unsloth/medgemma-4b-it
# unsloth/medgemma-27b-text-it
# unsloth/gemma-3-270m-it
# unsloth/gemma-3n-E2B-it
# unsloth/gemma-3n-E4B-it
# unsloth/gemma-3-4b-it
# unsloth/gemma-3-12b-it
# unsloth/gemma-3-27b-it
# unsloth/llama-3-8b-Instruct
# unsloth/Llama-3.1-8B-Instruct
# unsloth/Llama-3.2-1B-Instruct
# unsloth/Llama-3.2-3B-Instruct
MODEL_NAME=unsloth/gemma-3-27b-it
MODEL_MAX_SEQ_LENGTH=4096
MODEL_LOAD_IN_4BIT=true
MODEL_LOAD_IN_8BIT=false
MODEL_FULL_FINETUNING=false

# PEFT parameters
PEFT_FT_VISION_LAYERS=false
PEFT_FT_LANGUAGE_LAYERS=true
PEFT_FT_ATTENTION_MODULES=true
PEFT_FT_MLP_MODULES=true
# Use heuristics: PEFT_LORA_ALPHA = PEFT_RANK * 2
PEFT_RANK=64
PEFT_LORA_ALPHA=128
PEFT_LORA_DROPOUT=0.0
PEFT_LORA_BIAS=false

# Training parameters
TRAIN_PER_DEVICE_BATCH_SIZE=4
TRAIN_GRADIENT_ACCUMULATION_STEPS=4
TRAIN_NUM_EPOCHS=15
TRAIN_LEARNING_RATE=2e-4
TRAIN_WEIGHT_DECAY=0.01
TRAIN_WARMUP_STEPS=5
TRAIN_LR_SCHEDULER_TYPE=linear
TRAIN_SEED=42

SYSTEM_PROMPT="You are a medical entity extractor from clinical texts. Extract the entities from the text and return them in a structured JSON format."
UNIQUE_ENTITIES=true

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
echo "MODEL_MAX_SEQ_LENGTH=${MODEL_MAX_SEQ_LENGTH}"
echo "MODEL_LOAD_IN_4BIT=${MODEL_LOAD_IN_4BIT}"
echo "MODEL_LOAD_IN_8BIT=${MODEL_LOAD_IN_8BIT}"
echo "MODEL_FULL_FINETUNING=${MODEL_FULL_FINETUNING}"
echo "PEFT_FT_VISION_LAYERS=${PEFT_FT_VISION_LAYERS}"
echo "PEFT_FT_LANGUAGE_LAYERS=${PEFT_FT_LANGUAGE_LAYERS}"
echo "PEFT_FT_ATTENTION_MODULES=${PEFT_FT_ATTENTION_MODULES}"
echo "PEFT_FT_MLP_MODULES=${PEFT_FT_MLP_MODULES}"
echo "PEFT_RANK=${PEFT_RANK}"
echo "PEFT_LORA_ALPHA=${PEFT_LORA_ALPHA}"
echo "PEFT_LORA_DROPOUT=${PEFT_LORA_DROPOUT}"
echo "PEFT_LORA_BIAS=${PEFT_LORA_BIAS}"
echo "TRAIN_PER_DEVICE_BATCH_SIZE=${TRAIN_PER_DEVICE_BATCH_SIZE}"
echo "TRAIN_GRADIENT_ACCUMULATION_STEPS=${TRAIN_GRADIENT_ACCUMULATION_STEPS}"
echo "TRAIN_NUM_EPOCHS=${TRAIN_NUM_EPOCHS}"
echo "TRAIN_LEARNING_RATE=${TRAIN_LEARNING_RATE}"
echo "TRAIN_WEIGHT_DECAY=${TRAIN_WEIGHT_DECAY}"
echo "TRAIN_WARMUP_STEPS=${TRAIN_WARMUP_STEPS}"
echo "TRAIN_LR_SCHEDULER_TYPE=${TRAIN_LR_SCHEDULER_TYPE}"
echo "TRAIN_SEED=${TRAIN_SEED}"
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

$RUN_PYTHON src/training/train_unsloth.py \
    --train-dataset-file ${TRAIN_DATASET_FILE_PATH} \
    --output-dir ${TRAIN_OUTPUT_DIR} \
    --model-name-or-path ${MODEL_NAME} \
    --model-max-seq-length ${MODEL_MAX_SEQ_LENGTH} \
    --model-load-in-4bit ${MODEL_LOAD_IN_4BIT} \
    --model-load-in-8bit ${MODEL_LOAD_IN_8BIT} \
    --model-full-finetuning ${MODEL_FULL_FINETUNING} \
    --peft-ft-vision-layers ${PEFT_FT_VISION_LAYERS} \
    --peft-ft-language-layers ${PEFT_FT_LANGUAGE_LAYERS} \
    --peft-ft-attention-modules ${PEFT_FT_ATTENTION_MODULES} \
    --peft-ft-mlp-modules ${PEFT_FT_MLP_MODULES} \
    --peft-rank ${PEFT_RANK} \
    --peft-lora-alpha ${PEFT_LORA_ALPHA} \
    --peft-lora-dropout ${PEFT_LORA_DROPOUT} \
    --peft-lora-bias ${PEFT_LORA_BIAS} \
    --train-per-device-batch-size ${TRAIN_PER_DEVICE_BATCH_SIZE} \
    --train-gradient-accumulation-steps ${TRAIN_GRADIENT_ACCUMULATION_STEPS} \
    --train-num-epochs ${TRAIN_NUM_EPOCHS} \
    --train-learning-rate ${TRAIN_LEARNING_RATE} \
    --train-weight-decay ${TRAIN_WEIGHT_DECAY} \
    --train-warmup-steps ${TRAIN_WARMUP_STEPS} \
    --train-lr-scheduler-type ${TRAIN_LR_SCHEDULER_TYPE} \
    --train-seed ${TRAIN_SEED} \
    --model-system-prompt ${SYSTEM_PROMPT} \
    --unique-entities ${UNIQUE_ENTITIES}

echo "================================================"
echo "Testing the model..."
echo "================================================"
echo ""

$RUN_PYTHON src/training/evaluate_unsloth.py \
    --eval-dataset-file ${EVAL_DATASET_FILE_PATH} \
    --results-dir ${TEST_OUTPUT_DIR} \
    --model-dir ${TRAIN_OUTPUT_DIR} \
    --model-max-seq-length ${MODEL_MAX_SEQ_LENGTH} \
    --model-load-in-4bit ${MODEL_LOAD_IN_4BIT} \
    --model-load-in-8bit ${MODEL_LOAD_IN_8BIT} \
    --eval-batch-size ${TRAIN_PER_DEVICE_BATCH_SIZE} \
    --model-system-prompt ${SYSTEM_PROMPT} \
    --eval-unique-entities ${UNIQUE_ENTITIES}

echo "Removing the model directory..."
rm -rf ${TRAIN_OUTPUT_DIR}

# Print end time
echo ""
echo "End time: $(date)"
