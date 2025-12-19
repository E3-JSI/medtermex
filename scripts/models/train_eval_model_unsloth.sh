#! /bin/bash
# Train and evaluate the model using Unsloth

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
        echo "Warning: No virtual environment found. Please run 'make setup' first."
        exit 1
    fi
    RUN_PYTHON="python"
fi

# ===============================================
# Load the dataset directory parameters
# ===============================================

# Storage directories
BASE_STORAGE_DIR=.
BASE_PROJECT_DIR=.

# Subdirectories
DATASET_DIR=[dataset_dir]
MODELS_DIR=[models_dir]
RESULTS_DIR=[results_dir]

# Dataset file names
TRAIN_DATASET_FILE=[train_dataset_file]
TEST_DATASET_FILE=[test_dataset_file]

# ===============================================
# Load the experiment parameters
# ===============================================
# Model parameters

# unsloth/gemma-3-4b-it
# unsloth/gemma-3-270m-it
# unsloth/gemma-3n-E2B-it
# unsloth/gemma-3n-E4B-it
# unsloth/medgemma-4b-it
# unsloth/llama-3-8b-Instruct
# unsloth/Llama-3.1-8B-Instruct
# unsloth/Llama-3.2-1B-Instruct
# unsloth/Llama-3.2-3B-Instruct
MODEL_NAME=unsloth/Llama-3.1-8B-Instruct
MODEL_MAX_SEQ_LENGTH=4096
MODEL_LOAD_IN_4BIT=true
MODEL_LOAD_IN_8BIT=false
MODEL_FULL_FINETUNING=false

# PEFT parameters
PEFT_FT_VISION_LAYERS=false
PEFT_FT_LANGUAGE_LAYERS=true
PEFT_FT_ATTENTION_MODULES=true
PEFT_FT_MLP_MODULES=true
PEFT_RANK=8
PEFT_LORA_ALPHA=16
PEFT_LORA_DROPOUT=0.0
PEFT_LORA_BIAS=false

# Training parameters
TRAIN_PER_DEVICE_BATCH_SIZE=4
TRAIN_GRADIENT_ACCUMULATION_STEPS=4
TRAIN_NUM_EPOCHS=10
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

# Model store name
MODEL_STORE_NAME="$MODEL_NAME-$(date +%Y%m%d-%H%M%S)"

# Load the train output directory
TRAIN_OUTPUT_DIR=${BASE_STORAGE_DIR}/${MODELS_DIR}/${MODEL_STORE_NAME}
if [[ -d $TRAIN_OUTPUT_DIR ]]; then
    echo "Removing the existing train output directory..."
    rm -rf ${TRAIN_OUTPUT_DIR}
fi

# Load the test output directory
TEST_OUTPUT_DIR=${BASE_PROJECT_DIR}/${RESULTS_DIR}/${MODEL_STORE_NAME}

echo "================================================"
echo "Training the model..."
echo "================================================"

$RUN_PYTHON -m src.training.train_unsloth \
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

$RUN_PYTHON -m src.training.evaluate_unsloth \
    --eval-dataset-file ${EVAL_DATASET_FILE_PATH} \
    --results-dir ${TEST_OUTPUT_DIR} \
    --model-dir ${TRAIN_OUTPUT_DIR} \
    --model-max-seq-length ${MODEL_MAX_SEQ_LENGTH} \
    --model-load-in-4bit ${MODEL_LOAD_IN_4BIT} \
    --model-load-in-8bit ${MODEL_LOAD_IN_8BIT} \
    --model-system-prompt ${SYSTEM_PROMPT} \
    --eval-batch-size ${TRAIN_PER_DEVICE_BATCH_SIZE} \
    --eval-unique-entities ${UNIQUE_ENTITIES}
