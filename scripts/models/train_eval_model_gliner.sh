#!/bin/bash
# Train and evaluate the model using GLiNER

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

# GLiNER model options:
# urchade/gliner_small-v2.1
# urchade/gliner_medium-v2.1
# urchade/gliner_large-v2.1
# urchade/gliner_multi-v2.1
# urchade/gliner_multi_pii-v1
# knowledgator/gliner-multitask-large-v0.5
MODEL=urchade/gliner_multi-v2.1

# Training parameters
TRAIN_NUM_EPOCHS=3
TRAIN_BATCH_SIZE=8
TRAIN_LEARNING_RATE=5e-6
TRAIN_WEIGHT_DECAY=0.01

EVAL_THRESHOLD=0.5
EVAL_METRICS="exact,relaxed,overlap"
EVAL_USE_CPU=false

# ===============================================
# Prepare the training and output directories
# ===============================================

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

$RUN_PYTHON -m src.training.train_gliner \
    --train-dataset-file ${TRAIN_DATASET_FILE_PATH} \
    --model-name-or-path ${MODEL} \
    --model-output-dir ${TRAIN_OUTPUT_DIR} \
    --train-num-epochs ${TRAIN_NUM_EPOCHS} \
    --train-batch-size ${TRAIN_BATCH_SIZE} \
    --train-learning-rate ${TRAIN_LEARNING_RATE} \
    --train-weight-decay ${TRAIN_WEIGHT_DECAY} \
    --use-cpu ${EVAL_USE_CPU}


echo "================================================"
echo "Testing the model..."
echo "================================================"

$RUN_PYTHON -m src.training.evaluate_gliner \
    --eval-dataset-file ${EVAL_DATASET_FILE_PATH} \
    --results-dir ${TEST_OUTPUT_DIR} \
    --model-dir ${TRAIN_OUTPUT_DIR} \
    --eval-threshold ${EVAL_THRESHOLD} \
    --eval-metrics ${EVAL_METRICS} \
    --use-cpu ${EVAL_USE_CPU}
