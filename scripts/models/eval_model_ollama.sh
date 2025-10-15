#! /bin/bash
# Train and evaluate the model using Unsloth

set -e # exit on error

# ===============================================
# Load the dataset directory parameters
# ===============================================

# Storage directories
BASE_STORAGE_DIR=.
BASE_PROJECT_DIR=.

# Subdirectories
DATASET_DIR=[dataset_dir]
RESULTS_DIR=[results_dir]

# Dataset file names
TEST_DATASET_FILE=[test_dataset_file]

# ===============================================
# Load the experiment parameters
# ===============================================
# Model parameters

# gemma3:4b
# gemma3:12b
# gemma3:27b
# alibayram/medgemma:4b
# alibayram/medgemma:27b
# llama3:8b
# llama3.1:8b
# llama3.2:1b
# llama3.2:3b
MODEL_NAME=gemma3:27b
MODEL_MAX_SEQ_LENGTH=4096
LANGUAGE=[language]

# ===============================================
# Prepare the training and output directories
# ===============================================

# Load the training dataset file
EVAL_DATASET_FILE_PATH=${BASE_STORAGE_DIR}/${DATASET_DIR}/${TEST_DATASET_FILE}

# Model store name
MODEL_STORE_NAME="ollama/${MODEL_NAME}"

# Load the test output directory
TEST_OUTPUT_DIR=${BASE_PROJECT_DIR}/${RESULTS_DIR}/${MODEL_STORE_NAME}


echo "# ==============================================="
echo "# Parameters"
echo "# ==============================================="
echo ""
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_MAX_SEQ_LENGTH=${MODEL_MAX_SEQ_LENGTH}"
echo ""
echo "EVAL_DATASET_FILE_PATH=${EVAL_DATASET_FILE_PATH}"
echo "TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo ""

echo "Start time: $(date)"

echo "================================================"
echo "Testing the model..."
echo "================================================"

uv run python -m src.training.evaluate_ollama \
    --eval-dataset-file ${EVAL_DATASET_FILE_PATH} \
    --results-dir ${TEST_OUTPUT_DIR} \
    --model-name ${MODEL_NAME} \
    --model-max-seq-length ${MODEL_MAX_SEQ_LENGTH}

echo "End time: $(date)"
