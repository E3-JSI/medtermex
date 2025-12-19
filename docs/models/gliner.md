# GLiNER Model Documentation

## Overview

GLiNER (Generalist and Lightweight model for Named Entity Recognition) is a specialized NER model designed for efficient entity extraction. Unlike traditional NER models that require predefined entity types, GLiNER can extract entities for any given labels at inference time, making it highly flexible for medical term extraction tasks.

This implementation supports fine-tuning GLiNER models on custom medical datasets and evaluating their performance.

## Supported Models

GLiNER models from Hugging Face can be used, including:
- `urchade/gliner_small-v2.1`
- `urchade/gliner_medium-v2.1`
- `urchade/gliner_large-v2.1`
- `urchade/gliner_multi-v2.1`
- Or any custom fine-tuned GLiNER model

## Fine-tuning

### How to Fine-tune

Fine-tuning is performed using the `src.training.train_gliner` module:

```bash
uv run python -m src.training.train_gliner \
    --train-dataset-file <path/to/train_data.json> \
    --model-name-or-path <model_name> \
    --model-output-dir <output_directory> \
    --num-train-epochs 3 \
    --train-batch-size 8 \
    --train-learning-rate 5e-6 \
    --train-weight-decay 0.01
```

### Fine-tuning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--train-dataset-file` | str | *required* | Path to the training dataset file (JSON format) |
| `--model-name-or-path` | str | *required* | Pre-trained model name from Hugging Face or local path |
| `--model-output-dir` | str | *required* | Directory where the fine-tuned model will be saved |
| `--training-output-dir` | str | `models/tmp` | Temporary directory for training artifacts |
| `--train-validation-ratio` | float | `0.8` | Ratio of training vs validation split (0.8 = 80% train, 20% val) |
| `--train-num-epochs` | int | `3` | Number of training epochs |
| `--train-batch-size` | int | `8` | Batch size for training and evaluation |
| `--train-learning-rate` | float | `5e-6` | Learning rate for main model parameters |
| `--train-weight-decay` | float | `0.01` | Weight decay for regularization |
| `--train-other-lr` | float | `1e-5` | Learning rate for other parameters (embeddings, etc.) |
| `--train-other-weight-decay` | float | `0.01` | Weight decay for other parameters |
| `--use-cpu` | bool | `false` | Force CPU usage even if GPU is available |

### Training Data Format

GLiNER expects training data in a specific tokenized format:

```json
[
  {
    "tokenized_text": ["The", "patient", "has", "diabetes", "mellitus", "."],
    "ner": [
      [3, 5, "Disease"]
    ]
  },
  {
    "tokenized_text": ["Prescribed", "metformin", "500mg", "."],
    "ner": [
      [1, 2, "Medication"],
      [2, 3, "Dosage"]
    ]
  }
]
```

**Format Details:**
- `tokenized_text`: Array of tokens (words) from the text
- `ner`: Array of entity annotations, where each annotation is:
  - `[start_index, end_index, label]`
  - `start_index`: Token index where the entity starts (inclusive)
  - `end_index`: Token index where the entity ends (exclusive)
  - `label`: Entity type/label as a string

**Important Notes:**
- Indices are token-based, not character-based
- End index is exclusive (Python slice notation)
- Multiple entities can overlap or be adjacent

## Evaluation

### How to Evaluate

Evaluation is performed using the `src.training.evaluate_gliner` module:

```bash
uv run python -m src.training.evaluate_gliner \
    --eval-dataset-file <path/to/test_data.json> \
    --results-dir <output_directory> \
    --model-dir <path/to/fine-tuned/model> \
    --eval-threshold 0.5 \
    --eval-metrics "exact,relaxed,overlap"
```

### Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--eval-dataset-file` | str | *required* | Path to the evaluation dataset file (JSON format) |
| `--results-dir` | str | *required* | Directory where evaluation results will be saved |
| `--model-dir` | str | *required* | Path to the fine-tuned model directory |
| `--eval-threshold` | float | `0.5` | Confidence threshold for predictions (0.0 to 1.0) |
| `--eval-metrics` | str | `"exact,relaxed,overlap"` | Comma-separated list of evaluation metrics |
| `--use-cpu` | bool | `false` | Force CPU usage even if GPU is available |

### Evaluation Data Format

The evaluation dataset uses a different format than training:

```json
[
  {
    "text": "The patient has diabetes mellitus.",
    "entities": [
      {
        "label": "Disease",
        "text": "diabetes mellitus",
        "start": 16,
        "end": 33
      }
    ]
  },
  {
    "text": "Prescribed metformin 500mg.",
    "entities": [
      {
        "label": "Medication",
        "text": "metformin"
      },
      {
        "label": "Dosage",
        "text": "500mg"
      }
    ]
  }
]
```

**Format Details:**
- `text`: Raw text string to extract entities from
- `entities`: Array of ground truth entities, where each entity contains:
  - `label`: Entity type/label
  - `text`: The actual entity text
  - `start` (optional): Character position where entity starts
  - `end` (optional): Character position where entity ends

### Evaluation Metrics

The evaluation supports three matching strategies:

- **exact**: Entity boundaries and labels must match exactly
- **relaxed**: Labels must match, and there must be any overlap between predicted and true entities
- **overlap**: Partial overlap is acceptable

### Output Files

Evaluation produces two JSON files in the results directory:

1. **`true_pred_entities.json`**: Contains all true and predicted entities for inspection
2. **`performance.json`**: Contains performance metrics including:
   - Number of examples evaluated
   - Average inference time per example
   - Precision, Recall, and F1 scores for each metric type
   - Per-label performance breakdown

## Example Usage

### Using the Bash Script

A complete training and evaluation script is available at `scripts/models/train_eval_model_gliner.sh`:

```bash
#!/bin/bash
# Train and evaluate the model using GLiNER

set -e # exit on error

# Dataset directories and files
BASE_STORAGE_DIR=.
BASE_PROJECT_DIR=.
DATASET_DIR=data/final
MODELS_DIR=models
RESULTS_DIR=results

TRAIN_DATASET_FILE=train_gliner.json
TEST_DATASET_FILE=test_gliner.json

# Model parameters
MODEL=urchade/gliner_medium-v2.1

# Training parameters
TRAIN_NUM_EPOCHS=3
TRAIN_BATCH_SIZE=8
TRAIN_LEARNING_RATE=5e-6
TRAIN_WEIGHT_DECAY=0.01

# Evaluation parameters
EVAL_THRESHOLD=0.5
EVAL_METRICS="exact,relaxed,overlap"
EVAL_USE_CPU=false

# Prepare paths
TRAIN_DATASET_FILE_PATH=${BASE_STORAGE_DIR}/${DATASET_DIR}/${TRAIN_DATASET_FILE}
EVAL_DATASET_FILE_PATH=${BASE_STORAGE_DIR}/${DATASET_DIR}/${TEST_DATASET_FILE}
MODEL_STORE_NAME="${MODEL}-$(date +%Y%m%d-%H%M%S)"
TRAIN_OUTPUT_DIR=${BASE_STORAGE_DIR}/${MODELS_DIR}/${MODEL_STORE_NAME}
TEST_OUTPUT_DIR=${BASE_PROJECT_DIR}/${RESULTS_DIR}/${MODEL_STORE_NAME}

# Train the model
echo "Training the model..."
uv run python -m src.training.train_gliner \
    --train-dataset-file ${TRAIN_DATASET_FILE_PATH} \
    --model-name-or-path ${MODEL} \
    --model-output-dir ${TRAIN_OUTPUT_DIR} \
    --num-train-epochs ${TRAIN_NUM_EPOCHS} \
    --train-batch-size ${TRAIN_BATCH_SIZE} \
    --train-learning-rate ${TRAIN_LEARNING_RATE} \
    --train-weight-decay ${TRAIN_WEIGHT_DECAY} \
    --use-cpu ${EVAL_USE_CPU}

# Evaluate the model
echo "Testing the model..."
uv run python -m src.training.evaluate_gliner \
    --eval-dataset-file ${EVAL_DATASET_FILE_PATH} \
    --results-dir ${TEST_OUTPUT_DIR} \
    --model-dir ${TRAIN_OUTPUT_DIR} \
    --eval-threshold ${EVAL_THRESHOLD} \
    --eval-metrics ${EVAL_METRICS} \
    --use-cpu ${EVAL_USE_CPU}
```

## Hardware Requirements

- **GPU**: Recommended for training (CUDA-compatible GPU with at least 8GB VRAM)
- **CPU**: Can be used with `--use-cpu` flag, but will be significantly slower
- **RAM**: At least 16GB recommended
- **Storage**: Depends on model size (typically 500MB - 2GB per model)

## Best Practices

1. **Data Preparation**: Ensure your training data is properly tokenized and entity indices are correct
2. **Validation Split**: Use the `--train-validation-ratio` to monitor overfitting
3. **Threshold Tuning**: Experiment with `--eval-threshold` values (typically between 0.3-0.7)
4. **Batch Size**: Adjust based on GPU memory; smaller batches for larger models
5. **Learning Rate**: Start with default `5e-6` and adjust if needed
6. **Epochs**: Monitor validation performance to avoid overfitting (typically 3-5 epochs)

## Troubleshooting

### Out of Memory Errors
- Reduce `--train-batch-size`
- Use `--use-cpu` flag (slower but uses RAM instead of VRAM)
- Try a smaller model variant

### Poor Performance
- Increase number of training epochs
- Adjust learning rate (try `2e-6` or `1e-5`)
- Ensure data quality and correct entity annotations
- Lower the evaluation threshold

### Slow Training
- Enable GPU usage (remove `--use-cpu`)
- Increase batch size if GPU memory allows
- Use a smaller model for faster iteration
