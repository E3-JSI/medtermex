# Unsloth Model Documentation

## Overview

Unsloth is a fast and memory-efficient framework for fine-tuning Large Language Models (LLMs) using LoRA (Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning) techniques. It enables fine-tuning of large models like Llama, Gemma, and their medical variants with significantly reduced memory requirements.

This implementation uses Unsloth to fine-tune LLMs for medical entity extraction tasks, leveraging instruction-following capabilities and chat templates.

## Supported Models

### Llama Family
- `unsloth/llama-3-8b-Instruct`
- `unsloth/Llama-3.1-8B-Instruct`
- `unsloth/Llama-3.2-1B-Instruct`
- `unsloth/Llama-3.2-3B-Instruct`

### Gemma Family
- `unsloth/gemma-3-4b-it`
- `unsloth/gemma-3-270m-it`
- `unsloth/gemma-3n-E2B-it`
- `unsloth/gemma-3n-E4B-it`

### Medical Models
- `unsloth/medgemma-4b-it` (Medical-specialized Gemma variant)

### Other Models
- Any Unsloth-compatible model from Hugging Face

## Fine-tuning

### How to Fine-tune

Fine-tuning is performed using the `src.training.train_unsloth` module with LoRA/PEFT:

```bash
# Using uv (if installed)
uv run python -m src.training.train_unsloth \
    --train-dataset-file <path/to/train_data.json> \
    --output-dir <output_directory> \
    --model-name <model_name> \
    --model-max-seq-length 4096 \
    --model-load-in-4bit true \
    --peft-rank 8 \
    --peft-lora-alpha 16 \
    --train-num-epochs 10 \
    --train-learning-rate 2e-4

# Or using standard python (activate venv first)
source .venv-unsloth/bin/activate  # or .venv/bin/activate
python -m src.training.train_unsloth \
    --train-dataset-file <path/to/train_data.json> \
    --output-dir <output_directory> \
    --model-name <model_name> \
    --model-max-seq-length 4096 \
    --model-load-in-4bit true \
    --peft-rank 8 \
    --peft-lora-alpha 16 \
    --train-num-epochs 10 \
    --train-learning-rate 2e-4
```

> **Note:** When running Python directly (not through bash scripts), you must activate the appropriate virtual environment first if not using uv.

### Fine-tuning Parameters

#### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--train-dataset-file` | str | *required* | Path to the training dataset file (JSON format) |
| `--output-dir` | str | *required* | Directory where the fine-tuned model will be saved |

#### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model-name-or-path` | str | *required* | Pre-trained model name (e.g., `unsloth/Llama-3.1-8B-Instruct`) |
| `--model-max-seq-length` | int | `4096` | Maximum sequence length for model input/output |
| `--model-load-in-4bit` | bool | `true` | Load model in 4-bit quantization (reduces memory usage) |
| `--model-load-in-8bit` | bool | `false` | Load model in 8-bit quantization (alternative to 4-bit) |
| `--model-full-finetuning` | bool | `false` | Use full fine-tuning instead of LoRA (requires more memory) |
| `--model-hf-token` | str | `None` | Hugging Face token for accessing gated models |
| `--model-system-prompt` | str | See below | System prompt for instruction formatting |
| `--unique-entities` | bool | `true` | Whether to deduplicate extracted entities |

**Default System Prompt:**
```
You are a medical entity extractor from clinical texts. Extract the entities from the text and return them in a structured JSON format.
```

#### PEFT (LoRA) Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--peft-ft-vision-layers` | bool | `false` | Fine-tune vision layers (for multimodal models) |
| `--peft-ft-language-layers` | bool | `true` | Fine-tune language layers |
| `--peft-ft-attention-modules` | bool | `true` | Fine-tune attention modules (Q, K, V, O projections) |
| `--peft-ft-mlp-modules` | bool | `true` | Fine-tune MLP/feed-forward modules |
| `--peft-rank` | int | `8` | LoRA rank (higher = more parameters, better fit but slower) |
| `--peft-lora-alpha` | int | `16` | LoRA alpha scaling parameter (typically 2x rank) |
| `--peft-lora-dropout` | float | `0.1` | Dropout rate for LoRA layers |
| `--peft-lora-bias` | bool | `false` | Whether to add trainable bias to LoRA layers |

#### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--train-per-device-batch-size` | int | `16` | Batch size per GPU/device |
| `--train-gradient-accumulation-steps` | int | `4` | Number of steps to accumulate gradients |
| `--train-num-epochs` | int | `3` | Number of training epochs |
| `--train-learning-rate` | float | `2e-4` | Learning rate for optimization |
| `--train-weight-decay` | float | `0.01` | Weight decay for regularization |
| `--train-warmup-steps` | int | `5` | Number of warmup steps for learning rate scheduler |
| `--train-lr-scheduler-type` | str | `linear` | Learning rate scheduler type |
| `--train-seed` | int | `42` | Random seed for reproducibility |

**Effective Batch Size** = `train-per-device-batch-size` × `train-gradient-accumulation-steps` × num_gpus

### Training Data Format

Unsloth expects data in a simple JSON format with text and entities:

```json
[
  {
    "text": "The patient has diabetes mellitus.",
    "entities": [
      {
        "label": "Disease",
        "text": "diabetes mellitus"
      }
    ]
  },
  {
    "text": "Prescribed metformin 500mg twice daily.",
    "entities": [
      {
        "label": "Medication",
        "text": "metformin"
      },
      {
        "label": "Dosage",
        "text": "500mg"
      },
      {
        "label": "Frequency",
        "text": "twice daily"
      }
    ]
  }
]
```

**Format Details:**
- `text`: Raw text string containing medical information
- `entities`: Array of entities to extract, where each entity contains:
  - `label`: Entity type/category
  - `text`: The actual entity text as it appears in the input

**Note:** The system automatically formats this data into a chat template during training.

### Chat Templates

The training process automatically applies the appropriate chat template based on the model type:

- **Llama 3.x**: Uses Llama-specific chat format with `<|start_header_id|>` tags
- **Gemma 3.x**: Uses Gemma format with `<start_of_turn>` tags
- **Other models**: Appropriate templates are auto-detected

The formatter converts your data into instruction-response pairs:
- **Instruction**: "Extract medical entities from: [text]"
- **Response**: JSON-formatted entities

## Evaluation

### How to Evaluate

Evaluation is performed using the `src.training.evaluate_unsloth` module:

```bash
# Using uv (if installed)
uv run python -m src.training.evaluate_unsloth \
    --eval-dataset-file <path/to/test_data.json> \
    --results-dir <output_directory> \
    --model-dir <path/to/fine-tuned/model> \
    --model-max-seq-length 4096 \
    --eval-batch-size 4

# Or using standard python (activate venv first)
source .venv-unsloth/bin/activate  # or .venv/bin/activate
python -m src.training.evaluate_unsloth \
    --eval-dataset-file <path/to/test_data.json> \
    --results-dir <output_directory> \
    --model-dir <path/to/fine-tuned/model> \
    --model-max-seq-length 4096 \
    --eval-batch-size 4
```

> **Note:** When running Python directly (not through bash scripts), you must activate the appropriate virtual environment first if not using uv.

### Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--eval-dataset-file` | str | *required* | Path to the evaluation dataset file (JSON format) |
| `--results-dir` | str | *required* | Directory where evaluation results will be saved |
| `--model-dir` | str | *required* | Path to the fine-tuned model directory |
| `--model-max-seq-length` | int | `4096` | Maximum sequence length for generation |
| `--model-load-in-4bit` | bool | `true` | Load model in 4-bit quantization |
| `--model-load-in-8bit` | bool | `false` | Load model in 8-bit quantization |
| `--model-system-prompt` | str | Same as training | System prompt for evaluation |
| `--eval-batch-size` | int | `4` | Batch size for evaluation |
| `--eval-unique-entities` | bool | `true` | Whether to deduplicate predicted entities |
| `--eval-metrics` | str | `"exact,relaxed,overlap"` | Comma-separated list of evaluation metrics |

### Evaluation Data Format

Same format as training data:

```json
[
  {
    "text": "Patient presents with hypertension and type 2 diabetes.",
    "entities": [
      {"label": "Disease", "text": "hypertension"},
      {"label": "Disease", "text": "type 2 diabetes"}
    ]
  }
]
```

### Generation Settings

The evaluation automatically applies model-specific generation settings:

**Llama models:**
```python
temperature = 1.5
min_p = 0.1
```

**Gemma models:**
```python
temperature = 1.0
top_p = 0.95
top_k = 64
```

**Mistral models:**
```python
temperature = 0.8
top_p = 0.95
```

### Evaluation Metrics

Three matching strategies are supported:

- **exact**: Entity text and labels must match exactly
- **relaxed**: Labels must match with any text overlap
- **overlap**: Partial text overlap is acceptable

### Output Files

1. **`true_pred_entities.json`**: All predicted vs ground truth entities
2. **`performance.json`**: Detailed metrics including:
   - Precision, Recall, F1 for each metric type
   - Per-label performance breakdown
   - Average inference time

## Example Usage

### Using the Bash Script

A complete training and evaluation script is available at `scripts/models/train_eval_model_unsloth.sh`:

```bash
#!/bin/bash
# Train and evaluate the model using Unsloth

set -e # exit on error

# Dataset configuration
BASE_STORAGE_DIR=.
BASE_PROJECT_DIR=.
DATASET_DIR=data/final
MODELS_DIR=models
RESULTS_DIR=results

TRAIN_DATASET_FILE=train.json
TEST_DATASET_FILE=test.json

# Model parameters
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

# Prepare paths
TRAIN_DATASET_FILE_PATH=${BASE_STORAGE_DIR}/${DATASET_DIR}/${TRAIN_DATASET_FILE}
EVAL_DATASET_FILE_PATH=${BASE_STORAGE_DIR}/${DATASET_DIR}/${TEST_DATASET_FILE}
MODEL_STORE_NAME="${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
TRAIN_OUTPUT_DIR=${BASE_STORAGE_DIR}/${MODELS_DIR}/${MODEL_STORE_NAME}
TEST_OUTPUT_DIR=${BASE_PROJECT_DIR}/${RESULTS_DIR}/${MODEL_STORE_NAME}

# Automatically detect uv or python
if command -v uv &> /dev/null; then
    RUN_PYTHON="uv run python"
else
    RUN_PYTHON="python"
fi

# Train the model
echo "Training the model..."
$RUN_PYTHON -m src.training.train_unsloth \
    --train-dataset-file ${TRAIN_DATASET_FILE_PATH} \
    --output-dir ${TRAIN_OUTPUT_DIR} \
    --model-name ${MODEL_NAME} \
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
    --model-system-prompt "${SYSTEM_PROMPT}" \
    --unique-entities ${UNIQUE_ENTITIES}

# Evaluate the model
echo "Testing the model..."
$RUN_PYTHON -m src.training.evaluate_unsloth \
    --eval-dataset-file ${EVAL_DATASET_FILE_PATH} \
    --results-dir ${TEST_OUTPUT_DIR} \
    --model-dir ${TRAIN_OUTPUT_DIR} \
    --model-max-seq-length ${MODEL_MAX_SEQ_LENGTH} \
    --model-load-in-4bit ${MODEL_LOAD_IN_4BIT} \
    --model-load-in-8bit ${MODEL_LOAD_IN_8BIT} \
    --eval-batch-size ${TRAIN_PER_DEVICE_BATCH_SIZE} \
    --model-system-prompt "${SYSTEM_PROMPT}" \
    --unique-entities ${UNIQUE_ENTITIES}
```

> **Note:** The script automatically detects whether `uv` is available and falls back to standard `python` if not. When using standard python, the script automatically activates the appropriate virtual environment (`.venv-unsloth` or `.venv`).

## Hardware Requirements

### Minimum Requirements
- **GPU**: CUDA-compatible GPU with at least 12GB VRAM (for 4-bit quantization)
- **RAM**: 16GB+ system RAM
- **Storage**: 10-20GB per model (base model + adapters)

### Recommended Configuration
- **GPU**: NVIDIA A100, A6000, or RTX 4090 (24GB+ VRAM)
- **RAM**: 32GB+ system RAM
- **Storage**: SSD for faster data loading

### Memory Optimization
- Use `--model-load-in-4bit true` for lowest memory usage (recommended)
- Use `--model-load-in-8bit true` for better quality with moderate memory
- Reduce `--train-per-device-batch-size` if running out of memory
- Increase `--train-gradient-accumulation-steps` to compensate for smaller batches

## Running on SLURM

For running Unsloth training and evaluation on HPC clusters with SLURM, a ready-to-use script is available at `slurm/train_eval_model_unsloth.sh`.

### Quick Start with SLURM

```bash
# 1. Edit the SLURM script to set your parameters
# Update placeholders: [account_name], [base_storage_dir], [base_project_dir],
# [dataset_dir], [models_dir], [results_dir], [train_dataset_file], [test_dataset_file]

# 2. Submit the job
sbatch slurm/train_eval_model_unsloth.sh

# 3. Monitor the job
squeue -u $USER

# 4. Check logs
tail -f logs/train-model-unsloth-<job_id>.out
```

### Important Notes

- **Placeholders**: The SLURM script contains placeholders (in square brackets) that must be replaced with your actual paths and parameters before submission
- **GPU Requirements**: Unsloth requires at least 12GB VRAM (for 4-bit quantization). The script requests 1 GPU by default (`--gres=gpu:1`)
- **Time Limit**: Default is 8 hours (`--time=08:00:00`). LLM training typically takes longer than GLiNER
- **Memory Optimization**: The script uses 4-bit quantization by default to reduce memory usage
- **Hugging Face Cache**: Cache directories are configured to prevent filling your home directory

For more information about SLURM commands and job management, see the [SLURM documentation](../../SLURM.md).

## Best Practices

### LoRA Configuration
1. **Start with default rank (8)**: Good balance of quality and speed
2. **Higher ranks (16-32)**: Better performance but slower and more memory
3. **Lower ranks (4)**: Faster training but may underfit
4. **Alpha = 2 × Rank**: Standard rule of thumb

### Training Tips
1. **Learning Rate**: Start with `2e-4` for LoRA (higher than full fine-tuning)
2. **Epochs**: 3-10 epochs typically sufficient; monitor validation loss
3. **Batch Size**: Adjust based on GPU memory; aim for effective batch size of 16-64
4. **Gradient Accumulation**: Use to simulate larger batches without memory overhead

### Data Preparation
1. **Quality over quantity**: Clean, accurate annotations are crucial
2. **Balanced labels**: Ensure all entity types are well-represented
3. **Text diversity**: Include varied sentence structures and terminology
4. **Train/test split**: Use 80/20 or 90/10 split for evaluation

### System Prompt Design
- Be specific about the task
- Specify output format (JSON with entities array)
- Include any special instructions (e.g., language, formatting)

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
--train-per-device-batch-size 2

# Increase gradient accumulation to maintain effective batch size
--train-gradient-accumulation-steps 8

# Use 4-bit quantization
--model-load-in-4bit true
```

### Poor Entity Extraction
- Increase training epochs
- Adjust system prompt to be more specific
- Use a larger model (e.g., 8B instead of 1B)
- Increase LoRA rank for more capacity
- Check data quality and label consistency

### Slow Training
- Use 4-bit quantization
- Reduce `--model-max-seq-length`
- Use a smaller model for initial experiments
- Enable Flash Attention (automatic with Unsloth)

### Model Not Following Format
- Ensure system prompt is consistent between training and evaluation
- Check that training data has proper entity formatting
- Try adjusting generation temperature
- Increase training epochs

## Advanced Features

### Multi-GPU Training
Unsloth automatically detects and uses multiple GPUs with `device_map="balanced"`

### Custom Chat Templates
The system auto-detects chat templates, but you can customize by modifying `src/training/train_unsloth.py`

### Exporting Models
Fine-tuned models can be:
- Used directly with the evaluation script
- Merged with base model for deployment
- Exported to GGUF format for CPU inference
- Pushed to Hugging Face Hub
