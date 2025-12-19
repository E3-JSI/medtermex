# Ollama Model Documentation

## Overview

Ollama is a framework for running large language models locally. Unlike GLiNER and Unsloth, Ollama is designed for using pre-trained models directly without fine-tuning. This implementation evaluates Ollama models on medical entity extraction tasks using structured output formatting and JSON schema validation.

**Key Features:**
- No fine-tuning required (uses pre-trained models)
- Local inference with privacy preservation
- Structured output with JSON schema enforcement
- Support for various open-source LLMs

**Note:** This implementation only supports evaluation, not training. Use pre-trained models that already have medical knowledge or have been fine-tuned separately.

### When to Use Ollama

Ollama is ideal for:
- **Quick testing** of pre-trained models without training overhead
- **Local development** environments with limited compute resources
- **Privacy-sensitive** applications requiring on-premise inference
- **Rapid prototyping** to test different models before committing to fine-tuning

### SLURM/HPC Considerations

**Ollama is designed for local inference and does not require SLURM or HPC resources** since it:
- Performs evaluation only (no training)
- Runs efficiently on local machines with CPU or moderate GPU
- Is optimized for single-node inference

**For HPC environments requiring model fine-tuning**, consider using [Unsloth](./unsloth.md) instead, which provides:
- Full training capabilities with LoRA/PEFT
- Efficient multi-GPU training on SLURM clusters
- Better performance for custom medical entity extraction tasks

If you need to evaluate Ollama models on HPC clusters, you can run the evaluation scripts directly on compute nodes without special SLURM job scripts.

## Supported Models

Ollama supports a wide range of models available through the Ollama registry:

### Gemma Family
- `gemma3:4b`
- `gemma3:12b`
- `gemma3:27b`

### Llama Family
- `llama3:8b`
- `llama3.1:8b`
- `llama3.2:1b`
- `llama3.2:3b`

### Medical Models
- `alibayram/medgemma:4b`
- `alibayram/medgemma:27b`

### Other Models
- Any model available in the Ollama registry
- Custom models imported into Ollama

**Installation Note:** Models must be pre-installed via Ollama CLI:
```bash
ollama pull gemma3:27b
ollama pull llama3.1:8b
ollama pull alibayram/medgemma:4b
```

## Fine-tuning

**Ollama does not support fine-tuning through this implementation.**

To use custom fine-tuned models with Ollama:
1. Fine-tune your model using other frameworks (HuggingFace, Unsloth, etc.)
2. Convert to GGUF format if necessary
3. Import into Ollama using `ollama create`
4. Evaluate using the evaluation script

For fine-tuning LLMs for medical entity extraction, see the [Unsloth documentation](unsloth.md).

## Evaluation

### How to Evaluate

Evaluation is performed using the `src.training.evaluate_ollama` module:

```bash
# Using uv (if installed)
uv run python -m src.training.evaluate_ollama \
    --eval-dataset-file <path/to/test_data.json> \
    --results-dir <output_directory> \
    --model-name gemma3:27b \
    --model-max-seq-length 4096

# Or using standard python (activate venv first)
source .venv/bin/activate
python -m src.training.evaluate_ollama \
    --eval-dataset-file <path/to/test_data.json> \
    --results-dir <output_directory> \
    --model-name gemma3:27b \
    --model-max-seq-length 4096
```

> **Note:** When running Python directly (not through bash scripts), you must activate the appropriate virtual environment first if not using uv.

### Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--eval-dataset-file` | str | *required* | Path to the evaluation dataset file (JSON format) |
| `--results-dir` | str | *required* | Directory where evaluation results will be saved |
| `--model-name` | str | *required* | Ollama model name (e.g., `gemma3:27b`, `llama3.1:8b`) |
| `--model-system-prompt` | str | See below | System prompt for entity extraction |
| `--model-max-seq-length` | int | `4096` | Maximum sequence length for model context |
| `--unique-entities` | bool | `true` | Whether to deduplicate extracted entities |
| `--eval-metrics` | str | `"exact,relaxed,overlap"` | Comma-separated list of evaluation metrics |

### Default System Prompt

The default system prompt is optimized for medical entity extraction:

```
You are a medical entity extractor from clinical texts. Extract the entities from the text and return them in a structured JSON format.

**Instructions:**
- Extract the text as written in the clinical text language
- When there are multiple entities, return all of them
- Entity text should be a single word or phrase
- Output should be a valid JSON with entities array

**Output format:**
```json
{
    "entities": [
        {"label": "label1", "text": "text1"},
        {"label": "label2", "text": "text2"},
        {"label": "label3", "text": "text3"}
    ]
}
```

You can customize this with the `--model-system-prompt` parameter.

### Evaluation Data Format

The evaluation dataset uses a simple JSON format:

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
- `text`: Raw text string to extract entities from
- `entities`: Ground truth entities, where each entity contains:
  - `label`: Entity type/category
  - `text`: The actual entity text

**Note:** The system automatically creates a JSON schema based on unique labels in your dataset.

### JSON Schema Validation

The implementation uses Pydantic to enforce structured output:
- Automatically generates schema from dataset labels
- Validates model responses against schema
- Retries up to 3 times on invalid responses
- Returns empty array if all attempts fail

### Evaluation Metrics

Three matching strategies are supported:

- **exact**: Entity text and labels must match exactly
- **relaxed**: Labels must match with any text overlap
- **overlap**: Partial text overlap is acceptable

### Output Files

Evaluation produces two JSON files:

1. **`true_pred_entities.json`**: Contains all true and predicted entities
2. **`performance.json`**: Contains detailed metrics:
   - Number of examples evaluated
   - Average inference time per example
   - Precision, Recall, F1 for each metric type
   - Per-label performance breakdown

## Example Usage

### Basic Evaluation

```bash
# Using uv (if installed)
uv run python -m src.training.evaluate_ollama \
    --eval-dataset-file data/final/test.json \
    --results-dir results/gemma3-27b \
    --model-name gemma3:27b \
    --model-max-seq-length 4096

# Or using standard python (activate venv first)
source .venv/bin/activate
python -m src.training.evaluate_ollama \
    --eval-dataset-file data/final/test.json \
    --results-dir results/gemma3-27b \
    --model-name gemma3:27b \
    --model-max-seq-length 4096
```

### With Custom System Prompt

```bash
# Using uv (if installed)
uv run python -m src.training.evaluate_ollama \
    --eval-dataset-file data/final/test.json \
    --results-dir results/medgemma \
    --model-name alibayram/medgemma:4b \
    --model-system-prompt "Extract medical entities including diseases, medications, and dosages." \
    --model-max-seq-length 4096

# Or using standard python (activate venv first)
source .venv/bin/activate
python -m src.training.evaluate_ollama \
    --eval-dataset-file data/final/test.json \
    --results-dir results/medgemma \
    --model-name alibayram/medgemma:4b \
    --model-system-prompt "Extract medical entities including diseases, medications, and dosages." \
    --model-max-seq-length 4096
```

> **Note:** When running Python directly (not through bash scripts), you must activate the appropriate virtual environment first if not using uv.

### Using the Bash Script

A complete evaluation script is available at `scripts/models/eval_model_ollama.sh`:

```bash
#!/bin/bash
# Evaluate the model using Ollama

set -e # exit on error

# Dataset configuration
BASE_STORAGE_DIR=.
BASE_PROJECT_DIR=.
DATASET_DIR=data/final
RESULTS_DIR=results

TEST_DATASET_FILE=test.json

# Model parameters
MODEL_NAME=gemma3:27b
MODEL_MAX_SEQ_LENGTH=4096

# Prepare paths
EVAL_DATASET_FILE_PATH=${BASE_STORAGE_DIR}/${DATASET_DIR}/${TEST_DATASET_FILE}
TEST_OUTPUT_DIR=${BASE_PROJECT_DIR}/${RESULTS_DIR}/${MODEL_STORE_NAME}

echo "================================================"
echo "Parameters"
echo "================================================"
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

# Automatically detect uv or python
if command -v uv &> /dev/null; then
    RUN_PYTHON="uv run python"
else
    RUN_PYTHON="python"
fi

$RUN_PYTHON -m src.training.evaluate_ollama \
    --eval-dataset-file ${EVAL_DATASET_FILE_PATH} \
    --results-dir ${TEST_OUTPUT_DIR} \
    --model-name ${MODEL_NAME} \
    --model-max-seq-length ${MODEL_MAX_SEQ_LENGTH}

echo "End time: $(date)"
```

> **Note:** The script automatically detects whether `uv` is available and falls back to standard `python` if not. When using standard python, the script automatically activates an available virtual environment (`.venv`, `.venv-gliner`, or `.venv-unsloth`).

## Hardware Requirements

### Minimum Requirements
- **CPU**: Modern multi-core processor (x86-64)
- **RAM**: 8GB+ (16GB recommended for 7B models, 32GB+ for 27B+ models)
- **Storage**: 5-50GB depending on model size
  - Small models (1-4B): 2-5GB
  - Medium models (7-8B): 5-10GB
  - Large models (13-27B): 15-30GB

### GPU Support
- **Optional but recommended** for faster inference
- NVIDIA GPU with CUDA support
- AMD GPU with ROCm support
- Apple Silicon (M1/M2/M3) via Metal

### Performance Comparison
| Hardware | 7B Model Speed | 27B Model Speed |
|----------|---------------|-----------------|
| CPU only (16 cores) | ~5-10s/example | ~20-40s/example |
| GPU (RTX 3090) | ~1-2s/example | ~3-5s/example |
| Apple M2 Max | ~2-3s/example | ~8-12s/example |

## Model Selection Guide

### For Speed
- Use smaller models: `llama3.2:1b`, `gemma3:4b`
- Sacrifice some accuracy for faster inference

### For Accuracy
- Use larger models: `gemma3:27b`, `llama3.1:8b`
- Medical-specific models: `alibayram/medgemma:4b`

### For Balance
- 7-8B models offer good trade-off
- `gemma3:12b` provides middle ground

### For Medical Tasks
- **Best**: `alibayram/medgemma:4b` or `alibayram/medgemma:27b`
- **Alternative**: `gemma3:12b` or `gemma3:27b`
- **Fast**: `gemma3:4b` or `llama3.2:3b`

## Best Practices

### System Prompt Design
1. **Be explicit**: Clearly state the task and expected output format
2. **Include examples**: Show the desired JSON structure
3. **Specify constraints**: Entity granularity, text format, etc.
4. **Language specification**: Mention if entities should be in specific language

### Model Selection
1. **Start small**: Test with smaller models first (faster iteration)
2. **Scale up**: Move to larger models if accuracy is insufficient
3. **Try medical models**: For medical terminology, specialized models perform better
4. **Benchmark**: Compare multiple models on your specific dataset

### Performance Optimization
1. **Batch processing**: Evaluate multiple examples in sequence (not parallel)
2. **Model preloading**: Keep Ollama running to avoid model loading time
3. **Context length**: Use appropriate `--model-max-seq-length` (shorter = faster)
4. **Hardware acceleration**: Enable GPU support for faster inference

### Data Preparation
1. **Clean text**: Remove unnecessary formatting or special characters
2. **Consistent labeling**: Use consistent entity type names
3. **Representative examples**: Include diverse text samples
4. **Label coverage**: Ensure all entity types are represented

## Troubleshooting

### Model Not Found
```bash
# Pull the model first
ollama pull gemma3:27b

# List available models
ollama list
```

### Slow Inference
- Use a smaller model variant
- Enable GPU acceleration (check with `ollama ps`)
- Reduce `--model-max-seq-length`
- Close other applications to free up RAM

### JSON Parsing Errors
- The system retries up to 3 times automatically
- If persistent, try:
  - Simplifying the system prompt
  - Using a larger/better model
  - Adjusting temperature settings (in code)

### Poor Entity Extraction
- **Use medical-specific models** (`alibayram/medgemma`)
- **Improve system prompt** with more specific instructions
- **Try larger models** (e.g., upgrade from 4B to 12B or 27B)
- **Check data quality** and ensure examples are clear
- Consider fine-tuning with Unsloth instead

### Out of Memory
```bash
# Use smaller model
--model-name gemma3:4b  # instead of gemma3:27b

# Or close other applications and retry
```

### Connection Errors
```bash
# Ensure Ollama service is running
ollama serve

# Check if it's responsive
curl http://localhost:11434/api/tags
```

## Environment Setup

### Installing Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows
# Download from https://ollama.com/download
```

### Starting Ollama Service

```bash
# Start the service (runs in background)
ollama serve

# Or with Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### Pulling Models

```bash
# Download specific models
ollama pull gemma3:27b
ollama pull llama3.1:8b
ollama pull alibayram/medgemma:4b

# List installed models
ollama list
```
