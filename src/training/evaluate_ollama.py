import json
import logging
import sys
import time
from argparse import ArgumentParser
from importlib import reload
from pathlib import Path

import ollama
from tqdm import tqdm

import src.core.data.formatter as mfmt
import src.core.metrics as mmts
from src.core.interface import define_medical_entities_class
from src.core.utils.argument_parsers import str2bool

# reload custom modules to avoid caching issues
mfmt = reload(mfmt)
mmts = reload(mmts)

# ===============================
# Logging
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # override any existing handlers
)
logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ===============================
# Constants
# ===============================

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """
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
"""

# ===============================
# Model functions
# ===============================


def get_ollama_response(model_name, prompt, max_new_tokens, system_prompt, medical_entities_class):

    for _ in range(MAX_ATTEMPTS):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                format=medical_entities_class.model_json_schema(),
                options={"max_tokens": max_new_tokens},
            )
            return medical_entities_class.model_validate_json(response.message.content).model_dump(mode="json")[
                "entities"
            ]
        except Exception as e:
            logger.warning(f"Error getting response from {model_name}: {e}")
            continue

    logger.warning(f"Failed to get response from {model_name} after {MAX_ATTEMPTS} attempts")
    return []


# ===============================
# Main
# ===============================


def get_args():
    parser = ArgumentParser("Evaluate a Ollama model")
    # Data
    parser.add_argument("--eval-dataset-file", type=str, required=True, help="The path to the evaluation file")
    parser.add_argument("--results-dir", type=str, required=True, help="The path to the results directory")

    # Model
    parser.add_argument("--model-name", type=str, required=True, help="The ollama model name")
    parser.add_argument("--model-system-prompt", type=str, default=SYSTEM_PROMPT, help="The system prompt")
    parser.add_argument("--model-max-seq-length", type=int, default=4096, help="The maximum sequence length")

    # Validation parameters
    parser.add_argument("--unique-entities", type=str2bool, default=True, help="Whether to use unique entities")
    parser.add_argument("--eval-metrics", type=str, default="exact,relaxed,overlap", help="The metrics to evaluate")

    return parser.parse_args()


def main(args):

    if not Path(args.eval_dataset_file).exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {args.eval_dataset_file}")

    # =================================
    # Prepare the model and the dataset
    # =================================

    logger.info(f"Loading the evaluation dataset file '{args.eval_dataset_file}'...")

    # load the dataset
    with open(args.eval_dataset_file, "r") as f:
        test_dataset = json.load(f)

    formatter = mfmt.PromptFormatter(
        input_key="text",
        output_key="entities",
        system_prompt=args.model_system_prompt,
        unique_entities=args.unique_entities,
    )

    # prepare the examples
    examples = []
    for example in test_dataset:
        entities = [formatter.format_entities(e) for e in example["entities"]]
        if args.unique_entities:
            entities = formatter.get_unique_entities(entities)
        examples.append({"text": example["text"], "entities": entities})

    # =================================
    # Generate the model outputs
    # =================================

    # get the true and predicted entities
    true_pred_ents = {
        "true_ents": [],
        "pred_ents": [],
    }

    # prepare the example batches
    max_new_tokens = min(args.model_max_seq_length // 4, 1024)

    unique_labels = list(set(lbl["label"] for e in examples for lbl in e["entities"]))
    medical_entities_class = define_medical_entities_class(unique_labels)

    start_time = time.time()
    # iterate over the example batches
    for example in tqdm(examples, desc="Generating the model outputs", dynamic_ncols=True):
        pred_ents = get_ollama_response(
            args.model_name, example["text"], max_new_tokens, args.model_system_prompt, medical_entities_class
        )
        pred_ents = [formatter.format_entities(e) for e in pred_ents]
        if args.unique_entities:
            pred_ents = formatter.get_unique_entities(pred_ents)
        # append the true and predicted entities
        true_pred_ents["true_ents"].append(example["entities"])
        true_pred_ents["pred_ents"].append(pred_ents)

    end_time = time.time()
    avg_inference_time = (end_time - start_time) / len(examples)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # save the true and predicted labels
    with open(results_dir / "true_pred_entities.json", "w") as f:
        json.dump(true_pred_ents, f, ensure_ascii=False)

    # =================================
    # Evaluate the model
    # =================================

    logger.info("Computing the performance...")

    unique_labels.sort()
    # compute the performance
    performance = {
        "num_examples": len(examples),
        "avg_inference_time": avg_inference_time,
        "metrics": {
            "total": {},
            **{lbl: {} for lbl in unique_labels},
        },
    }

    eval_metrics = args.eval_metrics.split(",")
    metrics = mmts.NERMetrics(metrics=eval_metrics)
    for match_type in metrics.metrics:
        precision, recall, f1 = metrics.evaluate_ner_performance(
            true_pred_ents["true_ents"],
            true_pred_ents["pred_ents"],
            match_type=match_type,
        )
        performance["metrics"]["total"][match_type] = {"p": precision, "r": recall, "f1": f1}

        for label in unique_labels:
            precision, recall, f1 = metrics.evaluate_ner_performance(
                true_pred_ents["true_ents"],
                true_pred_ents["pred_ents"],
                match_type=match_type,
                label=label,
            )
            performance["metrics"][label][match_type] = {"p": precision, "r": recall, "f1": f1}

    # save the true and predicted labels
    with open(results_dir / "performance.json", "w") as f:
        json.dump(performance, f, ensure_ascii=False)

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    args = get_args()
    main(args)
