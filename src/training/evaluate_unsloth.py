import json
import logging
import sys
import time
from argparse import ArgumentParser
from importlib import reload
from pathlib import Path

import torch
from tqdm import tqdm
from unsloth import FastModel

import src.core.data.formatter as mfmt
import src.core.metrics as mmts
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
logger = logging.getLogger(__name__)


# ===============================
# Model functions
# ===============================


def get_recommended_settings_by_model(model_name):

    model_name_lower = model_name.lower()

    if "llama" in model_name_lower:
        return {"temperature": 1.5, "min_p": 0.1}
    elif "gemma" in model_name_lower:
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    elif "mistral" in model_name_lower:
        return {"temperature": 0.8, "top_p": 0.95}
    elif "qwen" in model_name_lower:
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    else:
        return {"temperature": 0.8, "top_p": 0.95}


# ===============================
# Main
# ===============================


def get_args():
    parser = ArgumentParser("Evaluate a Unsloth model")
    # Data
    parser.add_argument("--eval-dataset-file", type=str, required=True, help="The path to the evaluation dataset file")
    parser.add_argument("--results-dir", type=str, required=True, help="The path to the results directory")

    # Model
    parser.add_argument(
        "--model-dir", type=str, required=True, help="The path to the pre-trained model or the model name"
    )
    parser.add_argument("--model-max-seq-length", type=int, default=4096, help="The maximum sequence length")
    parser.add_argument("--model-load-in-4bit", type=str2bool, default=True, help="Whether to load the model in 4-bit")
    parser.add_argument("--model-load-in-8bit", type=str2bool, default=False, help="Whether to load the model in 8-bit")
    parser.add_argument(
        "--model-system-prompt",
        type=str,
        default="You are a medical entity extractor from clinical texts. Extract the entities from the text and return them in a structured JSON format.",
        help="The system prompt for the model",
    )

    # Validation parameters
    parser.add_argument("--eval-batch-size", type=int, default=4, help="The batch size for the evaluation")
    parser.add_argument("--eval-unique-entities", type=str2bool, default=True, help="Whether to use unique entities")
    parser.add_argument("--eval-metrics", type=str, default="exact,relaxed,overlap", help="The metrics to evaluate")

    return parser.parse_args()


def main(args):

    if not Path(args.model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    # Check if LoRA adapter files exist
    adapter_config_path = Path(args.model_dir) / "adapter_config.json"
    if not adapter_config_path.exists():
        logger.warning(f"No adapter_config.json found in {args.model_dir}. This may not be a fine-tuned LoRA model.")
        logger.info(f"Files in model directory: {list(Path(args.model_dir).iterdir())}")

    if not Path(args.eval_dataset_file).exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {args.eval_dataset_file}")

    # =================================
    # Prepare the model and the dataset
    # =================================
    logger.info(f"Preparing the model and tokenizer for model '{args.model_dir}'...")

    # prepare the model to be evaluated
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.model_max_seq_length,
        load_in_4bit=args.model_load_in_4bit,
        load_in_8bit=args.model_load_in_8bit,
        device_map="balanced",
    )

    # Enable inference mode
    model = FastModel.for_inference(model)

    # Log model info for verification
    logger.info(f"Model loaded from: {args.model_dir}")
    logger.info(
        f"Base model config: {model.config.name_or_path if hasattr(model.config, 'name_or_path') else 'unknown'}"
    )
    logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    logger.info(f"Loading the evaluation dataset file '{args.eval_dataset_file}'...")

    # load the dataset
    with open(args.eval_dataset_file, "r") as f:
        test_dataset = json.load(f)

    formatter = mfmt.PromptFormatter(
        input_key="text",
        output_key="entities",
        system_prompt=args.model_system_prompt,
        unique_entities=args.eval_unique_entities,
    )

    # prepare the examples
    examples = []
    for example in test_dataset:
        txt = formatter.format_test_example(example, tokenizer)
        entities = [formatter.format_entities(e) for e in example["entities"]]
        if args.eval_unique_entities:
            entities = formatter.get_unique_entities(entities)
        examples.append({"text": txt, "entities": entities})

    # =================================
    # Generate the model outputs
    # =================================

    model_settings = get_recommended_settings_by_model(args.model_dir)

    # get the true and predicted entities
    true_pred_ents = {
        "true_ents": [],
        "pred_ents": [],
    }

    # prepare the example batches
    max_new_tokens = min(args.model_max_seq_length // 4, 1024)
    example_batches = [examples[i : i + args.eval_batch_size] for i in range(0, len(examples), args.eval_batch_size)]

    # iterate over the example batches
    start_time = time.time()
    for batch in tqdm(example_batches, desc="Generating the model outputs"):
        # get the inputs
        inputs = tokenizer(
            [example["text"] for example in batch], padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        # generate the outputs for the batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **model_settings,
            )
        # iterate over the batch outputs
        for idx, (example, output) in enumerate(zip(batch, outputs)):
            # get only the newly generated tokens
            output = output[inputs.input_ids[idx].shape[0] :]
            # decode the output (use tokenizer.decode for just one output)
            output = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # extract the entities from the output
            output = formatter.extract_entities_from_text(output)
            # append the true and predicted entities
            true_pred_ents["true_ents"].append(example["entities"])
            true_pred_ents["pred_ents"].append(output)

    end_time = time.time()
    avg_inference_time = (end_time - start_time) / len(examples)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # save the true and predicted entities
    with open(results_dir / "true_pred_entities.json", "w") as f:
        json.dump(true_pred_ents, f, ensure_ascii=False)

    # =================================
    # Evaluate the model
    # =================================

    logger.info("Computing the performance...")

    unique_labels = list(set(lbl["label"] for e in examples for lbl in e["entities"]))
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

    # save the true and predicted entities
    with open(results_dir / "performance.json", "w") as f:
        json.dump(performance, f, ensure_ascii=False)

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    args = get_args()
    main(args)
