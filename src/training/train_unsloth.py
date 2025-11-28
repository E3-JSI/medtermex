import logging
import os
import sys
from argparse import ArgumentParser
from importlib import reload
from pathlib import Path

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


import src.core.data.formatter as mfmt
from src.core.utils.argument_parsers import str2bool

# reload custom modules to avoid caching issues
mfmt = reload(mfmt)

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


def auto_detect_target_modules(model):
    """Automatically detect appropriate target modules for LoRA."""
    target_modules = []

    for name, module in model.named_modules():
        # Look for common attention and MLP layer patterns
        if any(pattern in name.lower() for pattern in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            target_modules.append(name.split(".")[-1])
        elif any(pattern in name.lower() for pattern in ["gate_proj", "up_proj", "down_proj"]):
            target_modules.append(name.split(".")[-1])
        elif any(pattern in name.lower() for pattern in ["dense", "linear"]):
            target_modules.append(name.split(".")[-1])

    # Remove duplicates and return
    return list(set(target_modules)) or ["q_proj", "v_proj"]  # fallback


def get_chat_template_name(model_name):
    """Get the correct chat template name for unsloth."""
    model_name_lower = model_name.lower()

    if "Llama-3.2" in model_name_lower:
        return "llama-3.2"
    elif "Llama-3.1" in model_name_lower:
        return "llama-3.1"
    elif "llama" in model_name_lower or "Llama" in model_name_lower:
        return "llama-3"
    elif "gemma" in model_name_lower and ("3n" in model_name_lower or "3-n" in model_name_lower):
        return "gemma-3n"
    elif "gemma" in model_name_lower:
        return "gemma-3"  # works for gemma-3, medgemma
    elif "qwen" in model_name_lower:
        return "qwen-3"
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_train_on_responses_only_params(tokenizer, model_name):
    """Get appropriate chat template based on model type."""

    model_name_lower = model_name.lower()

    # Llama 3, 3.1, 3.2 models
    if "llama" in model_name_lower or "Llama" in model_name_lower:
        return {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        }

    # Gemma 2, Gemma 3, Gemma 3N, MedGemma models
    elif "gemma" in model_name_lower:
        return {"instruction_part": "<start_of_turn>user\n", "response_part": "<start_of_turn>model\n"}

    elif "qwen" in model_name_lower:
        return {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        }

    # Mistral models
    elif "mistral" in model_name_lower:
        return {"instruction_part": "[INST]", "response_part": "[/INST]"}

    # ChatML format (used by many models)
    elif hasattr(tokenizer, "chat_template") and "chatml" in (tokenizer.chat_template or "").lower():
        return {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n"}

    # Fallback: try to extract from chat template
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# ===============================
# Main
# ===============================


def get_args():
    parser = ArgumentParser("Train a Unsloth model")
    # Data parameters
    parser.add_argument("--train-dataset-file", type=str, required=True, help="The path to the training dataset file")
    parser.add_argument("--output-dir", type=str, required=True, help="The path to the output directory")

    # Model parameters
    parser.add_argument(
        "--model-name-or-path", type=str, required=True, help="The path to the pre-trained model or the model name"
    )
    parser.add_argument("--model-max-seq-length", type=int, default=4096, help="The maximum sequence length")
    parser.add_argument("--model-load-in-4bit", type=str2bool, default=True, help="Whether to load the model in 4-bit")
    parser.add_argument("--model-load-in-8bit", type=str2bool, default=False, help="Whether to load the model in 8-bit")
    parser.add_argument("--model-full-finetuning", type=str2bool, default=False, help="Whether to use full finetuning")
    parser.add_argument("--model-hf-token", type=str, default=None, help="The Hugging Face token")
    parser.add_argument(
        "--model-system-prompt",
        type=str,
        default="You are a medical entity extractor from clinical texts. Extract the entities from the text and return them in a structured JSON format.",
        help="The system prompt for the model",
    )
    parser.add_argument("--unique-entities", type=str2bool, default=True, help="Whether to use unique entities")

    # PEFT parameters
    parser.add_argument(
        "--peft-ft-vision-layers",
        type=str2bool,
        default=False,
        help="Whether to finetune the vision layers",
    )
    parser.add_argument(
        "--peft-ft-language-layers",
        type=str2bool,
        default=True,
        help="Whether to finetune the language layers",
    )
    parser.add_argument(
        "--peft-ft-attention-modules",
        type=str2bool,
        default=True,
        help="Whether to finetune the attention modules",
    )
    parser.add_argument(
        "--peft-ft-mlp-modules",
        type=str2bool,
        default=True,
        help="Whether to finetune the MLP modules",
    )
    parser.add_argument("--peft-rank", type=int, default=8, help="The rank for the LoRA")
    parser.add_argument("--peft-lora-alpha", type=int, default=16, help="The alpha for the LoRA")
    parser.add_argument("--peft-lora-dropout", type=float, default=0.1, help="The dropout for the LoRA")
    parser.add_argument("--peft-lora-bias", type=str2bool, default=False, help="Whether to use bias for the LoRA")

    # Training parameters
    parser.add_argument("--train-per-device-batch-size", type=int, default=16, help="The batch size per device")
    parser.add_argument(
        "--train-gradient-accumulation-steps",
        type=int,
        default=4,
        help="The gradient accumulation steps",
    )
    parser.add_argument("--train-num-epochs", type=int, default=3, help="The number of training epochs")
    parser.add_argument("--train-learning-rate", type=float, default=2e-4, help="The learning rate")
    parser.add_argument("--train-weight-decay", type=float, default=0.01, help="The weight decay")
    parser.add_argument("--train-warmup-steps", type=int, default=5, help="The warmup steps")
    parser.add_argument(
        "--train-lr-scheduler-type", type=str, default="linear", help="The learning rate scheduler type"
    )
    parser.add_argument("--train-seed", type=int, default=42, help="The seed for the training")
    return parser.parse_args()


def main(args):

    logger.info(f"Preparing the model and tokenizer for model '{args.model_name_or_path}'...")

    # prepare the model to be trained
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.model_max_seq_length,
        load_in_4bit=args.model_load_in_4bit,
        load_in_8bit=args.model_load_in_8bit,
        full_finetuning=args.model_full_finetuning,
        device_map="balanced",
        token=args.model_hf_token or os.getenv("HF_TOKEN", None),
    )

    logger.info("Preparing the model for training with PEFT...")

    # prepare model for training with PEFT
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=args.peft_ft_vision_layers,
        finetune_language_layers=args.peft_ft_language_layers,
        finetune_attention_modules=args.peft_ft_attention_modules,
        finetune_mlp_modules=args.peft_ft_mlp_modules,
        target_modules=auto_detect_target_modules(model),
        r=args.peft_rank,
        lora_alpha=args.peft_lora_alpha,
        lora_dropout=args.peft_lora_dropout,
        lora_bias="none",  # args.peft_lora_bias,
        random_state=args.train_seed,
    )

    # prepare the tokenizer for chat template
    logger.info("Preparing the tokenizer for chat template...")

    # prepare the tokenizer for chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=get_chat_template_name(args.model_name_or_path),
    )

    logger.info("Defining the formatting prompt function...")

    formatter = mfmt.PromptFormatter(
        input_key="text",
        output_key="entities",
        system_prompt=args.model_system_prompt,
        unique_entities=args.unique_entities,
    )

    # define the formatting prompt function
    def formatting_prompts_func(examples):
        return formatter.format_train_example_batch(examples, tokenizer)

    logger.info(f"Loading and formatting the dataset '{args.train_dataset_file}'...")

    dataset = load_dataset("json", data_files=args.train_dataset_file, split="train")
    dataset = standardize_data_formats(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    # make sure the dataset only contains the "text" column (otherwise the trainer will fail)
    dataset = dataset.remove_columns(list(filter(lambda x: x not in ["text"], dataset.features.keys())))
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=args.train_seed)

    logger.info("Printing the first couple of examples...")
    for idx in range(5):
        logger.info(dataset["train"][idx])

    logger.info("Preparing the trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=SFTConfig(
            dataset_text_field="text",
            dataset_num_proc=1,
            per_device_train_batch_size=args.train_per_device_batch_size,
            gradient_accumulation_steps=args.train_gradient_accumulation_steps,
            warmup_steps=args.train_warmup_steps,
            num_train_epochs=args.train_num_epochs,
            learning_rate=args.train_learning_rate,
            weight_decay=args.train_weight_decay,
            lr_scheduler_type=args.train_lr_scheduler_type,
            optim="adamw_8bit",
            seed=args.train_seed,
            logging_steps=10,
            report_to="none",
        ),
    )

    logger.info("Updating the trainer to use the chat template...")

    # update the trainer to use the chat template
    trainer = train_on_responses_only(
        trainer,
        **get_train_on_responses_only_params(tokenizer, args.model_name_or_path),
    )

    logger.info("Starting the training...")

    # train the model
    trainer.train()

    logger.info("Saving the model...")

    # save the model
    model_path = Path(args.output_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    args = get_args()
    main(args)
