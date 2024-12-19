import gc
import re
import os
import sys
import json
import torch
from peft import LoraConfig
from datasets import Dataset
from typing import Tuple, List
from functools import partial
from argparse import ArgumentParser
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

from projects.llama.src.utils.prompts import Prompts

Prompts = Prompts()

def training_data_individual_labels(dataset: List[dict], labels: List[str], output_type: str = "instruction_training") -> Dataset:
    """
    This function processes each entry in the given dataset and creates multiple prompts,
    each designed to extract a single label. Depending on the chosen `output_type`, 
    it constructs prompts in either instruction or conversational style.

    Args:
        dataset: The dataset to format. It should be a list of dictionaries, where each dictionary contains a "text" key and a "labels" key.
        labels: The labels to use when formatting the data.
        output_type: The format to use when formatting the data. It should be one of "instruction_training" or "conversational_training".
    
    Returns:
        A Dataset object containing the formatted data.
    """
    if output_type not in ["instruction_training", "conversational_training"]:
        raise ValueError(f"Unknown output_type: {output_type}")

    train_format_dataset = []
    for example in dataset:
        relevant_labels = [label for label in example["labels"] if label["label"] in labels]
        grouped_entities = {label: [item for item in relevant_labels if item['label'] == label] for label in labels}
        for label in labels:
            if output_type == "instruction_training":
                message = Prompts.create_instruction_training_message_with_completion([label], example["text"], grouped_entities[label])
                train_format_dataset.append(message)
            elif output_type == "conversational_training":
                message = Prompts.create_conversational_training_message_with_completion([label], example["text"], grouped_entities[label])
                train_format_dataset.append({"messages": message})
    return Dataset.from_list(train_format_dataset)

def training_data_combined_labels(dataset: List[dict], labels: List[str], output_type: str = "instruction_training") -> Dataset:
    """
    The function processes each entry in the given dataset and creates a prompt that is
    designed to extract all specified labels in a single example. Depending on the chosen
    `output_type`, it constructs the prompt in either instruction or conversational style.
    
    Args:
        dataset: The dataset to format. It should be a list of dictionaries, where each dictionary contains a "text" key and a "labels" key.
        labels: The labels to use when formatting the data.
        output_type: The format to use when formatting the data. It should be one of "instruction_training" or "conversational_training".
    
    Returns:
        A Dataset object containing the formatted data.
    """
    if output_type not in ["instruction_training", "conversational_training"]:
        raise ValueError(f"Unknown output_type: {output_type}")

    train_format_dataset = []
    for example in dataset:
        relevant_labels = [label for label in example["labels"] if label["label"] in labels]
        if output_type == "instruction_training":
            message = Prompts.create_instruction_training_message_with_completion(labels, example["text"], relevant_labels)
            train_format_dataset.append(message)
        elif output_type == "conversational_training":
            message = Prompts.create_conversational_training_message_with_completion(labels, example["text"], relevant_labels)
            train_format_dataset.append({"messages": message})
    return Dataset.from_list(train_format_dataset)


def instructions_formatting_function(prompt: str, completion: str, tokenizer: AutoTokenizer):
    """
    Formats a prompt and its corresponding completion according to the instruction format.
    
    Args:
        prompt: The prompt.
        completion: The completion.
        tokenizer: The tokenizer to use.
    
    Returns:
        A dictionary with the prompt and completion in the instruction format.
    """
    converted_sample = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    return tokenizer.apply_chat_template(converted_sample, tokenize=False, add_generation_prompt=False)
    
def conversations_formatting_function(examples: dict, tokenizer: AutoTokenizer):
    """Formats a dataset according to the conversational format.

    Examples:
        {'messages': [{'role': 'system',
                        'content': 'system prompt'},
                       {'role': 'user',
                        'content': 'user content'}]}
    """
    return tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)   
    
def formatting(examples, tokenizer, output_type="instruction_training"):
    """
    Formats the dataset according to the specified data format.

    Args:
        examples: The dataset to format.
        tokenizer: The tokenizer to use.
        output_type: The format to use. One of "instruction_training" or "conversational_training".

    Returns:
        The formatted dataset.
    """
    output_text = []
    if output_type == "instruction_training":
        for i in range(len(examples["prompt"])):
            # Format the instruction example
            formatted = instructions_formatting_function(
                examples["prompt"][i], examples["completion"][i], tokenizer
            )
            output_text.append(formatted)
        return output_text
    elif output_type == "conversational_training":
        for example in examples["messages"]:
            # Format the conversational example
            formatted = conversations_formatting_function(
                {"messages": example}, tokenizer
            )
            output_text.append(formatted)
        return output_text
    
def prepare_model_and_tokenizer(model_name: str, use_gpu: bool) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Prepares the model and tokenizer.

    Args:
        model_name: The name of the model to use.
        use_gpu: Whether to use GPU or not.

    Returns:
        The Huggingface model.
        The Huggingface tokenizer.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def release_model_memory(model, tokenizer):
    """
    Releases the memory used by the model and tokenizer.

    This is useful when loading a new model and tokenizer to avoid running out of memory.

    Args:
        model: The loaded model (e.g., a Hugging Face Transformers model).
        tokenizer: The loaded tokenizer associated with the model.
    """
    # Delete the model and tokenizer
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer

    # Empty the CUDA cache to free up memory
    torch.cuda.empty_cache()

    # Run garbage collection to free up any remaining memory
    gc.collect()

    print("Model and tokenizer memory released successfully.")

def training(model, tokenizer, dataset, output_type, save_model_path):
    print("Fine-tuning:", model.config.name_or_path)

    # formatting_func in SFTTrainer expects a function that takes only examples as input
    formatting_prompts_func = partial(formatting, tokenizer=tokenizer, output_type=output_type)

    # Extract the response template from the tokenizer's chat template.
    # The goal is to identify where the assistant's response begins in the template
    # and configure the data collator to focus on training the completion part only.
    template = tokenizer.chat_template
    pattern = r"{%- if add_generation_prompt %}\s*\{\{-(.*?)\}\}\s*{%- endif %}"
    match = re.search(pattern, template)
    collator = None
    if match:
        response_template = match.group(1)
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    if collator == None:
        print("The collator was not set correctly")
        return
        response_template = ""
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = TrainingArguments(
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        gradient_checkpointing=True,
        per_device_train_batch_size = 2,
        output_dir=save_model_path,
        report_to="none",
    )
    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=args,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(save_model_path + model.config.name_or_path)

    release_model_memory(model, tokenizer)

def main(args):
    # Load training args
    with open(args.script_args, "r") as f:
        script_args = json.load(f)

    # Load dataset
    with open(args.dataset, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # Load training args
    models = script_args["model_name"]
    training_type = script_args["training_type"]
    labels = script_args["labels"]
    output_type = script_args["output_type"]
    use_gpu = script_args["use_gpu"]

    print("Preapring dataset")
    if training_type == "training_data_one_by_one_label":
        dataset = training_data_individual_labels(dataset, labels, output_type)
    elif training_type == "training_data_all_labels":
        dataset = training_data_combined_labels(dataset, labels, output_type)
    print(dataset)
    print("Dataset prepared")

    for model_name in models:
        model, tokenizer = prepare_model_and_tokenizer(model_name, use_gpu)
        training(model, tokenizer, dataset, output_type, args.save_model_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--script_args", type=str, help="Path to the script arguments JSON file.")
    parser.add_argument("--dataset", type=str, help="Train dataset.")
    parser.add_argument("--save_model_path", type=str, help="Save model path.")
    args = parser.parse_args()
    main(args)

