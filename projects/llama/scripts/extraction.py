import re
import sys
import json
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from projects.llama.src.utils.prompts import Prompts
Prompts = Prompts()

def instructions_formatting_function(examples, tokenizer):
    """
    Input should look like a list of dictionaries:
        [ {'prompt': 'some prompt'} ]
    Or a single example like:
        {'prompt': 'some prompt'}
    """
    if isinstance(examples, list):
        output_texts = []
        for i in range(len(examples)):
            converted_sample = [
                {"role": "user", "content": examples[i]["prompt"]},
            ]
            output_texts.append(tokenizer.apply_chat_template(converted_sample, tokenize=True, return_tensors="pt", add_generation_prompt=True))
        return output_texts
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
        ]
        return tokenizer.apply_chat_template(converted_sample, tokenize=True, return_tensors="pt", add_generation_prompt=True)
    
def conversations_formatting_function(examples, tokenizer):
    """ 
    Input should look like a list of dictionaries: 
    [ 
        {'messages': [
            {'role': 'system',
                'content': 'system prompt'},
            {'role': 'user',
                'content': 'user content'}]},
        ...
    ]
    Or a single example like:
    { 'messages': [
            {'role': 'system',
                'content': 'system prompt'},
            {'role': 'user',
                'content': 'user content'}]}
"""
    if isinstance(examples, list):
        output_texts = []
        for i in range(len(examples)):
            output_texts.append(tokenizer.apply_chat_template(examples[i]["messages"], tokenize=True, return_tensors="pt", add_generation_prompt=True))
        return output_texts
    else:
        return tokenizer.apply_chat_template(examples["messages"], tokenize=True, return_tensors="pt", add_generation_prompt=True)
    
def prompt_formatting_function(example, tokenizer):
    return tokenizer.apply_chat_template(example, tokenize=True, return_tensors="pt", add_generation_prompt=True)

def prepare_model_and_tokenizer(model_name: str, use_gpu: bool, adapter_name: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Prepares the model and tokenizer.

    Args:
        model_name: The name of the model to use.
        use_gpu: Whether to use GPU or not.

    Returns:
        The Huggingface model.
        The Huggingface tokenizer.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=False)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config).to(device)
    print("Model used:", model_name)
    if adapter_name is not None:
        model = PeftModel.from_pretrained(model, adapter_name).to(device)
        print("Adding adapter:", adapter_name)
    return model, tokenizer

def generate_response(
    model,
    tokenizer,
    input_ids,
    temperature: float,
    top_p: float
) -> str:
    """Generate the response from the LLM.

    Args:
        model: The loaded LLM model.
        tokenizer: The tokenizer for the model.
        message: The message to generate the response from.
        temperature: The temperature to use for the generation.
        top_p: The top-p to use for the generation.

    Returns:
        The generated response.
    """
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1000,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    return response

def parse_response(response: str) -> str:
    """Parse the response from the LLM.

    Args:
        response: The response to parse.

    Returns:
        The parsed response.
    """
    try:
        # Use a regex pattern to find JSON-like structures
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            # Replace single quotes with double quotes for JSON compatibility
            json_str = json_str.replace("'", '"')
            # Attempt to load the JSON
            entities = json.loads(json_str)
            return entities
    except json.JSONDecodeError:
        pass
    # Fallback for cases where strict JSON parsing fails
    try:
        # Extract lines containing the output explicitly
        lines = response.splitlines()
        for line in lines:
            if "[" in line and "{" in line:
                # Extract possible JSON-like content
                json_str = line.strip()
                json_str = json_str.replace("'", '"')  # Handle single quotes
                entities = json.loads(json_str)
                return entities
    except json.JSONDecodeError:
        pass
    return []

def extract_label(
    model,
    tokenizer,
    text: str,
    label_to_extract,
    prompt_type: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> str:

    if prompt_type == "prompt_only":
        message = Prompts.create_prompt_only_prompt(label_to_extract, text)
        input_ids = prompt_formatting_function(message, tokenizer)
    elif prompt_type == "few_shot_prompting":
        message = Prompts.create_few_shot_prompt(label_to_extract, text)
        input_ids = prompt_formatting_function(message, tokenizer)
    elif prompt_type == "instruction_prompt":
        message = Prompts.create_instruction_message(label_to_extract, text)
        input_ids = instructions_formatting_function(message, tokenizer)
    elif prompt_type == "conversational_prompt":
        message = Prompts.create_conversational_message(label_to_extract, text)
        input_ids = conversations_formatting_function({"messages": message}, tokenizer)
    return parse_response(generate_response(model, tokenizer, input_ids, temperature, top_p))

def main(args):
    # load test dataset
    with open(args.dataset, "r", encoding="utf8") as file:
        data = json.load(file)

    # check if the path to save the output file exsists
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = prepare_model_and_tokenizer(args.model_name, args.use_gpu, adapter_name=args.adapter_name)

    labels = [
        "Age",
        "Sex",
        "Biological structure",
        "Sign symptom",
        "Diagnostic procedure",
        "Lab value",
        "Clinical event",
        "Personal background",
        "Detailed description",
        "Disease disorder",
        "Therapeutic procedure",
        "Distance",
        "Quantitative concept",
        "Nonbiological location",
        "Dosage",
        "Administration",
        "Frequency",
        "Medication",
        "Clinical event",
        "Severity",
        "Date",
        "Duration",
        "Coreference",
        "Family history",
        "History",
        "Outcome",
        "Activity",
        "Occupation",
        "Area",
        "Subject"
    ]
    # Add prompt_type argument to parser before main()

    predictions = []
    for example in tqdm(data):
        text = example["text"]
        extracted_labels = []
        
        if args.prompt_type in ["prompt_only", "few_shot_prompting"]:
            for label_to_extract in labels:
                label = extract_label(
                    model,
                    tokenizer,
                    text,
                    label_to_extract,
                    args.prompt_type,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                extracted_labels.extend(label)
        else:
            extracted_labels = extract_label(
                model,
                tokenizer,
                text,
                labels,
                args.prompt_type,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        
        predictions.append({
            "text": text,
            "labels": extracted_labels
        })

    output_path = args.output_file if args.output_file.endswith('.json') else args.output_file + '.json'
    with open(output_path, "w", encoding="utf8") as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct", help="Name of the model to use.")
    parser.add_argument("--adapter_name", type=str, default=None, help="Name of the adapter to use.")
    parser.add_argument("--dataset", type=str, help="Test dataset.")
    parser.add_argument("--prompt_type", type=str, default="prompt_only", 
                       choices=["prompt_only", "few_shot_prompting", "instruction_prompt", "conversational_prompt"],
                       help="Type of prompt to use")
    parser.add_argument("--output_file", type=str, help="Path to save the output JSON file with extracted labels.")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Flag to use GPU for inference.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p value for nucleus sampling.")
    args = parser.parse_args()
    main(args)
