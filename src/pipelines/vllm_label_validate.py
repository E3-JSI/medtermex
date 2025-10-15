import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import ollama
from tqdm import tqdm

from src.core.interface import define_medical_entities_class

# ===============================
# Logging
# ===============================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

SYSTEM_PROMPT = """
You are a medical entity extractor for clinical texts.

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
# Helper functions
# ===============================


def revise_entities(entity_text, entity_label, model_name, system_prompt, medical_entities_class):
    prompt = f"Medical entity text: {entity_text} - original label: {entity_label}"
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        format=medical_entities_class.model_json_schema(),
    )
    return medical_entities_class.model_validate_json(response.message.content)


# ===============================
# Main function
# ===============================


def get_args():
    parser = ArgumentParser("Validate entities in text using LLMs")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--model-name", type=str, default="gemma3:4b", help="Name of the ollama model")
    parser.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT, help="System prompt")
    parser.add_argument(
        "--validated-key", type=str, default="vllm_entities", help="Key to the validated entities in the output file"
    )
    return parser.parse_args()


def main(args):
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file {args.input_file} does not exist")

    # load the dataset
    with open(args.input_file, "r", encoding="utf8") as f:
        dataset = json.load(f)

    # get the medical entities class
    unique_labels = list(set(lbl["label"] for e in dataset for lbl in e["entities"]))
    medical_entities_class = define_medical_entities_class(unique_labels)

    vdataset = []
    for d in tqdm(dataset, desc="Validating entities", dynamic_ncols=True):
        entities = []
        for entity in d["entities"]:
            try:
                tmp_entities = revise_entities(
                    entity["text"], entity["label"], args.model_name, args.system_prompt, medical_entities_class
                )
                tmp_entities = tmp_entities.model_dump(mode="json")["entities"]
                tmp_entities = [e for e in tmp_entities if e["label"] != "" and e["text"] != ""]
                # if no entities are returned, use the original entity
                tmp_entities = tmp_entities if len(tmp_entities) > 0 else [entity]
                entities.extend(tmp_entities)
            except Exception as e:
                logger.error(f"Error validating entity {entity}: {e}")
                entities.append(entity)
                continue
        vdataset.append({**d, args.validated_key: entities})

    # save dataset
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(vdataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
