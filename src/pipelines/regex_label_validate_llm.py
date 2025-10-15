import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# ===============================
# Logging
# ===============================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================
# Helper functions
# ===============================


def get_unique_entities(entities):
    unique_entities = []
    span_set = set()
    for entity in entities:
        label = entity["label"]
        start = entity["start"]
        end = entity["end"]
        if (label, start, end) not in span_set:
            unique_entities.append(entity)
            span_set.add((label, start, end))
    return unique_entities


def filter_entities_in_text(example, mapping={}):

    skipped_entities = 0
    present_entities = []
    for entity in example["entities"]:
        ent_text, ent_label = entity["text"], entity["label"]

        ent_text_escaped = re.escape(ent_text.lower())
        regex1 = rf"\b{ent_text_escaped}\b"

        matches = [match for match in re.finditer(regex1, example["text"].lower())]
        if not matches:
            skipped_entities += 1
            continue

        for match in matches:
            mstart_idx, mend_idx = match.span()
            regex2 = rf"\b{ent_text_escaped}\b"
            for m in re.finditer(regex2, example["text"].lower()[mstart_idx:mend_idx]):
                start_index = mstart_idx + m.start()
                end_index = start_index + len(ent_text)
                text = example["text"][start_index:end_index]

                mapped_text = re.sub(r"\s+", " ", text.lower())
                for m_key, m_value in mapping.get(ent_label, {}).items():
                    if mapped_text in m_value:
                        mapped_text = m_key
                        break

                present_entities.append(
                    {
                        "label": ent_label,
                        "text": mapped_text,
                        "start": start_index,
                        "end": end_index,
                    }
                )
    # get unique entities
    unique_entities = get_unique_entities(present_entities)

    return {"text": example["text"], "entities": unique_entities}, skipped_entities


# ===============================
# Main function
# ===============================


def get_args():
    parser = ArgumentParser("Validate entities in text using regex for LLM format")
    parser.add_argument("--input-file", type=str, required=True, help="The path to the input file")
    parser.add_argument("--output-file", type=str, required=True, help="The path to the output file")
    parser.add_argument("--mapping-file", type=str, default=None, help="The path to the mapping file")
    parser.add_argument("--split-by-sentence", action="store_true", help="Split the text into sentences")
    parser.add_argument("--skip-empty-entities", action="store_true", help="Skip examples with no entities")
    parser.add_argument(
        "--entities-key", type=str, default="entities", help="The key to the entities in the input file"
    )
    return parser.parse_args()


def main(args):
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file {args.input_file} does not exist")

    with open(args.input_file, "r", encoding="utf8") as f:
        data = json.load(f)

    mapping = {}
    if args.mapping_file:
        with open(args.mapping_file, "r", encoding="utf8") as f:
            mapping = json.load(f)

    output_data = []
    total_skipped_entities = 0
    for example in tqdm(data, desc="Processing examples in dataset", dynamic_ncols=True):
        example = {"text": example["text"], "entities": example.get(args.entities_key, [])}

        # filter the entities in the whole text
        if not args.split_by_sentence:
            updated_example, skipped = filter_entities_in_text(example, mapping)
            output_data.append(updated_example)
            total_skipped_entities += skipped
            continue
        else:
            if not example["entities"]:
                output_data.append({"text": example["text"], "entities": []})
                continue
            # split the text into sentences (and retain the sentence labels)
            for sentence in sent_tokenize(example["text"]):
                # filter the entities in the sentence
                updated_example = {"text": sentence, "entities": example["entities"]}
                updated_example, skipped = filter_entities_in_text(updated_example, mapping)
                output_data.append(updated_example)
                total_skipped_entities += skipped

    print(f"Total skipped entities: {total_skipped_entities}")

    # skip examples with no entities
    if args.skip_empty_entities:
        output_data = [d for d in output_data if d["entities"]]

    # save the dataset in the output file
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
