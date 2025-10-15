import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path

from gliner.data_processing import WordsSplitter
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


def extract_gliner_entities_train(example, splitter):

    example_text = example["text"]
    ws_generator = splitter(example_text)

    token_info = []
    tokenized_text = []
    for token_idx, token in enumerate(ws_generator):
        token_info.append(
            # token text, start char, end char, token index
            (token[0], token[1], token[2], token_idx)
        )
        tokenized_text.append(token[0])

    skipped_entities = 0
    present_entities = []
    for entity in example.get("entities", []):
        ent_text, ent_label = entity["text"], entity["label"]

        ent_text_escaped = re.escape(ent_text.lower())
        regex1 = rf"\b{ent_text_escaped}\b"

        matches = [match for match in re.finditer(regex1, example_text.lower())]
        if not matches:
            skipped_entities += 1
            continue

        for match in matches:
            mstart_idx, mend_idx = match.span()
            match_ent = re.search(rf"\b{ent_text_escaped}\b", example_text.lower()[mstart_idx:mend_idx])
            start_index = match.start() + match_ent.start()
            end_index = start_index + len(ent_text)

            entity_tokens = [
                (token_text, token_idx)
                for token_text, token_start, token_end, token_idx in token_info
                if token_start >= start_index and token_end <= end_index
            ]

            if entity_tokens:
                present_entities.append(
                    [
                        entity_tokens[0][1],
                        entity_tokens[-1][1],
                        ent_label,
                    ]
                )

    return {"tokenized_text": tokenized_text, "ner": present_entities}, skipped_entities


# ===============================
# Main function
# ===============================


def get_args():
    parser = ArgumentParser("Validate entities in text using regex for GLiNER format")
    parser.add_argument("--input-file", type=str, required=True, help="The path to the input file")
    parser.add_argument("--output-file", type=str, required=True, help="The path to the output file")
    parser.add_argument("--skip-empty-entities", action="store_true", help="Skip examples with no entities")
    parser.add_argument("--format", type=str, default="train", help="Format to use for the output file")
    parser.add_argument(
        "--entities-key", type=str, default="entities", help="The key to the entities in the input file"
    )
    return parser.parse_args()


def main(args):
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file {args.input_file} does not exist")

    with open(args.input_file, "r", encoding="utf8") as f:
        data = json.load(f)

    # initialize the words splitter (default: universal)
    splitter = WordsSplitter("whitespace")

    if args.format == "train":
        extract_func = extract_gliner_entities_train
    elif args.format == "test":
        logger.warning(
            "To create the test format, please use the `regex_label_validate_llm`"
            "script (both gliner and llm have the same test format)"
        )
        exit()
    else:
        raise ValueError(f"Format {args.format} not supported")

    output_data = []
    total_skipped_entities = 0
    for example in tqdm(data, desc="Processing examples in dataset", dynamic_ncols=True):
        example_entities = example.get(args.entities_key, [])
        if not example_entities:
            # skip examples with no entities (empty examples are not allowed by GLiNER trainer)
            continue

        for sentence in sent_tokenize(example["text"]):
            updated_example = {"text": sentence, "entities": example_entities}
            updated_example, skipped = extract_func(updated_example, splitter)
            output_data.append(updated_example)
            total_skipped_entities += skipped

    print(f"Total skipped entities: {total_skipped_entities}")

    # skip examples with no entities
    if args.skip_empty_entities:
        output_data = [d for d in output_data if d["ner"]]

    # Save GLiNER format
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
