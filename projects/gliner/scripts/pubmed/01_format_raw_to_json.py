import json
import os
from argparse import ArgumentParser
from pathlib import Path


def get_unique_file_names(input_dir):
    txt_files = list(Path(input_dir).rglob("*.txt"))
    ann_files = list(Path(input_dir).rglob("*.ann"))

    txt_file_names = set([file.name.replace(".txt", "") for file in txt_files])
    ann_file_names = set([file.name.replace(".ann", "") for file in ann_files])

    return txt_file_names.intersection(ann_file_names)


def process_example(txt_file_path, ann_file_path):
    # load the example text
    with open(txt_file_path, "r", encoding="utf8") as ft:
        text = ft.read()

    # load the example annotations
    entities = []
    with open(ann_file_path, "r", encoding="utf8") as fa:
        for line in fa:
            parts = line.strip().split("\t")
            if parts[0].startswith("T"):
                entity_info = parts[1].split()
                entity = {
                    "label": entity_info[0].lower(),
                    "start": int(entity_info[1]),
                    "end": int(entity_info[-1]),
                    "text": parts[2],
                }
                entities.append(entity)

    return {"text": text, "entities": entities}


def main(args):
    if not Path(args.input_dir).exists():
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist")

    # Load the dataset
    unique_file_names = get_unique_file_names(args.input_dir)

    examples = []
    for file_name in unique_file_names:
        txt_file_path = os.path.join(args.input_dir, f"{file_name}.txt")
        ann_file_path = os.path.join(args.input_dir, f"{file_name}.ann")
        examples.append(process_example(txt_file_path, ann_file_path))

    # Save the dataset
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, help="path to the input dir containing the examples"
    )
    parser.add_argument("--output_file", type=str, help="path to the output file")
    args = parser.parse_args()
    main(args)
