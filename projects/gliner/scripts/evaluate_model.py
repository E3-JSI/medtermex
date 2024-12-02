import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from gliner import GLiNER
from tqdm import tqdm

from common.evaluate import evaluate_ner_performance


def main(args):
    if not Path(args.data_test_file).exists():
        raise FileNotFoundError(f"Test data file {args.data_test_file} does not exist")

    with open(args.data_test_file, "r", encoding="utf8") as f:
        data = json.load(f)

    # load the device (GPU or CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    )

    model = GLiNER.from_pretrained(
        args.model_name_or_path, load_tokenizer=True, local_files_only=True
    )
    model.to(device)

    performances = {
        "exact": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "relaxed": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "overlap": {"p": 0.0, "r": 0.0, "f1": 0.0},
    }

    # change the labels only to include those that you want to evaluate
    labels = list(set([e["label"] for e in data for e in e["labels"]]))

    true_ents = []
    pred_ents = []
    for example in tqdm(data, desc="Processing examples"):
        true_ents.append(
            [label for label in example["labels"] if label["label"] in labels]
        )
        with torch.no_grad():
            _pred_ents = model.predict_entities(
                example["text"], labels, threshold=args.threshold
            )
        pred_ents.append(_pred_ents)

    # evaluate the performance
    for match_type in performances:
        p, r, f1 = evaluate_ner_performance(true_ents, pred_ents, match_type)
        performances[match_type]["p"] = p
        performances[match_type]["r"] = r
        performances[match_type]["f1"] = f1

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(performances, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_test_file",
        type=str,
        help="path to the test data file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to the pre-trained model or model name",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="path to the output file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold for the prediction (default is 0.5)",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="whether to use CPU (default is GPU)",
    )
    args = parser.parse_args()
    main(args)
