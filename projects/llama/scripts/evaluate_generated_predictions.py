import json
from tqdm import tqdm
from argparse import ArgumentParser
from common.evaluate import evaluate_ner_performance

def main(args):
    with open(args.dataset_true, "r", encoding="utf8") as f:
        data_true = json.load(f)
    
    with open(args.dataset_pred, "r", encoding="utf8") as f:
        data_pred = json.load(f)

    performances = {
        "exact": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "relaxed": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "overlap": {"p": 0.0, "r": 0.0, "f1": 0.0},
    }

    # Labels to evaluate
    #labels = ["Age", "Sex", "Biological Structure", "Sign symptom", "Diagnostic procedure", "Lab value", "Detailed description"]
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

    # Filter the true labels
    true_ents = []
    for example in tqdm(data_true, desc="Processing examples"):
        true_ents.append(
            [label for label in example["labels"] if label["label"] in labels]
        )

    # Filter the predicted labels
    pred_ents = []
    for example in tqdm(data_pred, desc="Processing examples"):
        pred_ents.append(
            [label for label in example["labels"] if label["label"] in labels]
        )

    # Evaluate the performance
    for match_type in performances:
        p, r, f1 = evaluate_ner_performance(true_ents, pred_ents, match_type)
        performances[match_type]["p"] = p
        performances[match_type]["r"] = r
        performances[match_type]["f1"] = f1

    print(json.dumps(performances, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_true", type=str, default=0.95, help="Test dataset.")
    parser.add_argument("--dataset_pred", type=str, default=0.95, help="Predicted labels dataset.")
    args = parser.parse_args()
    main(args)