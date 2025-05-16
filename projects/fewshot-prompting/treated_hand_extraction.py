import json
import requests
import re
import argparse
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

OLLAMA_API_URL = "http://localhost:11434/api/generate"

few_shot_examples = [
    {
        "text": "De linkerhand werd behandeld na een breuk.",
        "treated_hand": "linkerhand"
    },
    {
        "text": "Rechterhand werd geopereerd wegens een blessure.",
        "treated_hand": "rechterhand"
    },
    {
        "text": "Beide handen werden behandeld na een val.",
        "treated_hand": "beide handen"
    },
    {
        "text": "Geen behandeling van de handen noodzakelijk.",
        "treated_hand": "none"
    },
    {
        "text": "Geen melding van behandeling van handen.",
        "treated_hand": "none"
    }
]

SYNONYM_MAP = {
    "links": "linkerhand",
    "rechts": "rechterhand",
    "beide": "beide handen",
    "geen behandeling": "none",
    "geen melding": "none",
    "null": "none",
    "": "none"
}


def normalize(label: str) -> str:
    if not label:
        return "none"
    label = label.lower().strip()
    return SYNONYM_MAP.get(label, label)


def evaluate_model(test_data, model_name, temperature):
    results = []

    print("Evaluating model on test set with few-shot structured prompts...")

    for test_sample in tqdm(test_data):
        label_name = test_sample.get("label_to_extract", "treated_hand")
        test_text = test_sample.get("text", "")
        labels = test_sample.get("labels", [])
        true_label = labels[0]["text"] if labels else "none"

        context = ""
        for example in few_shot_examples:
            context += f"Text:\n{example['text']}\nOutput:\n{{\"{label_name}\": \"{example['treated_hand']}\"}}\n\n"

        final_prompt = f"""
You are a medical assistant. Your task is to extract the field '{label_name}' from clinical texts in Dutch.

Respond ONLY with a JSON object of the form: {{"{label_name}": "extracted_value"}}.
Respond with ONE LINE only. Do NOT include explanations, headers, or any additional text.

Here are some examples:

{context.strip()}

Now process this text:
Text:
{test_text.strip()}
Output:
""".strip()

        payload = {
            "model": model_name,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            content = response.json()["response"].strip()

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', content)
                if match:
                    try:
                        parsed = json.loads(match.group())
                    except:
                        parsed = {label_name: "none"}
                else:
                    parsed = {label_name: "none"}

            pred_label = str(parsed.get(label_name, "none") or "none")

        except Exception as e:
            print("Error:", e)
            pred_label = "none"

        results.append((label_name, true_label, pred_label))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM treated hand extraction.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test JSON file.")
    parser.add_argument("--model_name", type=str, default="llama3.2:latest", help="Model name to use.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation.")
    args = parser.parse_args()

    with open(args.test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = evaluate_model(test_data, args.model_name, args.temperature)

    grouped = defaultdict(list)
    for label_name, true, pred in results:
        grouped[label_name].append((normalize(true), normalize(pred)))

    for label_name, pairs in grouped.items():
        y_true = [x[0] for x in pairs]
        y_pred = [x[1] for x in pairs]

        print(f"\nLabel: {label_name}")
        print("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
        print("F1-score (macro):", round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3))
        print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
