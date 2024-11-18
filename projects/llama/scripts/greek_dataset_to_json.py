import json
import re
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def find_date_year(sentence):
    output = []
    pattern = r"\b\d{4}\b"
    matches = list(re.finditer(pattern, sentence))
    if matches is not None:
        for match in matches:
            span = match.span(0)
            value = match[0]
            output.append((value, span))
    if len(output) == 0:
        return ""
    return output


def find_word(sentence, vocabs):
    output = defaultdict(list)
    for key in vocabs:
        for values in vocabs[key]:
            for inside_keys in values:
                for inside_value in set(values[inside_keys]):
                    pattern = r"\b{}(?![\w/])\b".format(re.escape(inside_value.upper()))
                    matches = list(re.finditer(pattern, sentence))

                    if matches is not None:
                        for match in matches:
                            span = match.span(0)
                            value = match[0]
                            output[key.upper()].append((value, span))

    found_dates = find_date_year(sentence)
    if found_dates != "":
        output["DATE"].extend(found_dates)

    if len(output) == 0:
        return ""

    return output


def main(args):
    data = args.data
    json_vocabulary = args.vocabulary
    json_save = args.output

    # Load excell data
    df = pd.read_excel(data)
    df["Personal Medical History upper"] = df["Personal Medical History"].str.upper()
    df = df[df["Personal Medical History"].notna()]

    # Load json vocabulary
    with open(json_vocabulary, "r") as file:
        json_data = file.read()

    json_data = json_data.replace(",]", "]")
    vocabs = json.loads(json_data)
    vocabs["measurements"].remove({"E": ["E"]})
    vocabs["measurements"].remove({"A": ["A"]})

    tqdm.pandas(desc="Progress!")

    df["output"] = df["Personal Medical History upper"].progress_apply(
        lambda x: find_word(x, vocabs)
    )

    df1 = df.loc[df["output"] != ""]

    data = []
    for _, row in df1.iterrows():
        entities = []
        index = 0
        for key, values in row["output"].items():
            for value in values:
                index += 1
                entity = {}
                entity.update({"id": index})
                entity.update({"label": key})
                value = row["Personal Medical History"][value[1][0] : value[1][1]]
                entity.update({"value": value})
                entity.update({"start_index": value[1][0]})
                entity.update({"end_index": value[1][1] - 1})
                entity.update({"entity_connection_id": None})
                entities.append(entity)

        row_to_json = {
            "patient_id": row["Patientid"],
            "text": row["Personal Medical History"],
            "entities": entities,
        }
        data.append(row_to_json)

    with open(json_save + ".json", "w") as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Greek dataset")
    parser.add_argument("--vocab", type=str, default=None, help="Greek vocabulary")
    parser.add_argument(
        "--output", type=str, default=None, help="path to save json data"
    )
    args = parser.parse_args()

    main(args)
